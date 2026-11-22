# Problem Statement: KirchhoffNet Bayesian Optimization Succeeds on Synthetic Benchmarks but Fails on the CTLE Inverse Task

**Date:** 2026-09-01
**Status:** Active investigation — no fix yet; document intended to be self-contained for an external LLM/reviewer with no codebase access
**Codebase:** `kirchhoffnet_realistic` (Windows + AllianceCan Slurm)
**Relevant outputs (authoritative, on disk):** `outputs_alliance_can/` (see §11 for deprecated warm-started directory)
**Absolute paths for key corrections:** `C:\Users\Atharva\Documents\kirchhoffnet_realistic\outputs_alliance_can\kn_bayes_opt\ctle_knet_e400_old_no_experts` (warm-started, deprecated), `C:\Users\Atharva\Documents\kirchhoffnet_realistic\dagger_distillation_trial_0002_recreation_no_vca_gate_shunt.out` (KNet cold-start 14.2%/11.0%), and `C:\Users\Atharva\Documents\kirchhoffnet_realistic\dagger_distillation_mlp_plain.out` (plain-MLP isolation 10.2%/8.1% — proves gap is not MoE-dependent)

---

## 1. Executive Summary

We have two Optuna-driven Bayesian Optimization (BO) drivers that share the same structure:

* `kn_bayes_opt.py` — tunes **KirchhoffNet (KNet)**, an analog-inspired Neural-ODE fabric.
* `mlp_bayes_opt.py` — tunes a standard **MLP baseline** (and a MoE-MLP for CTLE).

On three synthetic direct-regression benchmarks (`friedman1`, `friedman2`, `smooth2d`) **KNet BO outperforms the MLP BO** at matched parameter budgets. On the **CTLE inverse-design task** (4 target specs → 7 physical parameters, feasibility-gated, boundary-heavy) the same KNet BO pipeline — same sampler, same budget logic, same multi-fidelity extension — **reliably loses to the MLP BO**, despite extensive search and a dedicated multi-expert architecture. Crucially, a controlled ablation shows even a **plain MLP (no MoE) at the same parameter budget beats KNet** (10.2%/8.1% vs 14.2%/11.0%), so the failure is not because MoE is doing “extra work” — any MLP suffices. Increasing KNet capacity does not help; the current multi-expert fix is architecturally flawed. The contrast is the core puzzle: the identical optimizer and model succeed on smooth forward regression but fail on this inverse problem.

The question for an external reviewer is: *why does this specific BO loop succeed on smooth forward problems but fail on this inverse, thresholded, DAgger-driven problem, and what should change (search space, objective, model, or training loop) to close the gap?*

---

## 2. System Under Test

### 2.1 KirchhoffNet (KNet) — portable summary

KNet is an analog-inspired Neural ODE. The command below is the exact reference configuration (smooth2d, 5 stages) that the BO starts from; CTLE uses the same switches with different dims/budgets:

```
train_script.py --problem smooth2d --solver heun --epochs 600 --cell-library tanh_free
--hidden-family small_world --num-hidden 25 --num-stages 5 --small-world-k 4 --small-world-p 0.2
--boundary-fan-out '{"0":[0,1,2,6,8],"1":[3,4,5,7,9]}' --interstage-activation residual-relu-tanh
--freeze-read --temporal-readout --leak non-programmable --vca --vca-core --vca-separate-core-bus
```

* **State:** `x ∈ R^{B×N}`, differential voltages `v⁺−v⁻`. Evolution:
  `C_eff·dx/dt = Σ I_in − Σ I_out + I_drive + I_boundary + I_temporal − leak·x − clip(x)` with `C_eff=1`, `x_max=3`, `clip=0.05·(σ((x−x_max)/0.02)−σ((−x−x_max)/0.02))`.
* **Solver:** Fixed-step Heun (predictor-corrector, 2nd order), direct BPTT (no adjoint). `t_span=7.0`, `num_steps=70` split equally over stages → per-stage `t≈1.4`, 14 steps, `dt=0.1`.
* **Pipeline `KirchhoffNetWithIO` — Write / Evolve / Read:**
  - *Write:* This command uses BOUNDARY mode — no InputMapper; `x0=zeros(B,N_hidden)` (NullInputMapper, `hidden=25`, accumulator `+1` → `N=26` per stage). Input `u` enters ONLY via boundary OTA edges.
  - *Evolve:* 5 sequential `DifferentialStage` + `StageTransfer` (`x=stage.forward_heun(x,u,1.4,14)`, then `x=transfer(x)` = `x + W2·tanh(W1·x)+ReLU` residual).
  - *Read:* `y = OutputAffine(x[:, read_slice])`, `read_slice` = last node (accumulator); `y=gain·x+bias+tanh_gain·tanh(x)+relu_gain·ReLU(x−thr)`.
* **Per-stage graph:** `N=26` (25 hidden + 1 ODE accumulator). Hidden edges: Watts-Strogatz `small_world(n=25,k=4,p=0.2,seed=100)` → ~50 directed edges, `edge_repeats=2` → 100 core edges/stage with independent params.
  Each core edge: `I_e = gate_e·[I_tanh(Vsrc,Vdst)+G·(Vsrc−Vdst)]`, `gate_e=σ(z_logits[e])`, `z_logits` init 0→0.5. Cell `tanh_free`: `I_tanh=Isat·(tanh(u)+0.01·u)`, `u=gm·(s·(A·Vsrc−B·Vdst)+theta)`, `A=softplus(a_raw)`, `B=softplus(b_raw)`, `s=STE(sign(s_raw))=±1`, `gm/Isat∈[0.01,10]` via sigmoid, plus resistive shunt `G=softplus(g_raw)≈0.0067`.
* **Boundary fan-out:** 10 sparse OTA edges: `in0→[0,1,2,6,8]`, `in1→[3,4,5,7,9]`; `I_boundary=σ(boundary_z)·VCA_boundary(u)·I_OTA(u[src],x[dst])`, ideal source.
* **Temporal readout:** Accumulator node evolves with leak+clip; edges `hidden→accumulator` are sparse OTAs hidden→accumulator; final `x[acc]` read by OutputAffine.
* **Freezing (`--freeze-read`):** Core family frozen at stage entry `x0`: `i_edge_const=scatter(σ(z)·I_tanh(x0)·VCA_core(u))` computed once per stage, reused per Heun substep; resistive/boundary/temporal stay dynamic. Enables larger linear integration step.
* **VCA (`--vca --vca-core --vca-separate-core-bus`):** Low-rank input-driven gate `∈(0,2)`, identity `1.0` at init (`W=0`, `v∼N(0,0.01)`, `gate=2σ((u@W)·v)`). Separate core bus `W_core∈R^{2×rank}` isolated; applied as `I_e*=gate`; resistive `G` untouched unless `--vca-gate-shunt`.
* **Param count:** ~100 core·6 + 10 boundary·6 + 25 temporal·6 + VCA projections + OutputAffine → ~700/stage ×5 ≈3500 (leak non-programmable saves ~130). Configurable `t_span/num_steps`, solver, mapper/struct/dyn LR scales (`--struct-lr-scale 4` for `z` logits).

Pseudocode RHS:
```python
def rhs(x,u):
    acc = frozen_core_const if freeze_read else scatter(σ(z)*I_tanh_core(x))
    acc += resistive_shunt(x)*σ(z)
    acc += scatter_boundary(u,x)*VCA_bound(u)
    acc += scatter_temporal(x)*VCA_readout(u)
    return (acc - 0.0486*x - clip(x))/C_eff
```

For CTLE, `in_dim=4` specs → `out_dim=7` physical params, `x_max=4.0`, `num_hidden∈[10,20]`, `fanout_count` fixed to 2, `num_stages∈[2,5]` (search), VCA rank ∈{2,3,4}.

### 2.2 MLP baseline (for contrast)

* Direct regression tasks: plain MLP, hidden dim derived analytically from `--param-budget` and `num_layers` via `params=(L-2)h²+(in+L-1+out)h+out` to keep budget exact; BO tunes `lr`, `weight_decay`, `batch_size`, `activation`, `loss`.
* CTLE: `RegimeAwareMoE` via `generative-distillation-improved-dagger-nuance-mlp.py` — shared trunk (`width=44±`, `layers=2–3`) + `E=2–4` expert heads (each head is a full 7-parameter design), two bias-free `8→E` routing heads, `moe_param_count = 9W+(L-1)(W²+W)+16E+E·(7W+7)`. Hard WTA at inference, Gumbel-softmax in training, top-K re-ranking via physical verifier. This is the model KNet must beat.

### 2.3 BO drivers (shared mechanics, different search spaces)

* Sampler: `optuna.samplers.TPESampler(multivariate=True, group=True)`, seed 0 or 100.
* Pruner: `NopPruner` for synthetic; `SuccessiveHalvingPruner(min_resource=rung0, reduction_factor=2)` for CTLE multi-fidelity (rungs `1,2,4` DAgger iterations).
* Artifacts per trial: `trial_NNNN/log.txt`, `final_metrics.txt`, SQLite study `.db`, `best_hyperparams.txt`, `results.csv`, `objective_history.png`.
* Fixed budget per trial: `--epochs 400` (synthetic, KNet) / `800` (MLP synthetic, CTLE budgets); CTLE proxy is `4×100 = 400` epochs across 4 DAgger iterations, evaluated every 5 epochs, common eval 1000 specs.
* **Known bug / leakage:** Historical CTLE objective minimized **test** failure rate (`Test failure rate: X%` parsed from DAgger log). Validation failure rate (`Validation failure rate: X%`) was reported for pruning but not used as the final selection objective. Thus repeated BO on the same held-out test set is optimistic (test leakage). New multi-fidelity code reports validation for pruning but still returns the penalized test rate as the Optuna value.
* Recovery: abandoned `RUNNING` trials (AllianceCan Slurm preemption) are marked `FAIL` and re-enqueued via SQLite storage hack.
* Param-budget gating: `--count-params-only` preflight before training; over-budget returns finite `_over_budget_objective = 1e6*(params/budget)²` so Optuna learns the boundary instead of `inf`.

---

## 3. Tasks

### 3.1 Synthetic direct-regression benchmarks (where BO works)

* `friedman1` (10→1), `friedman2` (4→1), `smooth2d` (2→1), `housing` (8→1). Smooth, stationary data, Huber loss, standardized targets, single training phase, early stopping disabled, deterministic split.
* Noise: `friedman*` add `--target-noise-std 1.0`.
* KNet START_POINTS (validated 800-epoch configs) enqueued as trial 0 via `study.enqueue_trial` with exact boundary fan-out maps; subsequent trials explore 15 dims.

Evidence that KNet BO **outperforms** here (400-epoch KNet vs 800-epoch MLP, matched budgets):

* `friedman1_knet_e400` — `best_hyperparams.txt` exists, `best_trial=43` (NOT the seed), `best_val=0.000405` (Huber), `actual_params=7267`, `budget=7000`, params within 20% tolerance. **Beats** `friedman1_budget7000_e800` MLP (`best_val=0.000420`, `hidden=55, L=4, params=6821`).
* `friedman2` and `smooth2d` show the same pattern: KNet BO finds non-seed optima that beat the MLP BO at the same parameter budget (both have 50+ trials; logs show many non-pruned completions, smooth objective descent). The full winning margins are in `outputs_alliance_can/kn_bayes_opt/friedman*_knet_e400/best_hyperparams.txt` vs `outputs_alliance_can/mlp_bayes_opt/*_budget*_e800/best_hyperparams.txt`.
* This is not marginal or seed-lucky: on all three synthetic tasks the KNet-found configuration beats the MLP-found configuration at the same budget, despite MLP having 2× the epoch budget (800 vs 400) on those runs.

Qualitative: On these problems BO finds non-seed optima, objective curves descend smoothly, and capacity changes have monotonic/predictable effects. The 15-dim KNet space (topology+solver+optimizer+physics) is searchable by TPE **for forward regression**.

### 3.2 CTLE inverse-design benchmark (where BO fails)

**Problem definition:** Inverse of a physical equalizer: given 4 target specs (power, jitter, height, width — log-scaled inputs), predict 7 physical design parameters (`fW, current, ind, Rd, Cs, Rs, VDD`) bounded via `10^(log_lo+(log_hi−log_lo)·σ(logit))` with ranges e.g. `fW [1e-6,10]`, `current [5e-4,2.5]`, etc. Many-to-one (multiple designs can meet the same spec). Feasibility is judged by a **physical verifier (ZIG/CTLE)**: each predicted design is simulated; failure = any of the 4 specs violates threshold/cap criteria. **Failure rate** = fraction of 1000 held-out specs where the predicted design fails (lower is better). Reported stratified: aggregate, boundary, interior. **Boundary** cases dominate difficulty (near feasibility frontier; small output shifts flip pass/fail).

**Training:** DAgger distillation (`dagger-nuance-distillation-kirchhoffnet.py` / `generative-distillation-improved-dagger-nuance-mlp.py`):

* Flow teacher: conditional spline flow `I → 7 params` + ZIG validity/consistency filtering; k-NN fallback.
* DAgger loop: train student → find failed specs → query flow teacher for new labels after ZIG validity (`threshold 0.5`) + consistency filtering (strict/relaxed) → augment dataset (20k→~20.3k) → repeat, 4 iterations total, LR cosine restart each iter.
* Shared `ctle_dagger_common.py` (131KB) holds `setup`, `run_dagger_training`, `BoundedMLP`, `HybridHurdleModel`, `ConditionalSplineFlow`, etc. CTLE uses pure distillation (no Phase 2 fine-tuning).
* Common eval: 1000 specs (`--common-eval-size 1000`) evaluated every 5 epochs; best checkpoint per iteration restored for reporting.
* Outputs logged: `Validation failure rate: X%` (shared COMMON_EVAL, post-restore per iteration) and `Test failure rate: X% (boundary Y% interior Z%)`.

**Success criterion for KNet:** Beat MLP baseline at comparable budget:

| Model | Best trial | Validation (aggregate) | Test (aggregate) | Params | Boundary val | Boundary test |
|---|---|---:|---:|---:|---:|---:|
 | **MLP MoE** (`mlp_bayes_opt/ctle_budget7000_e800`, 26 trials) | `0021` | **5.1%** | **4.6%** | 5,573 | 8.0% | 8.61% |
| | `0011` | 5.6% | 4.7% | ~5k | — | — |
| | `0020` | 6.9% | 5.3% | — | — | — |
| **MLP plain (no MoE) — isolation ablation** (`dagger_distillation_mlp_plain.out`) | plain W48 L3 | **10.2%** | **8.1%** | 5,479 | 15.2% | 13.65% |
| **KNet non-expert — BO logs as reported** (`kn_bayes_opt/ctle_knet_e400_old_no_experts`, 22 logs) | `0002` | **7.4%** | 6.9% | 7,019* | — | — |
| | `0015`† | 7.9% | **5.1%** | 2,489 | — | — |
| | `0022` | 7.9% | 6.9% | 2,693 | — | — |
| | `0006` | 8.7% | 5.9% | 4,277 | — | — |
| **KNet non-expert — corrected recreation (no warm start)** | `0002` recreation | **14.2%** | **11.0%** | 7,019 | 24.6% | 24.93% |
| **KNet expert (MoE-style VCA-gated heads)** (`kn_bayes_opt/ctle_knet_e400`, 11 logs, multi-fidelity) | `0002` | **6.4%** | **5.2%** | 4,379 | 9.8% | 9.79% |
| | `0010` | 8.1% | 6.9% | 4,329 | — | — |
| | `0000` (seed of 0015) | 8.4% | 9.1% | 2,898 | — | — |

\* Exceeds nominal 6k target (CTLE `default_budget=6000`).
† Compact non-expert trial `0015` is the strongest KNet by test score **in the BO logs**; config:
```
kn-num-stages=2, kn-num-hidden=11, kn-small-world-k=2, kn-small-world-p=0.2,
kn-vca-rank=3, kn-x-max=4.0, boundary fan-out {"0":[0,4],"1":[1,5],"2":[2,6],"3":[3,7]},
lr=7.04e-3, weight-decay=1.53e-6, batch-size=1024, t-span=6.729, seed=0 (repro exact)
```

> **⚠️ Correction — warm-start artifact in `ctle_knet_e400_old_no_experts`:**
> All results in `outputs_alliance_can/kn_bayes_opt/ctle_knet_e400_old_no_experts` (including trials `0002`, `0015`, `0022` above) **mistakenly used a warm start** (checkpoint resumption / carried-forward state across BO trials). They are **not valid** cold-start measurements and are optimistically biased.
> The **corrected, cold-start** performance for the best-reported KNet config (trial `0002`: `stages=4, hidden=16, k=2, rank=3, xmax=4, lr=3.83e-3, wd=1.30e-4, batch=1024, t-span default`) is in:
> `dagger_distillation_trial_0002_recreation_no_vca_gate_shunt.out` (identical hyperparams, `KN_VCA_GATE_SHUNT=false`, cold start, 4×100 DAgger, 7019 params):
> - Iter 1 val 25.6% → Iter 2 val 17.5% → Iter 3 val 14.2% → Iter 4 val **14.2%** (boundary 24.6% / interior 3.8%)
> - Final **Test 11.0%** (boundary 24.93% / interior 3.92%)
> A second recreation with `VCA_GATE_SHUNT=true` (`dagger_distillation_trial_0002_recreation_vca_gate_shunt.out`) is even worse: val **22.9%**, confirming the warm-start inflation was ~7–8 pts val / ~4 pts test. **All conclusions about KNet CTLE must use 14.2%/11.0% (not 7.4%/6.9%) as the true non-expert baseline.** The `0015` 7.9%/5.1% log is likewise invalid; no cold-start recreation of `0015` has been run yet, so its true performance is unknown but expected to be substantially worse than logged.

> **MoE isolation check:** To test whether the MLP win depends on the MoE architecture, a **plain single-head MLP** (no MoE, no routing, no Gumbel-softmax, no per-expert heads) was trained with the **same parameter budget as the best MoE** and identical DAgger settings (`4×100`, flow teacher, ZIG filtering, `20405` final dataset). Config: `--param-budget 5559 → derived width 48` (`plain_param_count=5479, L=3, act=silu, LN=False, log_features=True`), `5480` trainable / `5493` total params, matched to MoE `0021`’s 5573. Log: `C:\Users\Atharva\Documents\kirchhoffnet_realistic\dagger_distillation_mlp_plain.out`:
> - Final **Val 10.2%** (102/1000, boundary 15.2% / interior 5.2%) — plateaued from epoch ~53 (`10.2%` best, no recovery in iters 2–4, iter 4 divergence-aborted at ep 10)
> - Final **Test 8.1%** (81/1000, boundary 13.65% / interior 5.28%)
> - vs MoE `0021` 5.1%/4.6% (MoE helps ~5 pts val, ~3.5 pts test) **but vs KNet corrected 14.2%/11.0% the plain MLP still wins by 4.0 pts val / 2.9 pts test**. The KNet failure is therefore **not** because MoE is “doing the heavy lifting” — even the simplest MLP at the same budget beats KNet.

**Best expert `0002` (for reference, cold-start, not warm-started):**
```
stages=2, hidden=20, k=2, vca-rank=2, experts=2, gate-rank=3, top-k=2, xmax=4,
lr=9.16e-3, weight-decay=1.74e-6, batch=512, t-span=4.01, seed=100
val 6.4% (boundary 9.8% / interior 3.0%), test 5.2% (boundary 9.79% / interior 2.87%), params 4379
```
Note: expert `0002` is still apparently better than the corrected non-expert `14.2%/11.0%`, but the expert and non-expert were not run under identical conditions (different seed, BO process, multi-fidelity), so this is not a controlled architecture comparison.

Key observations (updated for warm-start correction + plain-MLP isolation):
* **True KNet CTLE gap is much larger than BO logs suggest:** Corrected non-expert best is `14.2%/11.0%` vs MLP MoE `5.1%/4.6%` — a **9.1 pt validation gap and 6.4 pt test gap**, not the 1.3/0.6 pt gap implied by the warm-started logs. Even vs **plain MLP** `10.2%/8.1%` the gap is **4.0 pts val / 2.9 pts test**. The BO was selecting on inflated numbers.
* KNet expert `6.4%/5.2%` still loses to MLP MoE `5.1%/4.6%` (by 1.3/0.6 pts) but now appears to beat the corrected non-expert — however this may reflect the warm-start artifact rather than a real architectural win. It also still loses the isolation test: plain MLP `10.2%/8.1%` already beats corrected KNet, so MoE is not required to beat KNet.
* Plain-vs-MoE gap quantifies MoE value: MoE `5.1%/4.6%` beats plain `10.2%/8.1%` by **5.1 pts val / 3.5 pts test** at matched budget (~5.5k params, `W=48 L=3` plain vs `W≈45 L=3 E=3` MoE). MoE helps, but plain alone suffices to beat KNet — KNet is losing to the simplest MLP, not just to a sophisticated MoE.
* Larger-capacity KNet ablation (stages 4, hidden 13, k=4, rank3, xmax6) → **9.0% validation in warm-started logs** (worse than compact `7.9%`); its cold-start value is unknown but likely >14%. More params ≠ better.
* Seed confound: expert run 0 is the seed of old `0015` but with `seed=100` and MoE heads → `8.4%/9.1%` (much worse than the warm-started `7.9%/5.1%`), so expert-vs-non-expert is not a controlled comparison (seed + BO process + multi-fidelity differ).

---

## 4. What Has Already Been Tried (and Why It Didn't Close the Gap)

1. **Multi-fidelity SuccessiveHalving (`1→2→4` iterations):**
   - Implemented in `kn_bayes_opt.py`; trial runs 1 DAgger iter, reports validation to Optuna, pruned if uncompetitive; promoted trials `resume` from `<output>/dagger_checkpoint.pt` (fixed so standalone commands update checkpoint path). Uses `SuccessiveHalvingPruner`. Rungs parsed by `_ctle_fidelity_rungs`, always appending full budget.
   - *Status:* `py_compile` passes, smoke-tested, but **has not demonstrated better CTLE BO**; CTLE runs remain sparse (11 vs 22 trials) due to cluster preemptions and lack of local Optuna env. Pruning is validation-based but final objective still test-based (leakage persists).

2. **Multi-expert KNet (shared fabric + VCA-gated readout banks):**
   - One shared KNet fabric + `M=2–3` seven-output expert heads + low-rank VCA gate bus (shared `W∈R^{4×rank}`, per-expert `v_expert∈R^{rank}`, `gate=2σ((u@W)·v)`), hard argmax at inference, Gumbel-softmax in training, optional top-K + ZIG re-ranking, auxiliary load-balancing + diversity losses.
   - *Flaw (identified):* Main spec/physics/invalid/manifold loss applied **only to the routed/selected expert**; other heads get only diversity regularization, not independent feasibility training. So it is a gated ensemble where only one head is substantively trained per example — alternate candidates may be invalid, making top-2 re-ranking ineffective. Not a true multi-hypothesis inverse model (needs per-expert feasibility via best-of-M / Hungarian / assignment loss + per-expert verifier loss; gate learns to rank feasible candidates; no averaging of physical vectors).

3. **Capacity increase:** Directly scaling `num_hidden/k/stages/rank/xmax` hurt (9.0% vs 7.9%).

4. **Parser caveat discovered:** A standalone “large KNet” command used unsupported flags (`--kn-edge-repeats`, `--kn-cell-library`, `--kn-leak-mode`, `--kn-interstage-activation`, `--kn-freeze-read`, `--kn-temporal-readout`) that were **silently ignored** via `parse_known_args`. Only explicitly defined flags should be assumed active — many apparent ablations were no-ops.

---

## 5. Why Synthetic BO Works — Candidate Explanations

* Smooth, unimodal landscape with stationary target distribution; Huber/MSE objective correlates monotonically with params and LR.
* Single-phase training (no DAgger distribution shift); gradients stable, TPE can model the space.
* Inputs directly drive the fabric via boundary fan-out; no inversion, no feasibility cliff, no verifier discontinuity.
* KNet inductive bias (continuous ODE, small-world diffusion, VCA gates) matches smooth regression.

---

## 6. Hypotheses for Why CTLE BO Fails

An external reviewer should weigh these (non-exclusive) hypotheses against the evidence above and the KNet summary in §2.1:

**H1. Inverse many-to-one + thresholded objective → discontinuous BO landscape.**
  - CTLE maps `4→7`; many designs satisfy the same spec; failure rate is a 0/1 threshold on continuous verifier output, so small KNet output changes near the boundary flip 8% of cases. TPE’s smooth GP/kernel assumption is violated; validation signal is noisy and boundary-dominated (see §3.2: boundary 8–9% vs interior 2–3%). Synthetic’s MSE is smooth; CTLE’s failure rate is not.

**H2. DAgger nonstationarity makes BO trials incomparable/noisy.**
  - Each trial’s training distribution grows with its own failures (20k→20.3k via teacher). Two identical KNet configs with different random ordering see different DAgger augmentations; SuccessiveHalving at rung 1 (validation after 1 iter) may not predict rung 4 ranking. MLP faces the same but may be more robust to shift (simpler optimization).

**H3. Inductive bias mismatch: KNet (continuous diffusion ODE) vs piecewise multi-regime inverse.**
  - MLP MoE naturally partitions the spec space (`8` log features, trunk + per-expert heads). KNet’s fabric is a single dynamical system with low-rank VCA modulation; even with `rank=2–4`, it may not represent the discontinuous regime boundaries (e.g., feasibility cliffs) without many stages or explicit branching. The free-ODE accumulator + residual interstage may be insufficiently expressive for inversion.
  - **Update from plain-MLP isolation (§11):** Plain MLP (no MoE, no partitioning) *still* beats KNet (10.2%/8.1% vs 14.2%/11.0%), so the failure cannot be *solely* “MoE partitions but KNet cannot.” However, MoE *does* add 5.1/3.5 pts over plain (5.1%/4.6% vs 10.2%/8.1%), so piecewise expertise helps — KNet loses to both, just by a larger margin to MoE. The residual gap to plain suggests a more fundamental representation/optimization issue beyond expert count.

**H4. Search-space / budget misallocation.**
  - CTLE KNet searches 15 dims including `t_span∈[3,7]`, `num_stages∈[2,5]`, `vca_rank∈{2,3,4}`, `gm/isat_max∈[1,50]` but `x_max` is pinned to 4.0 (not searched), `small_world_p=0.2` fixed, `fanout_count` fixed to 2. MLP searches only 5 dims (`width fraction`, `layers`, `E`, `lr`, `wd`) at exact param budget. KNet’s 15-dim space at 11–22 trials is severely under-explored; TPE needs ~50 trials to beat the seed on synthetic but CTLE never got there. Param-count penalty (`1+0.25·params/budget`) and soft cap `budget*(1+0.1)` may also mis-penalize compact winners (best KNet `0015` is 2489 params — very small).

**H5. Optimization / credit assignment failure (multi-expert).**
  - §4.2 flaw: only the routed expert gets the feasibility loss; diversity loss pushes others away without making them valid. Best-of-M or Hungarian assignment per spec would be needed, plus per-expert ZIG loss. Current gate learns to pick the one good head; top-2 re-ranking is thus vacuous. Also `parse_known_args` silent drops mean some “improvements” were never active.

**H6. Evaluation methodology (test leakage + boundary weighting + warm-start bias).**
  - BO minimizes penalized **test** failure rate, so the reported gaps mix model quality with test-set overfitting. Validation is the correct pruner/selector; test should be held out for finalists only and reported stratified + multi-seed. **Crucially, the warm-started KNet logs inflated performance by ~7 pts val:** the true cold-start gap is `14.2% vs 5.1% val (9.1 pts)` and `11.0% vs 4.6% test (6.4 pts)`, not `7.4% vs 5.1%`. MLP’s advantage is clearest on boundary (8.0% vs 24.6% val cold-start, 8.61% vs 24.93% test) — aggregate alone hides where KNet fails.

**H7. Teacher / labeling loop favors MLP.**
  - Both use the same flow teacher and ZIG filtering, but `filter_by_zig_consistency` (52–60% pass) and `current` saturation (`39% below log_lo` in DAgger labels) may produce a label distribution that MLP fits easily (bounded sigmoid output) while KNet’s ODE output struggles with edge saturation (`sat_lo 13.7%` on test). MLP’s explicit `BoundedMLP.scale_input` (log→StandardScaler/Q75) vs KNet’s `LocalKirchhoffStudentWrapper.scale_input` (log→min-max→clamp `[-4,4]`) differ.

---

## 7. What We Need From an External Reviewer

Without codebase access, please comment on:

1. **Which hypotheses (H1–H7) best explain the synthetic-vs-CTLE gap** given the numbers and KNet inductive bias? Are any ruled out?
2. **Is the current 15-dim KNet CTLE search space sensible**, or should it be trimmed/reshaped (e.g., fix `gm/isat`, search `x_max`/`t_span` more narrowly, add `num_stages=1` or `vca_rank=1`)? What would you change first?
3. **What is the minimal controlled ablation** to separate architecture vs noise vs seed (e.g., `M=1` vs `M=2 top-1` vs `M=2 top-2` at fixed trunk/seed/budget, as proposed in §8)?
4. **How would you fix the multi-expert training** to be a true multi-hypothesis inverse model without averaging physical designs? Concrete loss (best-of-M, Hungarian, diverse teacher labels, per-expert verifier) and gate training prescription?
5. **Should CTLE BO optimize validation instead of test**, and how should boundary vs interior be weighted (e.g., regime-aware validation, boundary-heavy objective, or joint reporting)?
6. **Does KNet’s ODE formulation need a structural change for inverse tasks** (e.g., direct inverse mapping vs forward verifier in the loop, different readout, or hybrid KNet→MLP head) or is it a hyperparameter/training issue?
7. **What additional diagnostics** would you request (e.g., learning curves, failure-case embedding, per-spec logit histograms, teacher-label coverage)?

Ideal response: ranked hypotheses, concrete next 3 experiments (≤1 GPU-day each), and a sketch of a corrected multi-expert loss that an engineer could implement from this doc alone.

---

## 8. Recommended Next Actions (Current Plan, Not Yet Executed)

* **Under time pressure:** Do not run broad expert BO. Retain expert trial `0002` (6.4%/5.2%, cold-start valid), the **corrected** non-expert recreation `0002_no_vca_gate_shunt` (14.2%/11.0%, cold-start valid), and the **plain-MLP isolation** (`dagger_distillation_mlp_plain.out`, 10.2%/8.1%, 5479 params, no MoE) as baselines — **do not use** `ctle_knet_e400_old_no_experts` warm-started numbers (`0015` 7.9%/5.1% etc.) for decisions. The plain-MLP result proves the MLP advantage is not MoE-dependent. Focus on boundary failure (KNet 24.6% vs plain 15.2% vs MoE 8.0% val).
* **Minimal controlled ablation (1 GPU-day):** Same trunk/param budget/data/DAgger schedule at seeds `0` and `100`:
  1. `M=1` non-expert.
  2. `M=2, gate-rank 2–3, top-1` (hard WTA).
  3. `M=2, top-2 + physical ZIG/CTLE re-ranking`.
  Keeps everything fixed except `M`/routing; isolates (a) do experts help, (b) does routing hurt, (c) does re-ranking help, (d) seed noise.
* **If multi-hypothesis revisited:** Generate diverse teacher labels per spec, train every head to be feasible via best-of-M / Hungarian, apply per-expert spec/physics/validity loss (not just diversity), train gate to rank feasible candidates, deploy WTA or top-2 + verifier, never average 7-param vectors.
* **Methodology cleanup:** Optimize common validation (not test), reserve test for frozen finalists, report aggregate/boundary/interior, match budgets, multi-seed.

---

## 9. Reproducibility Pointers (for an engineer with repo access)

* KNet BO: `python kn_bayes_opt.py --dataset ctle --ctle-dagger-iterations 4 --ctle-epochs-per-iter 100 --ctle-common-eval-size 1000 --ctle-multifidelity --ctle-fidelity-rungs 1,2 --seed 0` (new) or `--seed 100` + multi-fidelity disabled for legacy. Output: `outputs_alliance_can/kn_bayes_opt/ctle_knet_e400[_old_no_experts]/trial_NNNN.log.txt`. **Caveat:** `ctle_knet_e400_old_no_experts` is warm-started and invalid; use `dagger_distillation_trial_0002_recreation_no_vca_gate_shunt.out` (cold-start, 14.2% val / 11.0% test) as the true KNet baseline.
* Plain-MLP isolation (proves gap not MoE-dependent): single-head MLP, no routing, `--param-budget 5559 → W=48 L=3 silu LN=False log_features=True` (5479 params, matched to MoE 5573), same DAgger `4×100` / same teacher / same ZIG filtering. Log: `C:\Users\Atharva\Documents\kirchhoffnet_realistic\dagger_distillation_mlp_plain.out` (Val 10.2% / Test 8.1%, boundary 15.2%/13.65% — still beats KNet 14.2%/11.0%).
* MLP BO: `python mlp_bayes_opt.py --dataset ctle --param-budget 7000 --epochs 800 --ctle-dagger-iterations 4 --ctle-epochs-per-iter 100 --ctle-common-eval-size 1000`; output: `outputs_alliance_can/mlp_bayes_opt/ctle_budget7000_e800/trial_NNNN/log.txt`.
* Single KNet DAgger (exact trial 15 repro): use `dagger-nuance-distillation-kirchhoffnet.py` with `kn-num-stages 2, kn-num-hidden 11, kn-small-world-k 2, kn-small-world-p 0.2, kn-vca-rank 3, kn-x-max 4.0, boundary-fan-out '{"0":[0,4],"1":[1,5],"2":[2,6],"3":[3,7]}', lr 7.04e-3, wd 1.53e-6, batch 1024, t-span 6.729, seed 0` — note: flags outside `dagger-nuance-distillation-kirchhoffnet.py`’s parser are silently dropped (see §4.4).
* KNet vs MLP BO comparison scripts: `kn_bayes_opt.py:972` (CTLE branch), `mlp_bayes_opt.py:481` (CTLE branch), `ctle_dagger_common.py:setup/run_dagger_training`.
* Metric parsing: `Validation failure rate: X%` (shared COMMON_EVAL, per iteration) vs `Test failure rate: X% (boundary Y% interior Z%)` (final).

---

## 10. Appendix: Full KNet Portable Summary (Reference)

*(Verbatim portable summary provided by author — sufficient to reason about the model without code):*

> KirchhoffNet (KNet) — Portable Summary for LLM — Command: `train_script.py --problem smooth2d --solver heun --epochs 600 --cell-library tanh_free --hidden-family small_world --num-hidden 25 --num-stages 5 --small-world-k 4 --small-world-p 0.2 --boundary-fan-out '{"0":[0,1,2,6,8],"1":[3,4,5,7,9]}' --interstage-activation residual-relu-tanh --freeze-read --temporal-readout --leak non-programmable --vca --vca-core --vca-separate-core-bus`
>
> 1. What it is — Analog-inspired Neural ODE. State `x∈R^{B×N}` diff voltages `v⁺−v⁻`. `C_eff·dx/dt=ΣI_in−ΣI_out+I_drive+I_boundary+I_temporal−leak·x−clip(x)`, `C_eff=1,x_max=3, clip=0.05·(σ((x−x_max)/0.02)−σ((−x−x_max)/0.02))`. Heun 2nd order + direct BPTT, `t_span=7.0,num_steps=70` → per-stage `t≈1.4, steps=14, dt=0.1`.
> 2. Pipeline `KirchhoffNetWithIO — Write/Evolve/Read` — WRITE: BOUNDARY mode, `x0=zeros(B,N_hidden)`, `output_ode_count=1` accumulator, `N=26` per stage, `u` via boundary OTAs only. EVOLVE: 5× `DifferentialStage+StageTransfer` (Heun + `x+W2·tanh(W1·x)+ReLU`). READ: `y=OutputAffine(x[:,read_slice])`, `read_slice` last node, `y=gain·x+bias+tanh_gain·tanh+relu_gain·ReLU(x−thr)`.
> 3. Per-stage `DifferentialStage` — `num_nodes=26` (25 hidden+1 accumulator). Hidden edges: Watts-Strogatz `small_world(n=25,k=4,p=0.2,seed=100)` → `E≈n·k/2=50` directed, `edge_repeats=2` →100 core edges/stage. Each: `I_e=gate_e·[I_tanh+G·ΔV]`, `gate=σ(z)`, `cell tanh_free: I_tanh=Isat·(tanh(u)+0.01u)`, `u=gm·(s·(A·Vsrc−B·Vdst)+theta)`, `A/B=softplus`, `s=STE(sign)`, `gm/Isat∈[0.01,10]` via σ, `G=softplus≈0.0067` shunt. Compliance gate `σ((x_max−|V|)/0.02)`. `leak=0.0486` fixed.
> 4. Boundary fan-out — 10 sparse OTAs `in0→[0,1,2,6,8] in1→[3,4,5,7,9]`, `I_boundary=σ(z)·VCA_bound(u)·I_OTA`.
> 5. Temporal readout — Accumulator OTAs `hidden→acc`, `acc[dst]+=I_temporal`, `src` not drained, leak+clip, read `x[acc]`.
> 6. Freezing `--freeze-read` — Core `i_edge_const=scatter(σ(z)·I_tanh(x0)·VCA_core(u))` folded once per stage; resistive/boundary/temporal dynamic.
> 7. VCA `--vca --vca-core --vca-separate-core-bus` — Low-rank gate `∈(0,2)` identity init `W=0,v∼N(0,0.01)→gate=1`; shared bus `gate_bound=2σ((u@W)·v_bound)` `W∈R^{2×rank}`, separate core `gate_core=2σ((u@W_core)·v_core)` cached+folded if frozen; `I_e*=gate` (nonlinear only, shunt untouched).
> 8. Training knobs — `heun`, `residual-relu-tanh`, mapper/struct/dyn LR scales `1.0/4/1.0`, `no-parallel`, `grad-log`, `seed 100`, param ~3500.

---

## 11. Critical Caveats: Warm-Start Inflation + MoE Isolation

* **Warm start — what happened:** `outputs_alliance_can/kn_bayes_opt/ctle_knet_e400_old_no_experts` was run with an unintended warm start (previous trial checkpoint / optimizer state carried forward). This makes every reported failure rate there artificially low. The directory should be considered **deprecated** for CTLE conclusions.
* **Warm start — ground truth:** A faithful cold-start recreation of its best config (trial `0002`, 7019 params, `stages=4 hidden=16 k=2 rank=3`) was run explicitly:
  - `dagger_distillation_trial_0002_recreation_no_vca_gate_shunt.out` — **Val 14.2% / Test 11.0%** (boundary 24.6%/24.93%) — this is the correct number to quote.
  - `dagger_distillation_trial_0002_recreation_vca_gate_shunt.out` — Val 22.9% / Test ~? (worse, with `--vca-gate-shunt` on).
* **Warm start — implication:** The “6.9% vs 4.6%” narrative in earlier handoffs understates the KNet failure by ~4–8 pts. The real gap is **14.2% vs 5.1% val (9.1 pts) and 11.0% vs 4.6% test (6.4 pts)**. Any BO analysis, expert comparison, or capacity ablation that used the old logs must be re-evaluated. The synthetic outperform-vs-CTLE underperformance contrast is therefore *stronger* than it first appeared.
* **MoE isolation — was MoE the reason MLP wins? No.** A **plain MLP (no MoE)** at the **same parameter budget** was trained to isolate architecture: `C:\Users\Atharva\Documents\kirchhoffnet_realistic\dagger_distillation_mlp_plain.out` (`--param-budget 5559 → W=48 L=3 silu`, 5479 params, matched to MoE `0021` 5573, same `4×100` DAgger / same teacher / same ZIG filtering, `outputs/dagger_output_plain_mlp_w48`).
  - Plain result: **Val 10.2% / Test 8.1%** (boundary 15.2%/13.65%, interior 5.2%/5.28%)
  - MoE result: **Val 5.1% / Test 4.6%** — MoE adds **5.1 pts val / 3.5 pts test** over plain at same budget, so MoE is valuable.
  - But plain **still beats KNet corrected 14.2%/11.0% by 4.0/2.9 pts** (and beats warm-started KNet logs too). The MLP→KNet advantage is therefore **not MoE-dependent**; even the simplest MLP suffices. Hypotheses about MoE-specific inductive bias (e.g., “only MoE can handle multi-regime inversion”) are ruled out as the *sole* explanation — the gap must be explained by something plain MLP also has and KNet lacks (optimization, representation, or input scaling, not just expert count).

---

## 12. Document History

* 2026-09-01 (v1): Initial problem statement compiled from handoff + `outputs_alliance_can` verification (KNet old non-expert 22 trials, expert 11 trials, MLP 26 trials) + `kn_bayes_opt.py`/`mlp_bayes_opt.py`/`ctle_dagger_common.py` inspection. Intended for external LLM review before next compute spend.
* 2026-09-01 (v2): Corrected CTLE KNet baseline per `dagger_distillation_trial_0002_recreation_no_vca_gate_shunt.out` (14.2%/11.0% cold-start vs 7.4%/6.9% warm-started). Added explicit synthetic-outperforms contrast and new §11 warm-start caveat. Old `ctle_knet_e400_old_no_experts` results marked deprecated.
* 2026-09-01 (v3): Added plain-MLP isolation ablation per `C:\Users\Atharva\Documents\kirchhoffnet_realistic\dagger_distillation_mlp_plain.out` (W=48 L=3 silu, 5479 params, 10.2% val / 8.1% test at same 5559 budget as MoE). Quantified MoE gain (5.1/3.5 pts) and proved KNet gap is not MoE-dependent (plain alone beats KNet 14.2%/11.0% by 4.0/2.9 pts). Updated §1, §3.2 table, §8, §9, §11.

