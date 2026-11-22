# Problem Statement: KNet Loses to MLP on Fixed Teacher Distillation (Multi-Target) While Beating It on Single-Target Friedman

**Date:** 2026-09-05
**Status:** Active investigation — no fix yet; self-contained for an external LLM/reviewer with no codebase access
**Author context:** `kirchhoffnet_realistic` repo (Windows + AllianceCan CUDA). All numbers below are from on-disk Alliance outputs.
**Related doc:** `docs/ctle-knet-bo-problem-statement.md` (older DAgger/CTLE BO; different pipeline — do NOT conflate with this fixed-distillation benchmark)

---

## 1. Executive summary

Two Bayesian-optimization (BO) runs distill the **same frozen PlainMLP teacher** into a student on a **fixed, immutable teacher-labelled dataset** (no DAgger, no data shift, pure `MSE(student_logits, teacher_logits)`):

- `fixed_knet_bo_round1` — student = **KirchhoffNet (KNet)**, analog Neural-ODE fabric, `--param-budget 7000`, 40 trials, rank-based halving (15ep probe → 100ep rung-2).
- `fixed_mlp_bo_plain_mlp_teacher` — student = **PlainMLP**, standard feed-forward net of the **same family as the teacher**, `--param-budget 5000`, 30 trials, full 100ep each (older script revision, no halving columns).

Result: **MLP wins by ~10× on validation logit-MSE** (best 0.00028 vs KNet best 0.00258; selected-config finals 0.00503 KNet val) and matches the teacher's ZIG failure rate more closely (MLP ~5–6% vs teacher 5.02%; KNet 6.8–8.1%). KNet additionally **fails** the 3pp isolation tolerance (val gap 3.67pp, test gap 4.39pp).

Contrast: on single-target direct regression (`friedman1`, 10→1, Huber), the *same two model families* under `kn_bayes_opt.py` / `mlp_bayes_opt.py` show **KNet marginally beating MLP** (best_val 0.000405 vs 0.000420 at ~7k params, despite MLP having 2× epochs).

The author does **not** want a same-budget rematch (KNet had the *larger* budget and still lost clearly). The leading hypothesis to evaluate: **single-target vs multi-target expressivity** — Friedmans are 1-output problems; distillation is **4 specs → 7 teacher logits** through one shared trunk, and KNet's shared fabric + linear-ish readout cannot supply the per-target degrees of freedom that a shared MLP trunk + linear head can. A secondary suspect is a recently added **final nonlinear `post_readout_transfer`** (identity at init, always-on since 2026-09-01, i.e. after the Friedman runs but before distillation) that may hurt rather than help.

Question for the reviewer: *is this a problem-construction artifact (same-family imitation + multi-output logit matching favoring MLP), a genuine KNet expressivity deficit on multi-target tasks, or a bug/regression in the new fixed-distillation script (optimizer, readout, halving)?*

---

## 2. Exact commands run

```bash
# KNet (round 1, new script with halving)
python fixed-knet-bayes-opt.py \
  --teacher-dir /home/annaik/Documents/kirchhoffnet_realistic/outputs/generative_plain_bayes_opt/plain_knet_budget25000_fproxy/trial_0019 \
  --zig-artifact-dir /home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2 \
  --dataset-dir /home/annaik/Documents/augmented-cvae-ctle \
  --fixed-dataset-path /home/annaik/Documents/kirchhoffnet_realistic/outputs/fixed_teacher_dataset.npz \
  --param-budget 7000 --epochs 100 --n-trials 20 \
  --output outputs/fixed_knet_bo_round1 \
  --device cuda --probe-epochs 15 --rung2-epochs 100 \
  --halving-eta 3 --halving-warmup 4 --resume
# Note: n_trials=20 requested but bo_summary reports 40 trials / 39 COMPLETE —
# study was resumed/extended. Treat as 40-trial run.

# MLP (older script revision, full-length trials)
python fixed-plainmlp-bayes-opt.py \
  --teacher-dir /home/annaik/Documents/kirchhoffnet_realistic/outputs/generative_plain_bayes_opt/plain_knet_budget25000_fproxy/trial_0019 \
  --zig-artifact-dir /home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2 \
  --dataset-dir /home/annaik/Documents/augmented-cvae-ctle \
  --fixed-dataset-path /home/annaik/Documents/kirchhoffnet_realistic/outputs/fixed_teacher_dataset.npz \
  --param-budget 5000 --epochs 100 --n-trials 30 \
  --output outputs/fixed_mlp_bo_plain_mlp_teacher --device cuda
```

Both are thin wrappers (`fixed-knet-bayes-opt.py` injects `--student-kind knet`, `fixed-plainmlp-bayes-opt.py` injects `--student-kind mlp`) over one shared controller `fixed-distillation-bayes-opt.py` driving one shared harness `fixed-mlp-distillation-kirchhoffnet.py` (`--student-kind` switches the student class only). Teacher/dataset/output/ZIG flags are identical.

---

## 3. Results (authoritative numbers)

### 3.1 KNet `outputs_alliance_can/kn_bayes_opt/fixed_knet_bo_round1`

Sources: `bo_summary.json`, `results.csv`, `trial_table.txt`, `dataset_prepare.log.txt`.

- Fixed dataset: `train=8928 / val=1116 / test=1117` (seed 100, one-time teacher labelling on CPU).
- Param window: `[5950, 7000]` (budget 7000, tol 0.15). 0 penalized. Feasible arch list: 37 prior-biased `(hidden, stages, k, rank)` tuples.
- Selected trial **15** (lexicographic min failure then MSE): `hidden=10, stages=4, k=4, rank=4, x_max=7.2935, solver_scale=0.8768, wd=3.64e-05, lr=4.717e-3, batch=256`, 5959 params, `val fail=6.81% / mse=0.004830`.
- Pareto set: trial 15 `(0.0681, 0.004830)` and trial **24** `(0.0806, 0.002578)` — trial 24 has the **best KNet MSE** (same arch `hidden=10, stages=4, k=4, rank=4`, different `x_max=3.148, scl=1.297, wd=1.15e-03, lr=4.34e-03, batch=128`).
- Other promoted full-run trials: 1 (`0.0896, 0.003050`), 3, 8, 10, 11, 16, 18, 21 — all MSE 0.003–0.010. ~30/40 trials killed at probe (`rung1-only (hard_cap)`: probe `mse>0.15` or `fail>40%`, typical probe MSE 0.2–2.1; 3 killed by rank).
- Selected-final refit (100ep, test enabled): `val logit_mse=0.005025, test=0.004131`; `log_param_mae val=0.01554`; ZIG: teacher fail 5.018% val / 4.924% test vs student 8.692% / 9.311% → gaps **3.67pp val / 4.39pp test → FAIL** (tolerance 3.0pp).
- Learning curve (trial 15 final): ep1 val 1.91 → ep5 0.298 → ep10 0.144 → ep25 0.0237 → ep50 0.0103 → ep99 0.00503. Smooth, no divergence — just plateaus ~10× above MLP.

### 3.2 MLP `outputs_alliance_can/mlp_bayes_opt/fixed_mlp_bo_plain_mlp_teacher`

Sources: `bo_summary.json` (added 2026-09-06; selected-final refit with test metrics) + `results_mlp_summary.csv`.

- Param window: `[4250, 5000]` (budget 5000, tol 0.15). 30/30 COMPLETE, 0 penalized.
- Selection is lexicographic (failure first, then MSE), so the winner is trial **8**, not the lowest-MSE trial: `W=45, L=3, silu, layernorm=True, lr=4.663e-3, batch=128`, 4957 params, `val fail=4.839% / mse=0.0003535`. Weight decay is 0.0.
- Best by MSE alone: trial **22** `mse=0.0002837, fail=5.824%`; trial 21 `0.0002943, 5.735%`; trial 4 `0.0003227, 5.376%` — all 4957 params (same `arch_idx=18` = 45×3+LN; Pareto set = trials 4/8/21/22, differing only in lr/activation: silu 1.28e-3 / silu 4.66e-3 / gelu 2.39e-3 / gelu 2.76e-3).
- Selected-final refit (100ep, test enabled): `val mse=0.0003535, test mse=0.0002949`; `log_param_mae val=0.00564, test=0.00547`; ZIG **PASS both splits** (teacher 5.018%/4.924% vs student 4.839%/5.013% → gaps **0.18pp val / 0.09pp test**, tolerance 3.0pp).
- Learning curve (trial 8 final): ep1 val 0.0806 → ep5 0.00728 → ep10 0.00253 → ep27 0.000924 → ep86 best 0.000353. MLP reaches KNet's *final* level (~0.005) by **epoch ~7**; KNet needs ~99 epochs to reach 0.005.
- Head-to-head finals: MLP `val 0.000354 / test 0.000295, PASS` vs KNet `val 0.005025 / test 0.004131, FAIL` → **~14× val / ~14× test MSE gap** on selected configs (~9× on best-vs-best: 0.00258 vs 0.000284).
- Caveat: MLP CSV schema is the **pre-halving** one (no `promoted/kill_reason/rung1_rank` columns that KNet's CSV has). The MLP run used an older controller revision: all 30 trials ran the full 100ep. Search effort is NOT matched (30 full vs ~10 full + 30 probes), but the best-full vs best-full and selected-final gaps (~10–14×) are too large to attribute to effort alone.

### 3.3 Friedman contrast — KNet demonstrably wins single-target (this architecture is not a failure)

On `friedman1` (10→1 smooth direct regression, Huber loss, standardized targets, deterministic split, seed 0, 50 trials each, matched ~7k budget) KNet beats the MLP **despite half the epoch budget**:

- KNet `kn_bayes_opt/friedman1_knet_e400/best_hyperparams.txt`: `raw best_val=0.000405` (penalized `best_value=0.000510` — 3.8% over nominal budget: `actual_params=7267`, `normalized=1.038`, still inside `param_limit=8400`), 400ep, ~489s. Original scale: `best_rmse=0.141508, best_mse=0.020025, best_mae=0.106891`.
- MLP `mlp_bayes_opt/friedman1_budget7000_e800/best_hyperparams.txt`: `best_val=0.000420`, 800ep, ~265s. Original scale: `best_rmse=0.144129, best_mse=0.020773, best_mae=0.107301`.
- Margin: **~3.6% better on raw Huber best_val, RMSE, and MSE** (0.000405 vs 0.000420; 0.1415 vs 0.1441; 0.020025 vs 0.020773), with **400 vs 800 epochs**. Same pattern reported for `friedman2`/`smooth2d` (`friedman*_knet_e400` vs `*_budget*_e800`).
- This was **not seed luck**: KNet `best_trial_number=43, best_trial_is_seed=False` — BO started from the validated seed (`hidden=25, stages=5, k=4, fanout=2, t_span=7.0, lr=1.2e-3, batch=4096`) and discovered a genuinely better config: `fanout_count=1` (one-to-one boundary map `{i:[i]}`), `hidden=26, stages=4, k=4, t_span=3.86, vca_rank=4, lr=2.37e-03, wd=4.9e-04, batch=2048, x_max=6.92, gm_max=25.1, isat_max=24.2, device_l2=9.45e-04, freeze_boundary=0, freeze_temporal_read=0`. MLP's winner was `hidden=55, L=4, tanh, lr=4.43e-03, wd=6.7e-03, batch=4096`.
- Success recipe (different from §3.1–3.2): `kn_bayes_opt.py` 15-dim search (topology + `t_span` + lr/wd/batch + `x_max` + `gm/isat_max` + `device_l2` + freezes) with **differential LR scales** `mapper 1.0 / struct 4.0 / dyn 1.0`, `--freeze-read --temporal-readout --cell-library tanh_free`, Heun, `residual-relu-tanh` interstage, large batches (2048/4096). MLP side: budget-exact width derivation, single AdamW, Huber.
- What this proves for the reviewer: the **same fabric, same sampler family, same readout+VCA machinery wins on smooth single-output forward regression** — the ODE, small-world diffusion, frozen-core integration, and temporal accumulator readout are all capable of beating a dense MLP head-to-head. The distillation failure is therefore **specific to the multi-target imitation setting** (7 simultaneous logits, same-family teacher, single-LR harness, post-transfer readout), not a general indictment of the architecture. Any fix should preserve what works here (differential LRs, large-batch Huber training, `t_span`/`x_max`/`gm/isat` search) while addressing only the multi-target gap.

---

## 4. System definitions (enough to reason without code)

### 4.1 Teacher (frozen, identical for both runs)

- `PlainMLP(trunk_width=48, trunk_layers=3, activation=SiLU, use_layernorm=False, use_log_features=False)` + `Linear(48→7)` head; checkpoint `trial_0019/dagger_student_plain.pt` (+ `teacher_config.json` schema-v2 scaler when present).
- Input preprocessing `"knet"`: `logs=log10(specs[4 dims])`, `scaled=2*(logs-lo)/(hi-lo)-1`, clamp `[-4, 4]`, where `lo/hi` = dataset log-min/max ±5% pad (`kn_input_log_pad_frac=0.05`), computed once over all CTLE CSV rows.
- Output: unbounded logits `(B,7)`; physical params via `log_lo+(log_hi-log_lo)*sigmoid(logit)` with per-param log bounds (`PARAM_LOG_BOUNDS`, 7 dims).
- Loaded via `load_plain_mlp_teacher(...)` and frozen. Labels generated **once** (`make_fixed_dataset`, RNG seed 100, shuffle, 80/10/10 split, `np.savez_compressed` with sha256 fingerprint over seed+teacher identity+rows+fractions+bounds). Every BO trial trains on byte-identical `(specs, teacher_logits, train/val/test_idx)`.

### 4.2 MLP student (`PlainMLPStudent : PlainMLP`)

- **Same class as the teacher.** Constructor maps BO suggestions: `trunk_width=student_width, trunk_layers=student_layers, activation∈{SiLU,GELU}, use_layernorm∈{False,True}, use_log_features=False`, then `attach_scaler(... input_preprocessing="knet", input_log_min/max)` from the fixed cache.
- Forward: `trunk: Linear(4→W) + (L-1)×[Linear(W→W) (+LN)] + activation per layer; head: Linear(W→7)`; `predict = log_lo+(log_hi-log_lo)*sigmoid(forward(x))`.
- Analytic params: `5W + (L-1)(W²+W) + (7W+7) + (2WL if layernorm)`. BO samples `(layers, width, layernorm)` **jointly** from the precomputed feasible list in `[low, budget]` (e.g. 48×3 no-LN = 5287) plus `lr log[2e-4,5e-3]`, `batch {128,256,512,1024}`, `activation {silu,gelu}`.
- Optimizer: **single** `AdamW(all_params, lr, weight_decay)`, `MSELoss` on logits, no grad-clip by default, `eval_every=1`, best-`val_logit_mse` checkpoint restored for metrics.

### 4.3 KNet student (`KirchhoffStudent` wrapping `KirchhoffNetWithIO`)

- Input scaling identical to teacher (`scale_input` with cached `lo/hi`, `input_rail=4.0` clamp).
- Fabric per stage: hidden graph Watts-Strogatz `small_world(n=num_hidden, k, p=0.2, seed=1, bidirectional=False)`, `edge_repeats=2`; per-edge cell `tanh_free`: `I_tanh=Isat*(tanh(u)+0.01u)`, `u=gm*(s*(A*Vsrc-B*Vdst)+theta)`, `A/B=softplus`, `s=±1` (STE), `gm/Isat∈[0.01,10]` via sigmoid, shunt `G=softplus(g)`; structural gates `σ(z_logits)` (init 0.5); fixed leak (non-programmable) + `x_max` clip.
- Boundary drive (no InputMapper): `x0=zeros`; input enters only via sparse boundary OTAs with diagonal fan-out `{i:[i,i+4]} for i in 0..3` (8 targets; same map used by sampler, analytic counter, and every trial so param counts are exact).
- Solver: Heun, per-stage `t_span = 7.0/num_stages * solver_scale`, `steps = round(70/num_stages * solver_scale)` (solver_scale is param-count-neutral by construction).
- Temporal readout: per-stage output-ODE accumulators fed by hidden→accumulator OTAs; frozen at stage entry when `freeze_read=True` (always in this benchmark); read at end through `OutputAffine` (per-dim `y=gain*x+bias+tanh_gain*tanh(x)+relu_gain*ReLU(x-thr)`, identity at init) and then through **`post_readout_transfer`** (see §5).
- VCA: low-rank input-driven gates on boundary+readout+core (`vca_rank∈{2,3,4}`, `W=0` init → gate 1.0, `v~N(0,0.01)`), separate core bus, no bias, no gate-shunt.
- BO arch: `(hidden 8–24, stages∈{3,4,5}, k∈{4,6} with k<hidden, rank∈{2,3,4})` sampled jointly from the feasible list in `[low, budget]` (37 arches at [5950,7000]); plus `lr log[2e-4,5e-3]`, `batch {128,256,384,512}`, `x_max [2,8]`, `wd log[1e-6,1e-2]`, `solver_scale [0.5,2]`. Fixed: `p=0.2, seed=1, edge_repeats=2, tanh_free, residual-relu-tanh interstage, freeze_read=True, temporal_readout=True`.
- Optimizer: **single** `AdamW(all_params, lr, weight_decay)` — same as MLP student, and **unlike** Friedman KNet which uses differential LR groups (`mapper 1.0 / struct 4.0 / dyn 1.0`). Same MSE-on-logits objective, same best-val checkpointing, same ZIG evaluation (frozen `HybridHurdleModel` + scalers; failure = predicted design violates spec thresholds on the shared 1000-spec COMMON eval).

### 4.4 Objective / selection (both runs)

- Per trial Optuna values: `(val_ZIG_failure_rate, val_logit_mse)`, both minimized; final selection lexicographic (failure first, then MSE); held-out test evaluated once for the selected config. Penalty `(1.0, 1e6)` (finite, never Pareto-optimal, never inf/None so multi-objective TPE doesn't crash) for preflight/parse/subprocess failures.
- KNet halving: probe at 15ep; hard kill if `probe_mse>0.15` or `probe_fail>40%` (returns finite probe values, never pruned); else rank-based `promote iff rank < max(1, N//eta)` with `eta=3`, first 4 trials always promoted; promoted trials `--resume` to 100ep. MLP run predates this gate (all full-length).

---

## 5. The `post_readout_transfer` change (user's prime suspect)

- Commit `95afe00 (2026-09-01) "permanently added output interstage transfer after output affine"` touches `kirchhoff_net.py` + both DAgger KNet scripts. Friedman Alliance runs are dated `20260822` (before); distillation runs `2026-09-04/05` (after). Chronology is consistent with a regression.
- What it is: `self.post_readout_transfer = StageTransfer(in=7, out=7, activation="residual-relu-tanh")` in `KirchhoffNetWithIO`, applied as `y = post_readout_transfer(output_mapper(x_read))` before the harness sigmoid. Per-dim: `y = w1*x + w2*tanh(x) + w3*ReLU(x-vth)`, init `w1=1, w2=w3=vth=0` → **identity at epoch 0** (bit-identical forward). Adds 28 trainable params (counted in both the analytic counter and `--count-params-only`, so budget accounting is consistent).
- Why it might hurt despite identity init: (a) 28 extra params trained at the **same single LR** as fabric/ODE (no `mapper-lr-scale`/`struct-lr-scale` separation in this harness); (b) places a learnable kink (ReLU-threshold + tanh) directly on the 7-dim bottleneck every target shares — gradients for all 7 heads couple through it; (c) related readout change `a805e7f "Added ReLU to temporal-readout output affine"` predates it. Author added it hoping for expressivity; observed effect was worse imitation.
- What rules it in/out: it cannot explain epoch-0 differences (identity), but can explain optimization trajectory differences. Needs an A/B (bypass vs freeze-at-identity vs current) at fixed seed/arch — not yet run.

---

## 6. Hypotheses for the reviewer (ranked by author's current belief)

**H1. Multi-target shared-trunk bottleneck (author's leading hypothesis).** Friedman outputs are 1-dim; distillation outputs are 7 logits through one shared representation. MLP's trunk (`W=32–64` dense, SiLU/GELU per layer) + per-target linear head rows gives each target an independent linear readout of a rich shared basis. KNet's shared object is a dynamical fabric whose entire 7-dim output passes through `OutputAffine` (mostly affine at init) + `post_readout_transfer` (7-dim diagonal nonlinearity, no cross-target mixing). The fabric itself (`hidden=8–19`, stages 3–5, sparse small-world, frozen core) may lack the per-target degrees of freedom for 7 simultaneous logit functions. Evidence for: gap appears exactly when moving 1-output → 7-output; capacity ablations in older CTLE work (bigger hidden/stages) didn't help, suggesting the bottleneck is structural (readout rank), not raw size. Evidence against / needed: per-dim MSE breakdown (does KNet match 5/7 heads and fail 2, or fail all uniformly?); a width-vs-depth slice at fixed budget.

**H2. Same-family imitation bias (problem construction, not a KNet bug).** The MLP student IS the teacher architecture (same `PlainMLP`, same knet scaler, same head, same sigmoid bonding). Distillation MSE on teacher logits rewards weight-copying; KNet must approximate a 48×3 SiLU map through an ODE + affine readout. On neutral ground (Friedman, both from scratch) KNet ties/wins. If true, no KNet feature addition is warranted — the benchmark measures family similarity, and the fix is methodological (report from-scratch + cross-family controls, e.g. distill a KNet teacher both ways).

**H3. Optimizer / credit-assignment asymmetry.** Friedman KNet uses differential LRs (struct 4× for `z_logits`, mapper/dyn 1×) tuned for the fabric; the fixed harness uses one LR for fabric + gates + VCA + OutputAffine + post_readout. Small-world structural params and readout params likely need different speeds (trial-15 winner used high LR 4.7e-3 + tiny wd — consistent with "readout starved, fabric overdriven" or vice versa). Single-factor ablation (LR groups only) is the cheapest test.

**H4. Readout-rank / post_readout regression.** Even setting H1 aside, the specific readout (OutputAffine + diagonal post_transfer, frozen-core accumulators, single read slice) may be the wrong kind of capacity: diagonal per-dim warps can't synthesize cross-target structure; the added ReLU kink may create dead/saturated logit regimes under MSE. Test: replace/bypass post_transfer; try wider read slice or low-rank mixing readout (still analog-plausible) as a controlled variant.

**H5. Halving / search artifact.** KNet's halving kills ~75% of trials at 15ep on hard caps calibrated for "obvious blow-ups" (`mse≤0.15, fail≤40%`). If KNet is a slow starter under single-LR (ep1 val 1.91 → ep5 0.30), good late-bloomers may be killed while MLP's full-length runs survive. Counter-evidence: promoted KNet trials still plateau ~0.003–0.005 at 100ep, far above MLP's 0.0003 — halving explains selection noise, not the 10× level gap. Still, rung-1 vs rung-2 rank correlation should be checked.

**H6. Logit-scale / loss-weighting artifact.** MSE weights all 7 logits equally; if teacher logits have very different variances (e.g. saturated `current` head vs broad `fW` head), KNet may minimize the easy heads while MLP copies all. ZIG failure then amplifies small logit errors near feasibility cliffs. Needs per-head teacher-variance + per-head student-MSE table (not yet computed).

Ruled OUT (per author, do not relitigate without new evidence): same-budget rematch (KNet's larger budget lost decisively); MoE-vs-plain as the explanation (older doc shows plain MLP alone beats KNet — here teacher/student are both plain anyway); test-leakage/DAgger nonstationarity (no DAgger in this benchmark — dataset is frozen).

---

## 7. What is wanted from an external reviewer (no codebase access)

1. Which of H1–H6 best explains **wins single-target, loses multi-target by 10×**? Rank them and state what evidence would flip your ranking.
2. Is H1 (shared-trunk multi-target bottleneck) mechanistically plausible for a Neural-ODE fabric with a 7-dim diagonal readout vs a dense MLP trunk + linear head? What minimal readout/fabric change would test it without breaking analog plausibility (e.g. per-head readout rank, mixing readout, head-specific time constants)?
3. Does H2 (same-family copying) invalidate the benchmark, or is a 10× gap too large for family bias alone? What control (from-scratch fit, KNet-teacher reverse distillation, per-head error profile) would decide?
4. Is the `post_readout_transfer` (identity-init diagonal `w1*x+w2*tanh+w3*ReLU(x-vth)` on the 7-dim bottleneck, single-LR training) more likely to help or hurt multi-target logit matching? Would you remove, freeze, or gate it behind a flag?
5. What 3 diagnostics (each ≤1 GPU-day, no code access assumed) would you request first — e.g. per-head teacher variance + per-head MSE, LR-group ablation, post_transfer bypass A/B, from-scratch control?
6. If H1 is true, is the fix architectural (readout rank, multi-head fabric, head-wise VCA) or is KNet simply the wrong tool for multi-target inverse/distillation tasks?

Ideal response: ranked hypotheses with one-paragraph mechanisms, the single most likely culprit, and 3 concrete next experiments with pass/fail thresholds (e.g. "if bypass-A/B closes <20% of the gap, drop H4").

---

## 8. Reproducibility pointers (for an engineer with repo access; reviewer may ignore)

- Controller: `fixed-distillation-bayes-opt.py` (halving `rung1_promotion_decision`, fingerprint `SAMPLER_REV=3`, `PENALTY_VALUES=(1.0,1e6)`, `knet_fanout(i)=[i,i+4]` diagonal rule, `_knet_param_analytic` via `build_net_from_config` + `tanh_free`).
- Harness: `fixed-mlp-distillation-kirchhoffnet.py` (`KirchhoffStudent` :226, `PlainMLPStudent` :278, single `AdamW` :518, frozen `make_fixed_dataset` :332, ZIG eval :413).
- Wrappers: `fixed-knet-bayes-opt.py`, `fixed-plainmlp-bayes-opt.py`, `fixed-mlp-distillation-plainmlp.py`.
- Model: `kirchhoff_net.py:362-374` (post_readout construction), `:521-529` (readout pipeline), `io_mapper.py:577` (`OutputAffine`), `stage_transfer.py:34` (`StageTransfer`), `topology.py:2318` (readout wiring), `ctle_dagger_common.py:1381` (`PlainMLP`).
- Outputs: `outputs_alliance_can/kn_bayes_opt/fixed_knet_bo_round1/{bo_summary.json,results.csv,trial_table.txt,results_live.csv,dataset_prepare.log.txt}`; `outputs_alliance_can/mlp_bayes_opt/fixed_mlp_bo_plain_mlp_teacher/results_mlp_summary.csv`; `outputs_alliance_can/kn_bayes_opt/friedman1_knet_e400/best_hyperparams.txt`; `outputs_alliance_can/mlp_bayes_opt/friedman1_budget7000_e800/best_hyperparams.txt`.
- History: `git show 95afe00` (post_readout), `git show a805e7f` (OutputAffine ReLU), trial logs dated 20260822 (Friedman, pre-transfer) vs 2026-09-04/05 (distillation, post-transfer).

---

## 9. Document history

- 2026-09-05 (v1): Created from Alliance outputs + `fixed-distillation-bayes-opt.py` / `fixed-mlp-distillation-kirchhoffnet.py` inspection + `git log` chronology. Supersedes hallway discussion; complements (does not replace) `docs/ctle-knet-bo-problem-statement.md` which covers the older DAgger CTLE loop.
- 2026-09-06 (v2): Added MLP `bo_summary.json` (selected trial 8, W45 L3 silu +LN, val 0.0003535/test 0.0002949, ZIG PASS 0.18/0.09pp). Selected-final gap is ~14× val/test; best-vs-best ~9×.
- 2026-09-06 (v3): Expanded §3.3 Friedman-1 success (raw 0.000405 vs 0.000420, RMSE/MSE/MAE margins, 400 vs 800ep, trial-43-not-seed discovery, differential-LR success recipe) to show the architecture wins single-target and the failure is multi-target-specific.
