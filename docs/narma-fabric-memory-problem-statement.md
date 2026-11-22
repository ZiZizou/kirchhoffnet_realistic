# Problem Statement: KirchhoffNet Fabric Has No Usable Memory on NARMA-10 (Plateau at NRMSE ~0.67 vs 0.15-0.29 Band)

**Date:** 2026-09-06
**Status:** Diagnosis phase complete, no fix -- all four lever families exhausted; self-contained for an external LLM/reviewer with no codebase access
**Author context:** `kirchhoffnet_realistic` repo (Windows + AllianceCan CUDA). All numbers below are from on-disk Alliance outputs (`outputs_alliance_can/narma_refresh2`, `outputs/narma_diag_*` legs, `narma_diag_latest_log.out`).
**Related docs:** `docs/specs/core-refresh-semantics.md` (refresh-interval semantics), `narma_diag_batch.sh` (diagnostics batch script; gitignored `*.sh` -- request a copy), repo `README.md` sections 1-3/11 (fabric math, Write/Evolve/Read pipeline, RHS pseudocode)

---

## 1. Executive summary

A KirchhoffNet (KNet) analog Neural-ODE fabric (25 hidden nodes, 5x5 torus, 100 edges, `tanh_free` cells, ~1261 params) trained on the **NARMA-10** sequence-memory benchmark plateaus at **test NRMSE ~= 0.66-0.68 (R^2 ~= 0.54, memory capacity ~= 0.2/20)** against a **0.15-0.29 success band** (standard-RNN parity). Seven follow-up experiments varying recurrence rate, readout learning speed, input drive strength, leak timescale, and state-carryover masking all land on the same wall (best 0.657, within single-seed noise of baseline 0.676).

The two load-bearing facts: (a) an **untrained** fabric with a linear ridge readout already scores **0.7194** -- training buys only ~0.05; (b) **zeroing the carried-over hidden state at eval time collapses NRMSE to 0.87 and memory to exactly 0.00** -- so the state *is* load-bearing, but it carries almost no *task-useful* memory. The author's TL;DR question for the reviewer: *we expected the node capacitances (state voltages) to hold fading memory of past inputs, and the evidence says they hold nearly none -- is this a reservoir-design/init failure, a saturation-dynamics failure, a readout bottleneck, or a task-setup artifact?*

---

## 2. System definitions (enough to reason without code)

### 2.1 The fabric (reservoir)
- Continuous-time node voltages `x` in R^26 (25 hidden + 1 output accumulator), evolved by `dx/dt = (I_edges + I_boundary + I_readout - I_leak)/C`, integrated with **Heun, `t_span=1.0`, 8 steps/sample** (`dt=0.125`).
- Core edges: `tanh_free` OTA cells, `I = Isat*(tanh(u)+0.01u)`, `u = gm*(s*(A*Vsrc-B*Vdst)+theta)`; `gm/Isat` in `[0.01, 10]` via sigmoid of raw params, **initialized at `gm_raw=isat_raw=-5.0` giving gm ~= 0.077 (near-linear, very weak coupling)**; resistive shunt + edge gates `sigmoid(z_logits)`; per-node learnable leak from `raw_leak` (init -3.0 giving leak ~= 0.05; trained mean -3.56 giving ~= 0.028, i.e. ~0.97 retention/sample, ~35-sample horizon).
- **Input enters ONLY via 25 boundary OTA edges** (input terminal fanned out to every hidden node; a `u_logits` drive path exists but is all-zeros/unused). Bipolar scaling maps `u` in `[0,0.5]` to the **full rail swing +-3.0V** (`x_max=3.0`).
- **Compliance gating**: every cell multiplies current by `sigmoid((x_max-|x|)/beta)` at both endpoints -- nodes approaching the rails **decouple from the fabric**.
- Readout: single output-ODE accumulator node + `OutputAffine` mapper (`y=gain*x+bias+tanh_gain*tanh(x)+relu_gain*ReLU(x-thr)`; trained gains: main 1.015, tanh/relu ~= 0.06-0.13, i.e. **readout stayed quasi-linear**).

### 2.2 The task and protocol
- **NARMA-10**: input `u ~ U[0, 0.5]`, target `y` depends nonlinearly (incl. product terms) on the last ~10 inputs/outputs. Train: 4 streams x 2500 samples, truncated BPTT (chunk 25, state carried across chunks, detached only at chunk boundaries, zeroed each epoch). Test: 1000-sample fresh stream, washout 200. Targets standardized; AdamW `lr=1e-3`, AMP on CUDA.
- Metrics: **NRMSE** (1.0 = mean predictor), **R^2**, **MC** = sum over k=1..20 of R^2 of ridge reconstruction of `u(n-k)` from states (Jaeger 2001; healthy reservoirs score 5-20+). **Caveat**: the MC collector starts from zero state with **no washout discard**, so with the slow learned leak it underestimates true MC -- but 0.2 is two orders from healthy regardless.
- **Baselines (ridge-30-tap / ESN / MLP / LSTM) were NOT run** (`--fabric-only` everywhere) -- the 0.15-0.29 band is literature-anchored, not in-harness calibrated.

### 2.3 Core-refresh context (closed line of inquiry)
- A `core_refresh_interval=k` knob recomputes the frozen core current every k Heun steps (k=8: 1 eval/sample ~= legacy frozen; k=2: 4 evals; k=1 ~= dynamic). Ladder {8,4,2} x seeds was the original experiment; **k=2, k=4, k=8 all plateau ~= 0.67-0.68** -- recurrence rate is ruled out as a lever.

---

## 3. Exact commands run

```bash
# Baseline k=2 (outputs/narma_refresh2): 80ep, early-stop 15, seed 0
python narma_experiment.py --order 10 --device cuda --compile --fabric-only \
  --cell-library tanh_free --epochs 80 --n-streams 4 --train-samples 2500 \
  --tbptt-chunk 25 --t-span 1.0 --num-steps 8 --seeds 0 --val-every 5 \
  --early-stop-patience 15 --no-progress --checkpoint-every 10 \
  --output ./outputs/narma_refresh2 --core-refresh-interval 2

# Diagnostics batch (narma_diag_batch.sh; sequential legs, same COMMON base +
# per-leg flags; E3 resumed from its epoch-79 checkpoint after a TIMEOUT kill)
#   E3:  --core-refresh-interval 8 --epochs 80 (+ --ridge-diagnostic on fresh path)
#   E4a: --core-refresh-interval 2 --epochs 35 --mapper-lr-scale 10 --ridge-diagnostic
#   E4b: --core-refresh-interval 2 --epochs 35 --input-scale 0.5
#   E4c: --core-refresh-interval 2 --epochs 35 --leak-constant 0.15
#   E4d: --core-refresh-interval 2 --epochs 35 --mapper-lr-scale 10 --input-scale 0.5
#   E2:  --eval-masks full,reset-hidden,reset-all --init-from <ckpt> (matched arch flags)
```

---

## 4. Results (authoritative numbers)

| leg | test NRMSE | R^2 | MC | note |
|---|---|---|---|---|
| k2 baseline (80ep, early-stop ep54) | 0.6762 | 0.5421 | 0.23 | train loss 0.99->0.58, flat from ~ep40; hidden RMS 0.02->1.57, still rising |
| k4 (earlier run) | ~0.67 (val, ep60) | -- | -- | killed by 2h limit; same wall |
| **E3 k8** (80ep) | **0.6783** | 0.5394 | **0.19** | frozen baseline = same wall; rail 0%->11.2%, max|x|/x_max ->1.31 |
| ridge diagnostic (untrained, k8) | **0.7194** | 0.4824 | -- | training buys ~0.04-0.05 over random reservoir + linear read |
| E4a mapperx10 (35ep) | 0.6710 | 0.5492 | 0.22 | hotter readout: nothing |
| E4b drivex0.5 (35ep) | 0.6726 | 0.5471 | 0.18 | half swing, rail only 1.3%: nothing |
| E4c leak 0.15 (35ep) | **0.6571** | 0.5677 | 0.19 | best by 0.02 (single-seed noise); 1235 params (no raw_leak) |
| E4d combo (35ep) | 0.6650 | 0.5572 | 0.21 | no synergy |

**E2 carry-mask ablation** (eval-only, no retraining; identical outcome on k2 and k8 checkpoints):

| mask | k2 NRMSE / MC | k8 NRMSE / MC |
|---|---|---|
| full (status quo) | 0.6762 / 0.23 | 0.6781 / 0.19 |
| reset-hidden (keep output accumulator only) | **0.8703 / 0.00** | **0.8670 / 0.00** |
| reset-all | 1.0197 / -0.04 | 1.0169 / -0.04 |

Post-hoc saturation from full-carry train states: max|x|/x_max **1.04** (k2) / **1.38** (k8), rail fraction **9.9% / 9.0%** -- nodes driven *past* the rail.

---

## 5. What is ruled OUT (do not relitigate without new evidence)

- **Recurrence granularity** (k=2/4/8 identical) -- the refresh-interval program is closed.
- **Readout learning speed** (mapper LR x10, readout stayed quasi-linear anyway).
- **Drive strength** (half swing still saturates to xmaxr~=1.0 and scores identically).
- **Leak timescale** (fixed 0.15 vs learned 0.028: -0.02, noise).
- **Resetting the core as a feature** (E2: -0.19 NRMSE, memory goes to 0.00 on two checkpoints).
- **Under-training** (loss/val flat for 15-40 epochs in every run; longer runs won't move).

---

## 6. Hypotheses for the reviewer (ranked by author's current belief)

**H1. Reservoir is near-linear and weak at init (author's leading hypothesis).** `gm~=0.077` at init means ~100 edges couple nodes through an almost-linear, tiny-gain map -- a slow linear integrator, not a nonlinear reservoir. NARMA needs *nonlinear* memory (product terms); a linear reservoir's ridge ceiling is low (0.72 observed), and gradient training then grows gains straight into compliance shutdown rather than into a useful nonlinear regime. Evidence for: ridge-before-training ~= trained score; E4b (gentler drive) can't fix what init lacks. Test: E0-level sweep (no training) over `gm_init`, drive scale, hidden size -- minutes per point.

**H2. Compliance-gating saturation shutdown.** Nodes near +-x_max decouple (edge currents go to 0); training pushes voltages outward in all runs (h_rms monotonic, rail% rising with fitting, xmaxr > 1). The reservoir may start mediocre and then *actively lose* mixing as it saturates. Evidence for: rail%/xmaxr trajectories track the plateau onset. Counter-evidence: E4b halves the drive yet still saturates and scores the same -- saturation accompanies fitting but may be symptom as much as cause. Test: clamp/regularize operating range (rail penalty weight, drive caps) and check whether MC recovers *before* NRMSE moves.

**H3. Single-accumulator readout bottleneck.** The entire 25-D reservoir funnels through one output-ODE node + quasi-linear affine map. Even with perfect reservoir memory, a 1-D leaky accumulator + linear read may lack rank for NARMA's 10-lag nonlinear dependence (contrast ESN practice: full-state linear readout). Evidence for: mapper gains never leave the linear regime despite 10x LR. Test: multi-tap readout (all hidden + delay taps) at eval time on frozen states -- pure readout-capacity probe, no retraining.

**H4. Credit-assignment through frozen core.** With k=8 the core current is computed once per window from the window-start state; gradients flow through that frozen quantity, never through per-step re-evaluation -- TBPTT through 25x8 Heun steps of a mostly-frozen graph may deliver unusable gradients to structural params (leak_grad ~= 1e-2, flat). Counter-evidence: k=2 (4 refreshes/window) scores identically, and loss does descend 0.99->0.59 -- gradients *work*, they just lead to a poor basin. Test: full-dynamic (k=1) training run; if it also lands 0.67, H4 is dead.

**H5. Task-setup artifact (missing calibration).** No baseline has ever run in this harness -- if ESN-25/LSTM also land ~0.6 here (data scaling, washout, metric, or u_max choices), the 0.15-0.29 band is miscalibrated *for this setup* and the fabric is being judged against an unreachable bar. Cheapest test in the whole document: run `--baselines-only` once.

**H6. Raw capacity (25 nodes / 100 edges).** Possibly just too small a reservoir for NARMA-10 at any init. Counter-evidence: ESN-25 classically reaches the band, and MC~=0.2 indicates a *qualitative* (mixing/saturation) failure, not a gradual capacity shortfall. Test: hidden 49 at E0 level.

---

## 7. What is wanted from an external reviewer (no codebase access)

1. Rank H1-H6 by mechanism plausibility for **"ridge-untrained ~= trained, and masking carryover destroys everything while tuning everything else changes nothing."** State what evidence would flip your ranking.
2. On the capacitance question: given E2 (carryover load-bearing) *and* MC~=0.2 (no task memory), is "the capacitors store state but not *useful* state" the right framing -- and does that point to init-regime (H1), shutdown dynamics (H2), or readout rank (H3)?
3. Is the H1xH2 interaction (weak-linear init -> training grows gains -> compliance shutdown) a known failure mode of analog/neuromorphic reservoirs, and what is the standard fix (spectral-radius-style init search, input scaling discipline, operating-range regularization)?
4. What 3 diagnostics (each <=1 GPU-day) would you request first -- e.g. in-harness baselines, gm_init E0 sweep, frozen-state multi-tap readout probe, k=1 dynamic training?
5. If H1 is true, is the fix a one-time init search, or does the fabric need an architectural addition (and if so, which, given E2 killed the reset/gating direction)?

Ideal response: ranked hypotheses with one-paragraph mechanisms, the single most likely culprit, and 3 concrete next experiments with pass/fail thresholds (e.g. "if the gm_init sweep never breaks ridge 0.6, drop H1").

---

## 8. Reproducibility pointers (for an engineer with repo access; reviewer may ignore)

- Experiment: `narma_experiment.py` (`train_fabric` TBPTT + carryover, `_forward_heun_sequence` integrator, `run_eval_masks` carry ablation with `carry_keep` mask, `collect_fabric_states` MC path, `ridge_readout_diagnostic`); batch: `narma_diag_batch.sh` (gitignored -- request copy); E4 flags `--mapper-lr-scale/--input-scale/--leak-constant`; E1 logging `xmaxr=/rail=` per epoch.
- Model: `differential_stage.py` (`_compute_i_edge_const`, `_heun_steps` refresh, `_compute_core_gate`, compliance gates in `rhs`), `cell_library.py` (`FreeTanhLibrary`, `gm_raw=-5.0` init), `config.py:make_narma_preset` (torus 5x5, boundary fan-out, `x_max=3.0`), `topology.py:build_net_from_config` (leak/mode cfg keys).
- Outputs: `outputs_alliance_can/narma_refresh2/` (k2 baseline + ckpt), `outputs/narma_diag_e3_k8` (k8 ckpt epoch 79 + resumed tables), `outputs/narma_diag_e4{a,b,c,d}_*`, `outputs/narma_diag_e2_{k2,k8}/carry_ablation.csv`, `narma_diag_latest_log.out` (full batch log, all legs DONE).

---

## 9. Potential path: multi-stage depth-per-sample (plan `narma-multistage-probe`)

All NARMA runs to date use a SINGLE stage (25 hidden nodes, no interstage transfers); depth was never a variable. The Friedman wins used 3-5 stages with `residual-relu-tanh` interstage transfers, so depth-per-sample is the one structural axis still untested. Note the mechanism precisely: transfers are instantaneous static maps (truncate/pad + pointwise warp, no time evolution) -- each sample's `t_span` window is split across stages, total integration per sample unchanged.

Two variants were considered; only the first is active:
- **A. Depth-per-sample (ACTIVE).** Each sample runs stage 1 -> transfer -> stage 2 -> ... -> stage N inside its window; the CONCATENATED multi-stage state carries across samples (E2 proved carryover load-bearing, so this is non-negotiable). Memory stays recurrent, but richer/higher-dimensional with static nonlinear warps between pools.
- **B. Systolic delay-line (DEFERRED).** Each stage keeps its OWN carried state across samples, like a conveyor belt: at time t, stage 1 holds a trace of `u_t`, stage 2 holds stage 1's output from t-1 (processed `u_{t-1}`), stage 3 holds processed `u_{t-2}`. The readout sees transformed present + 1-lag + 2-lag features simultaneously -- the analog-native version of the tapped delay line that makes NARMA trivial for ridge baselines. Genuinely different memory mechanism; bigger build; attempted only if A pulses.

Sequence: **P0** 49-wide single-stage control (needs only a `--hidden-dim` flag; separates "more dims" from "depth + transfers") -> generalize the single-stage-only NARMA paths (sequence integrator, `train_fabric`, eval/MC/collect, ckpt namespacing, per-stage carry-mask slices) per repo spec `multistage-sequence` -> build 3x16-hidden probe (~matched params to the 1261-param baseline, verified by exact counter; drive into stage 1 only; readout from last stage) -> **E0-gate it** (repo spec `narma-e0-gate`: train only on ridge clearly below 0.65 and/or MC clearly above 0.5) -> one short k=2 comparison train.

Kill criteria (pre-registered): if ridge ~= 0.72 and the short train lands ~= 0.67, depth is closed with no sunk-cost training, and work returns to H1 (init-regime sweep) and H3 (readout rank). The gate exists because training a poor reservoir only saturates it.

---

## 10. Document history

- 2026-09-06 (v1): Created from `narma_diag_latest_log.out` (all 7 legs), k2/k8 checkpoint inspections (histories, `raw_leak`, cell params), and the E2 ablation tables. Follows `docs/fixed-distillation-knet-mlp-gap-problem-statement.md` house style; complements it (different pipeline -- distillation vs reservoir memory; do NOT conflate).
- 2026-09-06 (v2): Removed the mistakenly-imported frozen-core-staleness section (belongs to the distillation freeze debate, not NARMA -- NARMA carries state so window-start x0 is nonzero, and Sec. 5 already closed refresh). Added Sec. 9 multi-stage depth-per-sample path (plan `narma-multistage-probe`, specs `multistage-sequence` + `narma-e0-gate`); systolic delay-line recorded as deferred variant.
- 2026-09-06 (v3): Advisor-ordered probe harness implemented (`narma_advisor_probes.py`, audit `probe_audit1.py` ALL_AUDITS_OK). Local CPU harness checks:
  - **Baselines calibration**: ESN NRMSE=0.5596 (26 trained/676 total params), above band top 0.29 but below the corrected pre-registered halt point 0.60; LSTM (7603 params) lands 0.2965, just inside band. Corrected decision: BAND-UNKNOWN, not HALT-TASK-SETUP. The task pipeline is sound, but the random-ESN reference is undertuned relative to the literature band; H5 remains open rather than resolved.
  - **Frozen-state probe** on k2 ckpt (epoch 54, tanh_free, k=2): ridge-from-full-hidden=0.6434 (≥0.6 -> INIT-REGIME-SOLE-PRIMARY), corrected ridge-from-5-lag-stack=0.5863, MLP-from-hidden=0.6645, participation ratio 1.96 (near 2 -> light rank-collapse), gate-zero core-edges ridge=0.6192 (only -0.024 delta -> core edges not purely decorative, but boundary-to-feedforward dominates), washout-corrected MC=0.2412 (matches prior).
  - **E0 (zero-training) sweep**: the earlier local 12-corner slice is superseded by audit corrections (wrong 6-step fallback and synthetic Jacobian operating points). The corrected harness pins `t_span=1.0`, `num_steps=8`, evaluates observed `(state[t], input[t+1])` transitions, and makes `--max-corners` an exact traversal prefix. Corrected 2-corner CLI wiring passes. No corrected full-sweep inference is claimed locally.
  - Full probes deferred to GPU on AllianceCan via `narma_advisor_probes_batch.sh` (P0 baselines, P1 frozen k2/k8, P2 full E0 in one Slurm allocation).
- 2026-09-06 (v4): P2 rerun implemented per external review (plan `narma-p2-rerun`, feature spec `p2-rerun-probes`). Ridge-parity-first gates (ridge ≤ 0.40, MC > 3, state-PR ≥ 6, rail ≤ 5%); per-corner washed-state PR, exact param counts, hidden dimension, and full transient gate-zero instruments (Ridge/Mc/PR/rail/Jacobian) added to `narma_advisor_probes.py`; explicit `hidden_dim` plumbing added to `_build_fabric_net`; `probe_audit1.py` extended (hidden-49 build, gate restoration, revised thresholds) and passing; local one-corner hidden-25/49 CPU smokes pass. P2-only batch `narma_advisor_probes_p2_batch.sh` (preflight + full hidden-25/hidden-49 grids, no P0/P1 recompute) is ready for Alliance; full 72-corner grids remain pending.
- 2026-09-07 (v5): Linear-controls harness implemented (plan `narma-linear-controls`, feature spec `linear-control-probes`, module `narma_linear_controls.py`, audit `probe_audit2.py` ALL_AUDITS_OK, CPU-only smokes). **Harness bug found and fixed**: `narma_experiment.memory_capacity` paired past-states with future-targets (`states[:T-k]` vs `targets[k:]`), i.e. forward prediction, not Jaeger k-step-back memory -- a noiseless 1-step delay line scored MC ~= 0.02 under the old convention and MC_1 = 1.00 after the fix (`X = states[k:]`, `y = targets[:T-k]`). All previously reported MC numbers (E0 ~= 0.19-0.23, frozen 0.24) were overfit-level artifacts of the swapped convention and must be re-read; gate thresholds are unchanged but must be re-anchored by the C0 control on the full stream. Local corrected-instrument checks (CPU, thin samples): C0 ESN (h25) MC = 10.18 with a textbook delay-decay curve (0.997 -> 0.18) and analytic Jacobian max|eig| = 0.83 -- pipeline validated, fabric verdicts stand; C1 linear-reservoir-in-fabric (h49, J = 0.945 in the [0.93, 0.97] band after 1 tuning iteration) MC = 1.76 with a real decay signature, PR ~= 1.1, ridge 0.71 vs raw-delay 0.38 -- does NOT pass (MC >= 3, PR >= 6, parity +0.03) on thin local samples; PASS/FAIL verdict deferred to the full-stream GPU run. Per-sample |J| ~= exp(rho_W - min_leak) is leak-dominated, so the C1 leak range is [0.06, 0.3] (slowest leak sets J ~= 0.96 in-band from the start). Full R0/C0/C1/C2-C4 grids deferred to GPU via `narma_linear_controls_batch.sh`.
- 2026-09-07 (v6): Alliance full-stream linear-controls results landed (`outputs_alliance_can/linear_controls`, H100, NARMA-10 seed 0, 1 stream x 2500 samples, washout 200; CPU/GPU preflight bit-identical):
  - **R0** (h25, 8 corners, 184 s): fabric ridge 0.8996-0.9042 across ALL variations (full-state vs hidden-only, k8-frozen vs dynamic, drive 1.0 vs 0.5 -- max spread 0.005, so the gap is structural, not diagnostic); long-stream leg (10000 samples) 0.7190, exactly reproducing the independent P0 untrained diagnostic 0.7194 (Ridge instrument cross-validated; short-stream 0.90s are noisy hundred-sample estimates). Raw-delay ridge 0.3838 -> matched parity Delta +0.5158 vs +0.03 tolerance: **rule_holds=False, decisively**.
  - **C0** (ESN h25, sr 0.9): ridge 0.771, **MC 8.07** with textbook delay decay (0.998 -> 0.13 over delays 1-10, floor after), analytic Jacobian max|eig| 0.72 (spread 0.13-0.72), **PR 3.02**. Pipeline valid (true memory measurable; fabric verdicts stand), but a *healthy* MC-8 ESN scores PR 3 -- the spec's PR 15-25 expectation is miscalibrated (memory lives in low-variance directions raw-covariance PR cannot see).
  - **C1** (h49, 2461 params, J = 0.9447 in-band after 1 tuning iteration, rails 0.5%): ridge **0.6179** (hidden-only readout identical), **MC 2.530** (per-delay 0.32 -> 0.07 smooth decay; hidden-only 2.543), **PR 1.12**, parity Delta +0.2119 vs raw-delay 0.4060. **FAIL on all three gates** (need MC >= 3, PR >= 6, Delta <= 0.03). Genuine but weak fading memory -- integrator/carryover/collection work; scalar input + weak mixing (W kept tiny for the J-band) leaves nodes as correlated leaky filters (PR ~1). Raw-delay taps (21 params) beat 50 reservoir dims on the same stream with C1 holding the overfit advantage, so the gap is real. Instrument wart: delay-19 MC = -159.1, one ill-conditioned ridge blowup (total protected by max(r,0)). C2C3C4 correctly skipped (`suppressor: c1-not-pass`); suppressor not localized. Note: zero-training h49 linear control ridge 0.6179 ties the trained 25-node fabric frozen-state ridge 0.6434 from two independent directions (stream provenance differs -- see re-measurement below).
  - External review verdicts adopted (plan `c1b-revision`, feature spec `c1b-protocol`): (a) void list -- E0 MC column (ridge/PR/rail stand), P1 MC 0.24, E2 "MC -> 0.00" leg (NRMSE collapse stands), v1/v2 "MC ~= 0.2" framing, Sec. 9 "MC clearly above 0.5" criterion, E0/C1 MC > 3 wordings pending re-anchor; (b) parity gate DROPPED for linear controls (delay line is the optimal linear basis -- C1 was gated to fail on arrival); (c) PR demoted to diagnostic, re-anchored to C0 (PR_ref = 3.02); new memory gate **MC >= 4** (~2x inside MC_ref = 8.07); (d) per-delay MC gets standardized features + truncated-SVD fallback; (e) long-stream R0 leg promoted to canonical reference, streams >= 2500 protocol constant.
  - **C1b revision implemented** (parallel agent; `probe_audit2.py` extended, ALL_AUDITS_OK incl. C1B/C1B-real install-restore roundtrips, QR unit-radius check, tuner runs, sidecar roundtrip + missing-.pt guard): C1b-override CLI (`c1b`: QR-of-Ginibre orthogonal W auto-scaled to J-band **[0.97, 0.99]**, uniform leak 0.05 non-programmable, PASS = MC >= 5 pre-registered 5-15 + rails <= 5% + J in band, tuned W/leak/G_in persisted to `c1b_tuned_base.pt`+JSON sidecar); C3 redefined as tanh-crossover sweep (`--c3-gm-grid` default -8..3, collapse = MC <= 0.5 x C1b MC) loading the sidecar; C2/C4 as degradation tests on the persisted base (fixes the latent bug where legs rebuilt an untuned base); standardized-state PR field; long-stream R0 reference constant.
  - **C1b-real deviation found during implementation (audit, same session)**: the C1b max|J| band [0.97, 0.99] is UNREACHABLE for a conductance Laplacian -- the near-zero Laplacian mode pins max|J| at the leak mode ~(1-dt*leak)^steps ~= 0.945 regardless of G (verified G/4 and G*4 both pin max ~0.945; max|J|(G) is U-shaped with minimum at G -> 0, band crossed only on the unstable shoulder). C1b-real (`c1b_real`: gm/isat -> min so shunt dominates, softplus-inverse G targets, real rhs, no override) therefore gates on spectral UNIFORMITY (min|J| >= 0.90, tiny-net measurement 0.8335 -> 0.9126 at G/4) with max|J| <= 0.995 as stability guard; max|J|-vs-band reported diagnostic-only.
  - **Pending (no results yet)**: C1b-override full-stream GPU run (falsifies mixing hypothesis iff MC >= 5); C1b-real GPU run (translation to real parameterization); C3 crossover sweep (needs passing C1b base); targeted high-gm E0 corners under fixed MC (no local corner ever hit PR >= 6 -- re-measure step contingent on P2 GPU grid); k2 frozen-state MC recollect on full stream locally (ckpt present); k8 recollect + full P2 E0 grid on Alliance.
- 2026-09-11 (v7): C1b-override / C1b-real / C3-sweep full-stream GPU results landed (`narma_linear_controls-58460596.out`, H100, NARMA-10 seed 0, h49, 1 stream x 2500 samples, washout 200; R0/C0/C1/C2C3C4 skipped as already-complete; CUDA preflight tiny-C1 reproduces local behavior):
  - **C1b: FAIL, decisively -- and worse than C1.** J = 0.9950 in the [0.97, 0.99] band after 13 tuning iterations (tuner works on GPU, same trajectory as the local stub), rails 2.7% pass -- every passable gate passes. The memory gate fails hard: **MC = 1.623** (need >= 5, pre-registered 5-15), ridge-from-states **NRMSE = 2.43 (R^2 = -4.89)** -- worse than predicting nothing. Mechanism: with clip disabled, the ~50 near-identical slow modes act as integrators (DC gain ~1/leak) and balloon state variance (sat_max_ratio 2.88) until the fixed-lambda ridge drowns. PR raw 2.15 / standardized 2.82. Parity diagnostic +2.02 (not a gate). Tuned sidecar written (`c1b_tuned_base.pt` + JSON), so C3 ran on the genuine base.
  - **Reading: weak-mixing hypothesis falsified as the sole cause.** The perfect mixer (MC 1.62) underperforms the random mixer C1 (MC 2.53, ridge 0.62): orthogonalizing destroyed timescale diversity -- dozens of ~200-step integrators on a 10-step task give the readout nothing distinguishable. Per the c1b-protocol fork this is the FAIL branch -> root-cause Heun/carryover/state-handling (instruments proven by C0 MC 8.07). Caveat: C1b's band may be too slow for NARMA-10, so the fail is overdetermined -- but C1 failing at the faster band keeps the verdict honest. The fabric's linear regime holds ~2 units of memory vs the ESN's 8; the deficit is real, not a band artifact.
  - **C1b-real: FAIL with asterisk (partially self-inflicted).** MC = 1.35, ridge 0.74, rails 0.0%, but **PR = 1.00** (total rank collapse) and the uniformity gate failed: the tuner ran 11 iterations then stopped at the MIN_SCALE floor (2^-11 ~= 1e-3), driving core G to ~zero chasing a uniformity floor unreachable on h49 -- the ref/output-ode tanh families (left at defaults) inject fast modes no core-G tuner can touch. Do NOT read as "the resistive core has no memory"; needs redesign (quiet ALL tanh families, or gate on the core-subspace spectrum) before its verdict counts.
  - **C3 sweep: crossover = None (vacuous as a gate, informative as physics).** Base MC 1.62 is too low for the 50%-collapse rule (threshold 0.81) to fire on any row. What the grid does show: gain brings memory -- MC peaks at **2.21** with best-in-file ridge **0.65** at gm_raw_fill = -2 -- but arrives hand-in-hand with rail saturation (1.4% -> 50% across the grid) and Jacobian explosion (**235 -> 1325**, i.e. unstable) at gm >= -2. The crossover is a cliff, not a sweet spot: no clean corner where nonlinearity arrives before saturation. Supports H2 (compliance/boundary-drive suppresses the echo-state regime).
  - **Anchors re-confirmed:** C0 ESN MC = 8.07 (memory instruments valid) though its readout NRMSE = 0.77 stays off-band (task-setup side-eye, cf. H5 -- does not invalidate MC comparisons); R0 parity rule_holds = False (delta +0.52, fabric ridge far above raw taps).
  - **Next (fork: Heun/carryover/state-handling):** cheapest discriminators are (a) frozen-state MC recollect on the C1b states -- is the 1.6 even in the readout-visible slice? -- vs (b) a clip-re-enabled Heun-map test -- does ridge recover when slow modes cannot blow up? C1b-real redesign deferred until the fail-branch root cause lands.


