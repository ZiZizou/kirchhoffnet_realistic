# Spec: knet-bo-fix

Scope: feature

# Fixed Distillation KNet BO

## Overview
Multi-objective Bayesian optimizer for KNet using the same fixed CTLE teacher-distillation benchmark as the MLP arm, but with critical fixes to sample efficiency and search focusing.

## Objectives
- Achieve KNet validation ZIG failure-rate comparable to the existing PlainMLP BO baseline (target ≤ 0.06% on validation, best KNet ~0.0073%?)
- Match or exceed the MLP BO best validation logit MSE (~0.00035) from trial 8 of `fixed-plainmlp-bayes-opt.py`
- Optimize within hard budget `[4250, 5000]` and 100 epochs (parity with existing BO arms)

## Rationale
The previous `fixed-knet-bayes-opt.py` ran a massively under-explored, suboptimal search (deep diag fanout, wide hyper-param ranges, no seed, no early-stopping). This is the corrected version:

- **Search hard-wired to the validated fanout map** used by your proven 6775-param manual run (derived from `base_command`/`fixed_distillation_bayes_opt.py` sample_config, not the trivial diag map)
- **Pin highly effective hyper-param corners** where the handcrafted manual run succeeded or the MLP BO excelled:
  - `lr ∈ [1e-3, 5e-3]` log (exclude low-lr catastrophes)
  - `batch ∈ {128, 256}` (exclude large-batch catastrophes)
  - `x_max` fixed at `4.0` (exclude other extremes)
- **Bias arch space toward priors** that match your Alliance-bo best and manual results: `k=4` fixed, `stages ∈ {4,5}`, `rank ∈ {2,3,4}`, `hidden ∈ {8..14}`. This keeps KNet param counts ≤ 5000 while exploring the sub-space where near-zero logit error occurs.
- **Seed trial 0** matches a proven-manual-style config (stages 4, k4, rank 2, lr 1e-3, batch 256, x_max 4.0, validated fanout).
- **TPE-safe multi-fidelity gate** on 25-epoch probe (harness with `--prepare-fixed-dataset` + student train 25 epochs + evaluation).
  - When 25-epoch logit MSE + failure rate trend suggests convergence to the Pareto frontier, advance to full 100 epochs via `--resume`.
  - Stragglers (likely over-optimized for one metric) return finite 25-epoch values (never `TrialPruned`/inf) to keep multi-objective TPE (with pair of loss dimensions) happy (avoid optuna#5260 raggedness, optuna#3676 infinities).
  - Gate threshold calibrated from a 4-6 arch/LR combo pilot: correlation of 25-ep vs 100-ep Pareto ranks, set to keep ≥70% recall of survivors.
- **Every trial uses the same fixed teacher-labelled dataset** as the MLP arm (shared cache). The harness (`fixed-mlp-distillation-kirchhoffnet.py`) already supports `--fixed-dataset-path`, and we use the same shared path for KNet and MLP.

## Technical details
- Implements `objective` in `fixed-distillation-bayes-opt.py` (renamed `objective_knet` but same module) with the fixes above, using the existing `base_command` and `flat_args` helpers.
- Supports `--resume` across multi-objective studies (previous KB, same Dir).
- Reuses the existing Optuna study naming: `fixed_ctle_knet`.
- Includes a `metrics` helper that computes ZIG failure-rate and teacher-logit MSE from the harness' `metrics.json`; fail-safe for malformed results.
- Penalizes infeasible over-budget or startup errors with finite `PENALTY_VALUES = (1.0, 1e6)` to keep TPE happy.
- In `main`, adds a `seed_trial_number` counter (0 for fresh run) that enqueues the manual-style start config *after* creating the study if not resuming. Seeds count as trial 0, subsequent trials from 1.
- If `--resume` is passed, refuses a study with any `RUNNING`/`PRUNED`/`FAIL` states (the multi-obj TPE breakdown feature). This protects against legacy studies left by `fixed-knet-bayes-opt.py` runs.

## Workflow when called (with defaults)
1. Allocate a fresh KNet output directory (e.g., `outputs/fixed_knet_bo_plain_mlp_teacher`).
2. Prepare fixed-teacher dataset (expensive one-time cost, uses `--device cpu` to leave GPU for training).
3. Check resume-protection (study.db age, distributions fingerprint match, no running trials).
4. Compute and cache joint feasible architecture set under `[4250,5000]` with the new priors.
5. Seed trial 0 with manual-style config (stages 4, k=4, rank=2, lr=1e-3, batch=256, x_max=4.0).
6. For each trial:
  - Sample one arch+hyper from fixed lists (architectural + bias).
  - Pre-flight param count check.
  - Run the harness with `base_command` + flat_args for a 25-epoch student train.
  - Parse `metrics.json` from the 25-epoch run to get trial values.
  - Gate: if (MSE/probability-of-success correlation > threshold) advance to 100 epochs via `--resume` (harness will continue from `last_checkpoint.pt`). Else count as penalised.
7. After all trials, select best complete trial and run it to full 100 epochs (--bo-mode / --no-bo-mode in harness), then perform ZIG evaluation and dump `metrics.json`, `bo_summary.json`, `results.csv`.

## Comparison targets
- MLP: best from `fixed-plainmlp-bayes-opt.py` (trial 8: failure 0.0384%, logit MSE 0.00035)
- KNet: best from `kn_bayes_opt.py` Alliance run on ctle dataset (trial 43: failure 0.051%, logit MSE 0.0004) — same data, lower cost, by delivering a ~0.1-0.2% failure rate, 0.0006-0.001 logit MSE in full evaluation.

## Path to parity with MLP
- Sample efficiency: our joint architecture sampling guarantees every trial is param-budget feasible (0% penalty) and the prior bias already cuts arch dimensions by ~3×.
- Early gate: 25-epoch probe prevents large wasted epochs (approx 1/4 cost) on doomed spaces.
- Proven connectivity (validated fanout) ensures the search area is actually promising; BO now explores within a sub-space known to support KNet distillation.
- Combined, this should shrink total useful eval time to match or beat the MLP arm, even with 6× slower per-trial cost.