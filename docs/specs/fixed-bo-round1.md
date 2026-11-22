# Spec: fixed-bo-round1

Scope: feature

# Spec: fixed-bo-round1 (feature scope)

## Goal
Close the 5x gap between the fixed-distillation KNet BO best
(`val_logit_mse=0.00397`, trial 3 of `fixed_ctle_knet`, 20 trials) and the
manual Kaggle KNet (`0.00073`) / MLP BO (`0.00035`) references, inside one
~6h Alliance allocation: **20 trials, two-rung rank-gated halving, fresh
study** (resumable with identical flags).

## Root causes this addresses (DB evidence, `outputs_alliance_can/study.db`)
- Fixed probe bar (`probe-mse-max 0.035` @25ep, applied mid-study) killed 7/13
  strict-phase trials inside the noise band; killed `arch_idx=1` twice at
  0.039/0.054 without a full run. Kill decisions must be **peer-ranked**, not
  thresholded.
- Manual winner `(hidden=14, stages=4, k=4, rank=2)` is outside the 17-arch
  `[4250,5000]` feasible set. Budget must admit it.
- `kn_bayes_opt` (synthetic, where KNet wins) searches `t_span`, `x_max`,
  `weight_decay`, wide lr/batch, gm/isat; fixed-BO searches arch+lr+batch only.

## Search space (6 dims, TPE-native: categoricals allowed, TuRBO rejected)
- `knet_arch_idx`: joint `(hidden, stages, k, rank)` categorical over the
  prior-biased feasible set at `--param-budget 7000 --param-tolerance 0.15`
  (window `[5950, 7000]`; admits hidden-14; exactly 37 tuples, verified).
  Fanout diagonal `{i:[i,i+4]}`, `p=0.2`, `tanh_free`, freeze-read/temporal on.
  (A wider window covering the literal DB arch `(10,3,4,4)`=4482 admits 71+
  tuples — too dilute for 20 trials; rejected during audit.)
- `lr` log-uniform `[2e-4, 5e-3]`; `batch_size` {128, 256, 384, 512};
  `x_max` `[2, 8]`; `weight_decay` log-uniform `[1e-6, 1e-2]`;
  `solver_scale` `[0.5, 2.0]` (scales per-stage `t_span` + `num_steps` together).
- All five are param-count-neutral by construction; arch tuple carries the count.
- Deferred to round 2: `gm/isat_max` (needs cell-library CLI plumbing),
  `small_world_p` (moves edge count, breaks exact counts).

## Harness contract (`fixed-mlp-distillation-kirchhoffnet.py`)
- New `--kn-solver-scale` (default 1.0) threading into `KirchhoffStudent`
  stage `t_span`/`num_steps`; `base_command` passes sampled `--weight-decay`
  (today forced `"0"`) and `--kn-x-max` (today default 4.0).
- **Count-invariance**: `--count-params-only` must be identical across all
  `solver_scale`/`x_max`/wd/batch corners for a fixed arch (verified, not assumed).

## Controller contract (`fixed-distillation-bayes-opt.py`)
- Sampler: `TPESampler(multivariate=True, group=True, seed, n_startup_trials=4)`.
- Seeds: `--seed-trial-from` (repeatable) enqueues TWO trial-0s by arch
  **tuple** (+lr/batch), re-resolving the categorical index against the new
  feasible list (indices shift with budget; never persist raw indices):
  manual `(14,4,4,2) lr=1e-3 b=256` (6775 params), DB trial-3 topology scaled
  to hidden-14 `(14,3,4,4) lr=3.3e-3 b=128` (6110 params; the literal DB arch
  `(10,3,4,4)`=4482 falls below the window and is skipped by design).
  First-`--halving-warmup` trials always promote, which covers both seeds.
- Multi-fidelity (two rungs, fewer moving parts): rung 1 = 15ep, rung 2 = 100ep
  via existing `--resume` checkpoint chain. Promote during warmup
  (first `--halving-warmup` trials) or iff lexicographic `(failure, mse)`
  rank `< max(1, N//eta)` (eta=3). Killed trials return
  **finite rung values** — never `TrialPruned` (optuna#5260), never inf
  (optuna#3676). Dual `MINIMIZE` objectives retained; no scalarization.
- Fingerprint now covers sampler settings + rung config `(15, 100, eta=3)` +
  dim ranges + pin/seeds, closing the mid-study threshold-change hole found
  in the DB (early trials gated at 0.15, later at 0.035).
- Live artifacts extended, not replaced: `trial_table.txt` gains lr/wd/xmax/
  solver-scale/batch/rung/promoted columns; `results_live.csv` gains rung +
  scalar-rank columns; all rank decisions appended to `trial_log.jsonl`.
- Resume: identical flags + `--resume` continues (stale RUNNING trials warn,
  per existing guard). Any flag change fails fast via fingerprint.

## Budget
20 trials: all pay 15ep (~5 min at 36 min/100ep); ~7 promoted pay +85ep.
Total ~= 900 epoch-equivs ~= **5-6h**, inside one allocation; `--resume`
covers allocation-boundary kills.

## Acceptance
- Fresh-study best in-window `val_logit_mse <= 0.001` (manual 0.00073 in reach
  once hidden-14 is legal; MLP 0.00035 is stretch).
- Zero `TrialPruned`/`None`-values/inf crashes; `probe_study_db.py` clean.
- `py_compile` clean; count-invariance check passes (6775 across solver_scale
  and x_max corners); 27-guard unit suite passes; CPU smokes cover every
  branch (hard-cap kill, rank kill with `kill_reason=rank`, warmup/rank
  promote, rung2 resume, select-from-killed-inclusive pool).
- Post-study: per-target (7-dim) error breakdown vs manual/MLP refs decides
  round 2 (readout/weighting work vs wider space).