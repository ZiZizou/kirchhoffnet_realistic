# Spec: dagger-epoch-best

Scope: feature

# DAgger per-epoch best-model checkpointing + seeded early-stop eval

## Problem
The DAgger student training loop in `dagger-nuance-distillation-kirchhoffnet.py` evaluates the
validation failure rate only every 10 epochs, and each evaluation draws a **fresh unseeded**
1000-spec set (`sample_validation_specs`, line 2152). Consequences:

1. A failure-rate dip between the 10-epoch checks (e.g. the 7.8%/7.9% moments in the log) is
   never observed, so the carried-forward model is only the best *checked* epoch.
2. The unseeded draws make failure rates fluctuate (7.8%–13.1%) largely from sampling noise, so
   the "best" model is not reliably identifiable and per-iteration `val/adaptive` numbers
   (fresh 2000-spec draws) never match the tracked best.

## Goal
Carry forward the best-performing student model **within each epoch** — i.e. the model with the
lowest failure rate on a fixed, seeded validation set — and make that the model that is restored
at iteration end and persisted in `dagger_checkpoint.pt`.

## Design
- **Seeded evaluation set per iteration.** `sample_validation_specs()` gains a `seed` parameter.
  One fixed set of size `EARLYSTOP_EVAL_SIZE` (default 300) is sampled once per DAgger iteration
  and reused for every early-stop check, so all checks compare identical specs (noise removed).
- **Every-epoch evaluation.** The early-stop failure-rate check runs every epoch
  (`EARLYSTOP_EVAL_EVERY = 1`) instead of every 10. Cost is bounded by using the smaller seeded
  set (~3x the prior early-stop eval cost overall). The verbose `log_failure_breakdown` line is
  still throttled to `EARLYSTOP_LOG_EVERY` (default 10) to avoid log spam.
- **Best-state tracking unchanged in spirit.** `best_failure_rate` / `best_failure_state` still
  update only on strict improvement, with `patience` early-stop logic intact. The val-loss-based
  `best_state` (line 3073) remains the fallback if no failure-rate eval produced a state.
- **Explicit carry-forward confirmation.** At iteration end, log the best failure rate observed
  and which state was restored (failure-rate best vs val-loss fallback). `save_checkpoint`
  (line 3273) snapshots `student.state_dict()` after the restore, so the best model is what
  resumes next iteration — verify and log this.

## Non-goals / scope
- The final per-iteration validation (`VALIDATION_SIZE = 2000`, line 3105) stays fresh/unseeded;
  it is the reporting metric for `dagger_history` and the convergence check.
- No changes to teacher labeling, dataset aggregation, optimizer, or scheduler behavior.
- No changes to other callers of `sample_validation_specs` (they pass seed=None).

## New hyperparameters
- `EARLYSTOP_EVAL_SIZE = 300` — specs per early-stop check (per-iteration seeded set).
- `EARLYSTOP_EVAL_EVERY = 1` — check failure rate every N epochs.
- `EARLYSTOP_LOG_EVERY = 10` — emit `log_failure_breakdown` every N epochs.
- Per-iteration seed: `EARLYSTOP_EVAL_SEED_BASE + dagger_iter * 7919` (stable within iteration).
- All registered in `_HYPERPARAMS` (lines 392-456) for the `[CFG]` / `[CFG_JSON]` log dump.

## Testing
- `python -m py_compile dagger-nuance-distillation-kirchhoffnet.py` (local, syntax-only).
- Remote smoke run: `EPOCHS_PER_ITER=3`, `EARLYSTOP_EVAL_SIZE=50`, `DAGGER_ITERATIONS=1` to
  confirm per-epoch eval, one seeded set per iteration, correct restore, and new [CFG] keys.