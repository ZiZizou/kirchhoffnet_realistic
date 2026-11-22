---
plan name: dagger-best-checkpoint
plan description: Carry forward best per-epoch model
plan status: active
---

## Idea
In dagger-nuance-distillation-kirchhoffnet.py the DAgger student training loop only evaluates the validation failure rate every 10 epochs (dagger-nuance-distillation-kirchhoffnet.py:3078), and each evaluation draws a fresh unseeded 1000-spec set (sample_validation_specs, line 2152), so (a) dips in failure rate between checks are never captured and (b) the 7.8%-13.1% swings are largely sampling noise. The user wants the best-performing model observed WITHIN each epoch to be the one carried forward (restored + persisted). Fix: (1) give sample_validation_specs a seed parameter and build ONE fixed seeded early-stop spec set per DAgger iteration so all checks compare the same specs; (2) evaluate failure rate every epoch (EARLYSTOP_EVAL_EVERY=1) on a smaller set (EARLYSTOP_EVAL_SIZE=300) to bound the ~3x added cost; (3) keep/strengthen best_failure_state tracking and add an explicit log line at iteration end reporting the best failure rate observed and which state was restored; (4) verify the restored best model is what gets persisted in dagger_checkpoint.pt (already the case via save_checkpoint at line 3273, which snapshots student.state_dict() after the restore). The existing val-loss-based best_state (line 3073) stays as fallback. Final per-iteration VALIDATION_SIZE=2000 evaluation stays fresh/unseeded as the reporting metric.

## Implementation
- Add a `seed=None` parameter to `sample_validation_specs()` (line 2152): when not None, pass `random_state=seed` to the boundary/interior `.sample()` calls and to the final shuffle so the returned spec set is deterministic. Existing callers (per-iteration val eval, line 3105) keep seed=None and behavior unchanged.
- Add new constants near the other DAgger knobs (around line 486): `EARLYSTOP_EVAL_SIZE = 300`, `EARLYSTOP_EVAL_EVERY = 1`, `EARLYSTOP_LOG_EVERY = 10`, and a per-iteration seed formula (e.g. `EARLYSTOP_EVAL_SEED_BASE + dagger_iter * 7919`). Register them in `_HYPERPARAMS` (lines 392-456) so they appear in the [CFG]/[CFG_JSON] log dump.
- In the iteration loop, before the epoch loop (near lines 3009-3014), build the fixed early-stop set ONCE per iteration: `earlystop_specs = sample_validation_specs(df, n_samples=EARLYSTOP_EVAL_SIZE, boundary_ratio=BOUNDARY_RATIO, seed=<per-iter seed>)`.
- Rework the early-stop block (lines 3077-3092): evaluate failure rate every epoch when `(epoch+1) % EARLYSTOP_EVAL_EVERY == 0` (default 1 => every epoch) using the fixed `earlystop_specs` instead of a fresh draw; keep `best_failure_rate`/`best_failure_state`/`patience` logic; only emit the `log_failure_breakdown` line when `(epoch+1) % EARLYSTOP_LOG_EVERY == 0` to avoid log spam.
- After the loop (lines 3094-3100), add an explicit log line reporting the best failure rate observed this iteration and whether `best_failure_state` (vs the val-loss `best_state` fallback) was restored, so the carried-forward model is verifiable in the log.
- Verify persistence: `save_checkpoint` (line 3273) stores `student.state_dict()` AFTER the restore, so the best model is already what resumes next iteration; add a guard/log so a missing `best_failure_state` (never evaluated) falls back to `best_state` and is logged rather than silently accepted.
- Test: run `python -m py_compile dagger-nuance-distillation-kirchhoffnet.py` locally (syntax-only, imports not executed), then on the remote host run a tiny smoke run (EPOCHS_PER_ITER=3, EARLYSTOP_EVAL_SIZE=50, DAGGER_ITERATIONS=1) to confirm per-epoch eval, single seeded set per iteration, restore of best_failure_state, and that the [CFG] log shows the new constants.

## Required Specs
<!-- SPECS_START -->
- lstm-baseline-complexity
- dagger-epoch-best
<!-- SPECS_END -->