---
plan name: kn-bayes-fixes
plan description: Fix BO seed, lambda, bounds issues
plan status: done
---

## Idea
Address the 4 audit findings from kn-bayes-trim in kn_bayes_opt.py:

1. SEED SEMANTICS (#1/#7): make `--seed` CLI flag always win over the per-problem EXTRA_FLAGS. Currently `_build_command` (kn_bayes_opt.py:356-359) appends EXTRA_FLAGS_FOR_PROBLEM (which hardcode `--seed 100`) and only emits `--seed {seed}` for non-EXTRA problems, so `--seed` is silently ignored for friedman1/friedman2/smooth2d and `best_hyperparams.txt` writes a misleading `seed: 0`. Fix: remove `--seed 100` from the 3 EXTRA_FLAGS entries, change `--seed` default 0→100 (preserves validated START_POINTS reproducibility), always emit `--seed {seed}` after EXTRA_FLAGS so CLI wins, update help text. Note: TPE sampler seed will now be 100 (was 0) — deterministic change to exploration stream, acceptable.

2. DEVICE_L2 'OFF' (#2): START_POINTS uses `device_l2_lambda: 1e-7` (effective-off) instead of true 0.0, so the seed trial does NOT exactly reproduce the validated run. True 0.0 cannot pass through `suggest_float(..., log=True)` (optuna `to_internal_repr` = math.log(0) → ValueError crash). Fix: change START_POINTS values to 0.0, and in objective() use a conditional distribution — linear `[0.0, device_l2_lambda_max]` for the seed trial (fixed 0.0 in range, no crash), log `[min, max]` for all other trials. train_script already treats `>0.0` as penalty-on, so 0.0 = penalty fully off.

3. NUM_HIDDEN_MIN (#5): `default_min` from DATASETS num_hidden_range is unpacked but unused (kn_bayes_opt.py:495-499); num_hidden_min falls back to `max(2, in_dim*1)`, allowing exploration below documented range (friedman1→10 vs 20, smooth2d→2 vs 8). Fix: `else max(default_min, in_dim * min(FANOUT_COUNT_CHOICES))`. Verify START_POINTS still pass (enqueue bypasses suggest bounds).

4. RESUME LOSES SEED (#4): enqueue gated on `not args.resume` (line 546-549) while is_seed_trial uses `trial.number==0` (line 618-622). Resuming an empty DB or a study where trial 0 failed/pruned never re-enqueues the seed, but a resumed trial-0 would be mislabeled seed → random small num_hidden with the friedman1 boundary map (targets up to 24) → train_script out-of-range crash. Fix: compute `seed_trial_number` before optimize — enqueue the START_POINTS if dataset is seeded and no COMPLETE trial has `user_attr seed_trial=True`, set seed_trial_number = max(existing numbers)+1 (0 for fresh); is_seed_trial becomes `seed_trial_number is not None and trial.number == seed_trial_number`. Also fixes the misleading `seed_trial=` print (line 564).

Verification: py_compile; 1-trial friedman2 seed smoke (asserts device_l2=0.0 emitted as --device-l2-lambda 0.0, --seed 100, seed boundary map); 2-trial smoke (1 seed + 1 TPE) to confirm optuna handles mixed device_l2 distributions; --resume-with-empty-DB smoke (new output dir) confirming seed re-enqueued and correctly labeled; --resume-after-seed-complete smoke confirming no double enqueue. Update project memory block kn-bayes-opt.

## Implementation
- Edit argparse `--seed` (kn_bayes_opt.py:411-413): default 0→100, update help text to state it controls both TPE sampler and every train_script subprocess seed and always overrides per-problem defaults
- Remove `--seed`, `100` from all 3 EXTRA_FLAGS_FOR_PROBLEM entries (kn_bayes_opt.py:164-198)
- Edit `_build_command` (kn_bayes_opt.py:356-359): replace if/else so `--seed {seed}` is always appended AFTER EXTRA_FLAGS (last-wins for argparse); keep friedman target-noise-std line
- Change START_POINTS `device_l2_lambda` from 1e-7 to 0.0 for friedman1/friedman2/smooth2d (lines 133, 146, 159)
- Fix `num_hidden_min` (kn_bayes_opt.py:496-499) to `else max(default_min, in_dim * min(FANOUT_COUNT_CHOICES))` and update the `--num-hidden-min` help text (line 424-426)
- Rework seed-enqueue logic in main() (kn_bayes_opt.py:546-549 + 564): compute `seed_trial_number` closure var — enqueue START_POINTS when dataset is seeded AND no COMPLETE trial has user_attr seed_trial=True; seed_trial_number = max(existing trial numbers)+1 (0 for fresh); print actual seed_trial_number; remove `not args.resume` gate
- Edit objective(): move is_seed_trial computation (was line 618-622) to top using `seed_trial_number` closure; use conditional device_l2_lambda distribution (linear [0.0, max] when is_seed_trial, log [min, max] otherwise) at lines 610-612
- Update best_hyperparams.txt `seed:` line comment if needed (now correctly shows args.seed=100); verify no stale references to trial.number==0 remain
- Run verification: py_compile; 1-trial friedman2 seed smoke asserting --device-l2-lambda 0.0, --seed 100, boundary map, device_l2 off; 2-trial seed+TPE smoke (mixed distribution); --resume empty-DB smoke (fresh dir) confirming seed enqueued + labeled; --resume after completed seed confirming no re-enqueue
- Update project memory block kn-bayes-opt (device_l2_lambda seed value 0.0, seed semantics, resume behavior) and mark plan done

## Required Specs
<!-- SPECS_START -->
- lstm-baseline-complexity
- bo-seed-fixes
<!-- SPECS_END -->