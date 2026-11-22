# Spec: bo-seed-fixes

Scope: feature

# BO seed/lambda/bounds/resume fixes (kn_bayes_opt.py)

Feature spec for plan `kn-bayes-fixes`. Addresses 4 audit findings from the kn-bayes-trim implementation.

## 1. Seed semantics: `--seed` CLI always wins
- Change argparse `--seed` default from 0 to 100 (preserves validated START_POINTS reproducibility; also seeds the TPE sampler — deterministic change to exploration stream, acceptable).
- Remove `--seed 100` from all 3 EXTRA_FLAGS_FOR_PROBLEM entries (friedman1/friedman2/smooth2d).
- In `_build_command`, always append `--seed {seed}` AFTER the EXTRA_FLAGS so argparse last-wins makes the CLI value authoritative.
- Update `--seed` help text: "Seed for both the TPE sampler and every train_script.py subprocess; overrides per-problem defaults (default: 100)."
- `best_hyperparams.txt` `seed:` line (writes `args.seed`) then correctly reflects the effective training seed.

## 2. device_l2_lambda "off" representation
- Change START_POINTS `device_l2_lambda` from `1e-7` to `0.0` for friedman1/friedman2/smooth2d so the seed trial EXACTLY reproduces the validated baseline (train_script treats `>0.0` as penalty-on, so 0.0 = penalty fully off).
- True 0.0 crashes optuna's `to_internal_repr` under `log=True` (math.log(0) ValueError), so in objective() use a conditional distribution:
  - seed trial: `suggest_float("device_l2_lambda", 0.0, device_l2_lambda_max)` (linear, fixed 0.0 in range → no crash, recorded in trial.params).
  - other trials: `suggest_float(..., min, max, log=True)` unchanged.
- Exploration can never hit 0.0 (log floor 1e-7) — only the seed can express "off". Acceptable.

## 3. Honor num_hidden_range default_min
- `num_hidden_min` fallback (kn_bayes_opt.py:496-499) currently ignores `default_min` from DATASETS num_hidden_range, letting exploration go below the documented range (friedman1→10 vs 20, smooth2d→2 vs 8).
- Fix fallback to `max(default_min, in_dim * min(FANOUT_COUNT_CHOICES))`.
- Verify START_POINTS unaffected (enqueue bypasses suggest bounds).
- Update `--num-hidden-min` help text accordingly.

## 4. Resume no longer loses the seed trial
- Replace the `not args.resume` enqueue gate + `trial.number == 0` is_seed_trial check with a `seed_trial_number` closure:
  - Before `study.optimize`: if dataset is in START_POINTS AND no COMPLETE trial has `user_attr seed_trial == True`, call `study.enqueue_trial(START_POINTS[dataset])` and set `seed_trial_number = max(existing trial numbers) + 1` (0 for a fresh study).
  - In objective(): `is_seed_trial = seed_trial_number is not None and trial.number == seed_trial_number`. Use this value for the boundary map, user attrs, `[SEED]` tag, and the device_l2 conditional distribution.
- Fixes: resume-on-empty-DB (previously trial 0 mislabeled seed → friedman1 boundary map targets up to 24 with random small num_hidden → train_script out-of-range crash); resume after failed/pruned trial 0 (seed silently never runs).
- Update the `seed_trial=` startup print (line 564) to report the resolved seed_trial_number.

## Verification (all CPU, --epochs 1)
1. `py_compile` clean.
2. 1-trial friedman2 seed smoke: log shows `--device-l2-lambda 0.000000e+00`, `--seed 100`, seed boundary map `{"0":[2,12],"1":[7,17],"2":[22,5],"3":[10,15]}`, `[SEED]` tag, results.csv device_l2_lambda=0.0, best_hyperparams seed: 100.
3. 2-trial (1 seed + 1 TPE) smoke: confirms optuna handles the mixed device_l2 distributions without error.
4. `--resume` with a fresh (empty) DB: seed re-enqueued at trial 0 and labeled `[SEED]`.
5. `--resume` after a completed seed trial: no double enqueue; new trials are not labeled seed.

## Acceptance criteria
- No `trial.number == 0` logic remains for seed detection.
- Seed trial params exactly equal START_POINTS (device_l2_lambda=0.0) and produce penalty-off in train_script.
- Exploration bounds respect documented num_hidden_range mins.
- `--seed N` on the CLI produces `--seed N` in every trial's logged command, overriding defaults.