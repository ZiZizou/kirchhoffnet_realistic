# Spec: kn-bayes-turbo

Scope: feature

# kn_bayes_opt.py Turbo Mode (optuna-native TuRBO)

## Purpose
Add a `--turbo` flag to `kn_bayes_opt.py` that switches the Optuna sampler from
`TPESampler` to the optuna-ecosystem TuRBO sampler (Trust-Region Bayesian
Optimization) instead of hand-rolling a TuRBO loop.

## Sampler choice (researched decision)
- optuna core 4.9.0 has NO built-in TuRBO sampler (`TPESampler` has no turbo
  param; `GPSampler` is global GP-BO without trust regions).
- optuna-integration's `BoTorchSampler` previously had `turbo`/`min_trials_for_turbo`
  but those params were removed; current signature is
  `BoTorchSampler(*, candidates_func, constraints_func, n_startup_trials,
  consider_running_trials, independent_sampler, seed, device)` and is NOT TuRBO.
- BoTorch's own TuRBO is a tutorial loop (`TurboState`, `generate_batch`,
  `update_state`) that must be re-implemented per project — NOT a drop-in class.
- THEREFORE: use **OptunaHub `samplers/turbo` → `TuRBOSampler`**
  (`optunahub.load_module(package="samplers/turbo")`). It is pure optuna (built on
  `optuna._gp`), needs only `scipy` + `torch` (both installed), and requires the
  small `optunahub` pip package. **No botorch dependency.**

## TuRBOSampler API
```
TuRBOSampler(*, n_startup_trials=4, n_trust_region=5, success_tolerance=3,
             failure_tolerance=5, seed=None, independent_sampler=None,
             deterministic_objective=False, warn_independent_sampling=True)
```
- `n_startup_trials` = initial trials PER TRUST REGION.
- Categorical parameters are NOT supported → pinned (see below).
- Int/float params are supported (discrete via `optuna._gp`).

## Categorical pinning
The 5 categorical dims are pinned to START_POINTS defaults in turbo mode:
- `fanout_count = 2`, `small_world_k = 4`, `batch_size = 4096`,
  `freeze_boundary = 0`, `freeze_temporal_read = 0`
The objective must NOT call `suggest_categorical` for these in turbo mode, so no
CategoricalDistribution is registered in the study search space.
Turbo therefore searches the 10 continuous dims:
`num_hidden` (int), `num_stages` (int), `vca_rank` (int), `t_span`, `lr` (log),
`weight_decay` (log), `x_max`, `gm_max` (log), `isat_max` (log),
`device_l2_lambda` (linear). `num_steps` stays derived; `sparsity_lambda`,
`entropy_lambda`, `small_world_p` stay fixed constants.

## Parallelism
- TuRBO keeps one trust region → force `n_jobs = 1` (serial trials).
- Print a notice when the user also passes `--n-workers > 1`.

## Seed trial / START_POINTS
- Keep enqueueing the START_POINTS seed trial (trial 0) in turbo mode.
- `study.enqueue_trial` with extra non-suggested keys is already tolerated
  (sparsity/entropy/num_steps precedent), so pinning via constants is safe.
- Seed-trial reproducibility is preserved because pinned categoricals equal the
  START_POINTS values by definition.

## Resume guard
- Optuna allows loading a study with a different sampler (no error), so resuming a
  prior TPE study under `--turbo` would leave COMPLETE trials with registered
  categorical distributions → incompatible with TuRBO's continuous search space.
- Turbo mode targets FRESH studies; if the study DB already exists at startup,
  print a strong warning recommending a fresh `--output` / study name, then
  proceed. No hard block.

## Failure policy
- If `optunahub` is missing, `load_module` fails, or the module is incompatible
  with optuna 4.9.0 → HARD ERROR with actionable message (no silent fallback to
  TPE).

## Reporting / artifacts
- `best_hyperparams.txt`: add `sampler`, `turbo`, `n_startup_trials`,
  `n_trust_region` lines.
- `results.csv` + `best_hyperparams.txt` params: turbo trials have no categorical
  keys in `trial.params` → fall back to the pinned values dict when writing CSV.
- `objective_history.png` title gains a `[turbo]` marker.
- Startup banner prints the active sampler class.

## CLI additions
- `--turbo` (store_true)
- `--turbo-n-startup` (int, default 4)
- `--turbo-n-trust-regions` (int, default 5)
- Validate: n_startup >= 1, n_trust_regions >= 1.

## Determinism
- Objective is deterministic (fixed per-trial seed) → `deterministic_objective=True`.

## Testing
- `py_compile` kn_bayes_opt.py.
- Smoke (`--epochs 1 --n-trials N`): seed trial runs; non-seed trial params contain
  only continuous dims; CSV shows pinned categoricals; `n_jobs` forced to 1;
  banner shows TuRBOSampler; resume-from-TPE-study warns.