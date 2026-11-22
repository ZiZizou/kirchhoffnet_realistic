---
plan name: kn-bayes-turbo
plan description: Optuna-native TuRBO flag for kn_bayes_opt
plan status: done
---

## Idea
Add a --turbo flag to kn_bayes_opt.py that switches the optuna sampler from TPESampler to the optuna-ecosystem TuRBO sampler (OptunaHub samplers/turbo TuRBOSampler) for trust-region Bayesian optimization. Research showed optuna core/integration has no usable TuRBO: optuna-integration BoTorchSampler dropped its turbo param, and BoTorch's own TuRBO is only a tutorial loop. The OptunaHub TuRBOSampler is pure optuna (optuna._gp), needs only scipy+torch (already present) plus the small optunahub package — no botorch. Because TuRBOSampler does not support categorical params, the 5 categorical search dims (fanout_count, small_world_k, batch_size, freeze_boundary, freeze_temporal_read) are pinned to validated START_POINTS defaults, leaving a 10-dim continuous search (num_hidden, num_stages, vca_rank, t_span, lr, weight_decay, x_max, gm_max, isat_max, device_l2_lambda). Turbo mode forces n_jobs=1 (single trust region), keeps the seed-trial/START_POINTS mechanism, hard-errors if the turbo module is unavailable, and warns when resuming a prior TPE study (categorical distributions incompatible).

## Implementation
- Add dependency + import guard: pip install optunahub; add helper _build_turbo_sampler(seed, n_startup, n_trust_regions) wrapping optunahub.load_module(package='samplers/turbo') in try/except that HARD-ERRORS with an actionable message (missing optunahub, load failure, or optuna 4.9.0 incompatibility) — no silent fallback to TPE.
- Add CLI flags: --turbo (store_true), --turbo-n-startup (int default 4), --turbo-n-trust-regions (int default 5); validate n_startup >= 1 and n_trust_regions >= 1 near the existing arg validation block.
- Objective changes (turbo mode): replace the 5 suggest_categorical calls (fanout_count, small_world_k, batch_size, freeze_boundary, freeze_temporal_read) with constants pinned to START_POINTS defaults (2, 4, 4096, 0, 0) so no CategoricalDistribution registers in the search space; keep num_hidden/num_stages/vca_rank as suggest_int and the continuous suggests unchanged; keep the small_world_k >= num_hidden guard (dead but harmless); keep seed-trial boundary_fan_out handling.
- Sampler selection + parallelism: in main(), when args.turbo build TuRBOSampler(seed=args.seed, n_startup_trials=..., n_trust_region=..., deterministic_objective=True, independent_sampler=RandomSampler(seed=args.seed)) else TPESampler(seed=args.seed); force n_workers=1 in turbo mode with a printed notice when --n-workers > 1; print active sampler class in the startup banner.
- Resume guard: if args.turbo and the study DB already exists (db_path.exists()), print a strong warning that prior TPE trials carry categorical distributions incompatible with TuRBO and recommend a fresh --output / study name; proceed without hard block.
- Reporting: write sampler/turbo/n_startup_trials/n_trust_region lines into best_hyperparams.txt; make results.csv and best_hyperparams.txt fall back to the pinned categorical values for turbo trials (trial.params lacks them); append [turbo] to objective_history.png title.
- Update module docstring: document --turbo, pinned categoricals, 10-dim continuous search, optunahub dependency, n_jobs=1, and resume caveat.
- Smoke test: py_compile kn_bayes_opt.py; run small configs (--epochs 1 --n-trials 4) to verify (a) seed trial runs, (b) non-seed trial params contain only continuous dims, (c) CSV/best_hyperparams emit pinned categoricals, (d) n_jobs forced to 1, (e) banner shows TuRBOSampler, (f) resume-from-TPE-study prints the warning.

## Required Specs
<!-- SPECS_START -->
- lstm-baseline-complexity
- kn-bayes-turbo
<!-- SPECS_END -->