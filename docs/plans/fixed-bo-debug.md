---
plan name: fixed-bo-debug
plan description: Diagnose Optuna TPE failure
plan status: done
---

## Idea
Debug the ValueError in optuna TPE sampler _calculate_weights_below_for_multi_objective (inhomogeneous shape) crashing fixed-distillation-bayes-opt.py at trial 11. Root hypothesis is mixed-length trial.values in the SQLite study (stale single-objective DB reused via load_if_exists=True, or inconsistent return shapes across runs). Plan covers: reproduce/inspect study.db trial states and directions, check optuna/numpy versions and git history for objective signature changes, verify prune vs complete return paths, then fix study creation/resume guards, ensure consistent 2-tuple returns, add safe CSV/selection handling, and validate with a small multi-objective smoke run.

## Implementation
- Inspect fixed-distillation-bayes-opt.py study setup (create_study directions, load_if_exists, resume path) and all objective return/prune paths
- Query the crashed study.db (sqlite/optuna) to list study directions, per-trial state and len(values) to confirm mixed 1-vs-2 vs None shapes
- Check optuna and numpy versions plus git history of fixed-distillation-bayes-opt.py for single-to-multi-objective changes
- Define fix: direction-mismatch guard, remove silent load_if_exists reuse without --resume, enforce consistent 2-tuple returns
- Add hardened post-processing (selection, results.csv, bo_summary) that handles None/short values safely
- Validate with a 10-12 trial smoke run (n_startup_trials small) that passes the TPE Parzen-estimator threshold where trial 11 crashed

## Required Specs
<!-- SPECS_START -->
- lstm-baseline-complexity
- plain-mlp-config
- bo-crash-repro
<!-- SPECS_END -->