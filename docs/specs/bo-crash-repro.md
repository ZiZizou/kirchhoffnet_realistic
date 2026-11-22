# Spec: bo-crash-repro

Scope: feature

# BO Crash Repro + Fix Contract (feature scope)

## Error
`ValueError: setting an array element with a sequence` in
`optuna/samplers/_tpe/sampler.py:909 _calculate_weights_below_for_multi_objective`
at `lvals = np.asarray([t.values for t in below_trials])[is_feasible]`,
crashing `fixed-distillation-bayes-opt.py:83 suggest_float('lr', ...)` at trial 11.
Study is 2-objective (`directions=["minimize","minimize"]`), fresh `--output` dir (no `--resume` reuse per reporter).

## Repro (read-only, on failing host)
1. Record `optuna.__version__` + `numpy.__version__` from the failing ENV (`/home/annaik/.../ENV`, python3.10).
2. Dump crashed `study.db`: study directions, per-trial `(number, state, len(values) if values else None, values)`.
3. Confirm all COMPLETE trials return 2-tuples from `objective:161-182`; list any FAIL/PRUNED trials and their `values`.
4. Check `git log` for `fixed-distillation-bayes-opt.py` objective-signature changes (single→2-tuple).

## Hypotheses (ranked)
1. Optuna+NumPy incompat: FAIL/None values leak into `below_trials`; NumPy>=1.24 raises on ragged `asarray` where older NumPy made object arrays.
2. Stale/foreign trials in DB despite fresh dir (e.g. shared sqlite file, copied DB, `load_if_exists=True` reusing study name `fixed_ctle_knet`).
3. Inconsistent return shape within run (single float vs 2-tuple on some path).

## Fix contract (guard + fresh DB)
- On startup verify `study.directions == [MINIMIZE, MINIMIZE]`; fail fast with actionable message on mismatch (print expected vs actual, suggest fresh `--output`).
- Remove silent reuse: without `--resume`, do NOT `create_study(load_if_exists=True)` over an existing `study.db`; error or require explicit `--resume`/fresh dir.
- Ensure `objective` has exactly two outcomes: `raise TrialPruned` or `return (failure, imitation)` 2-tuple of finite floats; reject NaN/list payloads before return.
- Harden tail code (`selected.values`, `results.csv`, `bo_summary.json`) to guard `trial.values is None or len<2`.
- Pin/document minimum optuna version where multi-objective TPE ragged-array bug is fixed (or add workaround if pinned old).

## Acceptance
- Explains the `(2,)+inhomogeneous` shape: identifies which 2 trials/values mismatched.
- `study.optimize` passes trial 11 (TPE Parzen path) on a 10-12 trial smoke run with small `n_startup_trials`.
- Rerun with mismatched DB produces clear guard error instead of sampler ValueError.