# Spec: phase-a-fixes

Scope: feature

# FEATURE spec: phase-a-fixes (plan phase-a-debug-fixes)

Follow-up fixes to the canonical Phase-A implementation after the first
Alliance launch failed (log `run_phase_a_knet_bo.sh-58325309.out`). All three
are required before any Phase-A BO study can run.

## Bug 1 — stale checkpoint restore attempted on Phase-A runs
- Cause: `dagger-nuance-distillation-kirchhoffnet.py` unconditionally calls
  `load_checkpoint()` from the global `CHECKPOINT_PATH`. The `.sh` prepare step
  passes no `--output`, so it hit the default
  `/home/annaik/Documents/dagger_output/dagger_checkpoint.pt` (older arch:
  52 vs 56 edges, VCA rank 3 vs 2). Strict load failed → fell back to fresh
  start (no warm start occurred), but a matching-arch stale file would have
  silently warm-started the trial.
- Fix: compute `_phase_a_requested` right after `_bo_parser.parse_known_args()`
  and skip the restore (`ckpt = None`) when set. Also pass a scratch
  `--output` dir in the `.sh` prepare step so the shared default dir is never
  touched.

## Bug 2 — builder requires `_active_ctx`, scripts don't set it (integration bug)
- Cause: `create_canonical_flow_dataset` calls `_active_zig()/_active_scalers()/
  _active_device()`, which need `setup()` to have run. Only
  `generative-distillation-plain-mlp.py` calls `setup()`; nuance + MoE scripts
  wire their own locals → fatal `RuntimeError: _active_ctx['zig_model'] not set`,
  no `.npz` written.
- Fix: add `zig_model=None, scaler_X=None, device=None` kwargs to the builder,
  falling back to `_active_*()` when `None`; thread the explicit `device`
  through the internal `filter_by_zig_validity` calls; update all three harness
  gates to pass their locals. `setup()`-based callers keep working unchanged.

## Bug 3 — `UnboundLocalError: retry_params` in `kn_bayes_opt.py:924` (pre-existing)
- Cause: the retry-enqueue loop (922–931) runs before study create/load
  (deferred to ~1045–1068), so `retry_params`/`study`/`recovered_seed_indices`
  are unbound on any fresh run. Crashes before trial 0. Untouched by the
  Phase-A diff (confirm via `git blame`).
- Fix: move the loop to after the study create/load block, preserving the
  fingerprint-guard-before-recovery ordering. Pure reorder, no logic change.

## Out of scope / benign
- Eye-scale flow-vs-zig `WARNING`s (2–5% rel diff) and optuna
  `ExperimentalWarning`s: benign, no action.
- `test_bo_guard.py::test_sampler_rev_bumped` (asserts rev 2, source at rev 3):
  pre-existing stale assertion in an untracked file; separate issue.

## Verification
- `py_compile` on all touched files; `--help` flag surfacing re-check.
- Local CPU micro-smoke: builder with explicit zig args (`n_samples=8`) →
  assert `.npz` + fingerprint written (flow ckpt + CSVs exist locally).
- AST check: `retry_params` assigned on all paths before use.
- Alliance 1-trial smoke (`--n-trials 1`, small `--epochs`) proving study
  setup → trial dispatch end-to-end before the full 30-trial launch.