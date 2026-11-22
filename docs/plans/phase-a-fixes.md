---
plan name: phase-a-fixes
plan description: Fix Phase-A launch failures
plan status: active
---

## Idea
Repair the three failures from the first Alliance Phase-A launch (log run_phase_a_knet_bo.sh-58325309.out): skip stale checkpoint restore when Phase-A flags are set and point the prepare step at a scratch output dir; make create_canonical_flow_dataset accept explicit zig_model/scaler_X/device instead of requiring _active_ctx and update all three harness gates; move the retry-enqueue loop in kn_bayes_opt.py after study create/load. Verify with py_compile, a local CPU builder micro-smoke, an AST assignment check, and an Alliance 1-trial smoke before the full launch.

## Implementation
- Guard checkpoint restore in dagger-nuance script behind _phase_a_requested and add scratch --output to the .sh prepare step
- Add explicit zig_model/scaler_X/device kwargs to create_canonical_flow_dataset and pass locals from all three harness gates
- Move retry-enqueue loop after study create/load in kn_bayes_opt.py preserving fingerprint-guard order
- Verify locally: py_compile all touched files, CPU builder micro-smoke with n_samples=8, AST check on retry_params assignment
- Alliance 1-trial smoke with small epochs to prove study setup to trial dispatch, then clear for full launch

## Required Specs
<!-- SPECS_START -->
- phase-a-fixes
- canonical-ctle-bench
- advisor-probe-suite
- lstm-baseline-complexity
- multistage-sequence
- narma-e0-gate
- plain-mlp-config
<!-- SPECS_END -->