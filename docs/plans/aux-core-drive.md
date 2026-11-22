---
plan name: aux-core-drive
plan description: Eval-only twin probe
plan status: done
---

## Idea
Twin-level eval-only aux-core drive probe (plan aux-core-drive, feature spec aux-core-probe): a learnable feedforward aux core (echo-state, shift-register, slow-feature bank) replaces the static 8-tap delay bank as drive for the main twin. Same streams/seeds, aux-only and aux+scalar legs against taps8/scalar references with pre-registered SUCCESS/KILL verdicts. No fabric changes, no feedback, no training; fabric integration earned only by SUCCESS.

## Implementation
- Implement aux builders (echo-state, shift-register, slow-feature bank) plus four drive legs (aux-only, aux+scalar, taps8 ref, scalar ref) in new side-effect-free aux_core_probe.py reusing post_twin_gates instruments
- Implement sweep driver at Gate-1 best corner with itemized edge counts plus pure pre-registered SUCCESS/KILL verdict function (tap-subspace projection, void, no-gain-over-scalar)
- Wire CLI run mode with flavor subset, seed caps, and smoke flags plus CSV/JSON/TXT writers into leg output dir
- Write audit_aux_core.py covering builder determinism, shift-register chain exactness, drive dimensions, verdict stubs, CLI registration
- Verify with py_compile plus full audit green plus CPU-only lightweight smoke, then record verdict and fabric-integration decision in progress summary

## Required Specs
<!-- SPECS_START -->
- advisor-probe-suite
- canonical-ctle-bench
- lstm-baseline-complexity
- multistage-sequence
- narma-e0-gate
- per-node-diagnostics
- plain-mlp-config
- aux-core-probe
<!-- SPECS_END -->