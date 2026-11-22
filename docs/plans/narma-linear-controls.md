---
plan name: narma-linear-controls
plan description: Calibrated memory suppressor hunt
plan status: active
---

## Idea
Localize the NARMA-10 memory suppressor with calibrated controls before further sweeping, training, or architectural work. Run the R0 Ridge-instrument reconciliation matrix on one fixed seed and stream, then the C0 ESN-through-E0-instruments calibration control, then the C1 linear-reservoir-in-fabric control at hidden-49 with spectral radius pinned near 0.95 and linear injection matched to volt-drive RMS, then the C2/C3/C4 single-element bisection restoring boundary-OTA input, tanh core, and compliance one at a time. Collect maximal instrumentation throughout, revise gates only as the controls dictate, and record the localized suppressor or harness bug in project memory and the problem statement.

## Implementation
- Run the R0 reconciliation matrix varying diagnostic path, frozen-versus-dynamic core, drive scale, stream length, and raw-delay Ridge convention on one fixed seed and stream, then freeze the matched parity rule.
- Run the C0 ESN calibration control through the identical Ridge, memory-capacity, participation-ratio, and Jacobian instruments and use it to validate or revise the gates.
- Implement the probe-side linear-core mode, linear dense input injection matched to volt-drive RMS, and rail and clip bypasses, then tune hidden-49 conductances until measured spectral radius lands in 0.93-0.97.
- Run the C1 linear-reservoir-in-fabric control with maximal instrumentation and apply the MC-at-least-3 gate with PR and Ridge confirmations.
- Run the C2, C3, and C4 bisection restoring boundary-OTA input, tanh core, and compliance one at a time and identify the first restoration collapsing MC and PR.
- Write per-control JSON and CSV artifacts with config tags, thresholds, seeds, spectra, per-delay MC curves, saturation histograms, and paired deltas.
- Update the audit suite for linear-core construction, gate restoration, spectral targeting, matched parity, and expanded schema, then run compile checks and small CPU smoke tests.
- Record thresholds, code and batch provenance, results, and the localized suppressor or harness bug in project memory and the problem statement.

## Required Specs
<!-- SPECS_START -->
- advisor-probe-suite
- canonical-ctle-bench
- lstm-baseline-complexity
- multistage-sequence
- narma-e0-gate
- plain-mlp-config
- linear-control-probes
<!-- SPECS_END -->