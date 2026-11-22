---
plan name: c1b-revision
plan description: Orthogonal mixing memory controls
plan status: active
---

## Idea
Revise the NARMA-10 linear-controls suite per external review: drop the impossible parity gate for linear controls, re-anchor memory gating to C0 (MC>=4), demote PR to diagnostic with a standardized-state variant, harden per-delay MC with an SVD fallback, persist the tuned C1 base for valid bisection, then run C1b with orthogonal W plus uniform slow leak (J-band 0.97-0.99, pre-registered MC 5-15) followed by a resistive-core C1b-real variant, redefined C3 tanh-crossover sweep, and local plus Alliance re-measurements.

## Implementation
- Revise gates and instruments in narma_linear_controls.py: drop parity pass/fail for linear legs, set MC>=4 gate, add standardized-state PR diagnostic, add SVD fallback to per-delay MC, persist tuned W/leak/G_in sidecar, promote long-stream R0 reference
- Implement C1b-override CLI (w-mode, leak-mode, leak-value, j-low/j-high) with QR-orthogonal W construction and uniform-leak install path plus C1b PASS rule
- Extend probe_audit2.py for new options and sidecar roundtrip, then run py_compile, both audit suites, and CPU tiny smokes locally
- Implement C1b-real resistive-core mode (shunt-dominant fills, softplus-inverse G targeting, small-signal boundary input) and C3 gm-grid sweep with C2/C4 loading the persisted C1b base
- Run local re-measurements: targeted high-gm E0 corners under fixed MC and k2 frozen-state MC recollect on the full stream
- Stage and verify the Alliance batch for full-stream C1b legs (preflight green) and hand off GPU launch to user
- Document triage: problem-statement v6 void list, revised spec gate tables, and project memory outcomes

## Required Specs
<!-- SPECS_START -->
- advisor-probe-suite
- canonical-ctle-bench
- lstm-baseline-complexity
- multistage-sequence
- narma-e0-gate
- plain-mlp-config
- c1b-protocol
<!-- SPECS_END -->