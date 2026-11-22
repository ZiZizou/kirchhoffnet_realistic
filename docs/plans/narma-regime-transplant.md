---
plan name: narma-regime-transplant
plan description: Canary-first regime transplant grid
plan status: active
---

## Idea
Replace NARMA steps 4-6 with a regime transplant: E0 harness gains isat-decoupled fill, sub-0.25 swing floor, and per-library fills (probe-only, zero dynamics changes); canary leg on CPU pairs the exact-tensor k2 anchor (must read ridge 0.60-0.68, validating the instrument) with the advisor priority corner (high core Isat raw -2, core gm -3.5, leak raw -0.5, swing 0.2); expand to the full ~54-corner transplant grid on GPU only per the pre-registered outcome table; short k=2 train from any gate-passing init (MC>=4, rails<=5%, ridge<0.6); t-span sweep, auxiliary-loss branch, y-MC null, and the shelved forget-gate plan all wait behind the canary.

## Implementation
- Add isat-decoupled fill + sub-0.25 swing floor + per-library fills to E0 probe harness (probe-only, coupled single-fill path byte-identical, no dynamics changes)
- Audit: coupled-fill byte-identity, decoupled values land per-library, exact-tensor anchor load reproduces k2 checkpoint states
- Canary leg on CPU: exact-tensor anchor corner (expect ridge 0.60-0.68, MC ~2.6) + advisor priority corner (high core Isat, moderate gm, fast leak, swing 0.2)
- Read canary per pre-registered 4-outcome table; expand to full ~54-corner transplant grid on GPU only if anchor calibrated and regime alive
- Short k=2 train from winning init if any corner passes gates (MC>=4, rails<=5%, ridge<0.6)
- Conditional follow-ups only on their triggers: t-span sweep on best corner, auxiliary tap-loss/rail-schedule branch if grid passes but training stalls, y-MC null off main line; forget-gate plan stays shelved

## Required Specs
<!-- SPECS_START -->
- advisor-probe-suite
- canonical-ctle-bench
- lstm-baseline-complexity
- multistage-sequence
- narma-e0-gate
- plain-mlp-config
- regime-transplant-grid
<!-- SPECS_END -->