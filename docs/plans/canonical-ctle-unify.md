---
plan name: canonical-ctle-unify
plan description: Neutral CTLE benchmark unification
plan status: active
---

## Idea
Unify kn_bayes_opt and mlp_bayes_opt CTLE branches onto one shared canonical flow-labeled benchmark built in ctle_dagger_common.py: canonical dataset builder with frozen min-power tie-break, Phase-A Huber-only training mode in all dagger harnesses, KNet Friedman recipe (differential LRs, bias init, gm/isat), validation-failure objective in both drivers at 7000/7000 budget, then launch both Phase-A studies with E0 diagnostics and the H3 decision gate. Phase B deferred.

## Implementation
- Add create_canonical_flow_dataset (top-8 flow labels, min-power tie-break, fingerprint, npz schema) to ctle_dagger_common.py
- Add run_phase_a_training (static data, Huber-only, shared eval/checkpointing) to ctle_dagger_common.py
- Wire --canonical-dataset/--phase-a-epochs into all three dagger harnesses plus KNet Friedman recipe flags (diff LRs, bias init, gm/isat)
- Route both CTLE BO branches to Phase-A mode with validation-failure objective, 7000/7000 budget, fingerprint bumps
- Verify: py_compile, param-count bit-identical, 1-trial smokes, guard invariants, then launch both 30-trial studies and E0 diagnostics

## Required Specs
<!-- SPECS_START -->
- canonical-ctle-bench
- advisor-probe-suite
- lstm-baseline-complexity
- multistage-sequence
- narma-e0-gate
- plain-mlp-config
- canonical-phasea-run
<!-- SPECS_END -->