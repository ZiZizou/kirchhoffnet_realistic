---
plan name: audit-vca-shunt
plan description: Audit vca-gate-shunt flag wiring
plan status: active
---

## Idea
Audit whether --vca-gate-shunt actually changes model behavior/params. User observes identical 7057/7441 param counts and no visible change between runs with/without flag across friedman1/2 smooth2d with --freeze-read --temporal-readout --vca --vca-core --vca-separate-core-bus. Hypotheses: (1) param count should NOT change by design (reuses existing VCA core gate on resistive shunt only), (2) print line omits core gated edges hiding activation, (3) functional effect is tiny with freeze-read (only core resistive G~0.007 vs Isat~0.08), and boundary/temporal resistive are always VCA-gated regardless of flag, so delta is near-zero. Verify code plumbing from train_script.py CLI through topology.py build_net_from_preset to DifferentialStage.rhs, check _forward_heun frozen path, check grad flow and init (W zero-init → gate=1.0), and propose fixes: correct VCA ENABLED print, unify or document resistive-gating semantics, add determinist unit test and optional param/log assertion.

## Implementation
- Read CLI plumbing: train_script.py --vca-gate-shunt dest, topology.py build_net_from_preset/topology_to_stage pass-through, DifferentialStage.__init__ vca_gate_shunt attribute
- Read DifferentialStage.rhs and _forward_heun/_forward_heun_sequence frozen paths: confirm i_edge_const folds VCA tanh, i_res core branch guarded by `if vca_gate_core and vca_gate_shunt`, boundary/temporal i_res_b/i_res_o always gated
- Explain why param counts 7057 and 7441 are correct to stay identical: no extra parameters, reuses vca_W/vca_W_core/vca_v_core; compute expected breakdown (core tanh 700 + 500 core VCA embeds) and show gate_shunt adds 0 params
- Diagnose stale log line train_script.py:3682-3694 prints only boundary+temporal, omits core; note it misleads user into thinking core VCA inactive
- Quantify functional impact with freeze-read: estimate resistive vs tanh magnitude (G=softplus(-5)~0.0067 vs Isat~0.08 at init) so delta <10% of core current and zero on boundary/temporal (already always gated); explain why validation curves appear identical
- Check VCA init and optimizer: W zero-init → gate 1.0, grad = u ⊗ v * σ'(0), separate core bus doubles weak init; review mapper_lr_scale/struct_lr_scale grouping to explain static grad-log appearance
- Define verification tests: (a) unit test building stage with/without flag asserts vca_gate_shunt attr, param counts equal, rhs output differs for same x,u with non-zero W; (b) compile path test; (c) frozen path test that i_edge_const unchanged but i_res dynamic differs
- Propose fix options: Option A doc-only (clarify flag is core-resistive-only, fix print to include core edges and shunt state), Option B code unification (make boundary/temporal resistive gating also conditional on flag), with tradeoffs and recommendation
- Draft plan output: audit summary for user, file:line citations, pseudocode of correct vs current behavior, and next-step edit checklist for build agent

## Required Specs
<!-- SPECS_START -->
- lstm-baseline-complexity
<!-- SPECS_END -->