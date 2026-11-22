---
plan name: narma-node-activation
plan description: Mix-first bounding structural repair
plan status: active
---

## Idea
The fabric's memory ceiling is structural, not parametric: per-edge tanh saturates before summation while node voltage is bounded only by a hard clip rail, so no gain setting yields moderate-amplitude + high-gain coexistence (verified: rung-1 J max|λ|≈0.005, MC≈0.003). Add a per-node tanh activation so edges mix bounded signals (Y = x_max·tanh(V/x_max), Hopfield/ESN structure, self-regulation to the edge of chaos), plus a one-time single-shot spectral-radius normalization (shift gm_raw so the Heun-map Jacobian at x=0 hits rho≈0.95), then re-run rung-1 as the gate. All new cells, topologies, and flags are NARMA-only and unreachable from train_script (default-off + allow_experimental_cells guard). Roll back narma-gain-fix first (delete untracked narma_gain_norm.py + gain_norm_audit.py; zero tracked modifications exist; spec file stays as record). No E0 sweep until the rung-1 gate reports. Phase 5 (LinearOTA/DeadZoneOTA/AsymmetricOTA cells, esn_sparse_random/hierarchical_multiscale/random_dag topologies) runs only after a PASS; spiking/LIF deferred unless the J-spectrum diagnostic demands it.

## Implementation
- Step 0 - Roll back narma-gain-fix: delete untracked narma_gain_norm.py and gain_norm_audit.py, verify git status clean and py_compile green on narma_experiment/narma_advisor_probes/narma_linear_controls, mark narma-gain-fix superseded (spec file retained as record)
- Step 1 - Node activation in DifferentialStage default-off: add node_activation {none,tanh,identity} plain attr (no param/state_dict change); in rhs feed x_act=x_max*tanh(x/x_max) gathered edge inputs to all five cell families (core, resistive shunt, boundary, ref, readout/sense) while leak/clip/drive keep raw x; gate experimental keys behind allow_experimental_cells=True in build_net_from_config (only NARMA passes it); no train_script CLI additions; make_cell_library untouched
- Step 2 - Single-shot spectral normalization NARMA-side only: new NARMA-module normalize_spectral_radius(net, target_sr=0.95) linearizing one Heun step at x=0,u=0 via autograd jacobian (reuse _one_sample_transition), sr=max|eig(J)|, then gm_raw += log(target_sr/sr) under no_grad; warn-and-skip on non-finite sr; report sr_before/sr_after; init-only, no iterative bracketing, no leak retuning
- Step 3 - Rung-1 gate: add --node-activation + --spectral-radius-target CLI to narma_experiment.py only, thread through _build_fabric_net with fills gm_raw=-1/isat_raw=-5/leak=0.1/input_scale=0.2 (tanh_free, t_span=1.0, num_steps=8, h25, seed 0); CPU smoke locally then full run on Alliance GPU; PASS = MC>5/PR>4/rails<10% then train 40 epochs expecting NRMSE<0.4; FAIL = stop and reassess with no E0 sweep
- Step 4 - Fallback matrix only on Step-3 FAIL: --node-activation identity + --no-clip (NARMA-CLI-only, default-off, same guard); read decision table (unbounded-pass/bounded-fail = bounding guilty; both fail = pipeline/integration-KCL guilty, new debug path)
- Step 5 - Cells + topologies only after a PASS: new narma_cells.py (LinearOTA, DeadZoneOTA, AsymmetricOTA, FreeTanhLibrary-signature-compatible, spiking deferred) and narma_topologies.py (esn_sparse_random 10pct/random-signs/SR-normalized, hierarchical_multiscale via per-stage leak plumbing, random_dag; sw-k6 param-only); pre-search eigenvalue-spread diagnostic of recurrent J at Step-3 optimum decides whether topology search is justified
- Step 6 - Audits + batches + memory: new probe_audit_nodeact.py (epoch-0 equality at gm=isat=0, none-vs-identity no-op, spectral band, train-path isolation rejection + default byte-identity, registry unreachability from make_cell_library); gitignored Alliance batch for Step 3 shipped via git add -f; record verdict and thresholds in project memory

## Required Specs
<!-- SPECS_START -->
- advisor-probe-suite
- canonical-ctle-bench
- lstm-baseline-complexity
- multistage-sequence
- narma-e0-gate
- per-node-diagnostics
- plain-mlp-config
- narma-cell-isolation
<!-- SPECS_END -->