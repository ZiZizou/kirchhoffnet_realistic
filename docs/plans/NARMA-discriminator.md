---
plan name: NARMA-discriminator
plan description: NARMA reservoir computing experiment
plan status: active
---

## Idea
Implement the NARMA-10 and NARMA-20 benchmark as the discriminator experiment for the KirchhoffNet fabric. This experiment determines whether the evolving nonlinear core has computational value beyond the frozen core. Key changes: (1) state-carryover mode in KirchhoffNet forward (no reset between samples — the single most important code change), (2) boundary-fan-out mode for continuous-time input drive via OTA edges, (3) temporal readout accumulators that integrate during each sample window, (4) truncated BPTT through ~50-100 samples of history. Baselines: ridge regression on 30 tapped delays (proves task nonlinearity), ESN with 25 nodes + ridge readout (honest comparison at same node count), MLP on 30-tap window (digital feedforward reference), and fabric with freeze_read ON vs OFF (the actual experiment). Metrics: NRMSE on test set, memory capacity curve (R^2 for u(n-k) reconstruction, k=1..20). Decision rule: if freeze_read ≈ no-freeze on NARMA-20, the evolving core is dead and we commit to control-path paper. If no-freeze clearly wins (NRMSE ~0.1-0.2 vs ~0.3+), signal-path branch stays alive.

## Implementation
- Add state-carryover mode to KirchhoffNet forward(): accept an initial state (x0_initial) from the caller; do NOT reset state between samples; return final state for the caller to feed back on the next call. Must work with enable_boundary=True. The carr_over_state flag and initial state tensor passed through the core.
- Implement NARMA data generation in a new script (narma_experiment.py): narma(n_samples, order=10/20) producing u(n) in [0, 0.5], scaled to V_u(n) = 3.0 * u(n)/0.5 V as per spec. Train: 3000 samples after 200-sample washout. Test: 1000 samples, fresh seed. Standard reservoir splits.
- Build the fabric network using enable_boundary=True, boundary_fan_out from 1 input terminal to hidden nodes (e.g., grid/torus topology), temporal_readout accumulators for readout. Use a preset-agnostic builder or add narma preset.
- Implement the training loop with truncated BPTT: unroll ~50-100 samples, detach gradients beyond that window. Temporal accumulators integrate over each sample window; read at window end. Linear readout layer maps to y_hat(n). Use Boundary mode: state carries across samples.
- Implement NARMA baselines: (a) ridge regression on 30 tapped delays, (b) ESN with 25 nodes + ridge readout (echo state network from reservoir library or manual implementation), (c) MLP on 30-tap window. Ensure they run on same train/test splits.
- Implement freeze_read ON vs OFF experiment: train the same fabric network with freeze_read=True (frozen core — boundary and temporal-readout OTA edges only) vs freeze_read=False (evolving core — all internal edges dynamic). Compare NRMSE.
- Implement memory capacity evaluation: train linear readouts from the fabric to reconstruct u(n-k) for k=1..20. Compute MC = sum_k R^2_k. This directly characterizes the fading-memory property.
- Implement ESN baseline: 25-node echo state network with spectral radius ~0.9, ridge regression readout. This is the honest comparison — same node count as fabric, training cost of a linear solve.
- Run all experiments with multiple random seeds, collect NRMSE + MC metrics. Decision gate: if freeze_read ≈ no-freeze on NARMA-20, the evolving core is dead. Commit to control-path paper.
- Write results report: NARMA-10 and NARMA-20 NRMSE tables, memory capacity curves, freeze_read comparison, ESN comparison. This is the experiment that has been an afternoon away for three conversations.

## Required Specs
<!-- SPECS_START -->
- narma-experiment
<!-- SPECS_END -->