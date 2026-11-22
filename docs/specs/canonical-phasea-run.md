# Spec: canonical-phasea-run

Scope: feature

# FEATURE spec: canonical-ctle-unify Phase-A run config

Frozen run configuration for plan `canonical-ctle-unify` Phase A.
Any change to these values invalidates cross-student comparability; bump the
sampler fingerprint and re-freeze this spec first.

## Budget / search
- Param budget: 7000/7000, tolerance 0.15 → window [5950, 7000] both students.
- Trials: 30 each. Epochs: 300. Seed: 100. Halving/pruning: OFF (full-length trials).
- KNet space: prior-biased (hidden, stages, k, rank) joint arch + lr log[2e-4,5e-3] +
  batch {128,256,512,1024,2048} + x_max [2,8] + wd log[1e-6,1e-2] +
  solver_scale [0.5,2.0] + gm_max/isat_max log[1,50].
- MLP space: joint (layers, width, layernorm) arch + lr + batch + {silu,gelu}.

## Dataset / labels
- 20k specs, boundary_ratio 0.5, flow `top_k=8`, `valid_threshold=0.9`
  (0.5 gate / 0.3 relaxed fallback), tie-break = min predicted power.
- Fingerprint covers seed + teacher identity + top_k + tie-break + boundary_ratio.

## Training / objective
- Phase-A loss: Huber (delta 0.5) on log-params, both students.
- KNet differential LRs: mapper 1.0 / structural 4.0 / dynamic 1.0.
- Objective + selection: validation ZIG failure, then validation logit MSE;
  test reserved for frozen finalists; isolation tolerance 3pp.

## H3 decision gate
- KNet parity/tie → optimization/convergence story; Phase B (`--init-from`) may proceed.
- KNet loss ≥2× with full recipe → H3 dead; readout-rank probes become part 2.