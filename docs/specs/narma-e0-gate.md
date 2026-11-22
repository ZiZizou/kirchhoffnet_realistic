# Spec: narma-e0-gate

Scope: repo

# Spec: narma-e0-gate (repo scope)

E0 (zero-training) gating protocol for NARMA fabric experiments. No training run may start until the probe passes -- training a poor reservoir only saturates it (observed: rail 0%->~10%, NRMSE gain ~0.05 over 35-80 epochs).

## Probe procedure (minutes, eval-only)
1. Build the candidate net from the exact training feature flags (order, cell library, t_span, num_steps, refresh, leak, drive scale, hidden dim, stage count).
2. Ridge diagnostic on the UNTRAINED net: forward the train stream once, fit offline ridge states->targets, report NRMSE/R^2 (existing `ridge_readout_diagnostic`, washout 200).
3. Memory capacity on the untrained net: `collect_fabric_states` on train stream 0 + `memory_capacity` (max_delay 20). Note the no-washout caveat (underestimates under slow leak); compare against same-pipeline numbers only.
4. Saturation snapshot: max|x|/x_max and rail fraction (>0.9*x_max) on the probe states (same semantics as live `xmaxr`/`rail` logging).

## Gate thresholds (NARMA-10, tanh_free, single seed 0)
- PASS (train): ridge NRMSE clearly below 0.65 (trained baseline 0.676; untrained k8 baseline 0.7194) AND/OR MC clearly above 0.5 (baselines 0.19-0.23).
- FAIL (do not train): ridge ~= 0.72 and MC ~= 0.2 -- the configuration is in the known-dead region; go back to init/regime levers.
- Thresholds are relative deltas by design (single seed, eval-only): the gate screens configurations, it does not certify them.

## Reporting
- Each probe logs one line: `E0 <config-tag>: ridge_NRMSE=x.xxxx R^2=x.xxxx MC=x.xx sat=x.xx rail=x.x% -> PASS|FAIL`, plus the full `carry_ablation`-style row if masks are probed.
- Killed/duplicate probes are never re-run silently: probe outputs land in the leg output dir next to (not inside) training tables.