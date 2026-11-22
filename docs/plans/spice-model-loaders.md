---
plan name: spice-model-loaders
plan description: MoE and KNet loader support
plan status: active
---

## Idea
Extend spice_failure_check.py with two new --model-kind loaders plus test-split spec selection so trial_0011 (KNet) and trial_0024 (MoE-MLP) can be evaluated: (1) moe loader rebuilds RegimeAwareMoE(W34/L3/E2, SiLU, log-features off) from CLI flags, loads phase_a_best.pt raw state_dict, attaches knet scaler from canonical npz input_log_min/max, forwards to (N,7) log-params via log_lo/log_hi affine map; (2) knet-distill loader rebuilds LocalKirchhoffStudentWrapper via topology.build_net_from_config with trial_0011 run_command flags (stages 7, hidden 11, k2/p0.2, vca-rank 3, moe 3 experts gate-rank 2 top-k 1, x-max 4.0, gm/isat max, shared-x2 readout, learnable-clip-sharpness, GLN B4 rank2 boundary+readout, t-span 4.660583, boundary fan-out map), loads phase_a_best.pt, forwards with knet preprocessing; (3) --specs-idx-key test_idx restricts sampling to the npz held-out split; (4) local verify with py_compile plus mocked E2E for both loaders including a determinism check that both models see the identical 200-spec subset at seed 100.

## Implementation
- Add --model-kind choices moe and knet-distill plus flags --moe-trunk-width/--moe-trunk-layers/--moe-num-experts/--moe-activation, --kn-config-json consumption, --specs-idx-key with default none
- Implement MoE loader: RegimeAwareMoE rebuild, phase_a_best.pt state load, knet scaler attach from npz input_log_min/max, batched forward to log-params
- Implement KNet-distill loader: LocalKirchhoffStudentWrapper rebuild from trial_0011 flags via build_net_from_config, phase_a_best.pt load, batched forward to log-params
- Implement --specs-idx-key test_idx subselection before seeded shuffle in load_specs
- Local verify: py_compile, mocked E2E for moe and knet-distill paths, identical-subset determinism check at seed 100
- Hand over ssr0 run commands for knet_trial0011_test200 and mlp_trial0024_test200

## Required Specs
<!-- SPECS_START -->
- spice-loaders-spec
<!-- SPECS_END -->