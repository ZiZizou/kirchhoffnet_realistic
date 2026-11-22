# Spec: spice-loaders-spec

Scope: feature

# SPICE model loaders for trial_0011 / trial_0024 (feature spec)

## Goal
Evaluate `knet_model_files_manual_trial_0011/phase_a_best.pt` (KNet) and
`mlp_model_files_bo_trials_0024/phase_a_best.pt` (RegimeAwareMoE) with
`spice_failure_check.py` on the same seeded 200-spec test subset.

## Checkpoints (both raw state_dicts, cf. ctle_dagger_common run_phase_a_training)
- KNet: `.../knet_model_files_manual_trial_0011/phase_a_best.pt`
- MoE: `.../mlp_model_files_bo_trials_0024/phase_a_best.pt`
- Neither dir carries a scaler sidecar; knet preprocessing takes
  `input_log_min/max` from `canonical_ctle_phase_a_v2.npz`.

## New CLI surface
- `--model-kind {mlp,knet,moe,knet-distill}` (extends existing `{knet,mlp}`;
  bare `knet` keeps its current SystemExit-to-bridge behaviour).
- MoE flags: `--moe-trunk-width` (34), `--moe-trunk-layers` (3),
  `--moe-num-experts` (2), `--moe-activation {silu,gelu}` (silu).
- KNet-distill: rebuild driven by explicit `--kn-*` flags mirroring
  trial_0011's `run_command.txt` (stages 7, hidden 11, k 2, p 0.2,
  vca-rank 3, moe 3 experts / gate-rank 2 / top-k 1, x-max 4.0,
  gm-max 4.241970e+00, isat-max 2.868855e+01, readout shared-x2,
  learnable-clip-sharpness, GLN B=4 rank=2 families boundary+readout,
  t-span 4.660583, boundary fan-out map); `--kn-config-json` reserved.
- `--specs-idx-key` (default none): when set (e.g. `test_idx`), sampling is
  restricted to `npz[key]` indices before the seeded shuffle.

## Loader contracts
- MoE: `RegimeAwareMoE(W,L,E, SiLU, use_log_features=False)`; strict
  `load_state_dict`; knet scaler attach from npz bounds; forward =
  `sigmoid(logits)` -> `log_lo/log_hi` affine map -> (N,7) log10 params in
  PARAM_COLS order.
- KNet: `LocalKirchhoffStudentWrapper` via `topology.build_net_from_config`;
  strict state load; knet-preprocessed batch forward -> (N,7) log10 params.
- Both reuse the existing Stage-2/Stage-3 path unchanged (mocked parser,
  resume, scoring, summary).

## Verification (local, no Cadence)
- `py_compile`; mocked E2E for `moe` and `knet-distill` with `--mock-template`;
  determinism check: identical sampled subset for both kinds at `--seed 100`.