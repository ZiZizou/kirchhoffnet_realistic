# Spec: multistage-sequence

Scope: repo

# Spec: multistage-sequence (repo scope)

Reusable semantics for multi-stage per-sample integration in NARMA-style streaming experiments. Applies wherever `DifferentialStage`s are chained per sample with cross-sample carryover.

## Per-sample data flow (depth-per-sample)
- Each sample's `t_span` window is SPLIT across stages (stage_times[i] sum to the per-sample window; stage_steps[i] split proportionally, each >= 1). Total integration per sample stays ~= the single-stage `t_span` (no extra time from depth).
- Within one sample: stage1 integrates its sub-window from the carried state -> `StageTransfer` (instantaneous truncate/pad + pointwise activation, NO time evolution) -> stage2 integrates -> ... -> stageN. Readout reads the LAST stage's output accumulator slice.
- Input drive (`u_t`) enters stage 1 only (pipeline semantics). Boundary fan-out is built per stage only where drive is wanted; stages > 1 receive no fresh external drive by default.
- Transfers default to the Friedman recipe `residual-relu-tanh` (identity at init: w1=1, w2=0) so day-0 behavior is sane and ablations start from a no-op.

## Cross-sample carryover (non-negotiable)
- The carried state is the CONCATENATED multi-stage state vector. E2 carry ablation proved carryover is load-bearing (reset-hidden: NRMSE 0.68->0.87, MC->0.00 on two checkpoints) -- any multi-stage design MUST preserve full carryover by default.
- Carry masks (`carry_keep`) generalize to per-stage slices: callers build the mask from each stage's width; zeroing conventions (`reset-hidden`, `reset-all`) apply to the concatenated vector with documented slice boundaries.

## Per-stage independence
- `core_refresh_interval`, compile (`enable_sequence_compile`), freeze flags, leak mode, and cell library are per-stage settings; the NARMA ladder convention applies the same `k` to all stages unless a leg says otherwise.
- Checkpoints namespace histories AND stage weights per stage index; strict `load_state_dict` must fail loudly on stage-count/width mismatch (never silently score the wrong net).
- Param accounting reports per-stage + total; matched-param probes (e.g. 3x16 vs 1x25) verify counts at build time via exact counter, not estimates.

## Integration points (kirchhoffnet_realistic)
- Sequence path: generalize `_forward_heun_sequence` callers (`train_fabric` inner loop, `_evaluate_fabric_direct`, `collect_fabric_states`) from `stages[0]` to a stage loop with transfers. `train_fabric`'s single-stage assertion is retired by this spec, not bypassed.
- State width becomes the sum over stages; `read_slice`, `hid_count`/`proj_count`/`output_ode_count` bookkeeping must expose per-stage offsets for masks, MC slicing, and saturation stats (reported per concatenated state, same `xmaxr`/`rail` semantics as single-stage).