---
plan name: narma-multistage-probe
plan description: Multi-stage NARMA memory probe
plan status: active
---

## Idea
Test whether depth-per-sample (3 torus stages with residual-relu-tanh interstage transfers, ~matched params to the 1261-param single-stage baseline) gives the NARMA-10 fabric usable memory where the single-pool design plateaus at NRMSE ~0.67. Sequence: P0 49-wide single-stage control to separate dims from depth (new --hidden-dim flag + E0 probe), then generalize the NARMA sequence/train/eval/MC/ckpt paths from hardcoded single-stage to N stages per the multistage-sequence spec, E0-gate the built 3-stage net (train only on PASS), run one short k=2 comparison train, and decide: advance only on ridge clearly below 0.65 / MC clearly above 0.5 / train beating 0.66, else close depth and return to init-regime (H1) and readout-rank (H3). Cross-sample concatenated carryover preserved throughout (E2 proved it load-bearing); systolic delay-line explicitly deferred.

## Implementation
- Add --hidden-dim CLI flag to narma_experiment run path (preset torus requires perfect square; default 25 preserves all current behavior) and run P0 49-wide E0 probe (ridge + MC + saturation snapshot) with matching k=2/tanh_free/t_span=1.0/num_steps=8 flags
- Generalize the NARMA sequence integration path to N stages per multistage-sequence spec (per-sample stage loop with transfers, window split of t_span/steps, input drive to stage 1 only, readout from last stage accumulator)
- Generalize train_fabric and its callers (remove single-stage assertion, hoist per-stage refs, concatenated carryover state across chunks, per-stage compile/refresh, read-slice/MC/eval/collect paths, per-stage ckpt namespacing, per-stage carry-maskslices, sat stats over concatenated state)
- Build the 3x16-hidden probe net (~matched params, verified by exact counter), run the E0 gate (ridge/MC/saturation) and record PASS/FAIL per narma-e0-gate thresholds
- On E0 PASS run one short k=2 comparison train (35ep, same E4-batch data regimen) against the single-stage baseline; on E0 FAIL stop with no training
- Decide and document: advance only if probe/train beats ridge 0.65 / MC 0.5 / NRMSE 0.66; otherwise close the depth axis, update the crossroads problem-statement doc and project memory, and return H1/H3 as the live hypotheses

## Required Specs
<!-- SPECS_START -->
- multistage-sequence
- narma-e0-gate
- lstm-baseline-complexity
- plain-mlp-config
<!-- SPECS_END -->