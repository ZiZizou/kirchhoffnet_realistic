---
plan name: knet-gated-memory
plan description: Spike-first Heun fabric upgrade
plan status: active
---

## Idea
Fix KirchhoffNet recurrent memory/nonlinearity for NARMA-10: spike dynamic-leak + hidden-conditioned VCA in differential_stage.py first (epoch-0 bit-equality), then extend the existing e0_sweep with a gm×isat operating-point axis, add a native-linear rung-1 control complementing C1/C1b, and only then consider multi-stage depth and FreeTanh dead-zone cells. Thin CPU smokes locally; all full grids go to Alliance GPU via gitignored Slurm batches. Reuses narma_advisor_probes / narma_linear_controls / narma_revised_plan Round-2 plumbing instead of duplicating it.

## Implementation
- Spike 3a dynamic leak: add per-node dyn_leak_a/b/c params + --dynamic-leak flag in differential_stage.py with a=0,b=0,c=raw_leak init so epoch-0 forward is bit-identical; local 1-sample smoke + audit assert
- Spike 3b hidden-conditioned VCA: extend vca_W to (in_dim+num_nodes, rank) with hidden block zero-init + --vca-use-hidden flag; verify gate==1.0 at epoch-0 and grads flow; local smoke
- Spike train test: 20-epoch NARMA-10 run from best-known operating point with --dynamic-leak --vca-use-hidden on Alliance; go/no-go on MC/NRMSE vs baseline decides whether Phases 1-2 grids are needed
- Phase 1: extend narma_advisor_probes.e0_sweep with gm_raw x isat_raw axes at fixed leak_constant=0.1 (softplus^-1 mapping), reusing apply_gain_override + metric fns + audit style; add Alliance batch leg
- Phase 2: implement native-linear rung-1 (small gm/Isat init, x_max far-rail, leak matched to ESN per-sample retention, input_scale matched to ESN swing) using ESN winner read from run_baselines logs per seed; compare MC~14 target
- Phase 4/5 decision: multi-stage (num_stages=3 preset) comparison only if spike shows nonlinearity-without-depth; dead-zone FreeTanh variant only as follow-up plan after spike verdict

## Required Specs
<!-- SPECS_START -->
- advisor-probe-suite
- canonical-ctle-bench
- lstm-baseline-complexity
- multistage-sequence
- narma-e0-gate
- per-node-diagnostics
- plain-mlp-config
- knet-gating-spec
<!-- SPECS_END -->