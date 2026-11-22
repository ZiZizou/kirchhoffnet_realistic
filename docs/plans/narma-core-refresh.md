---
plan name: narma-core-refresh
plan description: Efficient recurrence cost study
plan status: active
---

## Idea
Add variable core-refresh interval to DifferentialStage so NARMA-10 can sweep refresh every 8 / 4 / 2 Heun steps (1 = fully dynamic, deferred unless all three fail) without frozen-core baselines. Thread the knob through KirchhoffNet/build_net_from_config and narma_experiment.py CLI (--core-refresh-interval, --epochs, --compile, --fabric-only), keep boundary/readout/shunt paths dynamic and independent, validate with py_compile + short CPU/GPU smokes, then run the NARMA-10 tanh_free grid (order 10, 4 streams x 2500 samples, t_span 1.0, num_steps 8, seeds 0,1,2, 80 epochs + early stop patience 15) on Alliance targeting 4-6h wall-clock and NRMSE 0.15-0.29 to decide if cheap recurrence matches standard RNN.

## Implementation
- Add core_refresh_interval attribute + setter to DifferentialStage with validation (None/0=once-per-sample frozen, k>=1 recompute every k Heun steps, k=1 fully dynamic)
- Refactor _forward_heun_sequence and _heun_steps so i_edge_const is recomputed inside the Heun loop every k steps while i_boundary_const/i_readout_const follow their own freeze flags and resistive shunt stays per-step dynamic; handle k>num_steps case (frozen across whole window, equivalent to k=legacy frozen)
- Thread refresh knob through KirchhoffNet stage construction, topology.build_net_from_config, and config.make_narma_preset without changing default behavior when unset
- Add CLI flags to narma_experiment.py run_fabric_condition (--core-refresh-interval choices 1,2,4,8, --epochs/early-stop patience support) plus result logging of refresh interval, cell_lib evals per sample, and wall-clock per condition
- Validate with py_compile, CPU smoke (2 epochs, order 10, refresh 1/2/4/8), and single-chunk GPU smoke with --compile checking finite states/loss and that k=8 matches legacy freeze_read=True loss trajectory
- Run Alliance NARMA-10 grid: tanh_free, t_span 1.0, num_steps 8, n-streams 4, train-samples 2500, tbptt-chunk 25, --compile --fabric-only, refresh {8,4,2} x seeds {0,1,2} x 80 epochs with val_every 5 and early-stop patience 15; fallback dynamic k=1 only if all three fail the 0.15-0.29 NRMSE band
- Analyze NRMSE/R2/MC vs refresh rate and evals-per-sample cost, compare against 0.15-0.29 RNN band and ridge baseline, record decision on whether k=2 gives cheap-recurrence win or k=1 is required

## Required Specs
<!-- SPECS_START -->
- lstm-baseline-complexity
- plain-mlp-config
- core-refresh-semantics
<!-- SPECS_END -->
