---
plan name: narma-training-overhaul
plan description: Fix t_span TBPTT and training bugs
plan status: active
---

## Idea
Complete overhaul of the NARMA fabric training pipeline to fix three critical bugs identified in the debug analysis:

1. **t_span too large (7.0 → 1.0)**: Per-sample decay of 0.71 gives only ~3.4 sample memory horizon, insufficient for NARMA-10 (needs 10+). Also causes dt=1.167 which is too large for accurate Heun integration.

2. **200 optimizer steps total**: B=60 covers all 60 windows in one micro-batch → 1 optimizer step/epoch × 200 epochs = 200 Adam steps. LSTM baseline gets 12,000 steps.

3. **Per-sample detach() + zero-state init**: Training state distribution (zero-init per window) doesn't match eval (continuous stream). Loss includes washout transients.

Fix: Reduce t_span to 1.0, rewrite train_fabric() with sequential TBPTT using B=4-8 parallel streams, carried state, detach only at chunk boundaries. Add target standardization, bipolar input drive, ridge diagnostic, and instrumentation.

## Implementation
- [config.py] Add t_span parameter to make_narma_preset(), default 1.0. Change PRESET_NARMA10/20 construction to use t_span=1.0 instead of SOLVER["t_span"] (7.0).
- [narma_experiment.py] Rewrite train_fabric() — sequential TBPTT loop: B parallel independent streams, state carried across chunks, detach() ONLY every K=50 samples (not per-sample). Full BPTT within each chunk. No gradient accumulation needed (each chunk = 1 optimizer step).
- [narma_experiment.py] Rewrite data generation for training: generate B independent NARMA sequences of length n_train_samples each (default 2500), stack as (B, n_train_samples). Test data stays as single (1000,) sequence.
- [narma_experiment.py] Add target standardization: normalize y_train to zero-mean/unit-variance before training, denormalize predictions at evaluation for NRMSE/R². Store y_mean/y_std for denormalization.
- [narma_experiment.py] Switch scale_input_to_rails to bipolar mode: map u in [0,0.5] to V_u in [-1.5, +1.5] instead of [0, +3]. Add --bipolar/--unipolar CLI toggle.
- [narma_experiment.py] Add ridge-on-frozen-states diagnostic: run untrained fabric on train stream, collect 25 node states, fit ridge readout, report NRMSE. This separates 'fabric can't remember' from 'optimizer can't learn'.
- [narma_experiment.py] Add instrumentation: log (a) RMS of hidden node voltages during training, (b) per-parameter-group gradient norms (especially raw_leak).
- [narma_experiment.py] Update CLI args: --t-span, --train-samples, --tbptt-chunk, --n-streams, --unipolar, --ridge-diagnostic. Remove --accum-steps (no longer needed). Update defaults.
- [narma_experiment.py] Update evaluate_fabric() to accept y_mean/y_std for denormalization. Keep _evaluate_fabric_direct() working with the new flow.
- [narma_experiment.py] Update run_fabric_condition() to pass new params through, handle multiple streams, standardization, and diagnostic.

## Required Specs
<!-- SPECS_START -->
- narma-training-overhaul
<!-- SPECS_END -->