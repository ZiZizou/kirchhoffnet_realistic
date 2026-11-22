# Spec: narma-training-overhaul

Scope: feature

# NARMA Training Overhaul

## Three Critical Bugs

### Bug 1: 200 optimizer steps total
- Cause: B = min(128, 60) = 60 → single micro-batch covers all 60 non-overlapping windows → only 1 optimizer.step() per epoch × 200 = 200 total
- LSTM baseline: 60 steps/epoch × 200 = 12,000 optimizer steps
- Fix: Use B=4-8 parallel independent streams, ~2500 samples per stream → ~50 chunks per stream per epoch → ~200-400 optimizer steps per epoch

### Bug 2: t_span=7.0 gives ~3 sample memory horizon
- leak ≈ 0.049, C_eff = 1 → τ ≈ 20 time units
- Per-sample decay: exp(−0.049 × 7.0) ≈ 0.71 → effective memory ≈ 1/(1−0.71) ≈ 3.4 samples
- NARMA-10 needs 10+ sample memory
- Also: dt = 7.0/6 = 1.167 (too large for accurate Heun integration)
- Fix: t_span = 1.0 → per-sample decay ≈ 0.95 → ~20 sample memory horizon. dt = 0.167.

### Bug 3: Training protocol doesn't match evaluation
- Training: zero-state init per window, first ~10 samples are transient-dominated
- Evaluation: continuous stream, trained on washout distribution, evaluated on steady-state
- Fix: Sequential TBPTT with carried state, detach only at chunk boundaries

## TBPTT Algorithm
```
B = 4-8 parallel independent streams (generated with different seeds)
n_samples = ~2500 per stream
K = 50 samples per chunk

state = zeros(B, 26)  # carried across chunks
for chunk_start in range(0, n_samples - K, K):
    loss = 0.0
    for s in range(K):
        y_pred, _, state = net(u_stream[:, chunk_start+s], initial_state=state)
        loss += mse(y_pred, y_stream[:, chunk_start+s])
    (loss / K).backward()
    clip_grad_norm_(params, 1.0)
    optim.step()
    optim.zero_grad()
    state = state.detach()  # ONLY detach here
```

## Additional improvements
1. Target standardization: zero-mean/unit-variance y, denormalize at eval
2. Bipolar input drive: map [0, 0.5] → [-1.5, +1.5] instead of [0, +3]
3. Ridge-on-frozen-states diagnostic: untrained fabric + ridge readout
4. Instrumentation: hidden node RMS, per-param-group gradient norms
5. CLI args: --t-span, --train-samples, --tbptt-chunk, --n-streams, --unipolar

## Expected outcomes
- Ridge baseline at 0.38 is the bar (linear model on 30 taps)
- Fabric with genuine recurrence should beat ridge
- LSTM_large at 0.28 is aspirational target
- NRMSE < 0.38 means experiment is working