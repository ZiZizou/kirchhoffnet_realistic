# Spec: narma-experiment

Scope: feature

# NARMA Discriminator Experiment Protocol

## Overview
The NARMA (Nonlinear AutoRegressive Moving Average) benchmark determines whether the KirchhoffNet's evolving nonlinear core provides computational value beyond the frozen-core (boundary + temporal-readout only) configuration. This is the discriminator experiment: if freeze_read ≈ no-freeze on NARMA-20, the evolving core is dead and we commit fully to the control-path paper.

## Task & Data

### NARMA Generation
```python
def narma(n_samples, order=10, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, n_samples + order + 1)
    y = np.zeros(n_samples + order + 1)
    for n in range(order, n_samples + order):
        y[n + 1] = (
            0.3 * y[n]
            + 0.05 * y[n] * np.sum(y[n - order + 1 : n + 1])
            + 1.5 * u[n - order + 1] * u[n]
            + 0.1
        )
    return u[order : order + n_samples], y[order + 1 : order + n_samples + 1]
```

- Test NARMA-10 and NARMA-20
- Train: 3,000 samples (after 200-sample washout)
- Test: 1,000 samples, fresh seed
- Standard splits from reservoir literature

### Input Scaling
- V_u(n) = 3.0 * u(n)/0.5 V (fills rails)
- Zero-order held during sample window (no smooth interpolation)

## Mapping onto Fabric

### State Carryover (Critical Change)
- **Do NOT reset state between samples.** The final state of window n is the initial state of window n+1.
- Without state carryover, there is no memory and NARMA is impossible.
- Implemented via a `carry_over_state` flag in KirchhoffNet.forward() and/or KirchhoffNetWithIO.forward().

### Time Integration
- One integration window per sample
- E.g., 14 Heun steps per sample with per-stage dt
- Configuration: t_span=total_time, num_steps per window

### Input Mode
- Use `enable_boundary=True` with `boundary_fan_out` mapping 1 input terminal to multiple hidden nodes
- Input drives OTA edges continuously throughout integration
- No separate InputMapper write phase

### Readout
- Temporal accumulators integrate during each window
- Read at window end
- Linear layer to y_hat(n)

### Truncated BPTT
- Backprop through ~50–100 samples of history
- Detach gradients beyond that window

## Network Architecture
- Topology: grid or torus, ~25 nodes (matching ESN baseline)
- Cell library: t-rex or v2 for good nonlinear dynamics
- 1-3 stages
- freeze_read ON vs OFF for the core experiment

## Baselines

| Baseline | Purpose |
|---|---|
| Ridge regression on 30 tapped delays | Proves task is nonlinear (fails on NARMA-20) |
| ESN, 25 nodes, ridge readout | Honest comparison — same node count as fabric, linear solve training |
| MLP on 30-tap window | Digital feedforward reference |
| Fabric, freeze_read ON | Frozen-core (boundary + temporal edges only) |
| Fabric, freeze_read OFF | Evolving core (all internal edges dynamic) |

### ESN Implementation
- 25 internal nodes
- Spectral radius ~0.9
- Input scaling, leak rate tuned
- Ridge regression readout (no backprop)
- Same train/test splits

## Metrics & Decision Rule

### Primary Metric
- NRMSE on test set

### Memory Capacity Evaluation
- Train linear readouts to reconstruct u(n-k) for k = 1..20
- MC = sum_k R^2_k
- Cheap, standard, directly characterizes fading-memory property

### Decision Rule (Pre-registered)
- If freeze_read ≈ no-freeze on NARMA-20 → evolving nonlinear core is dead → commit fully to control-path paper
- If no-freeze clearly wins (expect NRMSE ~0.1–0.2 vs ~0.3+) → signal-path branch stays alive

### Experimental Rigor
- Run each condition with 3-5 random seeds
- Report mean ± std across seeds
- All conditions use identical train/val splits

## Files to Create
- `narma_experiment.py`: main experiment script with training loop, baselines, evaluation
- State-carryover modifications in `kirchhoff_net.py` and `differential_stage.py`
- Config additions for NARMA preset in `config.py`