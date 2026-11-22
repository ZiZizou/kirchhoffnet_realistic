# NARMA Discriminator Experiment

**Status:** Active · **Priority:** HIGH — do before anything else  
**Cost:** ~days · **Decision gate:** determines whether evolving core has reason to exist

---

## What this decides

If `freeze_read ≈ no-freeze` on NARMA-20, the evolving nonlinear core is dead. Commit fully to the control-path paper.

If `no-freeze` clearly wins (expect NRMSE ~0.1–0.2 vs ~0.3+), the signal-path branch stays alive.

---

## Prerequisite code changes (shared with Experiments 2 & 3)

### P1: State-carryover mode in `KirchhoffNet.forward()`

**The single most important code change.** Without it, no memory → NARMA impossible.

Current behaviour: `KirchhoffNet.forward()` always starts from a fresh `x0` (output of InputMapper or zero-init).

New behaviour: add an `initial_state: Tensor | None` kwarg. When provided, skip the zero-init/InputMapper and use this state directly as the starting point for ODE integration. Return the final state alongside the output so the caller can feed it back next sample.

Signature change:
```python
def forward(self, x0, ..., initial_state: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
```

- When `initial_state is not None`, use it instead of `x0` as the integration starting point.
- Always return `(y, trajs, final_state)` — caller stores `final_state` and passes it back on next sample.
- Must work with `enable_boundary=True` (boundary mode).
- Must work with multi-stage networks (state carries through all stages + transfers).

**Files to modify:** `kirchhoff_net.py` (both `KirchhoffNet.forward` and `KirchhoffNetWithIO.forward`), `differential_stage.py` (forward signature).

### P2: Boundary-mode improvements for sequential processing

`enable_boundary=True` currently works for single-pass forward. Verify it works correctly when the same network is called repeatedly with carried-over state (no zero-init of the hidden state on the second call).

**Files to verify:** `kirchhoff_net.py` (the `enable_boundary` path in `forward`), `differential_stage.py` (boundary current computation in `rhs`).

### P3: NARMA preset in `config.py`

Add a `make_narma_preset(order=10)` function returning a preset dict:

- 1 input (the driven voltage), 1 output (y_hat)
- Hidden topology: torus or grid, ~25 nodes (matching ESN baseline)
- `enable_boundary=True`, `boundary_fan_out` mapping input 0 to several hidden nodes
- `enable_temporal_readout=True` with `output_ode_count` = readout dimension
- `freeze_read` flag controllable (for ON vs OFF experiment)
- Per-stage `t_span` and `num_steps` appropriate for one sample window (~14 steps)

---

## Experiment design

### Data

```python
def narma(n_samples, order=10, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, n_samples + order + 1)
    y = np.zeros(n_samples + order + 1)
    for n in range(order, n_samples + order):
        y[n + 1] = (0.3 * y[n]
            + 0.05 * y[n] * np.sum(y[n - order + 1 : n + 1])
            + 1.5 * u[n - order + 1] * u[n]
            + 0.1)
    return u[order : order + n_samples], y[order + 1 : order + n_samples + 1]
```

- Tasks: NARMA-10, NARMA-20
- Train: 3,000 samples (after 200-sample washout)  
- Test: 1,000 samples, fresh seed  
- Input voltage: `V_u(n) = 3.0 * u(n) / 0.5` (fills ±3V rails), zero-order held per window

### Network mapping

| Aspect | Detail |
|--------|--------|
| Input | One boundary terminal driven by V_u(n), zero-order held |
| Integration | One Heun window per sample (e.g. 14 steps with per-stage dt) |
| State | **Carried across samples** — final state of window n = initial state of n+1 |
| Readout | Temporal accumulators integrate per window; read at window end → linear layer → y_hat(n) |
| BPTT | Truncated at ~50–100 samples, detach beyond |

### Conditions

| Condition | freeze_read | Internal edges | Purpose |
|-----------|-------------|----------------|---------|
| A | OFF (False) | Dynamic | Evolving core — full fabric |
| B | ON (True) | Frozen | Only boundary + temporal-readout edges learn |
| C | N/A | N/A | ESN baseline (25 nodes, ridge readout) |
| D | N/A | N/A | Ridge regression on 30 tapped delays |
| E | N/A | N/A | MLP on 30-tap window |

### Training

Standard AdamW, cosine annealing, grad clip. Loss = MSE on y_hat(n) vs y(n).  
Train Conditions A and B identically (same epochs, lr, schedule).  
Conditions C/D/E use their own optimal procedures (ridge solve for C/D, Adam for E).

### Evaluation

- **Primary:** NRMSE on test set
- **Memory capacity:** train linear readouts to reconstruct u(n-k) for k=1..20; MC = Σ R²_k
- **Seeds:** 3–5 random seeds per condition; report mean ± std

### Decision rule

| Outcome | Interpretation | Action |
|---------|---------------|--------|
| freeze_read ≈ no-freeze on NARMA-20 | Evolving core adds no value | Kill signal-path, commit to control-path |
| no-freeze clearly wins (NRMSE ~0.1–0.2 vs ~0.3+) | Evolving core computes | Keep signal-path alive |
| Both fail (NRMSE > 0.5) | Fabric fundamentally unsuitable | Reconsider architecture |

---

## Script structure

```
narma_experiment.py   — main entry point
  ├── narma()          — data generation
  ├── build_fabric()   — build KirchhoffNetWithIO from narma preset
  ├── ESN class        — echo state network baseline
  ├── train_fabric()   — truncated BPTT loop with state carryover
  ├── train_esn()      — ridge regression readout
  ├── evaluate()       — NRMSE + memory capacity
  └── main()           — orchestrate all conditions, print table
```

---

## Deliverables

1. Working state-carryover in KirchhoffNet/KirchhoffNetWithIO
2. `narma_experiment.py` that reproduces all conditions
3. Results table (NRMSE + MC) for NARMA-10 and NARMA-20
4. Decision: evolve core alive or dead
