# NARMA Fabric Debug

- [The NARMA Problem](#the-narma-problem)
- [Baselines](#baselines)
- [KirchhoffNet Architecture for NARMA](#kirchhoffnet-architecture-for-narma)
- [Training Loop (BPTT)](#training-loop-bptt)
- [Problem: Training MSE vs Test NRMSE](#problem-training-mse-vs-test-nrmse)
- [Root Cause: Per-Sample `detach()`](#root-cause-per-sample-detach)
- [Potential Fixes](#potential-fixes)

---

## The NARMA Problem

### NARMA-10 equation

```
y(n+1) = 0.3·y(n) + 0.05·y(n)·Σ_{k=n-9}^{n} y(k) + 1.5·u(n-9)·u(n) + 0.1
```

where `u(n) ~ Uniform[0, 0.5]`. The output `y(n)` has an autoregressive nonlinear term (`y(n) · sum of past 10 y`), a product of delayed inputs, and a DC offset. This makes it a standard benchmark for reservoir computing — good models must **remember** the last 10+ samples and handle **multiplicative nonlinear** interactions.

### Data generation

```python
def narma(n_samples, order=10, seed=0, u_max=0.5):
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(n_samples + order + 1, generator=g) * u_max  # [0, 0.5]
    y = torch.zeros(n_samples + order + 1)
    for n in range(order, n_samples + order):
        y[n+1] = (0.3 * y[n]
                  + 0.05 * y[n] * torch.sum(y[n-order+1:n+1])
                  + 1.5 * u[n-order+1] * u[n]
                  + 0.1)
    return u[order:order+n_samples], y[order+1:order+n_samples+1]
```

Return shapes are `(n_samples,)` each. The first `order` samples are washout (discarded).

### Train/test split

```python
u_train, y_train = narma(3000, order=10, seed=0)     # 3000 train samples
u_test,  y_test  = narma(1000, order=10, seed=10000)  # 1000 test samples (different seed)
```

### Input scaling

Before feeding to the fabric:

```python
u_train = scale_input_to_rails(u_train)   # u ∈ [0, 0.5] → V_u ∈ [0, 3.0]
u_test  = scale_input_to_rails(u_test)
```

This maps the input to the OTA voltage rail `x_max = 3.0` so boundary OTA edges operate in their active range.

---

## Baselines

All baselines share the same data (including washout convention). They serve as reference performance levels.

### 1. Ridge regression — 30 tapped delays

**Parameters**: 31 (30 tap weights + 1 bias)

**How it works**: Builds a feature matrix `X` where row `t` contains `[u(t), u(t-1), ..., u(t-29)]`. Solves `y = XW + b` in closed form with L2 regularization.

```
X = [u(t)  u(t-1)  ...  u(t-29)]          shape: (T-29, 30)
W = (X^T X + λI)^{-1} X^T y               ridge regression
```

```python
class RidgeRegressor:
    def _features(self, u):
        T = u.shape[0]
        return torch.stack([
            u[self.n_taps-1-lag : T-lag] for lag in range(self.n_taps)
        ], dim=1)   # (T - n_taps + 1, n_taps)

    def fit(self, u, y):
        X = self._features(u)                            # (2971, 30)
        X_aug = torch.cat([X, torch.ones(T-29, 1)], 1)  # + bias column
        W = solve(X_aug^T X_aug + λI, X_aug^T y_aligned)
        self.W, self.b = W[:-1], W[-1]
```

**Strengths**: Simple, fast, closed-form, no hyperparameter tuning beyond `λ`.

**Weaknesses**: Linear readout of the last 30 inputs — cannot capture the NARMA nonlinear recurrence (only 0.3·y(n) + 0.1 would be learnable; the `y(n)·sum(y)` term requires state).

### 2. Echo State Network — 25 tanh nodes

**Parameters**: 26 trained (readout) / 676 total (reservoir + readout + input weights)

**How it works**: A fixed random reservoir with spectral radius 0.9 is driven by the input. Only the linear readout `W_out` is trained via ridge regression.

```python
class ESN:
    def __init__(self, n_reservoir=25, spectral_radius=0.9, ...):
        W = randn(n_reservoir, n_reservoir)                     # fixed
        W *= spectral_radius / max(|eigvals(W)|)                # rescale
        self.W_in = randn(n_reservoir, 1) * input_scaling       # fixed

    def _run(self, u):
        x = zeros(n_reservoir)
        states = []
        for t in range(len(u)):
            pre = self.W @ x + self.W_in.squeeze() * u[t]
            x_new = tanh(pre)
            x = (1 - leak) * x + leak * x_new                   # leaky integration
            states.append(x)
        return stack(states)                                     # (T, 25)

    def fit(self, u, y):
        S = self._run(u)
        S_aug = cat([S, ones(T, 1)], 1)
        W_out = solve(S_aug^T S_aug + λI, S_aug^T y)            # ridge readout
        self.readout_W, self.readout_b = W_out[:-1], W_out[-1]
```

**Grid search**: Tries `input_scaling ∈ {0.2, 0.5, 1.0}` and `leak ∈ {0.3, 0.5, 1.0}` on a validation set (last 500 train samples).

**Why it works for NARMA**: The reservoir's recurrent dynamics create a nonlinear **fading memory** of the input history. The readout simply learns to combine these node voltages to reconstruct `y(n)`.

### 3. MLP — 30-tap window, 25 hidden

**Parameters**: 801 (30×25 + 25 + 25×1 + 1)

```python
class MLPRegressor(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(30, 25)    # tanh activation
        self.fc2 = nn.Linear(25, 1)      # linear output

    def forward(self, u_window):
        return self.fc2(torch.tanh(self.fc1(u_window)))
```

Trained for 200 epochs with AdamW, MSE loss. Takes a 30-sample window of `u` and predicts `y` at the window's last index. Feedforward — no state, no recurrence.

### 4. MLP_large — 30-tap window, 237 hidden

**Parameters**: 7,585 (≈KNet parameter count, for fair comparison)

Same architecture as MLP but with 237 hidden units.

### 5. LSTM — streaming u(t), 25 hidden

**Parameters**: 4,513 (4 layers × 25² + bias terms + input/output projections)

```python
class LSTMRegressor(nn.Module):
    def __init__(self, hidden_dim=25):
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, u_seq, state=None):
        # u_seq: (B, T)  or  (T,) — entire sequence in one call
        out, final_state = self.lstm(u_seq.unsqueeze(-1), state)
        return self.fc(out).squeeze(-1), final_state
```

**Grid search**: `hidden_dim ∈ {16, 25, 32}`, `lr ∈ {1e-3, 5e-4}`.

**Key difference from fabric**: The LSTM processes the **entire 50-sample BPTT window as one graph** — no `detach()` between timesteps. Gradients flow through all 50 steps.

### 6. LSTM_large — streaming u(t), 42 hidden

**Parameters**: 7,603 (≈KNet). Grid search on `lr` only.

### Baseline expected performance

| Baseline | NRMSE (seed 0) | Why it works |
|----------|----------------|--------------|
| Ridge | 0.38 | Linear memory of 30 taps captures ~85% of variance |
| ESN | 0.56 | Fixed reservoir with ridge readout — fading memory helps |
| MLP | 0.76 | Feedforward only — no recurrence, poor on temporal task |
| MLP_large | 0.40 | More capacity helps but still no recurrence |
| LSTM | 0.33 | Full BPTT, gated recurrence, well-suited for NARMA |
| LSTM_large | 0.28 | More capacity + full BPTT — best baseline |

---

## Metrics

```python
def nrmse(y_pred, y_true):
    """Normalized RMSE: RMSE / std(y_true).
    NRMSE < 1 means better than predicting the mean."""
    rmse = sqrt(mse_loss(y_pred, y_true))
    return rmse / y_true.std().clamp(min=1e-9)

def r2(y_pred, y_true):
    """R² = 1 - SS_res / SS_tot.
    R² = 1 is perfect; R² = 0 is predicting the mean; R² < 0 is worse than the mean."""
    ss_res = sum((y_pred - y_true)^2)
    ss_tot = sum((y_true - mean(y_true))^2)
    return 1 - ss_res / ss_tot

def memory_capacity(states, targets, max_delay=20):
    """For each delay k=1..20, fit a ridge regression:
       states[:T-k] → targets[k:]
    MC = Σ_k R²_k  (Jaeger 2001)"""
    for k in 1..20:
        X = states[:T-k]
        y = targets[k:]
        W = solve(X^T X + λI, X^T y)
        y_pred = X @ W
        r2_list.append(r2(y_pred, y))
    return sum(max(r, 0) for r in r2_list)
```

---

## KirchhoffNet Architecture for NARMA

### NARMA Preset (config.py:1146)

The NARMA preset is built by `make_narma_preset(order=10, hidden_dim=25)`:

```python
PRESET_NARMA10 = {
    "stages": [{
        "num_inputs": 1,
        "num_hidden": 25,           # 5×5 torus grid
        "num_proj": 0,
        "num_outputs": 0,
        "hidden_family": "torus",
        "hidden_kwargs": {"height": 5, "width": 5, "kernel_size": 3},
        "edge_repeats": 1,
        "input_pattern": "none",    # no direct input connections
        "output_pattern": "all_to_all",
        "proj_pattern": "none",
        "t_span": 7.0,
        "num_steps": 6,             # ← Heun steps per sample
    }],
    "out_dim": 1,
    "boundary_fan_out": {0: list(range(25))},  # input → every hidden node
    "enable_temporal_readout": True,
    "output_ode_count": 1,
}
```

### Network construction (topology.py → `build_net_from_preset`)

```
build_net_from_preset("narma10", cell_lib, boundary_fan_out=..., ...)
  │
  ├── 1. Torus topology: 25 hidden nodes, kernel_size=3
  │     (each node connects to ~8 neighbors in the 5×5 grid)
  │
  ├── 2. Boundary edges: 25 OTA edges from input 0 → each hidden node
  │     (the physical input signal enters here, NOT through an input mapper)
  │
  ├── 3. Temporal-readout edges: 25 OTA edges, each hidden node → output ODE node
  │     (hidden voltages drive current into the single accumulator)
  │
  ├── 4. DifferentialStage (1 stage):
  │     - num_nodes = 25 (hidden) + 0 (proj) + 1 (output_ode) = 26
  │     - E core edges ≈ 25 × ~8 / 2 (undirected) × edge_repeats(1) ≈ 100
  │     - 25 boundary edges (src=0, dst=0..24)
  │     - 25 temporal-readout edges (src=0..24, dst=25)
  │
  ├── 5. KirchhoffNet (core): wraps the stage
  │
  ├── 6. NullInputMapper: returns zeros (bypassed in boundary mode)
  │
  ├── 7. OutputAffine(out_dim=1): y = gain·x_ode + bias + tanh_gain·tanh(x_ode) + relu_gain·relu(x_ode)
  │     (initialized to identity: gain=1, bias=0, tanh_gain=0, relu_gain=0)
  │
  └── 8. KirchhoffNetWithIO: input_mapper → core → output_mapper
```

### Cell library

`make_cell_library("tanh")` returns a `SimpleEdgeLibrary(num_edges, mode="tanh")`.

Each edge computes:

```
I = tanh(p0 · V_src + p1 · V_dst + p2) × compliance_gate
```

where `p0, p1, p2` are per-edge learnable parameters (shape `[3, E]`), and the compliance gate is:

```
gate = sigmoid((x_max - |V_src|) / β) · sigmoid((x_max - |V_dst|) / β)
```

This gates off the edge when either terminal approaches the ±3.0 rail (soft voltage clamp).

### The RHS: per-node ODE (differential_stage.py:655)

Per-node dynamics:

```
C_eff · dx_j/dt = Σ I_edge  -  Σ I_edge  + I_boundary + I_output_ode
                   ^incoming   ^outgoing
                 - leak_j · x_j
                 - clip(x_j)

where:
  I_edge      = tanh(p0·V_src + p1·V_dst + p2) × σ(z_logits) × gate  (core edges)
  I_boundary  = tanh(·, ·) × σ(boundary_z_logits)                    (25 edges, u→hidden)
  I_output_ode = tanh(·, ·) × σ(output_ode_z_logits)                 (25 edges, hidden→ODE)
  leak        = softplus(raw_leak) ≈ 0.049                            (per-node learnable)
  clip        = sigmoid((x - x_max)/β) - sigmoid((-x - x_max)/β)     (soft rail clamp)
  C_eff       = 1.0
```

**Key points**:

- **Boundary edges are read-only**: current flows into the hidden node only. No current is drained from the input terminal (ideal voltage source). The input signal modulates the OTA current via the source voltage `V_u(t)`.

- **Temporal-readout edges are read-only**: current flows into the output ODE node only. Hidden nodes are not drained — they drive the OTA but keep their charge.

- **Core edges are bidirectional**: current flows source→destination via KCL, so both src and dst experience current injection/drain.

- **Edge gates** `σ(z_logits)`: learnable per-edge switches. Initially `σ(0) = 0.5`.

### Heun integration: `_forward_heun` (differential_stage.py:903)

For a single sample with input `u(t)`:

```python
def _forward_heun(x0, u, t_span=7.0, num_steps=6):
    dt = 7.0 / 6 = 1.167
    x = x0                                          # (B, 26) — initial state
    for _ in range(6):                              # 6 predictor-corrector steps
        k1 = rhs(x, u=u)                            # dx/dt at current state
        x_pred = x + dt * k1                        # Euler prediction
        k2 = rhs(x_pred, u=u)                       # dx/dt at predicted state
        x = x + 0.5 * dt * (k1 + k2)                # corrected state
    return x                                        # (B, 26) — final state
```

**dt ≈ 1.167 is very large** for an ODE over a nonlinear system. The default SOLVER config uses `num_steps=70` for `t_span=7.0` (dt=0.1), but the NARMA preset overrides this to `num_steps=6`.

### The full forward path: KirchhoffNetWithIO.forward (kirchhoff_net.py:408)

```python
def forward(self, u, initial_state=None, return_final_state=False):
    # Input setup
    if initial_state is not None:
        x0_full = initial_state          # carry-over: use previous state directly
    else:
        # boundary mode: zero-init hidden + output ODE
        x0 = u.new_zeros(u.size(0), 25)  # hidden nodes start at 0 V
        proj = zeros(..., 0)              # no projection nodes
        ode  = zeros(..., 1)              # output ODE node starts at 0 V
        x0_full = concat([x0, proj, ode], dim=1)  # (B, 26)

    # Core integration
    x_final, _ = self.core(x0_full, u=u)  # KirchhoffNet → DifferentialStage._forward_heun
    # x_final: (B, 26)

    # Readout
    x_read = x_final[:, 25:26]            # read_slice = slice(25, 26) → output ODE node
    y = OutputAffine(x_read)              # y = gain·x_ode + bias + tanh_gain·tanh(x_ode) + ...
    # y: (B, 1)

    if return_final_state:
        return y, None, x_final           # return prediction + state for carryover
    return y, None
```

---

## Training Loop (BPTT)

### Setup (`train_fabric`, narma_experiment.py:446)

```python
u_train: (3000,)    → moved to device
y_train: (3000,)    → moved to device
n_samples = 3000
bptt_window = 50
n_windows = 3000 // 50 = 60    # non-overlapping
batch_size = 128
B = min(128, 60) = 60           # one micro-batch covers all 60 windows
epochs = 200
```

### Per-epoch loop

```python
net.train()
window_starts = randperm(60, device=device) * 50
# random order of [0, 50, 100, 150, ..., 2950]

chunks = window_starts.split(60)    # single chunk of 60 elements

optim.zero_grad()
total_loss = 0.0
state = None  # ← ZERO initial state for each batch of windows

for step_idx, chunk in enumerate(chunks):    # just 1 iteration (60 windows)
    # Build index tensor: (60, 50) — 60 non-overlapping windows of 50 samples
    idx = chunk[:, None] + arange(50)[None, :]
    u_chunk = u_train[idx]    # (60, 50)
    y_chunk = y_train[idx]    # (60, 50)

    # ──────── 50 sequential timesteps ────────
    state = None  # ← ZERO initial state for EVERY batch, EVERY epoch
    for s in range(50):
        u_b = u_chunk[:, s].unsqueeze(-1)  # (60, 1)
        y_b = y_chunk[:, s]                # (60,)

        y_pred, _, final_state = net(u_b, initial_state=state, return_final_state=True)
        # y_pred: (60, 1) → squeeze → (60,)

        loss_step = F.mse_loss(y_pred, y_b)
        total_loss = total_loss + loss_step

        state = final_state.detach()        # ← THE BUG: gradient severed here

    total_loss = total_loss / 50            # mean over 50 timesteps

    # Backward
    scaler.scale(total_loss).backward()     # gradients computed for this micro-batch
    scaler.unscale_(optim)
    clip_grad_norm_(net.parameters(), 5.0)
    scaler.step(optim)
    optim.zero_grad()
```

### Key details

| Detail | Value | Implication |
|--------|-------|-------------|
| Window overlap | None (each sample seen once per epoch) | 60 independent 50-sample trajectories |
| Initial state per window | Zero (`None`) | Model always starts from `x = 0V` |
| State carryover within window | Yes (physical), but `detach()` at every sample | Gradient only flows through 1 sample (6 Heun steps) |
| Gradient backprop depth | **6 ODE steps** | Cannot learn dynamics longer than 1 sample |
| Micro-batches per epoch | 1 (60 windows / B=60) | Single optimizer step per epoch |
| Gradient accumulation | `accum_steps=2`, but only 1 micro-batch | No actual accumulation |

### Evaluation path (`_evaluate_fabric_direct`, narma_experiment.py:619)

```python
stage = net.core.stages[0]
x0 = zeros(1, 26)                                     # (1, 26) — zero initial state
all_states = stage._forward_heun_sequence(
    x0=x0, t_span=7.0, num_steps=6, u_seq=u_test      # (1000, 1, 26)
)
x_read = all_states[:, :, 25:26]                       # (1000, 1, 1) — output ODE node
y_preds = output_mapper(x_read)                        # OutputAffine: (1000, 1, 1)
preds = y_preds.view(1000)                             # (1000,)
```

The direct evaluation bypasses KirchhoffNetWithIO and calls `_forward_heun_sequence` which runs the same 6-step Heun integration per sample, threading the state through all 1000 samples. Output ODE node readout is identical to training.

---

## Problem: Training MSE vs Test NRMSE

### Full results (seed 0, fabric_fr_off)

```
Epoch  0: MSE = 3.1857  (RMSE = 1.785)
Epoch 50: MSE = 0.3826  (RMSE = 0.619)
Epoch 100: MSE = 0.2240 (RMSE = 0.473)
Epoch 150: MSE = 0.1175 (RMSE = 0.343)
Epoch 199: MSE = 0.0823 (RMSE = 0.287)

Test:  NRMSE = 2.2152  R² = -3.9132  MC = 0.18
```

### Converting MSE to NRMSE

```
std(y_test) ≈ training_RMSE / test_NRMSE
            = 0.287 / 2.2152 ≈ 0.130

training_NRMSE = training_RMSE / std(y_train)
               ≈ 0.287 / 0.130 ≈ 2.2
```

**The training NRMSE is also ~2.2** — there is no train/test gap. The model is equally bad on both sets. The loss *looks* like it's improving (40× drop from 3.18 to 0.08) but the absolute floor of 0.08 MSE corresponds to an RMSE of 0.287 — which is **30-100× worse** than the baselines.

### Comparison with baselines

| Method | Test NRMSE | Test R² | Test RMSE | MSE equivalent |
|--------|-----------|---------|-----------|----------------|
| Ridge | 0.38 | 0.86 | 0.049 | 0.0024 |
| ESN | 0.56 | 0.69 | 0.073 | 0.0053 |
| MLP | 0.76 | 0.42 | 0.099 | 0.0098 |
| LSTM | 0.33 | 0.89 | 0.043 | 0.0018 |
| LSTM_large | 0.28 | 0.92 | 0.036 | 0.0013 |
| **Fabric** | **2.22** | **-3.91** | **0.287** | **0.0823** |

The fabric's **training** MSE (0.0823) is worse than every baseline's **test** MSE. The model fundamentally cannot solve NARMA.

### Why the fabric fails

**Primary: Training/serving gradient mismatch (the `detach()` bug).**

The model is trained with `detach()` at every sample boundary. Each sample's 6-Heun-step computation graph is independent. The model learns to map `(state_s, u(s)) → (state_{s+1}, y_pred)` but only through 6 ODE steps at a time. The long-timescale dynamics of NARMA (order-10 recurrence) are invisible to the gradient.

Compare with the LSTM baseline:

```python
# LSTM — full 50-step graph, NO detach
for epoch in range(200):
    window_starts = randperm(n_windows, device=device) * bptt_window
    for w_start in window_starts:
        u_win = u_train[w_start:w_start+50].unsqueeze(0)  # (1, 50)
        y_win = y_train[w_start:w_start+50].unsqueeze(0)  # (1, 50)
        preds, _ = model(u_win)                            # single forward → 50 steps
        loss = F.mse_loss(preds, y_win)
        loss.backward()                                    # gradients flow through ALL 50 steps
```

**Secondary: Large dt (1.167).**

For the default solver settings (`num_steps=70`, `t_span=7.0`), dt = 0.1. The NARMA preset uses `num_steps=6`, giving `dt ≈ 1.167`. At this step size, the Heun integrator cannot accurately resolve the nonlinear ODE dynamics:

```
With dt = 0.1:   k1 = rhs(x, u), x_pred = x + 0.1·k1  →  small, stable step
With dt = 1.167: k1 = rhs(x, u), x_pred = x + 1.167·k1 →  large step, may overshoot
```

The built-in `_effective_leak` is `softplus(-3.0) ≈ 0.049`. The ODE's natural time constant is `τ = C_eff / leak ≈ 1/0.049 ≈ 20` time units. With `t_span = 7.0`, each Heun integration only covers about `7/20 ≈ 0.35τ` — less than half a time constant per sample. After 6 Heun steps, the state has barely evolved from its initial condition.

**Tertiary: Zero-state initialization for every training window.**

Each BPTT window starts from `state = None` (all zero voltages). The model learns dynamics starting specifically from the zero state. During evaluation, the state evolves continuously for 1000 samples, reaching voltage regimes the model has never seen during training. While a well-designed contractive system would forget the initial condition, the large dt and weak leak mean the state trajectory is dominated by the initial transient for many samples.

---

## Root Cause: Per-Sample `detach()`

### What the code does

```python
state = None   # start of BPTT window
for s in range(50):
    y_pred, _, final_state = net(u_b, initial_state=state, return_final_state=True)
    loss_step = F.mse_loss(y_pred, y_b)    # contributes to total_loss
    state = final_state.detach()            # ↑ GRAPH CUT
```

### What `detach()` does to the autograd graph

```
Sample 0:                    Sample 1:                    Sample 2:
                                                                    
state=0 ──► Heun(u0)        state1 ──► Heun(u1)          state2 ──► Heun(u2)    
  │            │              │            │                │            │        
  │            ▼              │            ▼                │            ▼        
  │        x_final0          │        x_final1              │        x_final2    
  │            │              │            │                │            │        
  │       ┌────┴────┐        │       ┌────┴────┐           │       ┌────┴────┐   
  │       │        detach()  │       │        detach()     │       │        ...
  │       ▼         ▼        │       ▼         ▼           │       ▼            
  │   y_pred0   state1═══════╪═══► state1    x_final1      │   y_pred2          
  │       │                  │       │                     │                    
  └───┬───┘                  └───┬───┘                     └───┬───┘            
loss0 ──┐                   loss1 ──┐                     loss2 ──┐            
        └── total_loss ◄────────────┴───────────────────────────┘             
                 │                                                            
           ∂/∂θ  ↓ ← gradients from loss0 see only 6 Heun steps in sample 0
           ∂/∂θ  ↓ ← gradients from loss1 see only 6 Heun steps in sample 1
           ∂/∂θ  ↓ ← gradients from loss2 see only 6 Heun steps in sample 2
```

The `detach()` means `state` at sample 1 is a *constant tensor* with no connection to the parameters that produced it from sample 0. PyTorch still passes the *numeric values* forward (the physics is correct — the ODE starts from the correct voltage), but the gradient chain is broken.

### Impact

| Metric | With `detach()` | Without `detach()` (ideal) |
|--------|----------------|---------------------------|
| Gradient depth | 6 ODE steps (1 sample) | 300 ODE steps (50 samples) |
| Learnable timescale | < 1 integrator time constant | 50× the integrator budget |
| Effective sequence length | 1 sample | 50 samples |
| NARMA order-10 visibility | Invisible (needs 10+ sample memory) | Visible (50 > 10) |

The model is trained to be a **1-step predictor with state carryover** but needs to be a **50-step sequence model** to compete with the baselines. The `detach()` collapses the temporal dimension of the gradient, making the fabric functionally equivalent to a **feedforward network with a state shortcut** — the state provides the *value* of the next initial condition but no *learning signal* flows through it.

### Why `detach()` was added

Memory. Without `detach()`, a single micro-batch of 60 parallel windows × 50 samples × 6 Heun steps = **300 sequential ODE integration steps** in one autograd graph. Each `rhs()` call involves:

- `cell_lib(x_src, x_dst)` — per-edge tanh computation (~100 edges)
- KCL scatter-add via `index_add_` (sparse accum, 2 calls per rhs)
- Boundary/temporal-readout OTA currents (25 edges each)
- Leak, clip, drive computations

At 300 steps with backprop through each step (~2× memory due to saved activations for gradient), this would require 6-8 GB just for the integration graph — before any optimizer states or parameter gradients. On a Kaggle GPU (typically 16 GB T4/P100), this would OOM with `batch_size=60`.

---

## Potential Fixes

### Option A — Detach less frequently (recommended first step)

```python
DETACH_EVERY = 10   # hyperparameter

state = None
for s in range(50):
    y_pred, _, final_state = net(u_b, initial_state=state,
                                  return_final_state=True)
    loss_step = F.mse_loss(y_pred, y_b)
    total_loss = total_loss + loss_step

    if s % DETACH_EVERY == DETACH_EVERY - 1:
        state = final_state.detach()    # cut every K samples
    else:
        state = final_state              # gradient flows through
```

| DETACH_EVERY | Gradient depth (ODE steps) | Memory multiplier |
|-------------|---------------------------|-------------------|
| 1 (current) | 6 | 1× |
| 5 | 30 | ~5× |
| 10 | 60 | ~10× |
| 50 (full) | 300 | ~50× |

Start with `DETACH_EVERY=5` or `10`. Reduce `batch_size` if memory is tight.

### Option B — Full BPTT + reduced batch size

Remove `detach()` entirely. Reduce `batch_size` from 60 to e.g. 4, and increase `accum_steps` to 15.

```python
batch_size = 4          # 4 parallel windows
accum_steps = 15        # 4 × 15 = 60 windows per epoch
state = None            # still zero-initialized
for s in range(50):
    y_pred, _, final_state = net(u_b, initial_state=state,
                                  return_final_state=True)
    loss_step = F.mse_loss(y_pred, y_b)
    total_loss = total_loss + loss_step
    state = final_state    # ← NO detach
```

**Tradeoff**: Full 300-step gradient depth, but only 4 parallel sequences. The optimizer sees 60 windows per epoch accumulated over 15 micro-batches (same effective batch size). Memory: 4 windows × 300 ODE steps = 1200-step graph, which may still be tight but more manageable than 60 × 300.

### Option C — Gradient checkpointing

Wrap each Heun step or each `rhs()` call with `torch.utils.checkpoint`:

```python
for _ in range(num_steps):
    k1 = torch.utils.checkpoint.checkpoint(self.rhs, x, u, ...)
    x_pred = x + dt * k1
    k2 = torch.utils.checkpoint.checkpoint(self.rhs, x_pred, u, ...)
    x = x + 0.5 * dt * (k1 + k2)
```

Checkpointing discards intermediate activations during forward and recomputes them during backward. This trades **compute for memory** — each `rhs()` call runs twice in backward (once to recompute, once for the actual gradient). With `batch_size=60` and full 300-step BPTT, this might make the memory budget work.

**Caveat**: `index_add_` scatter ops inside `rhs()` may interact poorly with checkpointing (PyTorch may not save/replay them correctly). Testing required.

### Option D — Per-epoch state carryover

Instead of resetting `state = None` at the start of each epoch, carry the final state from the previous epoch as the initial state for the new epoch:

```python
# Before training loop
state_carried = torch.zeros(1, 26, device=device)  # single-sequence state

for epoch in range(epochs):
    # Use only 1 sequence (batch_size=1), no parallel windows
    for s in range(0, 3000 - 50, 50):
        u_win = u_train[s:s+50]
        y_win = y_train[s:s+50]
        state = state_carried
        for t in range(50):
            y_pred, _, final_state = net(u_win[t].view(1,1),
                                          initial_state=state,
                                          return_final_state=True)
            loss_step = F.mse_loss(y_pred.squeeze(), y_win[t].unsqueeze(0))
            total_loss += loss_step
            state = final_state if full_bptt else final_state.detach()
        state_carried = final_state.detach()  # carry to next epoch
```

This eliminates the zero-state vs continuous-state mismatch entirely. The model would see the same state distribution during training and evaluation. Combined with Option A (moderate detach interval), this may give the best results.

### Summary of fix options

| Option | Gradient depth | Memory cost | Complexity | Expected improvement |
|--------|---------------|-------------|------------|---------------------|
| A (detach every 10) | 60 ODE steps | ~10× | Low | Moderate |
| B (full BPTT, small B) | 300 ODE steps | ~50× → reduced with small B | Medium | High |
| C (checkpointing) | 300 ODE steps | Low (~2×) | High (may break on scatter ops) | High |
| D (state carryover) | Varies | Low | Medium | Moderate (addresses init mismatch) |
| A + D combined | 60 ODE steps + no zero-init | ~10× | Low-Medium | High |
