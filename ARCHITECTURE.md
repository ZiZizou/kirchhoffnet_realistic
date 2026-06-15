# Reduced Differential KirchhoffNet — Architecture & Reference

> **Version:** Idealized (v3)  
> **Design target:** Tapeout-plausible analog compute fabric  
> **Key decisions:** Differential signaling, sparse topology only, cell-library-based edge parameterization (L/S/P/Z), direct BPTT through Heun integration, three-phase fit-compress-prune schedule

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Core Design Decisions](#2-core-design-decisions)
3. [System Architecture](#3-system-architecture)
4. [Module Reference](#4-module-reference)
5. [Data Flow](#5-data-flow)
6. [Training Pipeline](#6-training-pipeline)
7. [Task Presets](#7-task-presets)
8. [Configuration Reference](#8-configuration-reference)
9. [Sparse Solver Subsystem](#9-sparse-solver-subsystem)
10. [Testing](#10-testing)
11. [File Map](#11-file-map)

---

## 1. High-Level Overview

### What this is

A PyTorch implementation of an **analog-inspired neural ODE** based on KirchhoffNet. Each logical node is a **differential pair** of physical voltages (v⁺, v⁻), and the computation happens via **sparse transconductor edges** that mediate current flow between nodes. The network evolves in continuous time, integrated with a fixed-step Heun solver. Training is direct backpropagation-through-time (BPTT).

### Relationship to the original KirchhoffNet paper

The original paper described a graph neural ODE dressed in circuit language. This reimplementation fixes the paper's biggest omissions:

| Original paper | This implementation |
|---|---|
| Branch law = `ReLU(θ₁·Δv + θ₂)` (impossible passive device) | Branch law = library of realizable transconductor surrogates (L/S/P/Z families) |
| Implicitly powered edges (no energy source) | Explicit rail-powered differential transconductor cells + node leak + rail clamps |
| Implied fully connected hardware | Sparse topologies only (line/ring/cluster/grid/empty) |
| Adaptive ODE solver + adjoint | Fixed-step Heun + direct BPTT |
| Single-ended 0-to-VDD encoding | Differential signaling: xⱼ = vⱼ⁺ − vⱼ⁻ |
| No variation/robustness story | SimContext injects PVT + per-edge mismatch at training time |
| Arbitrary topology rewiring between "layers" | Fixed physical stages with simple truncation/padding transfer |
| Per-edge raw device geometry training | Soft library selection over pre-characterized cell families |

### What this is NOT

- **Not a SPICE-accurate circuit simulator.** The edge currents use idealized tanh surrogates with compliance gating, not full transistor models.
- **Not a tapeout-ready netlist.** It omits common-mode dynamics, non-diagonal capacitance, sample-and-hold circuitry, and post-layout parasitics.
- **Not the final signoff model.** It is a training scaffold for architecture search and task feasibility experiments.

---

## 2. Core Design Decisions

### 2.1 Differential signaling

Each logical node `j` is represented by two voltages:

```
xⱼ = vⱼ⁺ − vⱼ⁻        (differential signal — the useful state)
cⱼ = (vⱼ⁺ + vⱼ⁻) / 2   (common-mode — assumed held by feedback)
```

The reduced model assumes strong common-mode feedback so we only simulate `xⱼ`. This gives natural signed representation without awkward "negative in 0-to-VDD" hacks.

**Units (R7).** The conductance, current, and capacitance values in
`config.py` are NORMALIZED to plausible analog ranges, not calibrated to
SI. `x_max` was raised to 3.0 to give the solver more dynamic
range and avoid saturation. `t_span=5` and `num_steps=50` are the
defaults. `C_eff` is a pure scaling. No pretense is made that `g/C`
is a real analog time constant. `V_CM` has been removed from `PHYS`
since it is not simulated.

### 2.2 Sparse topologies only

No fully connected layers. Supported graph primitives:
- `line_graph` — 1D chain with neighborhood radius
- `ring_graph` — 1D periodic chain
- `grid_graph` — 2D local grid with kernel neighborhood
- `cluster_graph` — Erdős-Rényi random sparse graph
- `empty_graph` — no edges (for ablation or projection-only stages)

All primitives emit a single directed edge per unique pair. L/S/P cells
(odd or asymmetric I-V) provide implicit bidirectional conduction via
sign reversal or threshold gating.

### 2.3 Cell library instead of raw device parameters

Edges don't learn raw W/L or bias voltages. They soft-select from 4 cell families:

| Cell | Meaning | gm (norm.) | I_sat (norm.) | ρ (feedback) | θ (threshold) | β (softness) | gleak |
|------|---------|------------|---------------|--------------|---------------|--------------|-------|
| **L** | Weak linear transconductor | 0.2 | 10.0 | 1.0 | 0.0 | 1.0 | 0.01 |
| **S** | Saturating transconductor | 1.0 | 0.5 | 1.0 | 0.0 | 1.0 | 0.01 |
| **P** | Smooth bounded rectifier | 1.0 | 1.0 | 1.0 | 0.0 | 0.1 | 0.0 |
| **Z** | Disabled / zero branch | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |

The standard edge current (L, S, Z) for cell family `q` is:

```
I_standard = I_sat · tanh((gm · u + bias) / I_sat) + g_leak · u
```

The rectifier cell (P) uses a smooth bounded rectifier formula:

```
I_P = I_sat · tanh(gm · softplus((u − θ) / β) / I_sat)
```

where `u = x_src − ρ · x_dst` and compliance gating multiplies by
`σ((x_max − |x_src|)/β_soft) · σ((x_max − |x_dst|)/β_soft)`.

The P cell provides directionality (softplus thresholding), bounded
current (tanh sat), differentiability, and physical plausibility as an
active rail-powered branch.

Per-edge learnable parameters:
- **logits** `[E, Q]` — soft library selection (log-probabilities)
- **raw_mult** `[E]` — edge multiplicity `m = softplus(raw_mult)` (R3.3, can approach 0)
- **raw_leak** `[N]` — per-node weak stabilization leak `g = softplus(raw_leak)`

**Honest I/O split (R1).** The InputMapper writes ONLY to the
hidden-node portion of the differential state vector. Projection nodes
are zero-initialized and remain so until the ODE moves them. The
OutputMapper reads ONLY from the projection-node portion; if no
projection nodes exist (legacy/ablation), it falls back to hidden
positions with a warning. This forces the ODE core to do the work.

### 2.4 Fixed-step Heun integration (not adjoint)

The ODE `C · dx/dt = Σ I_in − Σ I_out − leak · x − clip(x)` is integrated with Heun's method (predictor-corrector, 2nd order):

```
k1 = rhs(x)
x_pred = x + dt · k1
k2 = rhs(x_pred)
x_next = x + (dt/2) · (k1 + k2)
```

Gradients flow directly through the unrolled steps (BPTT). No continuous adjoint. This is more reliable for stiff, physically constrained systems.

### 2.5 Write/Evolve/Read pipeline

The network operates in three phases:
1. **Write:** `InputMapper` (dense), `SparseInputMapper` (one-to-one), or `FanOutInputMapper` (multi-target) maps raw input `u` to bounded initial differential state `x_hidden(0)`. Hidden nodes NOT in `write_idx` are zero-initialized. Projection portion `x_proj(0) = 0`.
2. **Evolve:** Multi-stage ODE core integrates for fixed time horizons. The core evolves hidden + projection nodes together.
3. **Read:** `OutputMapper` linearly projects either the full state slice (sparse `read_idx`) or the projection portion (dense) to output `ŷ`.

**Sparse I/O mapping (sparse-io-mapping spec).** Each preset specifies `write_idx` and `read_idx` lists, and the default write/read modes are:
- `--write-mode one_to_one` (default): each input feature `u_i` writes to exactly one hidden node `h_{write_idx[i]}` via `SparseInputMapper`. Hidden nodes not in `write_idx` are zero-initialized at t=0. Parameter count: `2 * in_dim` (vs `in_dim * hid_count + hid_count` for dense `InputMapper`).
- `--write-mode fan_out`: each input feature writes to K>1 hidden nodes via `FanOutInputMapper` with per-target (gain, bias) pairs. Used by `smooth2d_grid` preset.
- `--read-mode sparse` (default): `OutputMapper` gathers only from `read_idx` full-state indices (a learnable linear of size `len(read_idx) -> out_dim`).
- `--write-mode dense` / `--read-mode dense`: original `InputMapper` / `OutputMapper` behavior for baseline comparison.

Preset defaults (SR4.2):
- `sinx`: `write_mode=one_to_one`, `write_idx=[0]`, `read_mode=sparse`, `read_idx=[7]` (out of 8 hidden + 2 proj = 10)
- `housing`: `write_mode=one_to_one`, `write_idx=[0..7]`, `read_mode=sparse`, `read_idx=[15]` (out of 16 hidden + 4 proj = 20)
- `smooth2d_grid`: `write_mode=fan_out`, `write_fan_out={0:[0,4,8], 1:[3,7,11]}`, `read_mode=sparse`, `read_idx=[16,17,18]`

CLI flags (SR5): `--write-mode {one_to_one,dense,fan_out}`, `--read-mode {sparse,dense}`, `--write-idx "0,2,4"`, `--read-idx "7"`. The index overrides take precedence over preset values.

### 2.6 Staged regularizer warm-up (RR-A)

By default, the five auxiliary regularizers (sparsity, edge_gate, node_gate,
power, capacitance) are **ramped in** over the first 150 epochs so the
network can learn the task first without penalty fighting:
- **Epochs 0–99 (`W=100`)**: all five are multiplied by 0.0 (free phase).
- **Epochs 100–149 (`W+A=150`)**: linear anneal from 0.0 → 1.0.
- **Epochs 150+**: full penalty value.

The schedule is controlled by `reg_schedule(epoch)` in `train.py` and
configurable via `OPTIM["reg_warmup_epochs"]` and
`OPTIM["reg_anneal_epochs"]` in `config.py`. **Note (fix-z-death)**:
`rail` is intentionally NOT in `_REG_KEYS`; it is a safety voltage
clamp on differential node states and is applied at full strength at every
epoch. The rail loss uses a **ReLU² quadratic barrier**
(`F.relu(|x| - x_max).pow(2).mean()`) which has zero loss and zero
gradient inside `[-x_max, x_max]` — unlike the previous `softplus`
formulation which had a non-zero floor creating constant gradient drag
toward zero voltages. The entropy bonus retains its own τ-dependent
scaling.

### 2.7 Per-component complexity regularizers (complexity-pruning-v2)

The old single `complexity` key is **decomposed into 4 physically
motivated terms** for finer-grained control over the pruning process:

| Regularizer | What it penalizes | Default weight |
|-------------|-------------------|----------------|
| `edge_gate` | `Σ σ(z_e)` — open edge count | 5e-4 |
| `node_gate` | `Σ σ(u_j)` — open node count | 1e-4 |
| `power` | `Σ_e σ(z_e) · m_e · Σ_q p(L\|e,q) · gm_q` — static power proxy | 1e-4 |
| `capacitance` | `Σ_j σ(u_j)` — node capacitance area proxy | 1e-5 |

The edge gate regularizer encourages inactive edges to close, the node
gate regularizer encourages inactive nodes to close, the power term uses
the cell library's gm values as a static power proxy, and the capacitance
term penalizes node count as a proxy for routing area.

In addition, **learnable gate parameters** are added to every
`DifferentialStage`:
- `z_logits [E]` — edge open-logit (init 2.0 → `σ(2) ≈ 0.88`).
  Each `i_edge` is multiplied by `σ(z_logits[e])` in `rhs()`,
  so a closed edge contributes zero current.
- `u_logits [N]` — node open-logit (init 2.0 → `σ(2) ≈ 0.88`).
  The state `x` is multiplied elementwise by `σ(u_logits)` before
  voltage differences are computed, so a closed node is pinned to ~0.

Note: the gate init was reduced from 5.0 to 2.0 (fix-z-death) to improve
gradient flow. At 5.0, `σ(5) ≈ 0.993` but `dσ/dz ≈ 0.007` — the gate
gradient was 14× weaker. At 2.0, `dσ/dz ≈ 0.10`, a 2× reduction from
the ideal `z=0` but a 14× improvement over `z=5`.

Helper methods on `DifferentialStage`:
- `edge_gates()` — returns `σ(z_logits)`
- `node_gates()` — returns `σ(u_logits)`
- `active_edge_mask(threshold=0.01)` — edge gates above threshold
- `active_node_mask(threshold=0.01)` — node gates above threshold
- `parameter_breakdown()` — returns dict with gate values

The gate regularizers (edge_gate, node_gate, power, capacitance) go
through the same staged warm-up as sparsity (RR-A, free phase
for first 50 epochs).

Old single `complexity` key is removed from `LAMBDAS`. Backward
compatibility with checkpoints is handled via `dict.get()` fallback
in `compute_loss()` and `compute_solver_loss()`.

### 2.8 Complexity pruning pipeline

After training with gate parameters, the network can be structurally
pruned to remove low-value edges and nodes. The pruning has evolved
significantly from the original version:

**Joint Z+gate criterion.** An edge is kept if its effective activity
score exceeds a threshold:

```
eff_score(e) = (1 - P(Z|e)) · σ(z_e) > edge_threshold
```

This folds both the Z-cell probability (gm_Z ≈ 0 ⇒ no current) and the
edge gate into a single criterion. The default threshold is 0.1 (up from
0.01 in early versions) for the post-x_max=3.0 regime.

**Connectivity backstop.** After gate-based pruning, a BFS from
`write_idx` verifies that all `read_idx` are reachable. Dead islands
(nodes not in any I/O-connected component) are purged. If fewer than
`min_read_nodes` survive, pruning raises `ValueError`.

**Protected nodes.** Write targets (input-side guard) are forced to
survive pruning regardless of gate value. Read nodes are NOT protected
(elastic readout is allowed, meaning pruned read nodes are dropped from
the read_idx list and the OutputMapper is rebuilt with fewer features).

**Edge-only mode.** When `prune_nodes_by_gate=False`, node gates are
ignored for independent pruning. Nodes are only removed via the
connectivity backstop (dead island purge). This preserves the maximum
number of edges consistent with I/O connectivity.

**I/O mapper transfer.** After pruning, the InputMapper and OutputMapper
are rebuilt with weights transferred from the pre-prune network.
`SparseInputMapper` weights are copied directly (indexed by input).
`FanOutInputMapper` targets are remapped through the stage node remap.
`InputMapper`/`RobustInputMapper` rows are selectively copied for
surviving nodes.

Key functions in `topology.py`:
- `prune_stage(stage, edge_threshold, node_threshold, transfer_params, write_idx, read_idx, protected_nodes, min_read_nodes, prune_nodes_by_gate)` — returns `(new_stage, node_remap)`
- `prune_network(core, ...)` — applies `prune_stage` to every stage, reinitializes `StageTransfer` modules, returns `(new_core, stage_remaps)`

The `train_script.py` CLI exposes:
- `--prune` — enable pruning after training
- `--retrain` / `--no-retrain` — retrain pruned network (default: retrain)
- `--prune-edge-threshold` — override edge gate threshold
- `--prune-node-threshold` — override node gate threshold
- `--prune-nodes-by-gate` / `--no-prune-nodes-by-gate` — toggle edge-only mode
- `--retrain-epochs` — retraining epochs (default: same as training, capped at half)
- `--fresh-init` — skip warm-start, reinitialize pruned network from scratch

### 2.9 Three-phase training schedule (three-phase-schedule plan)

A generic phased training pipeline that any preset can opt into via
`preset["schedule"] = "three_phase"`. Splits the total epoch budget
into three phases with independently configured tau annealing and
structural regularizer magnitudes:

| Phase | Epochs | Tau | Regularizers | Action |
|-------|--------|-----|-------------|--------|
| **A** (fit) | 0–30% | Fixed 1.0 | All zero — free fit | Network learns task without structure pressure |
| **B** (compress) | 30–70% | 1.0→0.6 anneal | Gate penalties ramped in (Strategy 2) | Gate logits are pushed toward 0 or 1 |
| **C** (retrain) | 70–100% | 0.6→0.1 anneal | Only sparsity (1e-5) + rail (unchanged) | Auto-prune at B→C boundary, retrain compact network |

**Phase A (fit, epochs 0–30%).** All structural regularizers are zeroed.
Tau stays at 1.0 (no hardening pressure). The network learns the task
freely. Rail is always active as a safety net.

**Phase B (compress, epochs 30–70%).** Tau anneals from 1.0→0.6
(gentle specialization — capping at 0.6 instead of 0.1 prevents Z-death
from prematurely locking edges to the zero cell). Structural
regularizers (edge_gate=5e-5, node_gate=1e-5, power=1e-5,
capacitance=1e-6, sparsity=1e-4) are ramped from 0 to full over the
first ~17% of Phase B (≈7 epochs at 800 total). Gate logits are pushed
toward 0 (closed) or stay at 1 (open).

**Phase C (retrain, epochs 70–100%).** At the B→C boundary, automatic
pruning removes edges/nodes below the schedule's thresholds
(edge_threshold=0.1, node_threshold=0.05). The compact network is
retrained from warm-start with aggressive tau annealing (0.6→0.1) and
only sparsity (1e-5) + rail (unchanged). Gate penalties are off
(irrelevant post-prune).

**Solidification diagnostics.** During Phases A and B, per-epoch
metrics are logged to `solidification.tsv`:
- `mean_max_cell_prob`: mean over all edges of max(softmax(logits/τ)).
- `mean_pZ`: probability mass on the Z cell.
- `mean_sigma_z/u`: average edge/node gate openness.
- `frac_sigma_z_below_0.1/0.05/0.01`: fraction of edges eligible for pruning.

**Argmax validation.** At each validation epoch, the network is
evaluated with τ→0.001 (effectively argmax cell selection) and the
task loss is compared against the soft-τ baseline. A small gap means
cell selection is solidified.

The `--schedule {legacy,three_phase}` CLI flag selects the mode.
`smooth2d_grid` defaults to `three_phase` via its preset config.

### 2.10 Per-preset lambda overrides (RR-D)

Each preset may optionally contain a `"lambdas"` dict that is merged on
top of the global `LAMBDAS`. This allows per-task tuning of a single
regularizer weight without redefining the entire dictionary.

Active overrides:
- **sinx**: `{"rail": 1.0}`.
- **smooth2d_grid**: `{"sparsity": 1e-4, "edge_gate": 5e-5, "node_gate": 1e-5, "power": 1e-5, "capacitance": 1e-6, "rail": 0.1}`.
- **housing**: no override.

### 2.11 Deprecated `temp_c` sampling (RR-C)

The `temp_c` field of `SimContext` is preserved for API compatibility
but **no longer used** by the analog model. `sample_random_context`
always returns `VARIATION["temp_c_default"]` (27.0°C) and ignores the
`temp_choices` argument. External callers that explicitly pass a
non-default `temp_c` value receive a `DeprecationWarning`. The
`legacy_temp=True` flag exists temporarily for code migration and also
emits a warning.

### 2.12 Variation-aware training (R6.3)

By DEFAULT, training uses a `SimContext()` with no mismatch and no
temperature drift, so the optimization sees a clean deterministic
forward. To evaluate robustness, pass `--variation` to `train_script.py`,
which then samples a fresh `SimContext` per training iteration:
- `temp_c` — junction temperature (deprecated, always defaults to 27°C)
- `global_gain_shift` — log-normal global gm drift (σ = 5%)
- `edge_mismatch` — per-edge per-cell log-normal mismatch (σ = 5%)

Mismatch is held fixed over the full transient but resampled each iteration. At validation time, `edge_mismatch=None` (nominal).

### 2.13 Training infrastructure

**AMP (mixed precision).** Enabled by default when CUDA is available
(`--amp` / `--no-amp`). Uses `torch.cuda.amp.GradScaler` with
autocast on forward+loss. The loss is split into task+rail (data-dependent)
and structural (parameter-only) components so DataParallel averages
only the data-dependent part.

**torch.compile.** Enabled by default when CUDA is available
(`--compile` / `--no-compile`). Uses `torch.compile` on
`cell_library.forward` and `stage.rhs` for kernel fusion (~1.3–2×
throughput on T4 Tensor Cores). Disabled when DataParallel is active.

**DataParallel.** Automatically enabled when ≥2 GPUs are detected
(`--parallel` / `--no-parallel`). The regularizer computation is
monkey-patched to unwrap DataParallel before accessing per-stage
parameters.

**Per-stage LR scaling (stage-lr-scaling).** Multi-stage networks
suffer from vanishing gradients in early stages. `make_optimizer()`
can create separate param groups with geometrically increasing LRs:
`stage i LR = base_lr × scale^(S−1−i)`. Controlled via
`--stage-lr-scale` (default 1.0, disabled). Retrain has a separate
`--retrain-stage-lr-scale` (default 1.0) to avoid over-aggressive
warm-start updates.

**Gradient logging.** `--grad-log` enables periodic per-parameter-group
L2 gradient norms written to `grad_norms.txt`. Each row shows per-stage
(logits, raw_mult, raw_leak, z_logits, u_logits) plus mapper norms.

**Batch-size-aware LR auto-scaling.** The learning rate is computed as
`lr = BASE_LR × (batch_size / BASE_BATCH_SIZE)` (Goyal et al., 2017)
where `BASE_LR=3e-4` and `BASE_BATCH_SIZE=1024`. At `batch_size=2048`,
this gives `lr=6e-4`.

---

## 3. System Architecture

```
                          ┌──────────────────────────────────────────┐
                          │           KirchhoffNetWithIO              │
                          │                                          │
   raw input u ──────────►│  InputMapper / SparseInputMapper         │
                          │  / FanOutInputMapper                     │
                          │    xⱼ(0) = x_max · tanh(W·u + b)         │
                          │    or per-feature (gain, bias) pairs     │
                          │                                          │
                          │  KirchhoffNet (ODE core)                 │
                          │  ┌────────────────────────────────────┐  │
                          │  │  Stage 0  ──►  Transfer  ──► ...  │  │
                          │  │  (Heun integration, t_span=5, 50  │  │
                          │  │   steps)                          │  │
                          │  └────────────────────────────────────┘  │
                          │                                          │
                          │  OutputMapper                            │
   prediction ŷ ◄─────────│    ŷ = Linear(x_final)                   │
                          │    or x_final[read_idx] → Linear         │
                          │                                          │
                          └──────────────────────────────────────────┘
```

### Inside one DifferentialStage

```
  x_src[j]  ──►  CellLibrary  ──►  i_edge[e]  ──►  scatter-add to dst
  x_dst[j]  ──►  (L/S/P/Z soft)    (μA)           subtract from src
                 × σ(z_logits)      × multiplicity
                 × compliance gate
                                                    ┌──────────────┐
                              ┌─────────────────────►│  acc[dst] += │
                              │                      │  acc[src] -= │
                              │                      └──────────────┘
                                                    ┌──────────────┐
                              ┌─────────────────────►│  -leak·x     │
                              │                      │  -clip(x)    │
                              │                      └──────────────┘
                                                    ┌──────────────┐
                              └─────────────────────►│  dx/dt = Σ/C │
                                                     └──────────────┘
```

### Node types in SparseTopology (pre-filter)

| Kind | Role | In ODE core? |
|------|------|-------------|
| `input` | Receives write-path initialization from InputMapper | No — filtered out |
| `hidden` | Internal dynamical node | Yes |
| `proj` | Projection node (global coupling, readout tap) | Yes |
| `output` | Readout placeholder for OutputMapper | No — filtered out |

---

## 4. Module Reference

### `config.py`
**Single source of truth for all tunable constants.**

- `CELL_LIBRARY` — L/S/P/Z cell parameters (gm, isat, rho, gleak, bias, theta, beta)
- `PHYS` — physical constants (x_max=3.0, C_eff=1.0, beta_softness=0.02, clip_current=0.05, clip_softness=0.02)
- `OPTIM` — training hyperparameters (lr=6e-4, wd=1e-4, epochs=800, batch_size=2048, reg_warmup_epochs=100, reg_anneal_epochs=50, CosineAnnealingLR with T_max based on phase boundaries)
- `TAU` — temperature annealing schedule (init=1.0, final=0.1, min=0.15, hardening_epoch_frac=0.1, final_pretrain=0.8 for two-phase pre-prune)
- `LAMBDAS` — regularizer weights (sparsity=1e-3, rail=1.0, edge_gate=5e-4, node_gate=1e-4, power=1e-4, capacitance=1e-5, entropy=0.0)
- `PRUNE` — pruning thresholds (edge_threshold=0.1, node_threshold=0.05, prune_nodes_by_gate=True)
- `SCHEDULE_THREE_PHASE` — three-phase schedule config (frac_a=0.30, frac_b=0.40, frac_c=0.30; tau targets per phase; Phase B/C lambdas; warmup_frac_b)
- `SOLVER` — integration defaults (method=heun, t_span=5, num_steps=50)
- `INIT` — parameter initialization biases (logits_z_bias=0.0 for equal cell probability P(L)=P(S)=P(P)=P(Z)=0.25; z_logit_init=2.0, u_logit_init=2.0 → σ≈0.88, dσ/dz≈0.10; raw_mult_init=0.0, raw_leak_init=-3.0, gain_scale=1.0)
- `VARIATION` — PVT/mismatch defaults (temp_c=27.0, gain_shift_std=0.05, edge_mismatch_std=0.05; temp_c sampling deprecated)
- `PRESETS` — task-specific topology configs (sinx, housing, smooth2d, smooth2d_grid; supports per-preset lambdas, write_mode, schedule flag, write_fan_out)

### `cell_library.py`
**IdealizedCellLibrary** — Tanh-surrogate edge cell library with smooth rectifier support.

- `forward(x_src, x_dst, logits, raw_mult, x_max, ctx, tau)` → `i_edge [batch, E]`
- Supports both standard tanh formula (L/S/Z) and rectifier formula (P) via `_is_rect` mask
- Injects PVT mismatch multiplicatively on gm: `gm *= exp(edge_mismatch)`
- Injects global gain shift: `gm *= exp(global_gain_shift)`
- Compliance gate: sigmoid-based transition when |x_src| or |x_dst| approaches x_max
- Soft library selection: `weights = softmax(logits / tau)`
- Multiplicity: `m = softplus(raw_mult)` (R3.3: m → 0 as raw_mult → -inf)
- `compile_forward(backend)`: wraps `forward` with `torch.compile` for kernel fusion

### `topology.py`
**Graph construction, topology management, and pruning.**

Three-layer API:
1. **Primitives:** `line_graph()`, `ring_graph()`, `grid_graph()`, `cluster_graph()`, `empty_graph()`
2. **Connectors:** `connect_bipartite()`, `connect_projection()`
3. **Composer:** `StageTopologyBuilder`, `MultiStageTopology.from_config()`

Key data structures:
- `SparseTopology` — universal sparse graph with src/dst edge lists, node kinds, edge types
- `validate_topology()` — sanity checks (no self-loops, density limits, input/output connectivity)
- `validate_topology_degrees(write_idx, read_idx)` — hard-error check that write→read is >1 hop
- `topology_to_stage()` — filters I/O edges, remaps node IDs, builds DifferentialStage
- `prune_stage(stage, ...)` — structural pruning with joint Z+gate criterion, connectivity backstop, protected nodes, edge-only mode; returns `(new_stage, node_remap)`
- `prune_network(core, ...)` — applies `prune_stage` to all stages, returns `(new_core, stage_remaps)`
- `build_net_from_preset()` / `build_net_from_config()` — factory functions supporting all write/read modes

### `differential_stage.py`
**DifferentialStage** — A single ODE stage with sparse COO graph + Heun integration.

Per-node dynamics:
```
C_eff · dxⱼ/dt = Σ_{e: dst(e)=j} I_e − Σ_{e: src(e)=j} I_e − leakⱼ · xⱼ − clip(xⱼ)
```

- `rhs(x, ctx, tau)` — computes dx/dt at current state (applies node gates `σ(u_logits)` to x before computing voltages, and edge gates `σ(z_logits)` to i_edge after cell library evaluation; accumulates KCL via float32 scatter-add for AMP robustness)
- `forward(x0, ctx, t_span, num_steps, tau, store_trajectory)` — Heun integration, returns `(x_final, [batch, N, steps+1] trajectory)`
- Parameters: `logits [E, Q]`, `raw_mult [E]`, `raw_leak [N]`, `z_logits [E]`, `u_logits [N]`
- Buffers: `src`, `dst` (COO format edge lists)
- Helper methods: `edge_gates()`, `node_gates()`, `active_edge_mask()`, `active_node_mask()`, `parameter_breakdown()`
- `compile_rhs(backend)`: wraps `rhs` with `torch.compile`

### `sim_context.py`
**SimContext** — PVT + mismatch container for one forward pass.

- `SimContext(temp_c, global_gain_shift, edge_mismatch)` — dataclass with `.to(device)` method
- `sample_random_context(num_edges, num_cells, ...)` — factory for training variation injection
- temp_c is deprecated (RR-C): defaults to `VARIATION["temp_c_default"]` (27.0°C), no longer randomized

### `stage_transfer.py`
**StageTransfer** — Width-changing layer between stages. No learnable parameters.

- If `out_nodes < in_nodes`: truncates
- If `out_nodes > in_nodes`: zero-pads
- If equal: identity pass-through

### `io_mapper.py`
**Input/Output mappers** — Write and read phases.

- `InputMapper(in_dim, out_dim)` — `x_hidden(0) = x_max · tanh(Linear(u))`, Xavier init scaled by gain_scale. Used in `--write-mode dense`.
- `RobustInputMapper(in_dim, out_dim)` — adds per-feature learnable log-scale preconditioner. Used in `--write-mode dense` when `use_robust_input=True` (housing preset).
- `SparseInputMapper(in_dim, out_dim, write_idx)` — one-to-one writer. Each input feature `u_i` writes to `h_{write_idx[i]}` via independent `(gain_i, bias_i)` pair; non-write positions are zero. Parameter count = `2 * in_dim`. Used in `--write-mode one_to_one` (default). Raises `ValueError` if `in_dim > out_dim`.
- `FanOutInputMapper(in_dim, out_dim, fan_out_map)` — multi-target writer. Each input feature writes to K>1 hidden nodes via per-target (gain, bias) pairs. Hidden nodes NOT in the union of all targets are zero. Parameter count = `2 * K_total`. Used by `smooth2d_grid` preset (`--write-mode fan_out`).
- `OutputMapper(node_dim, out_dim, read_idx=None)` — `ŷ = Linear(x_read)`, no activation. With `read_idx=None` (dense), reads from projection portion. With `read_idx` (sparse, default), gathers from specified full-state indices.

### `kirchhoff_net.py`
**KirchhoffNet** and **KirchhoffNetWithIO** — Top-level network classes.

- `KirchhoffNet(stages, transfers, stage_times, stage_steps)` — multi-stage ODE core
  - Handles per-stage edge_mismatch slicing internally (splits `ctx.edge_mismatch` across stages)
  - `parameter_breakdown()` for diagnostics
- `KirchhoffNetWithIO(input_mapper, core, output_mapper, hid_count, proj_count, final_hid_count, final_proj_count, write_idx=None, read_idx=None)` — write/evolve/read pipeline
  - `hid_count` / `proj_count` enforce the honest I/O split
  - `final_hid_count` / `final_proj_count` define the final-stage read_slice
  - `write_idx` (sparse mode): list of hidden-node indices; hidden nodes not in write_idx stay at 0 at t=0
  - `read_idx` (sparse mode): list of full-state indices; OutputMapper gathers from these positions
  - When all `core.stage_times` are 0, forward is identity (mapper-only ablation)
  - `forward(u, ctx, tau, store_trajectory)` → `(ŷ, trajectories)`

### `train.py`
**Loss functions, regularizers, tau annealing, training loop, three-phase schedule.**

- `compute_loss(net, u, target, ctx, task_fn, ...)` — total = task + reg_scale·(sparsity + edge_gate + node_gate + power + capacitance) + rail − entropy_bonus. Splits into task+rail (data-dependent) and structural (parameter-only) components for DataParallel safety.
- `compute_solver_loss(net, b, x_star, A, ctx, ...)` — solver-specific: residual + 0.1·solution + regularizers (preserved on disk; not active in paper v1)
- `tau_for_epoch(epoch, total_epochs, tau_init, tau_final)` — monotonic exponential decay with smooth linear hardening in the last fraction of training
- `reg_schedule(epoch)` — piecewise linear warm-up: [0, W) off, [W, W+A) linear anneal, [W+A, ∞) full
- `apply_reg_schedule(lambdas, epoch)` — returns copy of lambdas with structural terms scaled by reg_schedule
- `phase_boundaries(total_epochs)` / `phase_for_epoch(epoch, ...)` / `three_phase_tau(...)` / `three_phase_lambdas(...)` — three-phase schedule infrastructure
- `compute_solidification_metrics(net, tau)` — returns dict of mean cell prob, P(Z), gate openness fractions
- `validate_argmax(net, val_loader, ...)` — validation with τ→0.001 (argmax cell selection) for solidification diagnostics
- `make_optimizer(net, lr, weight_decay, stage_lr_scale)` — AdamW with optional per-stage geometric LR scaling
- `apply_ablation(net, ablation)` — in-place structural ablation: 'none', 'mapper-only', 'empty-graph'
- `train_epoch(net, loader, optimizer, task_fn, ctx_factory, epoch, ...)` — single-epoch loop with AMP support

**Regularizer details:**

| Regularizer | What it penalizes | Weight |
|-------------|-------------------|-------|
| Sparsity | `Σ w[:, :Z_index]` (active non-Z cells) | 1e-3 |
| Rail (ReLU²) | `mean(ReLU²(|x| - x_max))` over trajectory | 1.0 |
| Edge gate | `Σ σ(z_e)` — open edge count | 5e-4 |
| Node gate | `Σ σ(u_j)` — open node count | 1e-4 |
| Power | `Σ_e σ(z_e) · m_e · Σ_q p(L\|e,q) · gm_q` — static power proxy | 1e-4 |
| Capacitance | `C_eff · Σ_j σ(u_j)` — node capacitance area proxy | 1e-5 |
| Entropy bonus | `−Σ w·log(w)` of logits/tau (off by default) | 0.0·tau |

### `train_script.py`
**Main training entry point** — CLI script supporting `--problem {sinx,housing,smooth2d,smooth2d_grid}`.

CLI flags:
- `--problem`, `--output`, `--epochs`, `--lr`, `--device`
- `--stage-lr-scale`, `--retrain-stage-lr-scale`
- `--amp` / `--no-amp`, `--compile` / `--no-compile`, `--parallel` / `--no-parallel`
- `--validate-every`, `--early-stop` / `--no-early-stop`, `--patience`, `--min-delta`
- `--ablation {none,mapper-only,empty-graph}`, `--variation`
- `--write-mode {one_to_one,dense,fan_out}`, `--read-mode {sparse,dense}`
- `--write-idx`, `--read-idx` (comma-separated)
- `--prune`, `--retrain` / `--no-retrain`, `--prune-edge-threshold`, `--prune-node-threshold`
- `--prune-nodes-by-gate` / `--no-prune-nodes-by-gate`
- `--retrain-epochs`, `--retrain-lr`, `--fresh-init`
- `--scheduler-type {cosine,warm_restarts}`, `--no-scheduler`
- `--grad-log`, `--grad-log-every`
- `--schedule {legacy,three_phase}`, `--no-argmax-val`

Outputs per run: `loss_history.txt` (with phase markers for three_phase), `loss_curve.png`, `model.pt`, `config_snapshot.txt`, per-stage graph/selection/trajectory plots, output fit, pipeline diagram, `solidification_metrics.txt` (three_phase), `grad_norms.txt` (when --grad-log enabled), `prune_summary.txt` (when pruning), `model_pruned.pt`.

### `visualize.py`
**Visualization utilities** (lazy-imports matplotlib/networkx).

- `plot_sparse_topology(topo)` — colored graph with input/hidden/proj/output nodes
- `plot_stage_graph(stage)` — post-filter ODE core graph
- `plot_multi_stage_topology(multi)` — side-by-side stage graphs
- `plot_trajectories(trajs)` — node voltage vs integration step
- `plot_cell_selection(logits)` — heatmap of P(cell | edge)
- `plot_output_fit(y_pred, y_true)` — scatter + residuals
- `plot_network(net)` — full pipeline visualization

### `gen_network_images.py`
**Standalone image generator** — Runs all 4 non-solver presets and saves visualizations to `network_visualization/`.

### `mlp_benchmark.py`
**MLP baselines** — MLPRegressor(2→H→1) for smooth2d Franke task, benchmark comparison.

---

## 5. Data Flow

### Training step (one batch) — legacy schedule

```
1. Sample SimContext (PVT + mismatch)
       │
2. InputMapper:  u [B, in_dim]  →  x0 [B, N_active₀]  (write phase)
       │
3. Stage 0 Heun (50 steps @ t_span=5):
        │    for step in 1..num_steps:
       │        x_src = x[:, src]; x_dst = x[:, dst]
       │        i_edge = CellLibrary(x_src, x_dst, logits, raw_mult, x_max, ctx, tau)
       │        i_edge *= σ(z_logits)                    (edge gate)
       │        acc[dst] += i_edge; acc[src] -= i_edge   (KCL scatter-add)
       │        dx/dt = (acc - leak·x - clip(x)) / C_eff
       │        x ← Heun step
       │
       ▼
4. StageTransfer: x [B, N₁] → x [B, N₂]  (truncate or zero-pad)
       │
5. Stage 1 Heun ... (repeat for all stages)
       │
6. OutputMapper:  x_final [B, N_active_last]  →  ŷ [B, out_dim]  (read phase)
       │
7. Compute reg_scale = reg_schedule(epoch)  (RR-A: 0.0 during free phase)
       │
8. Loss = task_loss + rail + reg_scale · Σ λ·regularizer  →  backward through steps
```

### Training step — three-phase schedule

```
1. Determine phase for current epoch (A/B/C via phase_boundaries).
2. Compute tau via three_phase_tau (phase-dependent annealing).
3. Compute effective_lambdas via three_phase_lambdas (phase-dependent weights,
   including warmup within Phase B).
4. Forward pass (same as legacy).
5. At B→C boundary: auto-prune (remove low-gate edges/nodes, transfer I/O mappers).
6. Phase C: retrain compact network with post-prune lambdas.
```

### Input/Output edge filtering

`SparseTopology` includes input/output edges for topology visualization and validation. `topology_to_stage()` **removes** them — only hidden and projection edges enter the ODE core. This means:
- Input nodes are placeholders consumed by InputMapper (write path)
- Output nodes are placeholders consumed by OutputMapper (read path)
- The ODE only evolves hidden + projection nodes

---

## 6. Training Pipeline

### Temperature annealing

τ controls the softness of library selection. High τ = soft mixture (exploration), low τ = hard selection (exploitation).

Schedule: monotonic exponential decay with smooth linear hardening in the last 20% of training:

```
τ_base = max(τ_min, τ_init · exp(−epoch / (total_epochs · 0.5)))
τ = τ_base for first 80% of training
τ = linear interpolate from τ_base → τ_final over last 20%
```

Configuration:
- `TAU.init = 1.0`, `TAU.final = 0.1`, `TAU.min = 0.15`
- `TAU.hardening_epoch_frac = 0.1` (10% on each side = 20% transition window)
- `TAU.final_pretrain = 0.8` (two-phase tau: pre-prune annealing stops at 0.8, retrain continues 0.8→0.1)

### Initialization

Initial logits are all zero, with `logits_z_bias=0.0` applied to the
Z index. With 4 cells (L/S/P/Z), this gives equal P=0.25 for each cell
at init (fix-z-death: was +1.0 bias giving P(Z)≈0.42, which combined with
tau annealing 1.0→0.1 amplified to ~99.99% Z by end of training).

Gate logits are initialized at +2.0 so `σ(2) ≈ 0.88` with `dσ/dz ≈ 0.10`.
This gives 88% open gates (strong signal flow) while keeping gate gradients
14× healthier than `z=5.0` (dσ/dz=0.007).

- `raw_mult`: zeros → multiplicity = softplus(0) = ln(2) ≈ 0.69
- `raw_leak`: −3.0 → leak = softplus(−3) ≈ 0.05
- InputMapper/SparseInputMapper gain: Xavier uniform scaled by `gain_scale=1.0`

### Gradient handling

- Clip gradients to norm 5.0
- AdamW optimizer with weight decay 1e-4 (lr=6e-4 at default batch_size=2048)
- Two scheduler types:
  - `'cosine'` (default): CosineAnnealingLR with `T_max` based on phase boundaries
  - `'warm_restarts'`: CosineAnnealingWarmRestarts (T_0=50, T_mult=1, eta_min=1e-5)
- SimContext is no_grad (variation doesn't get gradients)
- AMP autocast on forward+loss, GradScaler for grad scaling
- torch.compile on cell_lib and rhs (disabled with DataParallel)

### Per-stage LR scaling (stage-lr-scaling)

Multi-stage networks suffer from **vanishing gradients in early stages**: the
gradient norm of stage 0 parameters can be 4–5 orders of magnitude smaller
than stage 2 because gradient information must flow backward through multiple
ODE solves.

To compensate, `make_optimizer()` can create separate param groups per stage
with geometrically increasing learning rates for earlier stages:

```
stage i LR = base_lr × scale^(S − 1 − i)
```

Where `S` = number of stages. Example with 3 stages and `scale=10`:
- stage 0 (earliest, smallest gradient) → `lr × 100`
- stage 1 → `lr × 10`
- stage 2 → `lr × 1`

Non-stage parameters (I/O mappers, StageTransfers) receive the base LR by
default (one additional param group). Controlled via `--stage-lr-scale`
CLI flag (default `1.0`, which collapses to a single standard param group).

**Retrain uses a separate flag `--retrain-stage-lr-scale` (default `1.0`)
to avoid over-aggressive updates during warm-start fine-tuning.**

---

## 7. Task Presets

### `sinx` — Sine function regression
| | |
|---|---|
| Input | 1D (angle in [−π, π]) |
| Output | 1D (sin(angle)) |
| Architecture | 1 stage: 8 hidden + 2 proj, **line** topology (radius=2) |
| Loss | MSE |
| Train size | 8192 |
| Write/Read | `write_mode=one_to_one`, `write_idx=[0]`, `read_mode=sparse`, `read_idx=[7]` |
| Special | Lambda override: `{"rail": 1.0}` |

### `housing` — California housing price regression (appendix-only)
| | |
|---|---|
| Input | 8D normalized features |
| Output | 1D (price, standardized) |
| Architecture | 1 stage: 16 hidden + 4 proj, **line** topology (radius=2) |
| Loss | MAE |
| Train size | ~16.5K |
| Write/Read | `write_mode=one_to_one`, `write_idx=[0..7]`, `read_mode=sparse`, `read_idx=[15]` |
| Special | Uses `RobustInputMapper` (per-feature log-scale preconditioner) |

### `smooth2d` — Franke function 2D regression (line topology)
| | |
|---|---|
| Input | 2D (x, y) in [0, 1]² |
| Output | 1D (Franke function value, zero-mean unit-variance normalized) |
| Architecture | 1 stage: 10 hidden + 2 proj, **line** topology (radius=2) |
| Loss | MSE |
| Train size | 20K (4K val, 4K test, sigma=0.01 noise) |
| Write/Read | `write_mode=one_to_one`, `write_idx=[0,1]`, `read_mode=sparse`, `read_idx=[9]` (hidden readout) |
| Special | LHS-based sampling for training data. No lambda overrides. |

### `smooth2d_grid` — Franke 2D regression (3-stage grid topology)
| | |
|---|---|
| Input | 2D (x, y) in [0, 1]² |
| Output | 1D (Franke function value, normalized) |
| Architecture | **3 stages**: each 16 hidden (4×4 grid, 8-neighbor kernel_size=3) + 3 proj (all_to_all). StageTransfer identity (19→19). |
| Edges/Params | ~90 edges/stage (42 hidden + 48 proj), ~588 params/stage = ~1764 total |
| Integration | t_span=5/3 ≈ 1.667 per stage, 17 steps per stage, dt ≈ 0.098 constant |
| Loss | MSE |
| Train size | 20K (4K val, 4K test, sigma=0.01 noise) |
| Write/Read | `write_mode=fan_out`, `write_fan_out={0:[0,4,8], 1:[3,7,11]}` (6 write targets), `read_mode=sparse`, `read_idx=[16,17,18]` (3 proj readout) |
| Schedule | `schedule=three_phase` (auto-prune at B→C boundary) |
| Lambda override | `{"sparsity": 1e-4, "edge_gate": 5e-5, "node_gate": 1e-5, "power": 1e-5, "capacitance": 1e-6, "rail": 0.1}` |
| Special | Fan-out write spreads each input to 3 grid positions (left/right columns). Proj-only read. Structural regularizers zeroed in fit-first phase. Multi-stage helps vanishing gradient. |

### Removed presets (R4.2, R4.3)
- `xor` — removed from active PRESETS (weak analog motivation).
- `solver` — sparse linear system benchmark removed from active PRESETS
  (scope creep for paper v1). The supporting modules
  (`sparse_solver_data.py`, `sparse_solver_topology.py`,
  `sparse_solver_baseline.py`, `sparse_solver_track.py`) are preserved
  on disk for future work.

---

## 8. Configuration Reference

All constants live in `config.py`. Key groups (R7: units are normalized, not SI):

```python
# Cell library (4 families: L=linear, S=saturating, P=rectifier, Z=disabled)
CELL_L = {"gm": 0.2, "isat": 10.0, "rho": 1.0, "gleak": 0.01, "bias": 0.0, "theta": 0.0, "beta": 1.0}
CELL_S = {"gm": 1.0, "isat": 0.5,  "rho": 1.0, "gleak": 0.01, "bias": 0.0, "theta": 0.0, "beta": 1.0}
CELL_P = {"gm": 1.0, "isat": 1.0,  "rho": 1.0, "gleak": 0.0,  "bias": 0.0, "theta": 0.0, "beta": 0.1}
CELL_Z = {"gm": 0.0, "isat": 0.0,  "rho": 0.0, "gleak": 0.0,  "bias": 0.0, "theta": 0.0, "beta": 1.0}
CELL_ORDER = ["L", "S", "P", "Z"]; Z_INDEX = 3; NUM_CELLS = 4

# Normalized physical limits (R7: not SI-calibrated; x_max=3.0 for headroom)
PHYS = {"x_max": 3.0, "C_eff": 1.0,
        "beta_softness": 0.02, "clip_current": 0.05, "clip_softness": 0.02}

# Training (RR-A: reg_warmup_epochs=100 for longer free phase;
#              lr auto-scaled: BASE_LR=3e-4 * batch_size/1024 = 6e-4 at 2048)
OPTIM = {"lr": 6e-4, "weight_decay": 1e-4, "grad_clip_norm": 5.0,
         "epochs": 800, "batch_size": 2048,
         "reg_warmup_epochs": 100, "reg_anneal_epochs": 50,
         "scheduler_T_0": 80, "scheduler_T_mult": 1, "scheduler_eta_min": 1e-5}

# Temperature annealing (init=1.0, final=0.1, min=0.15,
#                        final_pretrain=0.8 for two-phase pre-prune)
TAU = {"init": 1.0, "final": 0.1, "min": 0.15,
       "T_0": 80, "hardening_epoch_frac": 0.1, "final_pretrain": 0.8}

# Regularization weights (CP: 4 per-component terms;
#                          rail=1.0 with ReLU² barrier)
LAMBDAS = {"sparsity": 1e-3, "rail": 1.0,
           "edge_gate": 5e-4, "node_gate": 1e-4,
           "power": 1e-4, "capacitance": 1e-5,
           "entropy": 0.0}

# Pruning thresholds (post-x_max=3.0 regime: 0.1/0.05, was 0.01/0.01)
PRUNE = {"edge_threshold": 0.1, "node_threshold": 0.05,
         "prune_nodes_by_gate": True}

# Three-phase schedule (fit 30% / compress 40% / retrain 30%)
SCHEDULE_THREE_PHASE = {"frac_a": 0.30, "frac_b": 0.40, "frac_c": 0.30,
                        "tau_a": 1.0, "tau_b_init": 1.0, "tau_b_final": 0.6,
                        "tau_c_init": 0.6, "tau_c_final": 0.1,
                        "warmup_frac_b": 1.0/6.0,
                        "lambdas_b": {"sparsity": 1e-4, "edge_gate": 5e-5,
                                      "node_gate": 1e-5, "power": 1e-5,
                                      "capacitance": 1e-6},
                        "lambdas_c": {"sparsity": 1e-5, "edge_gate": 0.0,
                                      "node_gate": 0.0, "power": 0.0,
                                      "capacitance": 0.0},
                        "prune_edge_threshold": 0.1,
                        "prune_node_threshold": 0.05,
                        "prune_nodes_by_gate": True}

# Integration defaults
SOLVER = {"method": "heun", "t_span": 5, "num_steps": 50}

# Parameter initialization (fix-z-death: equal cell probability, healthy gate gradients)
INIT = {"logits_z_bias": 0.0, "raw_mult_init": 0.0, "raw_leak_init": -3.0,
        "gain_scale": 1.0, "z_logit_init": 2.0, "u_logit_init": 2.0}

# Variation injection (R6.3: off by default at training time; temp_c deprecated)
VARIATION = {"temp_c_default": 27.0, "temp_c_choices": [0.0, 27.0, 75.0],
             "global_gain_shift_std": 0.05, "edge_mismatch_std": 0.05}
```

---

## 9. Sparse Solver Subsystem (preserved, not in active PRESETS)

The sparse solver benchmark is preserved on disk but **not active in
paper v1** (R4.3). Modules:

### `sparse_solver_data.py`
Generates random sparse symmetric positive-definite matrices with controlled condition number via:
1. Random sparse symmetric pattern (scipy.sparse.rand)
2. Diagonal dominance margin (guarantees SPD)
3. Low-rank perturbation to spread eigenvalues to target condition number
4. Bounded random solution `x* = x_max · tanh(randn)`, then `b = A @ x*`

### `sparse_solver_topology.py`
Accumulates matrix sparsity patterns across the training dataset into a fixed union supergraph. Edges appearing in ≥10% of samples become hidden graph edges. Uses `StageTopologyBuilder` to add input/projection edges.

### `sparse_solver_baseline.py`
Digital solvers for comparison against the analog network:
- **Jacobi:** `x_{k+1} = D⁻¹(b − R·x_k)`
- **Conjugate Gradient:** Batched CG with early stopping on convergence
- `compare_against_baselines(net, val_loader, n)` — runs all three methods on the same validation set and prints per-method statistics

### `sparse_solver_track.py`
Convergence diagnostic: captures ODE state snapshots during integration and plots 3-panel figure (residual vs time, solution error vs time, final state magnitude distribution).

---

## 10. Testing

### Smoke test (`test_smoke.py`)

**567 total test checks** (116 test functions; 565 pass + 2 pre-existing failures) covering the full pipeline:
R1-R7, RR-A through RR-D reviewer-residual cleanup, CP-1 through CP-5
complexity pruning, smooth2d and smooth2d_grid presets (including
three-phase schedule, fan-out write), MLP benchmark comparison,
stage-LR scaling, rail-loss-fix, three-phase schedule (TP-1 through TP-9),
edge-only pruning, and prune I/O mapper transfer.

Run with:
```bash
/home/annaik/Documents/ASPDAC_2026/venv/bin/python \
  kirchhoff_redesign/ideal/test_smoke.py
```

Coverage:
| # | Test | What it verifies |
|---|------|-----------------|
| 1 | Config loads | CELL_LIBRARY has L/S/P/Z, CELL_ORDER=['L','S','P','Z'], Z_INDEX=3, NUM_CELLS=4, PHYS x_max=3.0, LAMBDAS all keys, PRUNE thresholds, OPTIM, PRESETS |
| 2 | SimContext | Default values, sampled mismatch shape/finiteness |
| 3 | Topology primitives | line, ring, grid, cluster, empty — edge counts, single edge per pair, no self-loops |
| 4 | StageTransfer | Equal width, truncation, zero-padding |
| 5 | Heun convergence | No NaN/explosion on zero input, trajectory shape correct |
| 6 | Gradient flow | Gradients reach logits, raw_mult, raw_leak |
| 7 | Loss finite | All regularizer components are finite and ≥0 |
| 8 | Sparsity push | Training reduces P(L or S) and increases P(Z) |
| 9 | Tau annealing | Schedule values at epoch 0, mid, end; smooth hardening |
| 10 | Round-trip sinx | Forward + loss + backward + step with sinx preset |
| 11 | XOR preset removed | xor not in active PRESETS (R4.2) |
| 12 | Housing preset | SparseInputMapper (default) / RobustInputMapper (dense mode), forward shape correct |
| 13 | I/O filtering | topology_to_stage removes I/O nodes/edges from ODE |
| 14 | Topology validation | Valid topo passes, self-loop rejected |
| 15-17 | Visualization | Stage graph, sparse topology, trajectory plots save to disk |
| 18-24 | Solver subsystem | SPD gen, dataset, union topology, solver loss, baselines, tracker (modules preserved on disk) |
| 25 | Solver preset removed | solver not in active PRESETS (R4.3) |
| 26 | Honest I/O split | R1.1-R1.3: InputMapper writes to hidden only, projection zero-init, read_slice selects projection |
| 27 | No-proj fallback | R1.4: warning emitted, read_slice falls back to hidden |
| 28-29 | Mapper-only ablation | t_span=0 makes output = OutputMapper(InputMapper(u)); fast validate mode |
| 30 | Weighted power/area | R3.1/R3.3/R3.4: weighted proxy goes to 0 for Z edges, softplus(0)=log(2) |
| 31 | Active presets stage count | All active presets are correct topology/nodes |
| 32 | sinx uses line | sinx preset uses line topology, not ring |
| 33 | Tau monotonic | R6.1: tau is non-increasing |
| 34 | CLI flags | --ablation and --variation flags exist in train_script.py |
| 35 | Normalized units | R7: V_CM removed, LAMBDAS['C'] removed, "normalized" present |
| 36 | apply_ablation | All three ablation modes work |
| 37 | Sparse I/O preset defaults | SR4.2: write_idx/read_idx present in presets; default build uses SparseInputMapper |
| 38 | Sparse write zeros non-targets | SR1.3: SparseInputMapper leaves non-write positions at 0 |
| 39 | Sparse write validation | SR1.6: in_dim > out_dim, out-of-range, duplicate write_idx raise ValueError |
| 40 | Sparse read select | SR2.1: OutputMapper(read_idx=...) gathers only specified positions |
| 41 | Dense fallback | SR3/SR6: --write-mode dense / --read-mode dense = original behavior |
| 42 | Sparse I/O gradients | SR1/SR2: gradients flow to SparseInputMapper gain and OutputMapper projection |
| 43 | Sparse I/O CLI flags | SR5: --write-mode, --read-mode, --write-idx, --read-idx exposed |
| 44 | Complexity proxy | RR-B/R3: merged power+area with (1-p_Z) weighting, softplus multiplicity |
| 45 | Reg schedule curve | RR-A: piecewise linear warm-up (free → anneal → full) |
| 46 | smooth2d_grid sparsity zero override | grid preset's sparsity override is 1e-4 (not global 1e-3) |
| 47 | Tau anneal preset option | PRESETS may set tau_anneal=False to disable annealing |
| 48 | Tau override floor | tau_for_epoch with override tau_final enforces min >= tau_final |
| 49 | Tau override endpoints | Override tau_init/tau_final produces correct start/end values |
| 50 | Tau backward compat | tau_for_epoch with no overrides uses config defaults |
| 51 | Preset lambda overrides | RR-D: per-preset lambdas merge correctly over global LAMBDAS |
| 52 | Z-bias eliminated | fix-z-death: INIT['logits_z_bias'] == 0.0, P(Z)=0.25, equal cell prob |
| 53 | Gate initialization | z_logit_init=2.0, u_logit_init=2.0 → σ≈0.88, dσ/dz≈0.10 |
| 54 | Gates applied in rhs | Closed vs open gates produce different RHS output |
| 55 | Per-component regularizers | edge_gate, node_gate, power, capacitance all finite, go to 0 when gates closed |
| 56 | prune_stage | Removes low-gate edges/nodes, returns DifferentialStage without gating |
| 57 | Parameter transfer | Surviving parameters preserved after prune |
| 58 | All-removed raises | prune_stage raises ValueError when all edges removed |
| 59 | prune_network | Applies to KirchhoffNet core, preserves stage_times/stage_steps |
| 60 | Topology degree validation | validate_topology_degrees rejects ≤1-hop write→read |
| 61-63 | Joint Z+gate pruning | eff_score criterion, dead island removal, disconnected I/O raises |
| 64 | prune_network returns remap | Stage remaps dict enables I/O mapper transfer |
| 65-69 | Prune I/O transfer | Mapper weight transfer, forward pass preservation, elastic readout |
| 70 | Prune protects write targets | write_idx nodes forced to survive q |
| 71 | Min read nodes guard | prune fails if too few read nodes survive |
| 72-76 | smooth2d preset | 35 checks: topology (line), Franke dataset, forward shape, Sparse I/O |
| 77-111 | smooth2d_grid preset | 35 checks: 3-stage grid (4×4, kernel=3), fan-out write, three-phase schedule |
| 112 | MLP benchmark | MLPRegressor(2→H→1) on Franke, verifies val loss decreases |
| 113 | FanOutInputMapper | Basic forward, param count, gradients, overlap/missing/oob raises |
| 114 | Optimizer LR | LR is auto-scaled to batch_size |
| 115 | Patience default | Default patience raised to 500 |
| 116 | Scheduler config | OPTIM has scheduler config entries |
| 117 | Tau smooth hardening | hardening_epoch_frac controls smooth linear interpolation |
| 118 | Retrain warmup bounds | Retrain warmup uses correct defaults |
| 119 | Fresh init default | --fresh-init defaults to False |
| 120 | Retrain LR CLI flag | --retrain-lr exposed and functional |
| 121-123 | Loss history | Phase markers, retrain appended, format correct |
| 124-126 | Gradient logging | --grad-log CLI, collect_gradient_norms, file output |
| 127 | Three-phase (TP-1–9) | phase_boundaries, phase_for_epoch, three_phase_tau, three_phase_lambdas, solidification metrics, argmax validation, schedule file markers |

**Pre-existing failures (2):**
Both failures are the same issue — test expectations check `OPTIM["lr"] == 3e-4`
but the auto-scaling formula gives `6e-4` at batch_size=2048:
- `test_config_loads`: `OPTIM has lr=3e-4` (actual: 6e-4)
- `test_smooth2d_grid_preset`: `OPTIM.lr == 3e-4: got 0.0006`

These are cosmetic — the test expectations predate batch-size-aware LR scaling.

---

## 11. File Map

```
kirchhoff_redesign/ideal/
├── __init__.py                    # Package docstring
├── config.py                      # All tunable constants + presets (L/S/P/Z, three-phase schedule)
├── cell_library.py                # IdealizedCellLibrary (L/S/P/Z tanh + rectifier surrogates)
├── topology.py                    # Graph primitives, builder, stage conversion, pruning (joint Z+gate)
├── differential_stage.py          # DifferentialStage (COO graph + Heun ODE + gates + compile)
├── sim_context.py                 # SimContext (PVT + mismatch dataclass, temp_c deprecated)
├── stage_transfer.py              # StageTransfer (truncation/zero-padding)
├── io_mapper.py                   # InputMapper, RobustInputMapper, SparseInputMapper, FanOutInputMapper, OutputMapper
├── kirchhoff_net.py               # KirchhoffNet, KirchhoffNetWithIO
├── train.py                       # Loss, regularizers (4 decomposed CP terms), tau annealing, three-phase schedule,
│                                  #   solidification metrics, argmax validation, stage-LR scaling, training loop
├── train_script.py                # CLI training entry point (4 problems; prune/retrain; three-phase; AMP/compile/DP)
├── test_smoke.py                  # ~565-test smoke suite (116 test functions; P cell, gates, pruning, three-phase,
│                                  #   smooth2d/smooth2d_grid, fan-out write, MLP benchmark, gradient logging)
├── mlp_benchmark.py               # MLPRegressor baseline for smooth2d Franke task
├── visualize.py                   # Matplotlib/networkx visualization utilities
├── gen_network_images.py          # Generate network viz images for all presets
├── sparse_solver_data.py          # Random sparse SPD matrix + dataset generator (preserved, not active)
├── sparse_solver_topology.py      # Union-graph topology builder (preserved, not active)
├── sparse_solver_baseline.py      # Jacobi + CG digital solvers for comparison (preserved, not active)
├── sparse_solver_track.py         # Convergence diagnostic tracker (preserved, not active)
├── ARCHITECTURE.md                # This file
├── results/                       # Training run output directories (generated)
└── network_visualization/         # Generated PNGs from gen_network_images.py (if present)
```
