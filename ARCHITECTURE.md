# Reduced Differential KirchhoffNet — Architecture & Reference

> **Version:** Idealized (v2)  
> **Design target:** Tapeout-plausible analog compute fabric  
> **Key decisions:** Differential signaling, sparse topology only, cell-library-based edge parameterization, direct BPTT through Heun integration

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
| Branch law = `ReLU(θ₁·Δv + θ₂)` (impossible passive device) | Branch law = library of realizable transconductor surrogates (L/S/Z families) |
| Implicitly powered edges (no energy source) | Explicit rail-powered differential transconductor cells + node leak + rail clamps |
| Implied fully connected hardware | Sparse topologies only (NE/cluster/grid/ring/line) |
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
SI. `x_max` was raised from 0.3 to 1.0 to give the solver more dynamic
range and avoid saturation. `t_span=5` and `num_steps=50` are the
defaults (increased from 0.5/20 for longer integration). `C_eff` is a
pure scaling. No pretense is made that `g/C` is a real analog time
constant. The OLD `V_CM` value has been removed from `PHYS` since it is
not simulated.

### 2.2 Sparse topologies only

No fully connected layers. Supported graph primitives:
- `line_graph` — 1D chain with neighborhood radius
- `ring_graph` — 1D periodic chain
- `grid_graph` — 2D local grid with kernel neighborhood
- `cluster_graph` — Erdős-Rényi random sparse graph
- `empty_graph` — no edges (for ablation or projection-only stages)

### 2.3 Cell library instead of raw device parameters

Edges don't learn raw W/L or bias voltages. They soft-select from 3 cell families:

| Cell | Meaning | gm (norm.) | I_sat (norm.) | ρ (feedback) |
|------|---------|------------|---------------|--------------|
| **L** | Weak linear transconductor | 0.2 | 10.0 | 1.0 |
| **S** | Saturating transconductor | 1.0 | 0.5 | 1.0 |
| **Z** | Disabled / zero branch | 0.0 | 0.0 | 0.0 |

The edge current for cell family `q` is:

```
I_cell = I_sat · tanh((gm · u + bias) / I_sat) + g_leak · u
```

where `u = x_src − ρ · x_dst` and compliance gating multiplies by `σ((x_max − |x_src|)/β) · σ((x_max − |x_dst|)/β)`.

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
1. **Write:** `InputMapper` (dense) or `SparseInputMapper` (one-to-one) maps raw input `u` to bounded initial differential state `x_hidden(0)`. Hidden nodes NOT in `write_idx` are zero-initialized. Projection portion `x_proj(0) = 0`.
2. **Evolve:** Multi-stage ODE core integrates for fixed time horizons. The core evolves hidden + projection nodes together.
3. **Read:** `OutputMapper` linearly projects either the full state slice (sparse `read_idx`) or the projection portion (dense) to output `ŷ`.

**Sparse I/O mapping (sparse-io-mapping spec).** Each preset specifies `write_idx` and `read_idx` lists, and the default write/read modes are:
- `--write-mode one_to_one` (default): each input feature `u_i` writes to exactly one hidden node `h_{write_idx[i]}` via `SparseInputMapper`. Hidden nodes not in `write_idx` are zero-initialized at t=0. Parameter count: `2 * in_dim` (vs `in_dim * hid_count + hid_count` for dense `InputMapper`).
- `--read-mode sparse` (default): `OutputMapper` gathers only from `read_idx` full-state indices (a learnable linear of size `len(read_idx) -> out_dim`).
- `--write-mode dense` / `--read-mode dense`: original `InputMapper` / `OutputMapper` behavior for baseline comparison.

Preset defaults (SR4.2):
- `sinx`: `write_idx=[0]`, `read_idx=[7]` (out of 8 hidden + 2 proj = 10)
- `housing`: `write_idx=[0..7]`, `read_idx=[15]` (out of 16 hidden + 4 proj = 20)

CLI flags (SR5): `--write-mode {one_to_one,dense}`, `--read-mode {sparse,dense}`, `--write-idx "0,2,4"`, `--read-idx "7"`. The index overrides take precedence over preset values.

#### 2.7 Staged regularizer warm-up (RR-A)

By default, the five auxiliary regularizers (sparsity, edge_gate, node_gate,
power, capacitance) are **ramped in** over the first 150 epochs so the
network can learn the task first without penalty fighting:
- **Epochs 0–99 (`W=100`)**: all five are multiplied by 0.0 (free phase).
- **Epochs 100–149 (`W+A=150`)**: linear anneal from 0.0 → 1.0.
- **Epochs 150+**: full penalty value.

The schedule is controlled by ``reg_schedule(epoch)`` in ``train.py`` and
configurable via ``OPTIM["reg_warmup_epochs"]`` and
``OPTIM["reg_anneal_epochs"]`` in ``config.py``. **Note (fix-z-death)**:
``rail`` is intentionally NOT in ``_REG_KEYS``; it is a safety voltage
clamp on differential node states and is applied at full strength at every
epoch. The rail loss uses a **ReLU² quadratic barrier**
(``F.relu(|x| - x_max).pow(2).mean()``) which has zero loss and zero
gradient inside ``[-x_max, x_max]`` — unlike the previous ``softplus``
formulation which had a non-zero floor (≈0.313 at |x|=0) creating constant
gradient drag toward zero voltages. The entropy bonus retains its own
τ-dependent scaling.

### 2.8 Per-component complexity regularizers (complexity-pruning-v2)

The old single ``complexity`` key is **decomposed into 4 physically
motivated terms** for finer-grained control over the pruning process:

| Regularizer | What it penalizes | Default weight |
|-------------|-------------------|----------------|
| ``edge_gate`` | ``Σ σ(z_e)`` — open edge count | 5e-4 |
| ``node_gate`` | ``Σ σ(u_j)`` — open node count | 1e-4 |
| ``power`` | ``Σ_e σ(z_e) · m_e · Σ_q p(L|e,q) · gm_q`` — static power proxy | 1e-4 |
| ``capacitance`` | ``Σ_j σ(u_j)`` — node capacitance area proxy | 1e-5 |

The edge gate regularizer encourages inactive edges to close, the node
gate regularizer encourages inactive nodes to close, the power term uses
the cell library's gm values as a static power proxy, and the capacitance
term penalizes node count as a proxy for routing area.

In addition, **learnable gate parameters** are added to every
``DifferentialStage``:
- ``z_logits [E]`` — edge open-logit (init 5.0 → ``σ(5) ≈ 0.993``).
  Each ``i_edge`` is multiplied by ``σ(z_logits[e])`` in ``rhs()``,
  so a closed edge contributes zero current.
- ``u_logits [N]`` — node open-logit (init 5.0 → ``σ(5) ≈ 0.993``).
  The state ``x`` is multiplied elementwise by ``σ(u_logits)`` before
  voltage differences are computed, so a closed node is pinned to ~0.

Helper methods on ``DifferentialStage``:
- ``edge_gates()`` — returns ``σ(z_logits)``
- ``node_gates()`` — returns ``σ(u_logits)``
- ``active_edge_mask(threshold=0.01)`` — edge gates above threshold
- ``active_node_mask(threshold=0.01)`` — node gates above threshold
- ``parameter_breakdown()`` — returns dict with gate values

The gate regularizers (edge_gate, node_gate, power, capacitance) go
through the same staged warm-up as sparsity and rail (RR-A, free phase
for first 50 epochs).

Old single ``complexity`` key is removed from ``LAMBDAS``. Backward
compatibility with checkpoints is handled via ``dict.get()`` fallback
in ``compute_loss()`` and ``compute_solver_loss()``.

### 2.9 Complexity pruning pipeline (complexity-pruning-v2)

After training with gate parameters, the network can be structurally
pruned to remove low-value edges and nodes:

1. **Train** with per-component complexity regularizers (2.8).
   Gate parameters ``z_logits`` and ``u_logits`` are trained alongside
   the cell logits and multiplicities.
2. **Prune** edges with ``σ(z_logits) < threshold`` and nodes with
   ``σ(u_logits) < threshold`` (default ``edge_threshold=0.01``,
   ``node_threshold=0.01`` — configurable in ``PRUNE`` dict).
3. **Reconstruct** a compact ``DifferentialStage`` without gate parameters
   that only contains the surviving edges and nodes.
4. **Retrain** (optional, ``--retrain`` / ``--no-retrain``) the pruned
   network from scratch or with transferred parameters.

Key functions in ``topology.py``:
- ``prune_stage(stage, edge_threshold, node_threshold, transfer_params)``
  — returns a new ``DifferentialStage`` with only surviving edges/nodes.
  Raises ``ValueError`` if all edges are removed.
- ``prune_network(core, edge_threshold, node_threshold)`` — applies
  ``prune_stage`` to every stage in a ``KirchhoffNet`` core.

The ``train_script.py`` CLI exposes:
- ``--prune`` — enable pruning after training
- ``--retrain`` / ``--no-retrain`` — retrain pruned network (default: retrain)
- ``--prune-edge-threshold`` — override edge gate threshold
- ``--prune-node-threshold`` — override node gate threshold
- ``--retrain-epochs`` — retraining epochs (default: 200)

Pruned model, loss plot, and summary text are saved to a separate
output directory (``<output_dir>_pruned``).

### 2.10 Per-preset lambda overrides (RR-D)

Each preset may optionally contain a ``"lambdas"`` dict that is merged on
top of the global ``LAMBDAS``. This allows per-task tuning of a single
regularizer weight without redefining the entire dictionary.

Active overrides:
- **sinx**: ``{"rail": 1.0}`` (matches the new global default of 1.0,
  kept for backward compatibility with existing configs).
- **housing**: no override (uses global ``rail=1.0``).

### 2.10 Deprecated `temp_c` sampling (RR-C)

The ``temp_c`` field of ``SimContext`` is preserved for API compatibility
but **no longer used** by the analog model. ``sample_random_context``
always returns ``VARIATION["temp_c_default"]`` (27.0°C) and ignores the
``temp_choices`` argument. External callers that explicitly pass a
non-default ``temp_c`` value receive a ``DeprecationWarning``. The
``legacy_temp=True`` flag exists temporarily for code migration and also
emits a warning.

### 2.6 Variation-aware training (R6.3)

By DEFAULT, training uses a `SimContext()` with no mismatch and no
temperature drift, so the optimization sees a clean deterministic
forward. To evaluate robustness, pass `--variation` to `train_script.py`,
which then samples a fresh `SimContext` per training iteration:
- `temp_c` — junction temperature (randomly 0, 27, or 75°C)
- `global_gain_shift` — log-normal global gm drift (σ = 5%)
- `edge_mismatch` — per-edge per-cell log-normal mismatch (σ = 5%)

Mismatch is held fixed over the full transient but resampled each iteration. At validation time, `edge_mismatch=None` (nominal).

---

## 3. System Architecture

```
                          ┌──────────────────────────────────────────┐
                          │           KirchhoffNetWithIO              │
                          │                                          │
   raw input u ──────────►│  InputMapper                             │
                          │    xⱼ(0) = x_max · tanh(W·u + b)         │
                          │                                          │
                          │  KirchhoffNet (ODE core)                 │
                          │  ┌────────────────────────────────────┐  │
                          │  │  Stage 0  ──►  Transfer  ──► ...  │  │
                          │  │  (Heun integration, t=0.5, 20     │  │
                          │  │   steps)                           │  │
                          │  └────────────────────────────────────┘  │
                          │                                          │
                          │  OutputMapper                            │
   prediction ŷ ◄─────────│    ŷ = Linear(x_final)                   │
                          │                                          │
                          └──────────────────────────────────────────┘
```

### Inside one DifferentialStage

```
  x_src[j]  ──►  CellLibrary  ──►  i_edge[e]  ──►  scatter-add to dst
  x_dst[j]  ──►  (L/S/Z soft)      (μA)           subtract from src
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

- `CELL_LIBRARY` — L/S/Z cell parameters (gm, isat, rho, gleak, bias)
- `PHYS` — physical constants (x_max=1.0, C_eff=1.0)
- `OPTIM` — training hyperparameters (lr=1e-3, wd=1e-4, epochs=800, batch_size=2048, reg_warmup/anneal, CosineAnnealingWarmRestarts)
- `TAU` — temperature annealing schedule (1.0 → 0.1, hardening_epoch_frac=0.1)
- `LAMBDAS` — regularizer weights (sparsity=1e-3, rail=1.0, edge_gate=5e-4, node_gate=1e-4, power=1e-4, capacitance=1e-5, entropy=0.0; CP: decomposed from single complexity into 4 physically motivated terms; RR-D: per-preset overrides merge on top)
- `PRUNE` — pruning thresholds (edge_threshold=0.01, node_threshold=0.01)
- `SOLVER` — integration defaults (method=heun, t_span=5, num_steps=50)
- `INIT` — parameter initialization biases (fix-z-death: logits_z_bias=0.0 for equal P(Z)=P(L)=P(S)=P(P)=0.25 at init, was 1.0 which gave P(Z)≈0.42 and locked to Z via tau annealing; z_logit_init=2.0, u_logit_init=2.0 → σ≈0.88, dσ/dz≈0.10, was 0.0/1.0 which gave σ=0.5/0.73 starving gradient flow; raw_mult_init=0.0, raw_leak_init=-3.0, gain_scale=1.0)
- `VARIATION` — PVT/mismatch defaults (RR-C: temp_c sampling deprecated)
- `PRESETS` — task-specific topology configs (sinx, housing, smooth2d, smooth2d_grid; RR-D: optional lambdas overrides; fix-z-death: optional `tau_anneal` bool key — when False, the initial fitting phase keeps tau=1.0 throughout; retrain phase always uses tau annealing)

### `cell_library.py`
**IdealizedCellLibrary** — Frozen tanh-surrogate edge cell library.

- `forward(x_src, x_dst, logits, raw_mult, x_max, ctx, tau)` → `i_edge [batch, E]`
- Injects PVT mismatch multiplicatively on gm: `gm *= exp(edge_mismatch)`
- Compliance gate: turns off edges when |x_src| or |x_dst| approaches x_max

### `topology.py`
**Graph construction and topology management.**

Three-layer API:
1. **Primitives:** `line_graph()`, `ring_graph()`, `grid_graph()`, `cluster_graph()`, `empty_graph()`
2. **Connectors:** `connect_bipartite()`, `connect_projection()`
3. **Composer:** `StageTopologyBuilder`, `MultiStageTopology.from_config()`

Key data structures:
- `SparseTopology` — universal sparse graph with src/dst edge lists, node kinds, edge types
- `validate_topology()` — sanity checks (no self-loops, density limits, input/output connectivity)
- `topology_to_stage()` — filters I/O edges, remaps node IDs, builds DifferentialStage
- `prune_stage(stage, edge_threshold, node_threshold, transfer_params)` — structural pruning: removes edges/nodes with gate values below threshold, returns compact DifferentialStage without gate params (CP: complexity-pruning-v2)
- `prune_network(core, ...)` — applies prune_stage to all stages in a KirchhoffNet core
- `build_net_from_preset()` / `build_net_from_config()` — factory functions

### `differential_stage.py`
**DifferentialStage** — A single ODE stage with sparse COO graph + Heun integration.

Per-node dynamics:
```
C_eff · dxⱼ/dt = Σ_{e: dst(e)=j} I_e − Σ_{e: src(e)=j} I_e − leakⱼ · xⱼ − clip(xⱼ)
```

- `rhs(x, ctx, tau)` — computes dx/dt at current state (applies node gates `σ(u_logits)` to x before computing voltages, and edge gates `σ(z_logits)` to i_edge after cell library evaluation)
- `forward(x0, ctx, t_span, num_steps, tau, store_trajectory)` — Heun integration, returns `(x_final, [batch, N, steps+1] trajectory)`
- Parameters: `logits`, `raw_mult`, `raw_leak`, `z_logits`, `u_logits` (CP: edge/node gate logits)
- Buffers: `src`, `dst` (COO format edge lists)
- Helper methods: `edge_gates()`, `node_gates()`, `active_edge_mask()`, `active_node_mask()`, `parameter_breakdown()`

### `sim_context.py`
**SimContext** — PVT + mismatch container for one forward pass.

- `SimContext(temp_c, global_gain_shift, edge_mismatch)` — dataclass with `.to(device)` method
- `sample_random_context(num_edges, num_cells, ...)` — factory for training variation injection

### `stage_transfer.py`
**StageTransfer** — Width-changing layer between stages. No learnable parameters.

- If `out_nodes < in_nodes`: truncates
- If `out_nodes > in_nodes`: zero-pads
- If equal: identity pass-through

### `io_mapper.py`
**Input/Output mappers** — Write and read phases.

- `InputMapper(in_dim, out_dim)` — `x_hidden(0) = x_max · tanh(Linear(u))`, Xavier init scaled by 0.1. Writes only to the hidden portion of the differential state. Used in `--write-mode dense`.
- `RobustInputMapper(in_dim, out_dim)` — adds per-feature learnable log-scale preconditioner (for heterogeneous features like California housing). Used in `--write-mode dense` when `use_robust_input=True`.
- `SparseInputMapper(in_dim, out_dim, write_idx)` — one-to-one writer. Each input feature `u_i` writes to `h_{write_idx[i]}` via an independent `(gain_i, bias_i)` pair; non-write hidden positions are zero. Parameter count = `2 * in_dim`. Used in `--write-mode one_to_one` (default). Raises `ValueError` if `in_dim > out_dim` (SR1.6).
- `OutputMapper(node_dim, out_dim, read_idx=None)` — `ŷ = Linear(x_read)`, no activation. With `read_idx=None` (dense), reads from the projection portion of the final state. With `read_idx` (sparse, default), gathers from the specified full-state indices and projects via a learnable linear of size `len(read_idx) -> out_dim`.

### `kirchhoff_net.py`
**KirchhoffNet** and **KirchhoffNetWithIO** — Top-level network classes.

- `KirchhoffNet(stages, transfers, stage_times, stage_steps)` — multi-stage ODE core
  - Handles per-stage edge_mismatch slicing internally
  - `parameter_breakdown()` for diagnostics
- `KirchhoffNetWithIO(input_mapper, core, output_mapper, hid_count, proj_count, final_hid_count, final_proj_count, write_idx=None, read_idx=None)` — write/evolve/read pipeline (R1, sparse-io-mapping)
  - `hid_count` / `proj_count` enforce the honest I/O split: the InputMapper writes only to the hidden portion, projection portion is zero-initialized.
  - `final_hid_count` / `final_proj_count` define the final-stage read_slice: OutputMapper reads only from the final projection portion (or hidden if no projections; emits a warning).
  - `write_idx` (sparse mode): list of hidden-node indices. The InputMapper returns a `hid_count`-sized vector with zeros at non-write positions (non-write hidden nodes stay at 0 at t=0; SR1.3).
  - `read_idx` (sparse mode): list of full-state indices. The OutputMapper gathers from these positions and projects via `Linear(len(read_idx), out_dim)`.
  - `forward(u, ctx, tau, store_trajectory)` → `(ŷ, trajectories)`

### `train.py`
**Loss functions, regularizers, and training infrastructure.**

- `compute_loss(net, u, target, ctx, task_fn, ...)` — total = task + reg_scale·(sparsity + edge_gate + node_gate + power + capacitance) + rail − entropy_bonus (CP: decomposed from single complexity into 4 terms; RR-A: reg_scale from reg_schedule; fix-z-death: rail is NOT in reg_scale — always full strength)
- `compute_solver_loss(net, b, x_star, A, ctx, ...)` — solver-specific: residual + 0.1·solution + regularizers (preserved on disk; not active in paper v1; also uses reg_scale)
- `residual_loss(x_pred, b, A)` — ‖Ax − b‖²
- `solution_loss(x_pred, x_star)` — ‖x − x*‖²
- `tau_for_epoch(epoch, total_epochs)` — monotonic exponential decay, hardens at 90% (R6.1)
- `apply_ablation(net, ablation)` — in-place structural ablation: 'none', 'mapper-only', 'empty-graph' (R2.1-R2.4)
- `train_epoch(net, loader, optimizer, task_fn, ctx_factory, epoch, ...)` — single-epoch loop

**Regularizer details:**

| Regularizer | What it penalizes | Weight |
|-------------|-------------------|-------|
| Sparsity | `Σ w[:, :Z_index]` (active non-Z cells) | 1e-3 |
| Rail | `mean(softplus(|x| − x_max))` over trajectory | 10.0 |
| Edge gate (CP) | `Σ σ(z_e)` — open edge count | 1e-3 |
| Node gate (CP) | `Σ σ(u_j)` — open node count | 1e-4 |
| Power (CP) | `Σ_e σ(z_e) · m_e · Σ_q p(L\|e,q) · gm_q` — static power proxy | 1e-4 |
| Capacitance (CP) | `Σ_j σ(u_j)` — node capacitance area proxy | 1e-5 |
| Entropy bonus | `−Σ w·log(w)` of logits/tau (off by default, R6.5) | 0.0·tau |

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
**Standalone image generator** — Runs all 3 non-solver presets and saves visualizations to `network_visualization/`.

### `train_script.py`
**Main training entry point** — CLI script supporting `--problem {sinx,housing,smooth2d,smooth2d_grid}`.

Outputs per run: `loss_history.txt`, `loss_curve.png`, `model.pt`, `config_snapshot.txt`, per-stage graph/selection/trajectory plots, output fit, pipeline diagram.

---

## 5. Data Flow

### Training step (one batch)

```
1. Sample SimContext (PVT + mismatch)
       │
2. InputMapper:  u [B, in_dim]  →  x0 [B, N_active₀]  (write phase)
       │
3. Stage 0 Heun (50 steps @ t_span=5, configurable via SOLVER):
        │    for step in 1..num_steps:
       │        x_src = x[:, src]; x_dst = x[:, dst]
       │        i_edge = CellLibrary(x_src, x_dst, logits, raw_mult, x_max, ctx, tau)
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
8. Loss = task_loss + reg_scale · Σ λ·regularizer  →  backward through steps
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

Schedule: monotonic exponential decay (R6.1, no sinusoidal noise), then hardens to τ_final=0.1 at 90% of training:
```
τ_base = τ_init · exp(−epoch / (total_epochs · 0.5))
τ = max(τ_min, τ_base)
τ = τ_final if epoch > 90% of total_epochs
```

### Initialization

- Edge logits: all zeros, except Z_INDEX (disabled) column = +1.0 → softmax ≈ [0.11, 0.11, 0.73] (RR-A: lowered from +2.0 to give more active edges at start)
- `raw_mult`: zeros → multiplicity = softplus(0) = ln(2) ≈ 0.69 (R3.3)
- `raw_leak`: −3.0 → leak = softplus(−3) ≈ 0.05
- `z_logits`: +5.0 → σ(5) ≈ 0.993 (all edges open at init, CP)
- `u_logits`: +5.0 → σ(5) ≈ 0.993 (all nodes open at init, CP)
- InputMapper: Xavier uniform scaled by 0.1

### Gradient handling

- Clip gradients to norm 5.0
- AdamW optimizer with weight decay 1e-4 (lr=1e-3, up from 3e-4)
- CosineAnnealingWarmRestarts (T_0=50, T_mult=1, eta_min=1e-5)
- SimContext is no_grad (variation doesn't get gradients)

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
default (one additional param group). Controlled via the `--stage-lr-scale`
CLI flag (default `1.0`, which collapses to a single standard param group
for full backward compatibility).

The flag is applied to both initial training and retrain optimizer creation.
**Retrain uses a separate flag ``--retrain-stage-lr-scale`` (default ``1.0``)
to avoid over-aggressive updates during warm-start fine-tuning.** A pruned
network is nearly converged, and a 100× LR boost on stage 0 would destroy
the learned features. Set ``--retrain-stage-lr-scale`` to match
``--stage-lr-scale`` when geometric LR scaling is also desired during
retrain.

When `stage_lr_scale > 1.0`, the tqdm postfix shows the LR range
(`min..max`) and `log_gradient_norms` appends per-group LR columns (`lr0`,
`lr1`, ...).  The same applies to retrain when
``--retrain-stage-lr-scale > 1.0``.

---

## 7. Task Presets

### `sinx` — Sine function regression
| | |
|---|---|
| Input | 1D (angle in [−π, π]) |
| Output | 1D (sin(angle)) |
| Architecture | 1 stage: 8 hidden + 2 proj, **ring** topology (radius=2) (R4.1) |
| Loss | MSE |
| Train size | 8192 |
| Difficulty | Easy — warm-up / sanity check |

### `housing` — California housing price regression (appendix-only)
| | |
|---|---|
| Input | 8D normalized features |
| Output | 1D (price, standardized) |
| Architecture | 1 stage: 16 hidden + 4 proj, line topology (radius=2) |
| Loss | MAE |
| Train size | ~16.5K |
| Special | Uses RobustInputMapper (per-feature log-scale preconditioner) |

### `smooth2d` — Franke function 2D regression (line topology)
| | |
|---|---|
| Input | 2D (x, y) in [0, 1]² |
| Output | 1D (Franke function value, zero-mean unit-variance normalized) |
| Architecture | 1 stage: 10 hidden + 2 proj, **line** topology (radius=2) |
| Loss | MSE |
| Train size | 20K (4K val, 4K test, sigma=0.01 noise) |
| Special | Sparse I/O: `write_idx=[0,1]`, `read_idx=[9]` (hidden readout). 1-batch loss ~1.0. |
| Variation | No lambda overrides |

### `smooth2d_grid` — Franke 2D regression (grid topology)
| | |
|---|---|
| Input | 2D (x, y) in [0, 1]² |
| Output | 1D (Franke function value, normalized) |
| Architecture | 1 stage: 25 hidden (5×5 grid, 8-neighbor kernel_size=3) + 3 proj (all_to_all) |
| Loss | MSE |
| Train size | 20K (4K val, 4K test, sigma=0.01 noise) |
| Special | Sparse I/O: `write_idx=[0,1]`, `read_idx=[25,26,27]` (proj readout). 438 edges, 28 nodes, ~1294 learnable params. 1-batch loss ~0.99. |
| Variation | No lambda overrides. Degree validation skipped when all read_idx are proj nodes. |

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
# Cell library (normalized units, dimensionless for rho)
CELL_L = {"gm": 0.2, "isat": 10.0, "rho": 1.0, "gleak": 0.01, "bias": 0.0}
CELL_S = {"gm": 1.0, "isat": 0.5,  "rho": 1.0, "gleak": 0.01, "bias": 0.0}
CELL_Z = {"gm": 0.0, "isat": 0.0,  "rho": 0.0, "gleak": 0.0,  "bias": 0.0}

# Normalized physical limits (R7: not SI-calibrated; V_CM removed)
PHYS = {"x_max": 1.0, "C_eff": 1.0,
        "beta_softness": 0.02, "clip_current": 0.05, "clip_softness": 0.02}

# Training (RR-A: reg_warmup_epochs/reg_anneal_epochs for staged warm-up)
OPTIM = {"lr": 1e-3, "weight_decay": 1e-4, "grad_clip_norm": 5.0,
         "epochs": 800, "batch_size": 2048,
         "reg_warmup_epochs": 100, "reg_anneal_epochs": 50,
         "scheduler_T_0": 50, "scheduler_T_mult": 1, "scheduler_eta_min": 1e-5}

# Regularization weights (CP: decomposed into 4 per-component terms;
#                          R3.3: multiplicity m=softplus(raw_mult);
#                          entropy off by default, R6.5;
#                          RR-A: staged warm-up via reg_schedule;
#                          RR-D: per-preset overrides merge on top)
LAMBDAS = {"sparsity": 1e-3, "rail": 1.0,
           "edge_gate": 5e-4, "node_gate": 1e-4,
           "power": 1e-4, "capacitance": 1e-5,
           "entropy": 0.0}

# Integration defaults (increased from t_span=0.5/num_steps=20 for longer dynamics)
SOLVER = {"method": "heun", "t_span": 5, "num_steps": 50}

# Variation injection (R6.3: off by default at training time)
VARIATION = {"temp_c_default": 27.0, "temp_c_choices": [0.0, 27.0, 75.0],
             "global_gain_shift_std": 0.05, "edge_mismatch_std": 0.05}
```

---

## 9. Sparse Solver Subsystem (preserved, not in active PRESETS)

The sparse solver benchmark is preserved on disk but **not active in
paper v1** (R4.3, 2.10). Modules:

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

388 tests covering the full pipeline including R1-R7, RR-A through
RR-D reviewer-residual cleanup, CP-1 through CP-5 complexity pruning,
smooth2d and smooth2d_grid presets, and MLP benchmark comparison.
Run with:
```bash
~/Documents/ASPDAC_2026/venv/bin/python kirchhoff_redesign/ideal/test_smoke.py
```

Coverage:
| # | Test | What it verifies |
|---|------|-----------------|
| 1 | Config loads | CELL_LIBRARY, CELL_ORDER, PHYS, OPTIM, LAMBDAS, PRESETS (incl. smooth2d, smooth2d_grid) |
| 2 | SimContext | Default values, sampled mismatch shape/finiteness |
| 3 | Topology primitives | line, ring, grid, cluster, empty — edge counts, symmetry, no self-loops |
| 4 | StageTransfer | Equal width, truncation, zero-padding |
| 5 | Heun convergence | No NaN/explosion on zero input, trajectory shape correct |
| 6 | Gradient flow | Gradients reach logits, raw_mult, raw_leak |
| 7 | Loss finite | All regularizer components are finite and ≥0 |
| 8 | Sparsity push | Training reduces P(L or S) and increases P(Z) |
| 9 | Tau annealing | Schedule values at epoch 0, mid, end |
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
| 28 | Mapper-only ablation | R2.2: t_span=0 makes output = OutputMapper(InputMapper(u)) |
| 29 | Weighted power/area | R3.1/R3.3/R3.4: weighted proxy goes to 0 for Z edges, softplus(0)=log(2) |
| 30 | Active presets single-stage | R4/R5: all active presets are 1-stage with num_proj > 0 |
| 31 | sinx uses ring | R4.1/2.8: sinx preset uses ring topology, not cluster |
| 32 | Tau monotonic | R6.1: tau is non-increasing |
| 33 | CLI flags | R2/R6.3: --ablation and --variation flags exist in train_script.py |
| 34 | Normalized units | R7: V_CM removed, LAMBDAS['C'] removed, "normalized" present in config.py |
| 35 | apply_ablation | R2: all three ablation modes work |
| 36 | Sparse I/O preset defaults | SR4.2: write_idx/read_idx present in sinx and housing presets; default build uses SparseInputMapper |
| 37 | Sparse write zeros non-targets | SR1.3: SparseInputMapper leaves non-write positions at 0 |
| 38 | Sparse write validation | SR1.6: in_dim > out_dim, out-of-range, and duplicate write_idx all raise ValueError |
| 39 | Sparse read select | SR2.1: OutputMapper(read_idx=...) gathers only specified positions; projection weight shape = (out_dim, len(read_idx)) |
| 40 | Dense fallback | SR3/SR6: --write-mode dense / --read-mode dense produces original InputMapper/OutputMapper behavior |
| 41 | Sparse I/O gradients | SR1/SR2: gradients flow to SparseInputMapper gain and OutputMapper projection |
| 42 | Sparse I/O CLI flags | SR5: --write-mode, --read-mode, --write-idx, --read-idx exposed in train_script.py |
| 43 | Complexity proxy | RR-B/R3: merged power+area with (1-p_Z) weighting, softplus multiplicity |
| 44 | Reg schedule curve | RR-A: piecewise linear warm-up (free → anneal → full) |
| 45 | Preset lambda overrides | RR-D: per-preset lambdas merge correctly over global LAMBDAS |
| 46 | Z-bias eliminated | fix-z-death: INIT['logits_z_bias'] == 0.0, P(Z)=P(L)=P(S)=P(P)=0.25 at init (was 1.0 with P(Z)≈0.42) |
| 47 | Gate initialization | fix-z-death: z_logit_init=2.0, u_logit_init=2.0 → σ≈0.88, dσ/dz≈0.10 (was 0.0/1.0 with σ=0.5/0.73 starving gradients) |
| 48 | Gates applied in rhs | CP-2: closed vs open gates produce different RHS output |
| 49 | Per-component regularizers | CP-4: edge_gate, node_gate, power, capacitance all finite, go to 0 when gates closed |
| 50 | prune_stage | CP-5: removes low-gate edges/nodes, returns DifferentialStage without gating |
| 51 | Parameter transfer | CP-5: surviving parameters (logits, raw_mult, raw_leak) preserved after prune |
| 52 | All-removed raises | CP-5: prune_stage raises ValueError when all edges removed |
| 53 | prune_network | CP-5: applies to KirchhoffNet core, preserves stage_times/stage_steps |
| 54-88 | smooth2d preset | 35 checks: topology, Franke dataset, forward shape, 1-batch loss, Sparse I/O, read_idx from hidden |
| 89-123 | smooth2d_grid preset | 35 checks: grid topology (5×5, kernel=3), 438 edges, forward shape, proj readout, 1-batch loss |
| 124 | MLP benchmark | MLPRegressor(2→H→1) on Franke, verifies val loss decreases after 5-step training |
| 125-130 | fix-z-death: Z-bias init | 6 checks: logits_z_bias=0.0, P(Z)=0.25 at init, equal cell probability |
| 131-135 | fix-z-death: gate init | 5 checks: z_logit=2.0, u_logit=2.0, σ≈0.88, dσ/dz≈0.10 |
| 136-138 | fix-z-death: rail excluded | 3 checks: _REG_KEYS no longer contains 'rail'; apply_reg_schedule leaves rail untouched |
| 139 | stage-lr-scale backward compat | SLS-1: scale=1.0 produces single param group (no regression) |
| 140 | stage-lr-scale multi-group | SLS-2: scale=10.0 with 3 stages creates 4 groups (3 stage + other) with correct geometric LR ratios |
| 141 | stage-lr-scale scheduler compat | SLS-3: CosineAnnealingLR reduces all groups proportionally (ratios preserved) |
| 142 | ReLU² rail loss | RL-1: rail loss is exactly 0 when all |x| < x_max (was 0.313 floor with softplus) |
| 143 | Rail loss positive outside bounds | RL-2: rail loss > 0 when |x| > x_max |
| 144 | Retrain LR guard | RL-3: --retrain-stage-lr-scale defaults to 1.0 (single group, safe for warm-start) |

**Known test failures (5):**
These checks reference config values from an earlier version and are
outdated vs current config.py. All are cosmetic (the test expectations
haven't caught up with config changes):
- `PHYS x_max=0.3` (current: 1.0)
- `LAMBDAS rail=10.0` (current: 1.0)
- `LAMBDAS edge_gate=1e-3` (current: 5e-4)
- `smooth2d t_span=0.5` (current: 5.0 — via SOLVER default)
- `smooth2d num_steps=20` (current: 50 — via SOLVER default)

---

## 11. File Map

```
kirchhoff_redesign/ideal/
├── __init__.py                    # Package docstring
├── config.py                      # All tunable constants + presets
├── cell_library.py                # IdealizedCellLibrary (L/S/Z tanh surrogates)
├── topology.py                    # Graph primitives, builder, stage conversion
├── differential_stage.py          # DifferentialStage (COO graph + Heun ODE)
├── sim_context.py                 # SimContext (PVT + mismatch dataclass)
├── stage_transfer.py              # StageTransfer (truncation/zero-padding)
├── io_mapper.py                   # InputMapper, RobustInputMapper, OutputMapper
├── kirchhoff_net.py               # KirchhoffNet, KirchhoffNetWithIO
├── train.py                       # Loss, regularizers (CP: 4 decomposed terms), tau annealing, training loop, RR-A: staged warm-up
├── train_script.py                # CLI training entry point (4 problems: sinx, housing, smooth2d, smooth2d_grid; CP: --prune/--retrain/--no-retrain flags)
├── test_smoke.py                  # 466-test smoke suite (CP: gates + pruning; smooth2d/smooth2d_grid presets; MLP benchmark; stage-lr-scaling; rail-loss-fix)
├── mlp_benchmark.py               # MLPRegressor baseline for smooth2d Franke task
├── visualize.py                   # Matplotlib/networkx visualization utilities
├── gen_network_images.py          # Generate network viz images for all presets
├── sparse_solver_data.py          # Random sparse SPD matrix + dataset generator (preserved, not active)
├── sparse_solver_topology.py      # Union-graph topology builder (preserved, not active)
├── sparse_solver_baseline.py      # Jacobi + CG digital solvers for comparison (preserved, not active)
├── sparse_solver_track.py         # Convergence diagnostic tracker (preserved, not active)
├── ARCHITECTURE.md                # This file
└── network_visualization/         # Generated PNGs from gen_network_images.py (if present)
```
