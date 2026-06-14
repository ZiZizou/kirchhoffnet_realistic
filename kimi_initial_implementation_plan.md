Here is a concrete implementation plan. It follows the reviewer’s recommended reduced differential model, uses **direct backpropagation through time (BPTT) with fixed-step Heun integration** (not adjoint), and is structured so that the idealized tanh cells can be swapped later for your SPICE-fitted surrogates without touching the topology or training loop.

---

## 1. Solver Choice: Direct BPTT through Fixed-Step Heun

**Do not use `torchdiffeq` or the continuous adjoint.** For this physics, direct BPTT is more reliable.

```text
for each batch:
    x = x0
    for step in 0..N-1:
        k1 = rhs(x)
        k2 = rhs(x + dt*k1)
        x = x + 0.5*dt*(k1 + k2)   # Heun
    loss = task_loss(x) + regularizers(x_trajectory)
    loss.backward()               # autograd unrolls automatically
```

**Why:** The system may be stiff, you will later add trajectory regularizers (rail, power), and you may inject variation/mismatch. The adjoint method is unstable for stiff, constrained dynamics and does not cleanly backprop through regularizers that depend on the full state path.

---

## 2. Target Model (Reduced Differential)

Each logical node $j$ is a differential pair $(v_j^+, v_j^-)$. We train only the differential state:

$$x_j = v_j^+ - v_j^- \quad (\text{common-mode } c_j \approx V_{CM} \text{ enforced by design})$$

Per-node dynamics:

$$C_x \dot{x}_j = \sum_{\text{in}} I_e^\Delta - \sum_{\text{out}} I_e^\Delta - g_j x_j - I_j^{\text{clip}}(x_j)$$

Edge current (idealized surrogate):

$$I_e = m_e \sum_{q \in \{L,S,Z\}} p_{e,q} \cdot \underbrace{\Big[ I_{q,\text{sat}} \tanh\!\Big(\frac{g_{m,q}(x_s - \rho_q x_d) + b_q}{I_{q,\text{sat}}}\Big) + G_{q,\text{leak}}(x_s - \rho_q x_d) \Big]}_{\text{cell } q} \cdot \beta(x_s, x_d)$$

with soft-max library weights $p_{e,q} = \text{softmax}(\alpha_{e,q}/\tau)$ and multiplicity $m_e = 1 + \text{softplus}(\mu_e)$.

---

## 3. Recommended Code Structure

Build **five separate modules**. Do not write one monolithic class.

| Module | Responsibility |
|--------|----------------|
| `SimContext` | Holds PVT + mismatch tensors for a forward pass |
| `IdealizedCellLibrary` | Computes $I_e$ from $(x_s, x_d)$, logits, and context |
| `DifferentialStage` | Sparse graph stage: KCL scatter-add + leak + clip + Heun |
| `StageTransfer` | Padding/truncation between stage widths |
| `KirchhoffNet` | Sequences stages and transfers |

---

## 4. Step-by-Step Implementation

### Step 1: Define the Variation Context

Even for idealized validation, wire the context in now so the training loop does not change later.

```python
from dataclasses import dataclass
import torch

@dataclass
class SimContext:
    temp_c: float = 27.0
    global_gain_shift: float = 0.0          # e.g. ±5% global gm drift
    edge_mismatch: torch.Tensor | None = None  # [num_edges, num_cells]
```

**Usage:** Sample a new `ctx` every training iteration. For idealized validation, `edge_mismatch` can be `None` or small Gaussian noise.

---

### Step 2: Build the Idealized Cell Library

This is the only module that knows about L/S/Z physics. Start with a **differentiable tanh surrogate**. When your SPICE data arrives, replace the `forward` body with a lookup or fitted MLP, but keep the same signature.

**Recommended idealized parameters (Q=3):**

| Family | $g_m$ | $I_{\text{sat}}$ | $\rho$ | $G_{\text{leak}}$ | $b$ |
|--------|-------|------------------|--------|-------------------|-----|
| **L** (weak linear) | 0.2 | 10.0 | 1.0 | 0.01 | 0.0 |
| **S** (saturating) | 1.0 | 0.5 | 1.0 | 0.01 | 0.0 |
| **Z** (disabled) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

High $I_{\text{sat}}$ for L makes it linear in the operating range; low $I_{\text{sat}}$ for S makes it saturate quickly.

```python
import torch.nn as nn
import torch.nn.functional as F

class IdealizedCellLibrary(nn.Module):
    def __init__(self, gm, isat, rho, gleak, bias):
        super().__init__()
        self.register_buffer("gm", torch.tensor(gm, dtype=torch.float32))
        self.register_buffer("isat", torch.tensor(isat, dtype=torch.float32))
        self.register_buffer("rho", torch.tensor(rho, dtype=torch.float32))
        self.register_buffer("gleak", torch.tensor(gleak, dtype=torch.float32))
        self.register_buffer("bias", torch.tensor(bias, dtype=torch.float32))

    @property
    def num_cells(self):
        return int(self.gm.numel())

    def forward(self, x_src, x_dst, logits, raw_mult, x_max, ctx,
                tau=1.0, beta_softness=0.02):
        # x_src, x_dst: [batch, num_edges]
        # logits: [num_edges, Q]
        # raw_mult: [num_edges]
        batch, num_edges = x_src.shape
        Q = self.num_cells

        # Soft library selection
        weights = F.softmax(logits / tau, dim=-1)          # [E, Q]
        mult = 1.0 + F.softplus(raw_mult)                    # [E]

        # Differential drive variable
        u = x_src.unsqueeze(-1) - self.rho * x_dst.unsqueeze(-1)  # [B, E, Q]

        # Apply PVT/mismatch
        gm = self.gm
        if ctx.edge_mismatch is not None:
            gm = gm * torch.exp(ctx.edge_mismatch)           # [E, Q]
        gm = gm * torch.exp(torch.tensor(ctx.global_gain_shift))

        # Core surrogate
        i_q = self.isat * torch.tanh((gm * u + self.bias) / self.isat.clamp_min(1e-6))
        i_q = i_q + self.gleak * u

        # Compliance gating (turn off near rails)
        gate = torch.sigmoid((x_max - x_src.abs()) / beta_softness).unsqueeze(-1)
        gate = gate * torch.sigmoid((x_max - x_dst.abs()) / beta_softness).unsqueeze(-1)

        # Mix and scale
        i_edge = mult.unsqueeze(0).unsqueeze(-1) * weights.unsqueeze(0) * gate * i_q
        return i_edge.sum(dim=-1)   # [batch, num_edges]
```

**Important:** Keep `x_max` small (e.g., 0.2–0.3 V differential) so the reduced model stays valid.

---

### Step 3: Implement a Sparse Differential Stage

Use **COO (source, destination) lists** and `index_add_` for KCL. This avoids $O(N^2)$ memory.

```python
class DifferentialStage(nn.Module):
    def __init__(self, num_nodes, src, dst, cell_lib,
                 c_eff=1.0, x_max=0.3):
        super().__init__()
        self.num_nodes = num_nodes
        self.cell_lib = cell_lib
        self.c_eff = c_eff
        self.x_max = x_max

        self.register_buffer("src", torch.tensor(src, dtype=torch.long))
        self.register_buffer("dst", torch.tensor(dst, dtype=torch.long))

        E = len(src)
        Q = cell_lib.num_cells

        # Initialize sparsity: bias logits toward Z (assume Z is index 2)
        self.logits = nn.Parameter(torch.zeros(E, Q))
        with torch.no_grad():
            self.logits[:, 2] = 2.0   # start mostly disabled

        self.raw_mult = nn.Parameter(torch.full((E,), -2.0))  # start near 1.0
        self.raw_leak = nn.Parameter(torch.full((num_nodes,), -3.0))

        self.clip_current = 0.05
        self.clip_softness = 0.02

    def rhs(self, x, ctx, tau=1.0):
        # x: [batch, nodes]
        x_s = x[:, self.src]
        x_d = x[:, self.dst]

        i_edge = self.cell_lib(x_s, x_d, self.logits, self.raw_mult,
                               self.x_max, ctx, tau)

        # KCL: accumulate currents at each node
        acc = torch.zeros_like(x)
        acc.index_add_(1, self.dst, i_edge)      # current entering dst
        acc.index_add_(1, self.src, -i_edge)     # current leaving src

        # Weak leak (stabilization to 0 differential)
        leak = F.softplus(self.raw_leak).unsqueeze(0) * x

        # Soft rail clip in differential coordinates
        clip = torch.sigmoid((x - self.x_max) / self.clip_softness)
        clip = clip - torch.sigmoid((-x - self.x_max) / self.clip_softness)
        clip = self.clip_current * clip

        return (acc - leak - clip) / self.c_eff

    def forward(self, x0, ctx, t_span, num_steps, tau=1.0):
        dt = t_span / float(num_steps)
        x = x0
        # Optional: store trajectory for regularizers
        traj = [x]
        for _ in range(num_steps):
            k1 = self.rhs(x, ctx, tau)
            k2 = self.rhs(x + dt * k1, ctx, tau)
            x = x + 0.5 * dt * (k1 + k2)
            traj.append(x)
        return x, torch.stack(traj, dim=2)   # [batch, nodes, steps+1]
```

**Key detail:** `index_add_(dim, index, source)` is the scatter-add you need for unstructured sparse graphs. It is differentiable and works on GPU.

---

### Step 4: Stage Transfer (Width Changing)

The reviewer recommends simple truncation or zero-padding. Do not learn a linear projection here for the first validation.

```python
class StageTransfer(nn.Module):
    def __init__(self, in_nodes, out_nodes):
        super().__init__()
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes

    def forward(self, x):
        if self.out_nodes == self.in_nodes:
            return x
        if self.out_nodes < self.in_nodes:
            return x[:, :self.out_nodes]
        pad = torch.zeros(x.size(0), self.out_nodes - self.in_nodes,
                          device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=1)
```

---

### Step 5: Assemble the Multi-Stage Network

```python
class KirchhoffNet(nn.Module):
    def __init__(self, stages, transfers, stage_times, stage_steps):
        super().__init__()
        assert len(stages) == len(stage_times) == len(stage_steps)
        assert len(transfers) == max(0, len(stages) - 1)

        self.stages = nn.ModuleList(stages)
        self.transfers = nn.ModuleList(transfers)
        self.stage_times = stage_times
        self.stage_steps = stage_steps

    def forward(self, x0, ctx, tau=1.0):
        x = x0
        all_trajs = []
        for i, stage in enumerate(self.stages):
            x, traj = stage(x, ctx, self.stage_times[i],
                            self.stage_steps[i], tau)
            all_trajs.append(traj)
            if i < len(self.transfers):
                x = self.transfers[i](x)
        return x, all_trajs
```

---

### Step 6: Topology Generators (Sparse Only)

Do not use fully connected. For validation, generate **Neighbor-Emphasizing (NE)** and **Projection (Proj)** topologies.

```python
def build_ne_topology(n, k=2):
    """Local bidirectional edges within ±k neighbors."""
    src, dst = [], []
    for i in range(n):
        for j in range(max(0, i-k), min(n, i+k+1)):
            if i != j:
                src.append(i); dst.append(j)
    return src, dst

def build_proj_topology(n_nodes, n_proj, offset=0):
    """Bipartite edges between nodes and projection bank."""
    src, dst = [], []
    proj_base = offset + n_nodes
    for i in range(offset, offset + n_nodes):
        for p in range(proj_base, proj_base + n_proj):
            src.append(i); dst.append(p)
            src.append(p); dst.append(i)
    return src, dst
```

**Example:** Stage 1 has 16 logical nodes + 4 projection nodes. Build NE on the 16, then Proj between the 16 and the 4. Total logical nodes = 20.

---

### Step 7: Loss Function with Regularizers

You need the trajectory to penalize rail excursions and common-mode drift (here, differential drift beyond bounds).

```python
def compute_loss(net, x0, target, ctx, task_fn, lambdas, tau=1.0):
    out, trajs = net(x0, ctx, tau)
    loss_task = task_fn(out, target)

    loss_power = 0.0
    loss_sparsity = 0.0
    loss_area = 0.0
    loss_rail = 0.0

    for stage, traj in zip(net.stages, trajs):
        # Soft weights for this stage
        w = F.softmax(stage.logits / tau, dim=-1)   # [E, Q]

        # Power proxy: assume active cells (L,S) burn bias proportional to isat
        # Here we just use multiplicity as a proxy for total branch strength
        mult = 1.0 + F.softplus(stage.raw_mult)
        loss_power += mult.sum()

        # Area proxy
        loss_area += mult.sum()

        # Sparsity: sum of probabilities of choosing non-Z cells
        # Z is index 2
        p_active = w[:, :2].sum()
        loss_sparsity += p_active

        # Rail penalty: integrate softplus(|x| - x_max) over time
        x_max = stage.x_max
        rail = F.softplus(traj.abs() - x_max).mean()   # mean over (batch,node,time)
        loss_rail += rail

    total = (loss_task
             + lambdas['power'] * loss_power
             + lambdas['area'] * loss_area
             + lambdas['sparsity'] * loss_sparsity
             + lambdas['rail'] * loss_rail)
    return total, out
```

**Regularizer weights (start here):**
- `lambdas['sparsity'] = 1e-3` to `1e-2` (encourage Z selection)
- `lambdas['rail'] = 10.0` (hard constraint)
- `lambdas['power'] = 1e-4`, `lambdas['area'] = 1e-4`

---

### Step 8: Training Loop with Variation Injection

```python
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

for epoch in range(epochs):
    for x0, target in train_loader:
        # Sample a random context per batch (or per sample)
        ctx = SimContext(
            temp_c=random.choice([0.0, 27.0, 75.0]),
            global_gain_shift=torch.randn(1).item() * 0.05,
            edge_mismatch=torch.randn(num_edges, Q) * 0.05
        )

        optimizer.zero_grad()
        loss, out = compute_loss(net, x0, target, ctx,
                                 F.mse_loss, lambdas, tau=0.5)
        loss.backward()

        # Important: clip gradients because the ODE can be stiff
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)

        optimizer.step()
```

**Temperature annealing:** Start with `tau=1.0` (soft selection), anneal to `tau=0.1` during training to force crisp L/S/Z choices.

---

## 5. Validation Roadmap (Idealized → Real)

Use this order to validate the concept before your SPICE data is ready:

| Phase | Test | Pass Criteria |
|-------|------|---------------|
| **A** | Single stage, 8 nodes, no training | Set initial $x_0$, verify Heun converges to a steady state without exploding |
| **B** | Gradient sanity | `torch.autograd.gradcheck` on `rhs` for a single step |
| **C** | Sparsity test | Train with high `lambda_sparsity`; verify Z cell dominates most edges |
| **D** | Variation robustness | Train with `ctx.global_gain_shift=0`, test with `±0.1`; accuracy should not collapse |
| **E** | 2-stage function fitting | Fit a simple 1D regression or MNIST with 2 stages; verify transfer layers work |

Once these pass, your scaffolding is sound. The only future swap is:

```python
# Replace this:
cell_lib = IdealizedCellLibrary(gm=[...], isat=[...], ...)
# With this:
cell_lib = SpiceFittedCellLibrary(surrogate_ckpt="spice_fit.pt")
```

The rest of the code (stage, transfer, solver, loss) stays identical.

---

## 6. What to Explicitly Leave Out of This Code

To stay focused on the validation goal, **do not implement yet**:

1. **Full common-mode dynamics** — assume $c_j = V_{CM}$ exactly.
2. **Non-diagonal capacitance matrix** — use scalar $C_{\text{eff}}$ per node.
3. **Input analog front-end** — inject $x_0$ directly as a tensor.
4. **Output analog back-end** — read $x$ directly.
5. **Stochastic noise / SDE** — add only after deterministic validation works.
6. **Adjoint solver** — direct BPTT is the correct baseline here.

---

## 7. Summary of the BPTT vs Adjoint Decision

| | Direct BPTT (Heun) | Neural ODE Adjoint |
|---|---|---|
| **Stiffness** | Stable; you control step size | Can be unstable, wrong gradients |
| **Trajectory regularizers** | Trivial; you have `traj` tensor | Expensive; requires augmented state |
| **Variation / Mismatch** | Easy; sample per unroll | Hard; continuous adjoint assumes smoothness |
| **Memory** | $O(N \times T)$ (store trajectory) | $O(N)$ but recomputes forward |
| **Code complexity** | Simple for-loop | Requires `torchdiffeq` hooks |

For a network with $<200$ nodes and $<20$ time steps, the memory cost of BPTT is negligible. Use it.

---

If you want, I can next generate:
1. A **concrete hyperparameter sweep spec** (stage counts, time constants, $x_{\max}$ ranges),
2. A **topology generator** for NE+Proj with variable stage widths, or
3. A **unit-test skeleton** for the gradient and sparsity checks in Phase A–C.

---

Here is a detailed implementation plan for a **task-structured, programmer-configurable topology generator**. It follows the heuristics from the reference text: topology must match problem structure, avoid fully-connected graphs, support projection nodes for global coupling, and integrate cleanly with edge-gated pruning.

---

## 1. Design Philosophy

**Core rule from the reference:** *"Match topology to data geometry. Scalar/tabular → clustered sparse graph. Temporal → 1D line. Spatial → 2D grid."*

Therefore, the generator is not one function with dozens of flags. It is a **family of builders** that share a common sparse-graph representation, plus a lightweight **composer** that assembles stage-wise topologies.

**Key requirements:**
- **Declarative:** The programmer specifies intent ("1D line, radius 2, 4 projection nodes"), not raw edge lists.
- **Composable:** Input-to-hidden, hidden-to-hidden, and hidden-to-output are separate sub-graphs that get merged.
- **Stage-aware:** Each stage can have its own topology and width.
- **Pruning-ready:** Every edge is born with a gate parameter `z_e` and a multiplicity parameter `m_e`.
- **Hardware-sane:** No fully-connected hidden layers. Maximum fanout is bounded.

---

## 2. Data Structure: The `SparseTopology` Object

All builders return a lightweight dataclass. This is the universal currency of the generator.

```python
from dataclasses import dataclass
import torch

@dataclass
class SparseTopology:
    num_nodes: int          # total logical nodes in this stage
    src: list[int]          # COO source indices
    dst: list[int]          # COO destination indices
    edge_type: list[str]    # optional tags: "input", "hidden", "proj", "output"
    node_kind: list[str]    # "input", "hidden", "proj", "output" per node
    
    # These are populated by the composer, not the raw builder
    input_node_ids: list[int]
    output_node_ids: list[int]
    hidden_node_ids: list[int]
    proj_node_ids: list[int]
```

**Why lists of ints, not tensors?** Because topologies are built once on CPU, then converted to `torch.LongTensor` when attached to a `DifferentialStage`. This keeps the generator simple and debuggable.

---

## 3. API Overview: Three-Layer Design

| Layer | What the Programmer Uses | Example |
|-------|------------------------|---------|
| **Primitives** | `line_graph(n, radius)`, `ring_graph(n, radius)`, `grid_graph(h, w, kernel)`, `cluster_graph(n, p_connect)` | Low-level sparse builders |
| **Connectors** | `connect_bipartite(src_nodes, dst_nodes, pattern)`, `connect_k_nearest(points, k)` | Wiring between node groups |
| **Composer** | `StageTopology(input_cfg, hidden_cfg, output_cfg, proj_cfg)` | Assembles a full stage |

The programmer never hand-writes `src`/`dst` arrays unless they want a fully custom topology.

---

## 4. Primitive Topology Builders

### 4.1 1D Local Graphs (Temporal / Waveform)

```python
def line_graph(n_nodes, radius=1, bidirectional=True):
    """1D chain where each node connects to neighbors within ±radius."""
    src, dst = [], []
    for i in range(n_nodes):
        for r in range(1, radius + 1):
            if i + r < n_nodes:
                src.append(i); dst.append(i + r)
                if bidirectional:
                    src.append(i + r); dst.append(i)
    return SparseTopology(
        num_nodes=n_nodes, src=src, dst=dst,
        edge_type=["hidden"] * len(src),
        node_kind=["hidden"] * n_nodes,
        input_node_ids=[], output_node_ids=[], 
        hidden_node_ids=list(range(n_nodes)), proj_node_ids=[]
    )

def ring_graph(n_nodes, radius=1):
    """1D ring with wrap-around. Useful for periodic signals."""
    src, dst = [], []
    for i in range(n_nodes):
        for r in range(1, radius + 1):
            j = (i + r) % n_nodes
            src.append(i); dst.append(j)
            src.append(j); dst.append(i)
    # ... return SparseTopology ...
```

**Reference heuristic:** *"For equalization / waveform, use a 1D line, not a 2D grid."*

### 4.2 2D Grid (Spatial / Image Patches)

```python
def grid_graph(height, width, kernel_size=3):
    """2D local grid. Node id = row * width + col."""
    src, dst = [], []
    n = height * width
    pad = kernel_size // 2
    
    for r in range(height):
        for c in range(width):
            i = r * width + c
            for dr in range(-pad, pad + 1):
                for dc in range(-pad, pad + 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        j = nr * width + nc
                        src.append(i); dst.append(j)
    return SparseTopology(...)
```

**Reference heuristic:** *"Only use k×k for spatial data. For scalar regression, a 2D grid is arbitrary and usually wrong."*

### 4.3 Cluster / Sparse Graph (Tabular / Low-Dim Regression)

```python
def cluster_graph(n_nodes, edge_prob=0.3, seed=0):
    """Erdős-Rényi-like sparse graph, but with bounded degree."""
    import random
    rng = random.Random(seed)
    src, dst = [], []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                src.append(i); dst.append(j)
                src.append(j); dst.append(i)
    return SparseTopology(...)
```

**Reference heuristic:** *"For sin(x), toy regression: use small non-grid sparse hidden graph. N_h = 6 to 16."*

### 4.4 Empty / Disconnected Graph (for Z-cells)

```python
def empty_graph(n_nodes):
    """No edges. Useful for ablation or pure projection-node stages."""
    return SparseTopology(
        num_nodes=n_nodes, src=[], dst=[],
        edge_type=[], node_kind=["hidden"] * n_nodes,
        input_node_ids=[], output_node_ids=[],
        hidden_node_ids=list(range(n_nodes)), proj_node_ids=[]
    )
```

---

## 5. Connector Primitives (Wiring Between Groups)

These functions connect two *disjoint* node sets. They are used to wire inputs→hidden, hidden→projection, and hidden→outputs.

```python
def connect_bipartite(src_ids, dst_ids, pattern="all_to_all"):
    """
    pattern: 'all_to_all', 'one_to_one', 'k_random', 'none'
    Returns (src, dst) edge lists.
    """
    src, dst = [], []
    if pattern == "all_to_all":
        for s in src_ids:
            for d in dst_ids:
                src.append(s); dst.append(d)
    elif pattern == "one_to_one":
        assert len(src_ids) == len(dst_ids)
        for s, d in zip(src_ids, dst_ids):
            src.append(s); dst.append(d)
    elif pattern == "none":
        pass
    else:
        raise ValueError(pattern)
    return src, dst

def connect_projection(hidden_ids, proj_ids, pattern="all_to_all"):
    """Bidirectional bipartite between hidden and projection nodes."""
    s1, d1 = connect_bipartite(hidden_ids, proj_ids, pattern)
    s2, d2 = connect_bipartite(proj_ids, hidden_ids, pattern)
    return s1 + s2, d1 + d2
```

**Reference heuristic:** *"Projection nodes are your main way to add long-range coupling without going fully connected."*

---

## 6. The Composer: `StageTopologyBuilder`

This is the main class the programmer interacts with. It assembles a full stage from input, hidden, projection, and output sub-graphs.

```python
class StageTopologyBuilder:
    def __init__(self, num_inputs, num_outputs, num_hidden, num_proj=0):
        self.n_in = num_inputs
        self.n_out = num_outputs
        self.n_h = num_hidden
        self.n_p = num_proj
        
        # Fixed node ID allocation:
        # [0 .. n_in-1] = inputs
        # [n_in .. n_in+n_h-1] = hidden
        # [n_in+n_h .. n_in+n_h+n_p-1] = projection
        # [n_in+n_h+n_p .. n_in+n_h+n_p+n_out-1] = outputs
        self.in_ids = list(range(num_inputs))
        self.hid_ids = list(range(num_inputs, num_inputs + num_hidden))
        self.proj_ids = list(range(num_inputs + num_hidden,
                                   num_inputs + num_hidden + num_proj))
        self.out_ids = list(range(num_inputs + num_hidden + num_proj,
                                  num_inputs + num_hidden + num_proj + num_outputs))
        self.total_nodes = num_inputs + num_hidden + num_proj + num_outputs

    def build(self, hidden_topo: SparseTopology,
              input_pattern="one_to_one",
              output_pattern="all_to_all",
              proj_pattern="all_to_all"):
        """
        hidden_topo: topology over hidden nodes (from primitives above)
        input_pattern: how inputs feed hidden nodes
        output_pattern: how hidden/proj nodes feed outputs
        proj_pattern: how hidden and proj nodes interconnect
        """
        src, dst = [], []
        edge_type = []
        node_kind = ["input"] * self.n_in + ["hidden"] * self.n_h \
                  + ["proj"] * self.n_p + ["output"] * self.n_out

        # 1. Hidden-to-hidden edges (the core computation graph)
        # Remap hidden_topo node ids to global ids
        offset = self.n_in
        for s, d in zip(hidden_topo.src, hidden_topo.dst):
            src.append(s + offset)
            dst.append(d + offset)
            edge_type.append("hidden")

        # 2. Input → Hidden
        s, d = connect_bipartite(self.in_ids, self.hid_ids, input_pattern)
        src.extend(s); dst.extend(d); edge_type.extend(["input"] * len(s))

        # 3. Hidden ↔ Projection (bidirectional)
        if self.n_p > 0:
            s, d = connect_projection(self.hid_ids, self.proj_ids, proj_pattern)
            src.extend(s); dst.extend(d); edge_type.extend(["proj"] * len(s))

        # 4. Hidden/Proj → Output
        # By default, outputs read from hidden + projection nodes
        source_pool = self.hid_ids + self.proj_ids
        s, d = connect_bipartite(source_pool, self.out_ids, output_pattern)
        src.extend(s); dst.extend(d); edge_type.extend(["output"] * len(s))

        return SparseTopology(
            num_nodes=self.total_nodes,
            src=src, dst=dst, edge_type=edge_type, node_kind=node_kind,
            input_node_ids=self.in_ids, output_node_ids=self.out_ids,
            hidden_node_ids=self.hid_ids, proj_node_ids=self.proj_ids
        )
```

**Key customization:** The programmer controls `hidden_topo` (grid, line, cluster) and the four wiring patterns.

---

## 7. Multi-Stage Assembly

The reference text says: *"For silicon, instantiate D fixed stages. Do not try to make one giant analog fabric that rewires itself between intervals."*

So we build a list of `StageTopology` objects, one per stage. Widths can change via `StageTransfer`.

```python
class MultiStageTopology:
    def __init__(self, stage_topologies: list[SparseTopology]):
        self.stages = stage_topologies

    @staticmethod
    def from_config(configs: list[dict]):
        """
        configs[i] = {
            'num_inputs': ...,
            'num_hidden': ...,
            'num_proj': ...,
            'num_outputs': ...,
            'hidden_family': 'line' | 'grid' | 'cluster' | 'empty',
            'hidden_kwargs': {...},
            'input_pattern': ...,
            'output_pattern': ...,
            'proj_pattern': ...,
        }
        """
        topologies = []
        for cfg in configs:
            builder = StageTopologyBuilder(
                num_inputs=cfg["num_inputs"],
                num_outputs=cfg["num_outputs"],
                num_hidden=cfg["num_hidden"],
                num_proj=cfg.get("num_proj", 0)
            )
            # Dispatch to primitive builder
            family = cfg["hidden_family"]
            if family == "line":
                hid = line_graph(cfg["num_hidden"], **cfg.get("hidden_kwargs", {}))
            elif family == "grid":
                h, w = cfg["hidden_kwargs"]["height"], cfg["hidden_kwargs"]["width"]
                hid = grid_graph(h, w, **{k:v for k,v in cfg["hidden_kwargs"].items() if k not in ["height","width"]})
            elif family == "cluster":
                hid = cluster_graph(cfg["num_hidden"], **cfg.get("hidden_kwargs", {}))
            else:
                hid = empty_graph(cfg["num_hidden"])
            
            topo = builder.build(
                hid,
                input_pattern=cfg.get("input_pattern", "one_to_one"),
                output_pattern=cfg.get("output_pattern", "all_to_all"),
                proj_pattern=cfg.get("proj_pattern", "all_to_all")
            )
            topologies.append(topo)
        return MultiStageTopology(topologies)
```

---

## 8. Integration with `DifferentialStage`

The `SparseTopology` must be converted into the tensors expected by the PyTorch module.

```python
def topology_to_stage(topo: SparseTopology, cell_lib, c_eff=1.0, x_max=0.3):
    """Factory function: SparseTopology -> DifferentialStage."""
    # Filter to only hidden/projection edges for the ODE core
    # Input edges are treated as write-path initialization, not ODE branches
    # Output edges are treated as readout taps
    core_mask = [t in ("hidden", "proj") for t in topo.edge_type]
    
    core_src = [topo.src[i] for i, m in enumerate(core_mask) if m]
    core_dst = [topo.dst[i] for i, m in enumerate(core_mask) if m]
    
    # Remap global node ids to compact 0..N-1 for the stage's internal state
    # The stage only evolves hidden + projection nodes.
    active_nodes = sorted(set(topo.hidden_node_ids + topo.proj_node_ids))
    id_map = {old: new for new, old in enumerate(active_nodes)}
    
    remapped_src = [id_map[s] for s in core_src]
    remapped_dst = [id_map[d] for d in core_dst]
    
    stage = DifferentialStage(
        num_nodes=len(active_nodes),
        src=remapped_src,
        dst=remapped_dst,
        cell_lib=cell_lib,
        c_eff=c_eff,
        x_max=x_max
    )
    return stage, active_nodes  # active_nodes tells you the global→local mapping
```

**Important design choice:** Input and output edges are **not** part of the ODE graph. Inputs write to node capacitors during initialization. Outputs are read directly from node voltages. This matches the reference: *"input is not necessarily a continuously driven voltage source during inference; it is the starting node voltage v(0)."*

---

## 9. Programmer Customization Examples

### Example A: Scalar Regression (sin x)

```python
# Reference heuristic: N_h = 6 to 16, P = 0 to 2, D = 1 or 2
# Topology: small cluster graph, inputs feed all hidden, output reads all hidden

cfg = {
    "num_inputs": 1,
    "num_hidden": 8,
    "num_proj": 2,
    "num_outputs": 1,
    "hidden_family": "cluster",
    "hidden_kwargs": {"edge_prob": 0.35},
    "input_pattern": "all_to_all",   # single input broadcasts to all hidden
    "output_pattern": "all_to_all",  # output reads all hidden + proj
    "proj_pattern": "all_to_all"
}

topo = StageTopologyBuilder(**{k:v for k,v in cfg.items() if k not in ["hidden_family","hidden_kwargs"]})
hid = cluster_graph(cfg["num_hidden"], **cfg["hidden_kwargs"])
stage_topo = topo.build(hid, **{k:cfg[k] for k in ["input_pattern","output_pattern","proj_pattern"]})
```

### Example B: Windowed Equalization (16-tap)

```python
# Reference heuristic: 1D line, N_h = L to 2L, r = 1 or 2, P = 2 to 4, D = 2 or 3

L = 16
configs = []
for stage_idx in range(2):
    configs.append({
        "num_inputs": L if stage_idx == 0 else 0,  # only first stage sees raw input
        "num_hidden": 24,
        "num_proj": 4,
        "num_outputs": L if stage_idx == 1 else 0,
        "hidden_family": "line",
        "hidden_kwargs": {"radius": 2, "bidirectional": True},
        "input_pattern": "one_to_one",   # each sample -> one hidden node
        "output_pattern": "one_to_one",
        "proj_pattern": "all_to_all"
    })

multi_topo = MultiStageTopology.from_config(configs)
```

### Example C: 2D Image Patch (4×4)

```python
# Reference heuristic: 2D grid, 3×3 kernel, P = sqrt(N_h)

cfg = {
    "num_inputs": 16,
    "num_hidden": 16,
    "num_proj": 4,
    "num_outputs": 4,
    "hidden_family": "grid",
    "hidden_kwargs": {"height": 4, "width": 4, "kernel_size": 3},
    "input_pattern": "one_to_one",
    "output_pattern": "all_to_all",
    "proj_pattern": "all_to_all"
}
```

---

## 10. Edge Gating & Pruning Support

The topology generator must support the **overprovision-then-prune** strategy from the reference.

```python
class GatedTopology:
    """Wraps a SparseTopology with learnable edge gates."""
    def __init__(self, topo: SparseTopology):
        self.topo = topo
        self.num_edges = len(topo.src)
        self.z_logits = nn.Parameter(torch.zeros(self.num_edges))  # edge gates
        self.u_nodes = nn.Parameter(torch.zeros(topo.num_nodes))   # node gates (optional)
        
    def gated_adjacency(self, tau=0.1):
        z = torch.sigmoid(self.z_logits / tau)
        # Return effective edge weights for loss penalties
        return z
    
    def node_mask(self):
        return torch.sigmoid(self.u_nodes)
```

**Pruning workflow:**
1. Build overcomplete topology.
2. Attach `z_logits` and `u_nodes`.
3. Train with `L_sparsity = lambda_z * sum(z_e) + lambda_u * sum(u_j)`.
4. After convergence, threshold `z_e < 0.01` and `u_j < 0.01`.
5. Rebuild a new `SparseTopology` with pruned edges/nodes removed.
6. Retrain from scratch or fine-tune.

---

## 11. Validation & Sanity Checks

Every generated topology should pass these assertions before training:

```python
def validate_topology(topo: SparseTopology):
    assert len(topo.src) == len(topo.dst)
    assert len(topo.edge_type) == len(topo.src)
    assert max(topo.src + topo.dst, default=-1) < topo.num_nodes
    assert len(topo.node_kind) == topo.num_nodes
    
    # No self-loops (optional, depending on design)
    for s, d in zip(topo.src, topo.dst):
        assert s != d, "Self-loops not allowed in this design"
    
    # No fully-connected hidden core (sanity check)
    n_h = len(topo.hidden_node_ids)
    max_edges = n_h * (n_h - 1)  # worst case directed
    actual_hidden_edges = sum(1 for t in topo.edge_type if t == "hidden")
    assert actual_hidden_edges < max_edges * 0.5, "Hidden core too dense"
    
    # Every input node has at least one outgoing edge
    for i in topo.input_node_ids:
        assert i in topo.src, f"Input node {i} is isolated"
    
    # Every output node has at least one incoming edge
    for o in topo.output_node_ids:
        assert o in topo.dst, f"Output node {o} is isolated"
```

---

## 12. Summary of Programmer Interface

| What they want | How they do it |
|----------------|----------------|
| 1D temporal graph | `line_graph(n, radius=2)` |
| 2D spatial graph | `grid_graph(h, w, kernel_size=3)` |
| Unstructured tabular | `cluster_graph(n, edge_prob=0.3)` |
| Add global shortcuts | Set `num_proj > 0` in `StageTopologyBuilder` |
| Change wiring density | Change `input_pattern` / `output_pattern` |
| Multi-stage network | `MultiStageTopology.from_config([cfg1, cfg2])` |
| Prune later | `GatedTopology` wrapper + `L_sparsity` |

---

## 13. What to Explicitly Leave Out

To keep the generator focused and hardware-sane, **do not support**:
- Fully-connected hidden graphs (the builder should reject or warn if `n_h > 32` and density > 50%).
- Runtime dynamic rewiring (the reference says: *"Do not try to make one giant analog fabric that rewires itself"*).
- Arbitrary non-local connections in the hidden core (use projection nodes instead).
- Learnable topology during forward pass (topology is fixed; only edge *weights* and *gates* are learned).

---

If you want, I can next generate:
1. **Concrete JSON/YAML schema** for topology configuration files,
2. **Unit test skeleton** for the validation checks,
3. **A pruning-to-layout helper** that converts a pruned `SparseTopology` into a compact node/edge list for the `DifferentialStage` constructor.

---

**No — do not use MLPs for I/O mapping.** Use simple **bounded affine maps** (linear + tanh). MLPs are the wrong abstraction here: they add unnecessary parameters, risk introducing unwanted nonlinearities, and the reviewer explicitly warns against burning learnable complexity on "input scalers" that should be trivial peripheral circuitry.

Here is the correct software-only approach.

---

## Input Mapping: $x_j(0) = x_{\max} \tanh(a_j u_j + b_j)$

Each raw input feature $u_j$ is mapped to a differential voltage $x_j(0)$ via a **learned affine-tanh** block.

**Why this form:**
- **$\tanh$ naturally bounds** the output to $(-1, 1)$, so scaling by $x_{\max}$ keeps you inside the differential rail limits.
- **It is differentiable** — PyTorch backprops through it cleanly.
- **It is exactly what a DAC + write driver does physically**: gain, offset, then clamp to a bounded voltage range.

**Implementation:**

```python
class InputMapper(nn.Module):
    def __init__(self, in_dim, out_dim, x_max=0.3):
        super().__init__()
        self.x_max = x_max
        # One linear layer: no hidden MLP
        self.gain = nn.Linear(in_dim, out_dim, bias=True)
        
    def forward(self, u):
        # u: [batch, in_dim]
        return self.x_max * torch.tanh(self.gain(u))
```

**Key details:**
- `out_dim` equals the number of input nodes in your first stage (usually $N_h$ or $L$).
- Initialize `gain.weight` small (e.g., Xavier with small magnitude) and `gain.bias` near zero so the network starts with small initial voltages.
- Do **not** add LayerNorm, ReLU, or hidden layers here. That would be a "learnable divider" that the reviewer specifically criticizes.

---

## Output Mapping: $\hat{y} = W_{\text{out}} x_{\text{final}} + b_{\text{out}}$

Read the final node voltages and apply a **simple linear projection**.

**Why no MLP:**
- The analog core is already doing the nonlinear computation via the ODE and tanh branch cells.
- Adding an MLP readout would hide the fact that the Kirchhoff network itself is the compute engine.
- Physically, this is just an ADC followed by digital scaling.

**Implementation:**

```python
class OutputMapper(nn.Module):
    def __init__(self, node_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(node_dim, out_dim, bias=True)
        
    def forward(self, x_final):
        # x_final: [batch, num_nodes] — the final differential voltages
        return self.proj(x_final)
```

**Which nodes to read from:**
- By default, read from **all hidden + projection nodes** in the final stage.
- If you want sparsity, learn a node attention mask or simply select a fixed subset (e.g., last 4 nodes) — but keep the readout linear.

---

## Wiring It Together (Software-Only)

Even in software, preserve the **write → evolve → read** phase abstraction. It keeps the model honest for when you later swap in the physical surrogate.

```python
class KirchhoffNetWithIO(nn.Module):
    def __init__(self, input_mapper, core_net, output_mapper):
        super().__init__()
        self.input_mapper = input_mapper   # Linear + tanh
        self.core = core_net               # Your ODE stages
        self.output_mapper = output_mapper # Linear
        
    def forward(self, u, ctx, tau=1.0):
        # 1. WRITE: map raw input to bounded initial voltages
        x0 = self.input_mapper(u)
        
        # 2. EVOLVE: run the analog core
        x_final, trajs = self.core(x0, ctx, tau)
        
        # 3. READ: linear projection to output space
        return self.output_mapper(x_final), trajs
```

---

## Why MLPs Are Specifically Wrong Here

| MLP as I/O scaler | Simple Affine + Tanh |
|---------------------|----------------------|
| Adds 2–3× parameters for no benefit | Minimal parameters |
| Can learn pathological mappings that exploit the ODE in unstable ways | Bounded, stable, physically faithful |
| Violates the "peripheral circuitry should be simple" principle | Matches the physical intent |
| Makes pruning harder (more params to regularize) | Clean separation of core vs. I/O |

The reviewer was explicit: *"Do not waste area learning triode-region divider ratios inside the core"* and *"This mapping can be implemented in software if the source is already digital."*

---

## Optional: Input-Dependent Bias (One Small Refinement)

If your task has strongly varying input ranges (e.g., some features are $10^{-3}$ and others are $10^3$), you can add a **per-feature learnable log-scale** before the affine map:

```python
class RobustInputMapper(nn.Module):
    def __init__(self, in_dim, out_dim, x_max=0.3):
        super().__init__()
        self.x_max = x_max
        self.log_scale = nn.Parameter(torch.zeros(in_dim))
        self.gain = nn.Linear(in_dim, out_dim, bias=True)
        
    def forward(self, u):
        u_scaled = u * torch.exp(self.log_scale)
        return self.x_max * torch.tanh(self.gain(u_scaled))
```

This is still not an MLP — it is just a **diagonal preconditioner** that helps the linear layer handle heterogeneous feature scales. The core remains simple.

---

## Summary

- **Input:** `nn.Linear` + `tanh` → bounded differential voltages. No MLP.
- **Output:** `nn.Linear` only. No activation.
- **Rationale:** The ODE core is the compute engine. I/O should be simple, bounded, and physically interpretable as DAC/ADC peripherals.

If you want, I can write the exact `nn.Module` wrapper that combines the mapper, multi-stage core, and output projection into a single trainable pipeline with proper initialization.