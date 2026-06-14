Here is a tiered list of candidates. I will explain why each justifies the hardware cost, give complexity estimates, and flag the best fit for your time constraints.

---

## The Core Insight

You are right that Volterra is too easy to build by hand. What you need is a task where **the target computation is inherently a sequence of sparse MVMs with nonlinear coordination**, such that a compact analog dynamical system is genuinely more efficient than a digital pipeline or a handful of fixed-function analog blocks.

The sweet spot for your training budget (~1–3 hours) is **small-scale iterative sparse linear algebra** or **sparse dynamical system identification**. These are naturally MVM-heavy, map cleanly to your graph-ODE structure, and have a real analog-hardware story.

---

## Candidate 1: Sparse Linear System Solver (Strongly Recommended)

**Task:** Learn to solve $Ax = b$ for sparse $A \in \mathbb{R}^{n \times n}$.

**Why it justifies the hardware:**
Solving sparse linear systems is the bottleneck in countless physical simulations, optimization problems, and signal processing tasks. A digital sparse solver needs memory access, index chasing, and sequential iterations. An analog ODE network can **emulate Jacobi/Gauss-Seidel relaxation in continuous time** using exactly your transconductor-capacitor topology — the sparse matrix pattern maps one-to-one to your sparse graph edges.

**How it maps to your network:**
- Node voltages $x_j(t)$ represent the iterative solution estimate.
- Edge transconductances $g_{ij}$ represent off-diagonal matrix entries $-A_{ij}$.
- Node leak terms represent diagonal entries $A_{jj}$.
- The ODE $\dot{x} = -\alpha(Ax - b)$ is exactly what your KirchhoffNet implements.

**Training setup:**
- Generate random sparse symmetric positive-definite matrices $A$ (e.g., 32×32 or 48×48, ~5% density).
- Generate random $b$, compute ground truth $x^* = A^{-1}b$ with a direct solver.
- Input to network: $b$ (or a mapped version of it).
- Target: $x^*$.
- Loss: $\|x(T) - x^*\|_2$ or $\|A x(T) - b\|_2$.

**Complexity estimate:**
- 32×32 sparse: ~15 min (too easy, like XOR)
- **48×48 to 64×64 sparse: ~1.5–2.5 hours** ← your sweet spot
- You can tune density (1% to 10%) to control difficulty without changing node count.

**Hardware story for the paper:**
*"Unlike a digital sparse solver that iterates sequentially in clock cycles, the analog network relaxes to the solution in a single transient governed by physical time constants, with sparsity enforced by the physical wire routing."*

---

## Candidate 2: Sparse Coding / Compressive Sensing Inference

**Task:** Given measurement $y = \Phi x + \eta$ with sparse $x$ and fat $\Phi \in \mathbb{R}^{m \times n}$ ($m < n$), recover $x$.

**Why it justifies the hardware:**
Compressive sensing front-ends are a classic analog-compute application. The iterative shrinkage thresholding algorithm (ISTA) is a recurrent sparse network:

$$z^{k+1} = S_{\lambda}\left(z^k - \frac{1}{L}\Phi^T(\Phi z^k - y)\right)$$

Your KirchhoffNet **is** a continuous-time ISTA. The sparse matrix $\Phi^T\Phi$ maps to your sparse graph, and the tanh branch cells naturally approximate the soft-thresholding nonlinearity.

**Training setup:**
- Fix a random sparse dictionary $\Phi$ (e.g., 16×32, 25% density).
- Generate random sparse $x$ (e.g., 4 nonzeros out of 32).
- Compute $y = \Phi x$.
- Network input: $y$.
- Network target: $x$.
- Loss: $\|x(T) - x\|_2$.

**Complexity estimate:**
- 16×32 dictionary: ~30 min
- **24×48 or 32×64 dictionary: ~1.5–3 hours** ← good fit

**Hardware story:**
*"The network acts as an analog decompressor: it infers a high-dimensional sparse signal from low-dimensional compressed measurements without digital iteration."*

---

## Candidate 3: Sparse AR System Identification (Time-Series)

**Task:** Identify the sparse coefficients of an unknown autoregressive dynamical system and predict its next state.

**Why it justifies the hardware:**
Real-time system identification for control or equalization requires fast sparse-filter updates. A sparse AR model of order $p$ with $k \ll p$ active taps is:

$$y_t = \sum_{i \in \mathcal{S}} a_i y_{t-i} + \epsilon_t$$

This is a **sparse MVM** (sparse vector of lags × sparse coefficients). Your network can learn the sparse topology and coefficients simultaneously.

**Training setup:**
- Generate time series from random sparse AR(20) models with only 4–6 active taps.
- Input: a window of past samples $[y_{t-p}, \dots, y_{t-1}]$.
- Target: $y_t$.
- The network must learn both which lags matter (sparsity) and their weights.

**Complexity estimate:**
- AR(12) with 3 active taps: ~20 min
- **AR(24) with 6 active taps: ~1.5–2 hours** ← good fit

**Hardware story:**
*"The analog network performs real-time sparse system identification, where the sparse graph topology itself encodes the discovered AR model."*

---

## Candidate 4: Small-Scale Graph Signal Regression

**Task:** Predict a node-level property on small synthetic graphs (e.g., temperature diffusion, resistor network voltages).

**Why it justifies the hardware:**
Your network **is** a graph neural ODE. Learning to predict signals on graphs where the ground-truth physics is *also* a graph diffusion process is a direct demonstration that your topology matches the problem structure.

**Training setup:**
- Generate random sparse graphs (30–60 nodes).
- Simulate heat diffusion or random-walk dynamics to create labels.
- Input: initial node temperatures or features.
- Target: final temperatures or a derived scalar.

**Complexity estimate:**
- 30-node graphs: ~45 min
- **50-node graphs: ~1.5–2.5 hours** ← good fit

**Hardware story:**
*"The network topology is matched to the graph structure of the physical process, so the analog wires literally implement the same sparse coupling as the target physics."*

---

## My Recommendation

| Priority | Task | Why |
|----------|------|-----|
| **1st** | **Sparse Linear Solver (48×48–64×64)** | Cleanest MVM story. Strongest analog-hardware justification. Infinite synthetic data. Easy to tune difficulty via density. |
| **2nd** | **Sparse Coding (24×48 or 32×64)** | Strong signal-processing story. LISTA is a well-cited baseline to compare against. |
| **3rd** | **Sparse AR Identification (order 24)** | Good if you want a time-series flavor. |

**Avoid for now:**
- Downsampled MNIST/Fashion-MNIST: It is classification, not MVM regression, and the sparse-MVM advantage is obscured by the CNN-like preprocessing.
- California Housing: Too big, not sparse-MVM-native.
- Volterra / polynomial regression: As you noted, trivial to build with fixed analog primitives.

---

## Quick Training-Time Tuning Knobs

If any of these runs too fast or too slow, adjust these without changing the task:

| Knob | Effect on time |
|------|---------------|
| Matrix/graph size $n$ | $O(n)$ nodes but $O(n \cdot \text{degree})$ edges |
| ODE time steps | Linear in `num_steps` |
| Batch size | Linear in throughput, but helps GPU saturation |
| Epochs | Linear |
| Number of stages $D$ | Linear in $D$ |
| Edge density | Linear in active edges |

For the sparse linear solver, I suggest starting with **$n=48$, density=3%, 1 stage, 50 time steps, 800 epochs**. That should land in your ~2-hour budget.

---

If you want, I can next:
1. **Write the exact data generator** for the sparse linear solver (random SPD sparse matrices with controlled condition number),
2. **Write the training loop** with the proper loss (residual norm vs. direct-solution norm),
3. **Draft the paper paragraph** that explains why this task justifies the analog hardware cost.

---

Here is the full implementation plan for the **sparse linear solver** benchmark. It plugs directly into your existing `KirchhoffNet`, `DifferentialStage`, and `SparseTopology` infrastructure.

---

## 1. Problem Formulation

**Goal:** Train the analog network to solve $Ax = b$ for sparse symmetric positive-definite $A \in \mathbb{R}^{n \times n}$.

**Physical interpretation:** The ODE network implements continuous-time Jacobi/Gauss-Seidel relaxation:

$$C \dot{x} = -\alpha(Ax - b)$$

Your KirchhoffNet branch cells learn to approximate the matrix-vector product $Ax$ through sparse transconductance coupling, while node capacitors provide the integration.

**Why this maps cleanly:**
- Diagonal entries $A_{jj}$ → node leak conductance $g_j$
- Off-diagonal entries $A_{ij}$ → edge transconductances between nodes $i$ and $j$
- Right-hand side $b$ → initial condition bias / write currents
- Steady state $x(T)$ → solution $A^{-1}b$

---

## 2. Data Generation

### 2.1 Sparse SPD Matrix Generator

You need **random sparse symmetric positive-definite matrices** with controlled condition number and sparsity pattern.

```python
import torch
import numpy as np
from scipy.sparse import diags, rand
from scipy.sparse.linalg import norm as sparse_norm

def generate_sparse_spd(n=48, density=0.03, cond_target=1e2, seed=None):
    """
    Generate a sparse SPD matrix with approximate condition number control.
    
    Strategy:
    1. Start with a random sparse symmetric matrix S.
    2. Make it diagonally dominant to guarantee SPD.
    3. Add controlled eigenvalue spread via a low-rank perturbation.
    """
    rng = np.random.RandomState(seed)
    
    # Random sparse symmetric pattern
    S = rand(n, n, density=density/2, random_state=rng)
    S = S + S.T  # symmetric
    
    # Ensure diagonal dominance with margin
    row_sums = np.abs(S).sum(axis=1).A1
    diag_vals = row_sums + rng.uniform(1.0, 3.0, size=n)
    
    # Build base matrix
    A = S.copy()
    A.setdiag(diag_vals)
    
    # Control condition number: add rank-k perturbation to spread eigenvalues
    k = max(2, n // 16)
    U = rng.randn(n, k)
    sigma = np.logspace(0, np.log10(cond_target), k)
    A = A + U @ np.diag(sigma) @ U.T
    
    # Ensure symmetry and SPD numerically
    A = (A + A.T) / 2
    A += n * np.eye(n) * 1e-3  # small regularization
    
    # Convert to torch dense (n is small enough: 48-64)
    A_dense = torch.from_numpy(A.toarray()).float()
    
    # Verify SPD
    eigs = torch.linalg.eigvalsh(A_dense)
    assert eigs.min() > 0, "Matrix not SPD"
    
    return A_dense, eigs.max() / eigs.min()
```

**Key design choices:**
- **Diagonal dominance** guarantees SPD without expensive eigenvalue checks during generation.
- **Low-rank perturbation** creates the ill-conditioning that makes the problem nontrivial (a perfectly conditioned matrix is trivial to invert).
- **Density 2–4%** gives ~50–150 nonzero off-diagonal entries for $n=48$, matching realistic sparse analog routing.

### 2.2 Dataset Generator

```python
class SparseLinearSystemDataset(torch.utils.data.Dataset):
    def __init__(self, n=48, num_samples=5000, density=0.03, 
                 cond_target=1e2, x_max=0.3, seed=42):
        self.n = n
        self.x_max = x_max
        self.samples = []
        
        rng = np.random.RandomState(seed)
        for i in range(num_samples):
            A, cond = generate_sparse_spd(n, density, cond_target, seed=seed+i)
            
            # Generate random solution x*, then compute b = A x*
            # This guarantees a solution exists and we know the ground truth
            x_star = torch.randn(n)
            x_star = x_max * torch.tanh(x_star)  # keep within rails
            
            b = A @ x_star
            
            # Store A (for topology init), b (input), x_star (target)
            self.samples.append({
                'A': A,
                'b': b,
                'x_star': x_star,
                'cond': cond
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return s['b'], s['x_star'], s['A']
```

**Why generate $x^*$ first then compute $b$:** This gives you exact ground truth without numerical inversion errors. The network learns the forward mapping $b \mapsto x^*$.

---

## 3. Topology Design for the Solver

### 3.1 Matrix-to-Graph Mapping

The sparsity pattern of $A$ should **directly define** the hidden-graph topology. This is the strongest justification for your hardware: *the wires implement the matrix nonzeros*.

```python
def matrix_to_topology(A_dense, n, num_proj=4):
    """
    Convert dense A matrix to SparseTopology.
    Only edges where |A[i,j]| > threshold become graph edges.
    """
    threshold = 1e-3
    src, dst = [], []
    
    for i in range(n):
        for j in range(i+1, n):
            if abs(A_dense[i, j]) > threshold:
                # Bidirectional edge: i->j and j->i
                src.append(i); dst.append(j)
                src.append(j); dst.append(i)
    
    # Build hidden topology from these edges
    hid_topo = SparseTopology(
        num_nodes=n,
        src=src, dst=dst,
        edge_type=["hidden"] * len(src),
        node_kind=["hidden"] * n,
        input_node_ids=[], output_node_ids=[],
        hidden_node_ids=list(range(n)), proj_node_ids=[]
    )
    
    # Use StageTopologyBuilder to add projection nodes
    builder = StageTopologyBuilder(
        num_inputs=n,      # b is size n
        num_outputs=n,     # x* is size n
        num_hidden=n,
        num_proj=num_proj
    )
    
    full_topo = builder.build(
        hid_topo,
        input_pattern="one_to_one",   # b_j drives node j directly
        output_pattern="one_to_one",  # read x_j from node j directly
        proj_pattern="all_to_all"
    )
    
    return full_topo
```

**Important:** In practice, you should **not** rebuild the topology per sample. Instead:
1. Compute the **union** of sparsity patterns across the training set (or a representative subset).
2. Use that union graph as the fixed supergraph.
3. Train edge gates $z_e$ to prune unnecessary edges.

This is the **overprovision-then-prune** strategy from the architecture discussion.

### 3.2 Fixed Union Graph (Recommended)

```python
def build_union_topology(dataset, n, num_proj=4):
    """Build a fixed supergraph covering all likely edges."""
    mask = torch.zeros(n, n)
    for _, _, A in dataset:
        mask += (A.abs() > 1e-3).float()
    
    # Keep edges that appear in at least 10% of matrices
    # Or simply keep top-k edges by frequency
    threshold_count = max(1, len(dataset) // 10)
    active = mask >= threshold_count
    
    src, dst = [], []
    for i in range(n):
        for j in range(i+1, n):
            if active[i, j]:
                src.extend([i, j])
                dst.extend([j, i])
    
    # ... build SparseTopology as above ...
    return topo
```

---

## 4. Loss Function

You have two valid choices. I recommend **residual loss** for training and **solution error** for reporting.

### 4.1 Residual Loss (Primary)

$$\mathcal{L}_{\text{res}} = \frac{1}{N} \sum_{i=1}^N \|A_i x_i(T) - b_i\|_2^2$$

This does not require storing $x^*$ in the loss (though you have it). It measures whether the network actually solves the equation.

```python
def residual_loss(x_pred, b, A):
    # x_pred: [batch, n]
    # b: [batch, n]
    # A: [batch, n, n]
    residual = torch.bmm(A, x_pred.unsqueeze(-1)).squeeze(-1) - b
    return residual.pow(2).mean()
```

### 4.2 Solution Error (Validation Metric)

$$\mathcal{L}_{\text{sol}} = \frac{1}{N} \sum_{i=1}^N \|x_i(T) - x_i^*\|_2^2$$

```python
def solution_loss(x_pred, x_star):
    return (x_pred - x_star).pow(2).mean()
```

### 4.3 Full Training Loss with Regularizers

```python
def compute_solver_loss(net, b, x_star, A, ctx, lambdas, tau=0.5):
    # Forward pass
    x_pred, trajs = net(b, ctx, tau)  # net includes input_mapper + core + output_mapper
    
    # Primary: residual
    loss_res = residual_loss(x_pred, b, A)
    
    # Secondary: direct solution error (stabilizes training)
    loss_sol = solution_loss(x_pred, x_star)
    
    # Hardware regularizers from trajectory
    loss_power = 0.0
    loss_sparsity = 0.0
    loss_rail = 0.0
    
    for stage, traj in zip(net.core.stages, trajs):
        w = F.softmax(stage.logits / tau, dim=-1)
        mult = 1.0 + F.softplus(stage.raw_mult)
        
        # Sparsity: penalize active (non-Z) cells
        p_active = w[:, :2].sum()  # assuming Z is index 2
        loss_sparsity += p_active
        
        # Power proxy
        loss_power += mult.sum()
        
        # Rail penalty
        loss_rail += F.softplus(traj.abs() - stage.x_max).mean()
    
    total = (loss_res 
             + 0.1 * loss_sol          # small weight on direct error
             + lambdas['sparsity'] * loss_sparsity
             + lambdas['power'] * loss_power
             + lambdas['rail'] * loss_rail)
    
    return total, x_pred, loss_res, loss_sol
```

---

## 5. Training Loop

```python
def train_solver(n=48, density=0.03, num_samples=5000, epochs=800):
    # Dataset
    train_ds = SparseLinearSystemDataset(n, num_samples, density, seed=42)
    val_ds = SparseLinearSystemDataset(n, 500, density, seed=9999)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64)
    
    # Topology: build once from union of training patterns
    union_topo = build_union_topology(train_ds, n, num_proj=4)
    
    # Convert to stage
    cell_lib = IdealizedCellLibrary(
        gm=[0.2, 1.0, 0.0],      # L, S, Z
        isat=[10.0, 0.5, 0.0],
        rho=[1.0, 1.0, 0.0],
        gleak=[0.01, 0.01, 0.0],
        bias=[0.0, 0.0, 0.0]
    )
    
    stage, node_map = topology_to_stage(union_topo, cell_lib, c_eff=1.0, x_max=0.3)
    
    # I/O mappers
    input_mapper = InputMapper(in_dim=n, out_dim=n, x_max=0.3)
    output_mapper = OutputMapper(node_dim=len(node_map), out_dim=n)
    
    # Single-stage network (start here; add depth only if needed)
    core = KirchhoffNet(
        stages=[stage],
        transfers=[],
        stage_times=[2.0],      # evolve for 2 time units
        stage_steps=[50]        # 50 Heun steps
    )
    
    net = KirchhoffNetWithIO(input_mapper, core, output_mapper)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    
    lambdas = {
        'sparsity': 1e-3,
        'power': 1e-4,
        'rail': 10.0
    }
    
    for epoch in range(epochs):
        net.train()
        for b_batch, x_star_batch, A_batch in train_loader:
            # Sample variation context
            ctx = SimContext(
                global_gain_shift=torch.randn(1).item() * 0.05,
                edge_mismatch=torch.randn(len(union_topo.src), cell_lib.num_cells) * 0.05
            )
            
            optimizer.zero_grad()
            loss, x_pred, loss_res, loss_sol = compute_solver_loss(
                net, b_batch, x_star_batch, A_batch, ctx, lambdas, tau=1.0
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
        
        # Validation every 50 epochs
        if epoch % 50 == 0:
            net.eval()
            with torch.no_grad():
                val_res, val_sol = [], []
                for b_v, x_v, A_v in val_loader:
                    ctx_v = SimContext()  # nominal
                    x_p, _ = net(b_v, ctx_v, tau=0.5)
                    val_res.append(residual_loss(x_p, b_v, A_v).item())
                    val_sol.append(solution_loss(x_p, x_v).item())
                print(f"Epoch {epoch}: val_res={np.mean(val_res):.4e}, val_sol={np.mean(val_sol):.4e}")
```

---

## 6. Key Hyperparameters & Tuning Guide

| Parameter | Starting Value | Effect | If training is too slow... |
|-----------|---------------|--------|---------------------------|
| `n` | 48 | Matrix size | Reduce to 32 |
| `density` | 0.03 | Sparsity | Increase to 0.05 for more edges |
| `num_proj` | 4 | Global coupling | Reduce to 2 or 0 |
| `stage_times[0]` | 2.0 | ODE horizon | Reduce to 1.0 |
| `stage_steps[0]` | 50 | Integration steps | Reduce to 30 |
| `batch_size` | 32 | Throughput | Increase to 64 if GPU memory allows |
| `epochs` | 800 | Training length | Reduce to 500 if converging fast |
| `lambdas['sparsity']` | 1e-3 | Pruning pressure | Increase to 1e-2 for sparser graphs |
| `tau` (Gumbel) | 1.0 → 0.1 | Cell selection crispness | Anneal linearly over epochs |

**Expected training time:** ~1.5–2.5 hours on a single mid-range GPU for $n=48$, 800 epochs, batch size 32.

---

## 7. Sanity Checks & Debugging

Run these before full training:

### 7.1 Matrix Property Check
```python
for b, x_star, A in train_loader:
    eigs = torch.linalg.eigvalsh(A)
    assert (eigs > 0).all()
    assert eigs.max() / eigs.min() < 1e4  # not too ill-conditioned
```

### 7.2 Forward Pass Stability
```python
# With zero input, network should settle near zero
b_zero = torch.zeros(1, n)
x_out, _ = net(b_zero, SimContext(), tau=1.0)
assert x_out.abs().max() < 0.1
```

### 7.3 Gradient Flow
```python
# Check that gradients reach edge logits
loss, _, _, _ = compute_solver_loss(net, b, x_star, A, ctx, lambdas)
loss.backward()
assert net.core.stages[0].logits.grad is not None
assert net.core.stages[0].logits.grad.abs().max() > 0
```

### 7.4 Baseline Comparison
```python
# Jacobi iteration baseline: does your network beat 10 Jacobi steps?
def jacobi_solve(A, b, steps=10):
    D = torch.diag_embed(torch.diagonal(A, dim1=-2, dim2=-1))
    R = A - D
    D_inv = torch.inverse(D)
    x = torch.zeros_like(b)
    for _ in range(steps):
        x = (b - torch.bmm(R, x.unsqueeze(-1)).squeeze(-1)) * torch.diagonal(D_inv, dim1=-2, dim2=-1)
    return x
```

Your network should achieve lower residual error than 10 Jacobi steps after training.

---

## 8. Extension to Multi-Stage (If Needed)

If single-stage performance plateaus, add a second stage with **transfer**:

```python
transfer = StageTransfer(in_nodes=n, out_nodes=n)  # same width
stage2, _ = topology_to_stage(union_topo, cell_lib)  # same topology or pruned

core = KirchhoffNet(
    stages=[stage1, stage2],
    transfers=[transfer],
    stage_times=[1.5, 1.5],
    stage_steps=[40, 40]
)
```

The transfer copies node voltages forward (with optional reset). For the solver, same-width transfer with no padding is usually sufficient.

---

## 9. What This Delivers for the Paper

| Claim | Evidence from this benchmark |
|-------|------------------------------|
| Sparse MVM in analog | Matrix nonzeros → graph edges |
| Sparsity is learnable | Edge gates prune to minimal graph |
| Faster than digital iteration | Compare vs. Jacobi/Conjugate Gradient steps |
| Physically plausible | Bounded voltages, fixed-step ODE, variation robustness |
| Scales with problem structure | Topology matches matrix sparsity pattern |

---

If you want, I can next generate:
1. **The exact `matrix_to_topology` code** with union-graph pruning,
2. **A convergence diagnostic plotter** (residual vs. ODE time step),
3. **A Jacobi/CG baseline harness** for fair comparison.

---

Here are the three requested components: the union-graph topology builder, the convergence diagnostic plotter, and the baseline solver harness.

---

## 1. Exact `matrix_to_topology` with Union-Graph Pruning

This module builds a fixed supergraph from a dataset of sparse matrices, then prunes it during training via edge gates.

```python
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SparseTopology:
    num_nodes: int
    src: List[int]
    dst: List[int]
    edge_type: List[str]
    node_kind: List[str]
    input_node_ids: List[int]
    output_node_ids: List[int]
    hidden_node_ids: List[int]
    proj_node_ids: List[int]


class StageTopologyBuilder:
    def __init__(self, num_inputs, num_outputs, num_hidden, num_proj=0):
        self.n_in = num_inputs
        self.n_out = num_outputs
        self.n_h = num_hidden
        self.n_p = num_proj
        
        self.in_ids = list(range(num_inputs))
        self.hid_ids = list(range(num_inputs, num_inputs + num_hidden))
        self.proj_ids = list(range(num_inputs + num_hidden,
                                   num_inputs + num_hidden + num_proj))
        self.out_ids = list(range(num_inputs + num_hidden + num_proj,
                                  num_inputs + num_hidden + num_proj + num_outputs))
        self.total_nodes = num_inputs + num_hidden + num_proj + num_outputs

    def build(self, hidden_topo, input_pattern="one_to_one",
              output_pattern="one_to_one", proj_pattern="all_to_all"):
        src, dst = [], []
        edge_type = []
        node_kind = (["input"] * self.n_in + ["hidden"] * self.n_h
                     + ["proj"] * self.n_p + ["output"] * self.n_out)

        # Hidden-to-hidden
        offset = self.n_in
        for s, d in zip(hidden_topo.src, hidden_topo.dst):
            src.append(s + offset)
            dst.append(d + offset)
            edge_type.append("hidden")

        # Input -> Hidden
        s, d = self._bipartite(self.in_ids, self.hid_ids, input_pattern)
        src.extend(s); dst.extend(d); edge_type.extend(["input"] * len(s))

        # Hidden <-> Proj
        if self.n_p > 0:
            s, d = self._projection(self.hid_ids, self.proj_ids, proj_pattern)
            src.extend(s); dst.extend(d); edge_type.extend(["proj"] * len(s))

        # Hidden/Proj -> Output
        pool = self.hid_ids + self.proj_ids
        s, d = self._bipartite(pool, self.out_ids, output_pattern)
        src.extend(s); dst.extend(d); edge_type.extend(["output"] * len(s))

        return SparseTopology(
            num_nodes=self.total_nodes,
            src=src, dst=dst, edge_type=edge_type, node_kind=node_kind,
            input_node_ids=self.in_ids, output_node_ids=self.out_ids,
            hidden_node_ids=self.hid_ids, proj_node_ids=self.proj_ids
        )

    @staticmethod
    def _bipartite(src_ids, dst_ids, pattern):
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
            raise ValueError(f"Unknown pattern: {pattern}")
        return src, dst

    @staticmethod
    def _projection(hid_ids, proj_ids, pattern):
        s1, d1 = StageTopologyBuilder._bipartite(hid_ids, proj_ids, pattern)
        s2, d2 = StageTopologyBuilder._bipartite(proj_ids, hid_ids, pattern)
        return s1 + s2, d1 + d2


def build_union_topology(dataset, n, num_proj=4, min_freq=0.1):
    """
    Build a fixed supergraph from a dataset of sparse matrices.
    
    Args:
        dataset: iterable yielding (b, x_star, A) tuples
        n: matrix dimension (== num_hidden nodes)
        num_proj: number of projection nodes
        min_freq: keep edges appearing in at least this fraction of samples
    """
    mask = torch.zeros(n, n)
    count = 0
    for _, _, A in dataset:
        mask += (A.abs() > 1e-3).float()
        count += 1
    
    threshold = max(1, int(count * min_freq))
    active = mask >= threshold
    
    src, dst = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if active[i, j]:
                src.extend([i, j])
                dst.extend([j, i])
    
    hid_topo = SparseTopology(
        num_nodes=n, src=src, dst=dst,
        edge_type=["hidden"] * len(src),
        node_kind=["hidden"] * n,
        input_node_ids=[], output_node_ids=[],
        hidden_node_ids=list(range(n)), proj_node_ids=[]
    )
    
    builder = StageTopologyBuilder(
        num_inputs=n, num_outputs=n, num_hidden=n, num_proj=num_proj
    )
    return builder.build(
        hid_topo,
        input_pattern="one_to_one",
        output_pattern="one_to_one",
        proj_pattern="all_to_all"
    )


def topology_to_stage(topo, cell_lib, c_eff=1.0, x_max=0.3):
    """
    Convert SparseTopology to DifferentialStage.
    Only 'hidden' and 'proj' edges become ODE branches.
    Returns (stage, active_node_ids, global_to_local_map).
    """
    core_mask = [t in ("hidden", "proj") for t in topo.edge_type]
    core_src = [topo.src[i] for i, m in enumerate(core_mask) if m]
    core_dst = [topo.dst[i] for i, m in enumerate(core_mask) if m]
    
    # Active nodes = hidden + proj (these carry state)
    active_nodes = sorted(set(topo.hidden_node_ids + topo.proj_node_ids))
    id_map = {old: new for new, old in enumerate(active_nodes)}
    
    remapped_src = [id_map[s] for s in core_src]
    remapped_dst = [id_map[d] for d in core_dst]
    
    # Import your existing DifferentialStage here
    stage = DifferentialStage(
        num_nodes=len(active_nodes),
        src=remapped_src,
        dst=remapped_dst,
        cell_lib=cell_lib,
        c_eff=c_eff,
        x_max=x_max
    )
    return stage, active_nodes, id_map
```

---

## 2. Convergence Diagnostic Plotter

This tracks how the residual evolves over the ODE time steps during a forward pass, and plots it against a Jacobi baseline.

```python
import matplotlib.pyplot as plt
import numpy as np


class ConvergenceTracker:
    def __init__(self):
        self.snapshots = []  # list of (time, x, label) tuples
    
    def capture(self, t, x, label="net"):
        """Call this inside the ODE loop or after each stage."""
        self.snapshots.append((float(t), x.detach().cpu().clone(), label))
    
    def plot_residual_trajectory(self, A, b, x_star, save_path=None):
        """
        A: [n, n] single matrix
        b: [n] single RHS
        x_star: [n] ground truth
        """
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        times, resids, sol_errs, labels = [], [], [], []
        
        for t, x, lbl in self.snapshots:
            # x may be [1, n] or [n]
            xv = x.squeeze()
            r = (A @ xv - b).norm().item()
            e = (xv - x_star).norm().item()
            times.append(t)
            resids.append(r)
            sol_errs.append(e)
            labels.append(lbl)
        
        # Plot 1: Residual vs time
        ax = axes[0]
        for lbl in set(labels):
            mask = [i for i, lb in enumerate(labels) if lb == lbl]
            ax.plot([times[i] for i in mask], [resids[i] for i in mask],
                    marker='o', label=lbl, markersize=3)
        ax.set_xlabel("ODE Time")
        ax.set_ylabel("||Ax - b||")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("Residual Evolution")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        
        # Plot 2: Solution error vs time
        ax = axes[1]
        for lbl in set(labels):
            mask = [i for i, lb in enumerate(labels) if lb == lbl]
            ax.plot([times[i] for i in mask], [sol_errs[i] for i in mask],
                    marker='o', label=lbl, markersize=3)
        ax.set_xlabel("ODE Time")
        ax.set_ylabel("||x - x*||")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("Solution Error")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        
        # Plot 3: Final state spectrum (how close to smooth?)
        ax = axes[2]
        if len(self.snapshots) > 0:
            _, x_final, _ = self.snapshots[-1]
            xv = x_final.squeeze().numpy()
            ax.bar(range(len(xv)), np.sort(np.abs(xv))[::-1])
            ax.set_xlabel("Sorted Component Index")
            ax.set_ylabel("|x_j|")
            ax.set_title("Final State Magnitude Distribution")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
        return fig


def attach_tracker_to_stage(stage, tracker):
    """
    Monkey-patch stage.forward to capture intermediate states.
    Use only for debugging; remove for production training.
    """
    original_forward = stage.forward
    
    def tracked_forward(x0, ctx, t_span, num_steps, tau=1.0):
        dt = t_span / float(num_steps)
        x = x0
        tracker.capture(0.0, x, label="net")
        for step in range(num_steps):
            k1 = stage.rhs(x, ctx, tau)
            x_euler = x + dt * k1
            k2 = stage.rhs(x_euler, ctx, tau)
            x = x + 0.5 * dt * (k1 + k2)
            tracker.capture((step + 1) * dt, x, label="net")
        return x, torch.stack([s[1] for s in tracker.snapshots if s[2] == "net"], dim=2)
    
    stage.forward = tracked_forward
    return tracker
```

**Usage during training:**
```python
tracker = ConvergenceTracker()
attach_tracker_to_stage(net.core.stages[0], tracker)

# Run one validation batch
with torch.no_grad():
    x_pred, _ = net(b_val, ctx_val)

# Plot
tracker.plot_residual_trajectory(A_val[0], b_val[0], x_val[0],
                                 save_path="convergence_epoch_400.png")
tracker.snapshots.clear()  # reset for next time
```

---

## 3. Jacobi / Conjugate Gradient Baseline Harness

Fair comparison requires matching the "compute budget." For the analog network, budget = ODE time steps. For digital solvers, budget = iterations.

```python
class DigitalSolverBaseline:
    def __init__(self, n, max_iters=50):
        self.n = n
        self.max_iters = max_iters
    
    def jacobi(self, A, b, x0=None, steps=None):
        """Standard Jacobi iteration. x_{k+1} = D^{-1}(b - R x_k)."""
        if steps is None:
            steps = self.max_iters
        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0.clone()
        
        D = torch.diag_embed(torch.diagonal(A, dim1=-2, dim2=-1))
        R = A - D
        D_inv = torch.inverse(D)
        d_inv = torch.diagonal(D_inv, dim1=-2, dim2=-1)
        
        residuals = []
        for _ in range(steps):
            x = (b - torch.bmm(R, x.unsqueeze(-1)).squeeze(-1)) * d_inv
            r = (torch.bmm(A, x.unsqueeze(-1)).squeeze(-1) - b).norm(dim=-1)
            residuals.append(r.mean().item())
        return x, residuals
    
    def gauss_seidel(self, A, b, x0=None, steps=None):
        """In-place Gauss-Seidel. Slightly better convergence than Jacobi."""
        if steps is None:
            steps = self.max_iters
        x = torch.zeros_like(b) if x0 is None else x0.clone()
        
        residuals = []
        for _ in range(steps):
            for i in range(self.n):
                sigma = torch.zeros(b.shape[0], device=b.device)
                for j in range(self.n):
                    if i != j:
                        sigma += A[:, i, j] * x[:, j]
                x[:, i] = (b[:, i] - sigma) / A[:, i, i]
            r = (torch.bmm(A, x.unsqueeze(-1)).squeeze(-1) - b).norm(dim=-1)
            residuals.append(r.mean().item())
        return x, residuals
    
    def conjugate_gradient(self, A, b, x0=None, steps=None, tol=1e-6):
        """
        Batch CG. Note: this is O(n) memory per step but requires
        matrix-vector products, same as your network.
        """
        if steps is None:
            steps = self.max_iters
        batch_size = b.shape[0]
        device = b.device
        
        x = torch.zeros_like(b) if x0 is None else x0.clone()
        r = b - torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
        p = r.clone()
        rs_old = (r * r).sum(dim=-1)
        
        residuals = []
        for k in range(steps):
            Ap = torch.bmm(A, p.unsqueeze(-1)).squeeze(-1)
            alpha = rs_old / ((p * Ap).sum(dim=-1) + 1e-12)
            x = x + alpha.unsqueeze(-1) * p
            r = r - alpha.unsqueeze(-1) * Ap
            rs_new = (r * r).sum(dim=-1)
            
            r_norm = rs_new.sqrt().mean().item()
            residuals.append(r_norm)
            if r_norm < tol:
                break
            
            beta = rs_new / (rs_old + 1e-12)
            p = r + beta.unsqueeze(-1) * p
            rs_old = rs_new
        
        return x, residuals


def compare_against_baselines(net, val_loader, n, device="cuda"):
    """
    Run one validation epoch and compare network vs. digital baselines.
    Returns a dict of metrics.
    """
    baseline = DigitalSolverBaseline(n, max_iters=50)
    metrics = {
        "net_res": [], "net_sol": [],
        "jacobi_res": [], "jacobi_sol": [],
        "cg_res": [], "cg_sol": []
    }
    
    net.eval()
    with torch.no_grad():
        for b_batch, x_star_batch, A_batch in val_loader:
            b_batch = b_batch.to(device)
            x_star_batch = x_star_batch.to(device)
            A_batch = A_batch.to(device)
            
            # Network prediction
            ctx = SimContext()
            x_net, _ = net(b_batch, ctx, tau=0.5)
            res_net = (torch.bmm(A_batch, x_net.unsqueeze(-1)).squeeze(-1) 
                       - b_batch).norm(dim=-1)
            sol_net = (x_net - x_star_batch).norm(dim=-1)
            metrics["net_res"].extend(res_net.cpu().numpy())
            metrics["net_sol"].extend(sol_net.cpu().numpy())
            
            # Jacobi (match network ODE steps)
            x_jac, _ = baseline.jacobi(A_batch, b_batch, steps=50)
            res_jac = (torch.bmm(A_batch, x_jac.unsqueeze(-1)).squeeze(-1)
                       - b_batch).norm(dim=-1)
            sol_jac = (x_jac - x_star_batch).norm(dim=-1)
            metrics["jacobi_res"].extend(res_jac.cpu().numpy())
            metrics["jacobi_sol"].extend(sol_jac.cpu().numpy())
            
            # CG (match network ODE steps)
            x_cg, _ = baseline.conjugate_gradient(A_batch, b_batch, steps=50)
            res_cg = (torch.bmm(A_batch, x_cg.unsqueeze(-1)).squeeze(-1)
                      - b_batch).norm(dim=-1)
            sol_cg = (x_cg - x_star_batch).norm(dim=-1)
            metrics["cg_res"].extend(res_cg.cpu().numpy())
            metrics["cg_sol"].extend(sol_cg.cpu().numpy())
    
    # Summarize
    print(f"{'Method':<12} {'Mean Residual':<18} {'Mean Sol Error':<18}")
    print("-" * 50)
    for method in ["net", "jacobi", "cg"]:
        mean_res = np.mean(metrics[f"{method}_res"])
        mean_sol = np.mean(metrics[f"{method}_sol"])
        print(f"{method:<12} {mean_res:<18.4e} {mean_sol:<18.4e}")
    
    return metrics
```

**Expected outcome after training:** The network should achieve lower residual error than Jacobi at matched step count, and approach or beat CG depending on how well the learned branch cells approximate the optimal preconditioner.

---

## 4. Quick Integration Checklist

To wire this into your existing codebase:

1. **Replace your XOR dataset** with `SparseLinearSystemDataset(n=48, ...)`.
2. **Replace topology generation** with `build_union_topology(train_ds, n=48, num_proj=4)`.
3. **Replace loss** with `compute_solver_loss(...)` using `residual_loss` + `solution_loss`.
4. **Add validation call** to `compare_against_baselines(...)` every 50 epochs.
5. **Optional:** Attach `ConvergenceTracker` during the first validation batch to generate convergence plots.

---

If you want, I can next generate:
1. **A pruning-to-compact-topology script** that takes trained gates and exports a pruned `SparseTopology` with dead edges removed,
2. **A stage-width sweep helper** to automate the search over `n ∈ {32, 48, 64}` and `num_proj ∈ {0, 2, 4}`,
3. **A distributed-training wrapper** if you want to parallelize the architecture search across GPUs.