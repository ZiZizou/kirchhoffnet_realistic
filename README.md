# KirchhoffNet (Realistic) — Rebuild-Grade README

> Purpose: give an LLM with **no codebase access** enough precise detail to
> reimplement KirchhoffNet (KNet) from scratch: the physics, the math, every
> important flag, all init values, and a minimal build order. Values below are
> the actual repo defaults (`config.py`, `cell_library.py`,
> `differential_stage.py`, `topology.py`, `io_mapper.py`, `kirchhoff_net.py`,
> `stage_transfer.py`, `train_script.py`).

> **Current NARMA research handoff:** for the active causal delay-bank
> reservoir problem, its measured Phase 5--8 results, constraints, and what an
> outside proposer should answer, read
> [`docs/narma-phase8-temporal-diversity-problem-statement.md`](docs/narma-phase8-temporal-diversity-problem-statement.md)
> alongside this README and the progress summary. The README describes the
> general machine; the handoff brief defines the current experiment.

---

## 1. What KNet is (one paragraph)

KNet is an **analog-inspired Neural ODE**: a sparse graph of differential
voltage nodes connected by programmable OTA-like edge devices. There are no
Transformer blocks, no convolutions, no MLP hidden layers in the core — the
**ODE integration itself is the computer**. A run has three phases
(**Write / Evolve / Read**): map the input vector into an initial voltage
state `x(0)`, integrate KCL current equations with fixed-step Heun through a
chain of `num_stages` graph stages (with small learned transfers between
stages), then read a slice of the final state through a learned affine readout
into `out_dim` logits. Training is direct BPTT through the unrolled Heun steps
(no adjoint by default; an experimental DEQ fixed-point path exists).

---

## 2. Core math (implement exactly this)

### 2.1 Node ODE

Per node `j`, with effective capacitance `C_eff = 1.0`:

```
C_eff * dx_j/dt = Σ_in I_edge - Σ_out I_edge
                + I_drive(j) + I_boundary(j) + I_temporal(j) + I_ref(j)
                - leak_j * x_j - clip(x_j)
```

- `leak_j`: see §9 (programmable per-node `softplus(raw_leak)`, init
  `raw_leak = -3.0` → `leak ≈ 0.0486`; or fixed scalar `0.0486`
  when `--leak non-programmable`).
- Soft rail clip (always on):
  `clip(x) = 0.05 * (σ((x - x_max)/0.02) - σ((-x - x_max)/0.02))`,
  default `x_max = 3.0` (CTLE/distillation runs use `4.0`; BO searches
  `x_max ∈ [0.5, 8.0]`).
- KCL scatter: each directed edge `(src → dst)` carrying current `I_e`
  adds `+I_e` to `dst` and subtracts `I_e` from `src` — **except**
  temporal-readout and reference edges, which inject into the destination
  only (source is read-only, not drained). Shared-sense currents (§6.2–6.3)
  never enter KCL scatter at all: they are mixed into the accumulator tail
  through the crossbar `W`.

### 2.2 Edge current (tanh_free cell — the standard cell)

`--cell-library tanh_free` (class `FreeTanhLibrary`) per directed edge `e`:

```
u_e   = gm_e * (s_e * (A_e * Vsrc - B_e * Vdst) + theta_e)
I_tanh_e = Isat_e * (tanh(u_e) + 0.01 * u_e)     # 0.01 leak slope = gradient floor
I_res_e  = G_e * (Vsrc - Vdst),  G_e = softplus(g_raw_e)   # resistive shunt, NEVER gated/frozen
I_e      = gate_e * budget_gate_e * compliance_e * I_tanh_e + <VCA/shunt/GLN terms, §10–§12> + I_res_e
```

Parameterization per edge (all shape `[E]`):

| Param | Mapping | Init |
|---|---|---|
| `a_raw` | `A = softplus(a_raw) ≥ 0` | `0` (≈0.693) |
| `b_raw` | `B = softplus(b_raw) ≥ 0`, independent of A | `0` |
| `s_raw` | `s = sign(s_raw)` ∈ {±1}, STE backward (`s + s_raw - s_raw.detach()`) | `N(0, 0.1²)` (must be non-zero) |
| `gm_raw` | `gm = gm_min + (gm_max-gm_min)·σ(gm_raw)` | `-5.0` (≈0.077 when range is [0.01, 10]) |
| `isat_raw` | `Isat = clamp(gm-style…, min 1e-6)` | `-5.0` |
| `theta_raw` | additive bias, **only if** `bias_enabled=True` (default False) | `0` |
| `g_resistive_raw` | `G = softplus(g_raw)` | `-5.0` (≈0.0067) |
| `z_logits` | structural gate `gate_e = σ(z_logits_e)` | `0.0` (σ=0.5, max sigmoid slope — deliberate so gates can learn) |

- `gm_min/gm_max`, `isat_min/isat_max` default `0.01 / 10.0`
  (`config.TANH_REALISTIC_*`); BO searches `gm_max/isat_max ∈ [1, 50]`.
- **Compliance gating** (all cell libs): per-endpoint
  `gate_src = σ((x_max - |Vsrc|)/0.02)`, same for dst; multiply the tanh
  branch only. The resistive shunt is intentionally ungated.
- Other cell libs (same interface, rarely used): `simple` (relu/tanh of
  `p0·Vsrc+p1·Vdst+p2`, 3 params/edge), `tanh_realistic`
  (`tanh(A·Vsrc-B·Vdst+C)`, `A+B=1`), `tanh_realistic_upgrade`
  (`Isat·tanh(gm·(…))`), `tanh_anti` (rectified antiparallel slice).
  `make_cell_library(name)` factory builds any of them sized for `E` edges.

### 2.3 Heun integration (the only default solver)

`SOLVER = {method: heun, t_span: 7.0, num_steps: 70}`. Fixed-step
predictor–corrector, 2nd order, unrolled for BPTT:

```
x_{n+1} = x_n + dt/2 * (rhs(x_n) + rhs(x_n + dt*rhs(x_n))),  dt = t_span/num_steps
```

- `t_span`/`num_steps` are **split equally over stages**: per-stage
  `t = 7.0/num_stages`, `steps = round(70/num_stages)`. With 5 stages:
  `t≈1.4, steps=14, dt=0.1`.
- Distillation/BO adds `--kn-solver-scale s ∈ [0.5, 2.0]`
  (param-count-neutral): per-stage `t *= s`, `steps = round(steps*s)`.
- `--solver deq` switches to a fixed-point solver
  (`Phi(x) = x + dt·rhs(x)`, Anderson, `deq_step=0.1`, `leak_floor=0.05`,
  `f_max_iter=30, f_tol=1e-4`); Heun path uses `leak_floor=0.0`.
- `--core-refresh-interval k` (NARMA work): recompute the frozen core
  current every `k` Heun steps (`k ≥ num_steps` = legacy frozen,
  `k = 1` = fully dynamic).

---

## 3. Pipeline: Write / Evolve / Read (`KirchhoffNetWithIO`)

```
y, trajs = net(u)                      # u: (B, in_dim)
  x0_full  = assemble(write(u))        # §4; zeros for proj/output-ODE tail
  x_final  = core(x0_full)             # stages + transfers, Heun
  x_read   = slice/output-ODE region   # §6
  y        = post_readout_transfer(output_mapper(x_read))  # §6 + §8
  (+ skip_linear(u) if enabled)
```

- State width per stage: `N = hid_count + proj_count + output_ode_count`.
- Multi-stage with width changes uses `StageTransfer` between stages (§8);
  **boundary / temporal-readout / drive modes require all stages to have
  equal width** (constructor raises otherwise).
- `t_span=0` everywhere = mapper-only ablation (no integration).

---

## 4. Write path (input → `x(0)`)

Exactly one is active per build; chosen by `write_mode` / boundary flag:

| Mode | Class | Formula | Params |
|---|---|---|---|
| `dense` (default) | `InputMapper` | `x(0) = x_max·tanh(W·u + b)` | `in·hid+hid` |
| `robust` | `RobustInputMapper` | `x(0) = x_max·tanh(W·(u·e^{log_scale}) + b)` | + `in_dim` scales |
| `one_to_one` | `SparseInputMapper` | `x_{write_idx[i]}(0) = x_max·tanh(a_i·u_i + b_i)`, rest 0 | `2·in_dim` |
| `fan_out` | `FanOutInputMapper` | `x_{fan_out_map[i][k]}(0) = x_max·tanh(g_{i,k}·u_i + b_{i,k})` | `2·K_total` |
| `sparse_proj` | `ProjectedSparseInputMapper` | `x_{write_idx}(0) = x_max·tanh(Linear(u))` (one feature → many nodes) | `in·len(write_idx)+len` |
| `residual_tanh` encoder | `ResidualTanhInputMapper` | residual-tanh MLP encoder, no `x_max` saturation | `in·h+h·hid` |
| **boundary** | `NullInputMapper` | returns zeros; signal enters via boundary OTAs (§5) | 0 |

- Dense init: Xavier × `gain_scale=1.0`, bias 0. Sparse/fanout: gains
  init `1.0`, biases `0`.
- Only the **hidden** portion is written; proj/output-ODE slots are zero-init.
- Distillation harness preprocessing (before the mapper): 4-dim specs →
  `logs=log10(specs)`, `scaled = 2·(logs-lo)/(hi-lo) − 1`, clamp to
  `input_rail=4.0`; `lo/hi` = dataset log-min/max ±5% pad.

---

## 5. Boundary fan-out (the input path for all BO/CTLE runs)

- Flag: `--boundary-fan-out '{"0":[0,4],"1":[1,5],...}'`
  (JSON input-index → hidden-target list). Enables `enable_boundary=True`.
- Each pair `(src terminal i → hidden node j)` is a **sparse OTA edge**
  with its own cell params + structural gate `σ(boundary_z)` (init 0 → 0.5)
  + VCA gate (§10). Ideal voltage source: terminal voltage = scaled input,
  never loaded.
- `I_boundary = σ(z)·VCA_bound(u)·I_OTA(u[i], x[j])`, injected into `x[j]`.
- The standard round-1 map is the diagonal `{i:[i,i+4]}, i=0..3`
  (8 targets; needs `hidden ≥ 8`; the older validated map needs `hidden ≥ 13`).
- Constraints: targets unique across inputs, in `[0, hid_count)`;
  `x0 = zeros` in boundary mode (mapper bypassed); boundary family has its
  own `boundary_cell_lib` sized `len(pairs)`.
- Param cost ≈ 8 targets × (7 cell + 1 gate [+ VCA share]).

---

## 6. Read path (state → logits)

`--readout` (`train_script.py`, default `temporal`) selects one of four
readouts. The three OTA-tail families (`temporal`, `shared`, `shared-x2`)
all append `output_ode_count = out_dim` accumulator nodes per stage (state
tail, zero-init, leak+clip like regular nodes) and read the tail through
`OutputAffine` (below) + `post_readout_transfer`. `linear` has no tail and
no `OutputAffine`. (`--temporal-readout` is a deprecated alias for
`--readout temporal`; combining it with any other `--readout` choice is a
hard error. Distillation/BO harnesses mirror `--readout` as `--kn-readout`.)

1. **Temporal mesh** (`--readout temporal`, ON in all BO/CTLE runs):
   every hidden node → every accumulator has its own one-way OTA edge
   (source read-only, never drained; `hidden·out_dim` edges/stage,
   e.g. 10·7=70). Each edge has its own cell (7 params) + structural gate
   `σ(z)` (init `0` → `0.5`) + VCA tap. Requires equal stage widths;
   incompatible with `decoder_type='residual_tanh'` and `grouped_readout`.
2. **Shared sense** (`--readout shared`): one FreeTanh sense OTA per hidden
   node (`s·h` senses/stage, `s=1`), computing
   `I_OTA(x_src_j, Vref)` against a per-stage learnable
   `Vref = σ(raw_vref_sense)·x_max` (init `raw=0` → `Vref=x_max/2`, an ideal
   source — never drained). The hidden node is read-only. Sense currents
   are mixed into the accumulator tail by a plain dense crossbar
   `W[d_out × s·h]` (init `N(0, 1/√n_sense)`, no gates, no VCA on the taps):
   `ȧ_i = Σ_j W[i,j]·I_j − leak_i·a_i − clip_i(a_i)`.
   Each sense has its own cell + gate `σ(readout_sense_z_logits)` (init 0 →
   0.5, multiplies both the tanh and the resistive path); VCA
   (`vca_v_readout`) gates the sense currents only, never the crossbar.
3. **Shared-x2** (`--readout shared-x2`): `s=2` senses per hidden node +
   `W[d_out × 2h]`. The `k=1` sense of every node gets a `+0.5` jitter on
   `gm_raw` at build (senses stored interleaved, odd indices) so the two
   senses on one node start at different saturation points; the `k=0` sense
   stays at the default `-5.0` low-current init. Otherwise identical to
   `shared`.
4. **Linear** (`--readout linear`): pre-refactor default.
   `OutputMapper = Linear(read_dim → out_dim)` over proj nodes (or hidden
   if no proj, with warning R1.4). No accumulator tail, no `OutputAffine`.
5. Mapper variants (orthogonal to the above): `read_idx` gather,
   `GroupedOutputMapper` (per-group linears), `ResidualTanhOutputMapper`
   (residual-tanh encoder head).

Param cost per stage: temporal ≈ `h·d_out` × (7 cell + 1 gate);
shared/shared-x2 ≈ `s·h` × (7 cell + 1 sense gate) + `d_out·s·h` crossbar
weights + 1 `raw_vref_sense` scalar (+ shared VCA/GLN taps, §10–§11).

Freeze semantics: `--freeze-temporal-read` freezes the tanh contribution of
whichever readout family is active. In shared mode the sense tanh (+ sense
gate + VCA gate folded in) is precomputed once from `x0`; the sense
resistive shunt is recomputed every Heun substep (same convention as the
temporal mesh). Gotcha: `prune_stage` does not support shared-sense
readout (raises — use the OTA mesh for pruning runs).

`OutputAffine` (all three OTA-tail paths, per-dim `o`, identity at init):

```
y_o = gain_o·x_o + bias_o + tanh_gain_o·tanh(x_o) + relu_gain_o·ReLU(x_o − thr_o)
init: gain=1, bias/tanh_gain/relu_gain/thr = 0
```

`post_readout_transfer` (always-on since 2026-09-01, `StageTransfer
residual-relu-tanh` on the 7-dim readout, identity at init, 28 params):

```
y = w1·y + w2·tanh(y) + w3·ReLU(y − vth),  init w1=1, w2=w3=vth=0
```

Physical heads then apply `param = lo + (hi−lo)·σ(logit)` for CTLE
(7 log-bounded physical params), or use logits directly (Friedman).

---

## 7. Graph families (`topology.py`)

Per-stage hidden graph, single directed edge per pair by default
(`bidirectional=False`; L/S odd cells conduct both ways by sign reversal):

- `small_world(n, k=4, p=0.2, seed)` — Watts-Strogatz ring + rewiring
  (THE standard; `k` even, `2 ≤ k < n`, cap 14). E ≈ `n·k/2`.
- `line_graph(n, radius)`, `ring_graph`, `grid_graph(h, w, k=3)`,
  `torus_graph`, `empty_graph`.
- `edge_repeats=2`: duplicate every hidden edge with independent params
  (100 core edges/stage at n=25, k=4).
- `connect_bipartite / connect_projection`: hidden↔proj wiring when
  `proj_count > 0` (usually 0 in BO runs).
- `MultiStageTopology.from_config()` + `topology_to_stage()` remap global
  IDs → compact `0..N-1`; `build_net_from_config(cfg, …)` is the single
  constructor everything (training, BO analytic counter, harness) must share
  so param counts stay exact.

---

## 8. Inter-stage transfer (`StageTransfer`)

Between stages (width change and/or nonlinearity), `activation=`:

- `none`: truncate/pad. `relu`: `ReLU(x)`.
- `residual`: `w1·x + w2·tanh(x)`, init `w1=1, w2=0`.
- `residual-relu-tanh` (standard): `w1·x + w2·tanh(x) + w3·ReLU(x−vth)`,
  init `1/0/0/0` = identity. `drive_mask` nodes pass through unchanged.
- `residual_mixing`: + zero-init full/low-rank cross-node mixing term.

---

## 9. The five flags you must get right

1. **`--freeze-read` (`freeze_read`, `DifferentialStage`)** — core tanh
   current (+ edge gate + VCA gate folded in) computed **once from `x0`**
   per stage and held constant across all Heun substeps; resistive shunt,
   leak, clip, drive stay dynamic. Bigger stable `dt`, fewer dynamics.
   Default False in `train_script`, **True in all BO/distillation runs**.
2. **`--freeze-boundary` / `--freeze-temporal-read`** — same freeze idea
   for the boundary / temporal-readout tanh families (resistive shunts stay
   dynamic). Independent of `freeze_read`; no-op without that family.
3. **`--readout`** — §6. `temporal` = legacy `h·d_out` OTA mesh;
   `shared` / `shared-x2` = `s=1/2` sense OTAs per hidden node against a
   per-stage Vref rail + dense crossbar into the same accumulator tail;
   `linear` = pre-refactor `OutputMapper`, no tail. Changes param count and
   the read slice (tail accumulators for the three OTA families, hidden/proj
   for `linear`). (`--temporal-readout` is a deprecated alias for
   `--readout temporal`.)
4. **`--boundary-fan-out`** — §5. Selects boundary mode (NullInputMapper,
   `x0=0`). The map is part of the model: it changes params and must be
   identical in sampler, counter, and trial.
5. **`--leak`** — `programmable` (per-node `softplus(raw_leak)`, init −3.0)
   vs `non-programmable` (fixed `0.0486 = softplus(−3.0)`, saves
   `num_nodes` params/stage). BO/CTLE use non-programmable.

Plus: `--vca/--vca-core/--vca-separate-core-bus/--vca-gate-shunt/--vca-bias`
(§10), `--gln-rails/--gln-B/--gln-rank/--gln-alpha-init/--gln-families`
(§11; requires `--cell-library tanh_free`, readout family needs
`shared/shared-x2`), `--x-max` (rail), `--t-span/--num-steps` (horizon; per-stage split),
`--cell-library` (default `tanh_free`), `--hidden-family/--num-hidden/
--num-stages/--small-world-k/--edge-repeats`, `--interstage-activation`,
`--skip-linear` (adds `y += Linear(u)`), `--enable-drive`
(persistent per-node current, needs sparse write modes),
`--mapper-lr-scale/--struct-lr-scale/--dyn-lr-scale`
(defaults `0.1/2.0/1.0`; BO reference uses `1.0/4.0/1.0`),
`--sparsity/--entropy/--device-l2-lambda` regularizers,
`--gm-max/--isat-max` cell bounds, `--core-refresh-interval`,
`--enable-ref-edges` (per-node OTA to learned `Vref ∈ [0,x_max]`,
init `x_max/2`), `--budget*` (degree-budget competition).

---

## 10. VCA (low-rank input-driven gating)

- Per gated edge: `gate_e = 2·σ(b_e + (uᵀ·W)·v_e)`, `W: (in_dim×rank)`
  shared projection (zero-init → gate exactly 1.0 at epoch 0), `v_e: (rank)`
  per-edge embedding (`N(0, 0.01²)`), optional bias `b_e` (`--vca-bias`).
  Range (0,2): amplify AND attenuate.
- Buses: one shared `W` for boundary+readout; `--vca-separate-core-bus`
  gives core edges their own `W_core`. `--vca-core` also gates core edges
  (auto-on if no boundary/readout family exists). `--vca-gate-shunt`
  extends gating to the resistive shunt (default OFF — shunt untouched).
- Compatible with freezing (`u` constant per sample → gate computed once
  at stage entry).

---

## 11. GLN rails (input-conditioned log-gm modulation, F2)

GLN (`gln_rails.py`, flags `--gln-rails --gln-B --gln-rank
--gln-alpha-init --gln-families`) is the analog of programmable bias
currents derived from the input: `B` shared tanh half-space rails modulate
the **log-gm** of the boundary and readout-sense OTA families. The static gm
from the cell library's bounded sigmoid map becomes the *base* `gm0`; rails
add a log-space correction:

```
z_b      = tanh(alpha_b * (a_b·u + c_b)),  alpha_b = softplus(alpha_raw_b) > 0
delta_e  = Σ_b W[e,b] * z_b,               W = P @ Q  (per family, [E × B])
gm_e     = clamp(gm0_e * exp(delta_e), gm_min, gm_max)
```

- **Scope (v1):** one shared `GLNRails` core `(a, c, alpha_raw)` for the
  whole net (single module, tied across stages) + one factorized edge mix
  `P[E,rank] @ Q[rank,B]` per family (also tied across stages). Families:
  `boundary` (`E` = #boundary pairs) and `readout` (`E` = `s·h` sense OTAs).
  **Core hidden edges and resistive shunts are never gated.**
- **Identity init (load-bearing):** `Q = 0` exactly so `W = P @ Q = 0` →
  `gm = gm0` and the epoch-0 forward is bit-identical to the no-GLN model.
  The init is deliberately *asymmetric* so training is not a dead saddle:
  `a ~ N(0, 0.01²)`, `P ~ N(0, 0.01²)`, `c = 0`, `alpha_raw =
  softplus⁻¹(alpha_init)` (default `alpha_init = 1.0`). Rails output a small
  nonzero `z` at startup, so `d(gm)/dQ = P·z ≠ 0`: Q gets gradient on step 0,
  and once Q moves, P/a/c/alpha all receive signal. (Zeroing both `a`/`z`
  AND `W` would freeze every GLN gradient at zero forever.)
- **Mechanism:** the stage calls the cell library with `gm_override`
  (per-sense/per-boundary modulated gm, shape `[B, E]`) instead of the
  static sigmoid map; the parallel tanh-multiplier branch (unused by
  `tanh_free` without that option) is untouched. `gm_min/gm_max` are the
  same rails as the F1 cell bounds — the BO-searched `gm_max` stays the
  clamp ceiling *after* modulation.
- **Ordering vs VCA:** GLN acts *inside* the tanh argument (changes gm);
  VCA still multiplies the resulting current *outside*. Compliance gating
  is unchanged. Rails `z` are computed once per stage entry (`u` is constant
  per sample), so GLN is compatible with `freeze_read` / `freeze_boundary` /
  `freeze_temporal_read` at zero extra Heun-loop cost.
- **Constraints (hard errors in `build_net_from_config`):** requires
  `--cell-library tanh_free` (only `FreeTanhLibrary` exposes `gm_override`);
  the `readout` family requires `--readout shared/shared-x2` (there is no
  sense bank in mesh/linear mode); the `boundary` family requires
  `--boundary-fan-out`; at least one family must be listed.
  Defaults: `--gln-B 4 --gln-rank 2 --gln-families boundary,readout`.
- **Param cost** (`gln_param_count`, exact — one module net-wide, NOT
  per-stage): rails `B·(in_dim + 2)` (`a`, `c`, `alpha_raw`) + per family
  `E·rank + rank·B` (`P`, `Q`). Example: `in_dim=4, B=4, rank=2`,
  8 boundary pairs, `s·h=20` senses → `4·6 + (8·2+2·4) + (20·2+2·4) = 96`.
  Counted by `--count-params-only`; BO budgets include it in the feasible
  tuple (fixed per study, not a sampled dim).
- **Example:** add `--gln-rails --readout shared-x2` (plus optional
  `--gln-B 4 --gln-rank 2 --gln-families boundary,readout`) to the §14
  reference command.

---

## 12. RHS pseudocode (one Heun evaluation)

```python
def rhs(x, u):
    z = gln_rails.rails(u) if gln else None          # once per stage entry
    acc = zeros_like(x)
    acc += frozen_core_const if freeze_read else scatter(σ(z)*I_tanh_core(x)*VCA_core(u))
    acc += scatter(G_core*(Vsrc-Vdst))              # resistive, always dynamic (never VCA/GLN-gated)
    acc += scatter_boundary(u, x; gm=GLN_bound(gm0,u,z))*VCA_bound(u)  # frozen if freeze_boundary
    acc += scatter_temporal(x)*VCA_readout(u)       # mesh readout; frozen if freeze_temporal_read
    acc += crossbar(W, sense(x, Vref; gm=GLN_read(gm0,u,z))*sense_gate*VCA_readout(u))
                                                    # shared/shared-x2 readout instead of mesh
    acc += I_ref + I_drive                          # if enabled
    return (acc - leak*x - clip(x)) / C_eff
```

---

## 13. Losses, regularizers, optimizer

- Task losses: Huber (Friedman), MSE on teacher logits (distillation),
  ZIG-gated failure rate for CTLE model selection (not a gradient loss).
- Regularizers (`config.LAMBDAS`, scaled by `--*-lambda`):
  `edge_gate λ·Σσ(z)` (active-edge count), `power λ·Σ…` (static-power
  proxy), `entropy`, `tanh_sat = mean(tanh(u)²)`, `rail 0.1`,
  `skip_linear_l2 1e-4`. Node-gate/capacitance are deprecated (0.0).
- Optimizer: `AdamW`, cosine/restart schedules in tasks; **three LR groups**
  in `train_script`: mapper (`InputMapper/OutputAffine/post_readout`,
  `--mapper-lr-scale`), struct (`z/u` logits, `--struct-lr-scale`),
  dynamic (cells/leak/drive/VCA, `--dyn-lr-scale`). Fixed-distillation
  harness uses a single LR (known asymmetry vs Friedman BO).
- Param counting: `sum(requires_grad)` after build; `--count-params-only`
  prints `TRAINABLE_PARAMS=N`. BO preflight rejects out-of-window arches
  with finite penalty `(1.0, 1e6)` (never inf/None — keeps MO-TPE alive).

---

## 14. Key configs and entry points

- `config.py`: `SOLVER/DEQ/PHYS/REF/OPTIM/TAU/LAMBDAS/PRUNE/SCHEDULE_* /
  DEGREE_BUDGET/VCA/DRIVE/INIT/CELL_LIBRARIES/PRESETS`.
- `train_script.py`: generic trainer (all flags in §9 + data/schedule/prune
  flags). `kn_bayes_opt.py`: 15-dim TPE over
  topology+solver+optimizer+physics (+CTLE successive-halving).
  `mlp_bayes_opt.py`: budget-exact MLP baseline BO.
  `fixed-mlp-distillation-kirchhoffnet.py` + `fixed-distillation-bayes-opt.py`:
  frozen-teacher distillation + MO-BO `(ZIG-fail, logit-MSE)`.
  `ctle_dagger_common.py`: shared CTLE/DAgger infra (`PlainMLP`,
  `RegimeAwareMoE`, flow teacher, ZIG evaluator).
- Reference commands:
  `train_script.py --problem smooth2d --solver heun --epochs 600
  --cell-library tanh_free --hidden-family small_world --num-hidden 25
  --num-stages 5 --small-world-k 4 --small-world-p 0.2
  --boundary-fan-out '{"0":[0,1,2,6,8],"1":[3,4,5,7,9]}'
  --interstage-activation residual-relu-tanh --freeze-read --temporal-readout
  --leak non-programmable --vca --vca-core --vca-separate-core-bus`
- Worked sizes: smooth2d ref ≈ 3500 trainable (100 core·7 + boundary +
  temporal + VCA + OutputAffine per stage ×5, leak fixed). CTLE KNet
  `hidden=10, stages=4, k=4, rank=4` = 5959 (diagonal fan-out 8 targets).
  MLP param formula: `5W + (L−1)(W²+W) + 7W+7 + 2WL (if LN)`
  (45×3+LN = 4957; 48×3 no-LN = 5287).

---

## 15. Minimal rebuild order

1. `config.py` constants (SOLVER/PHYS/INIT/VCA tables above).
2. `cell_library.py`: `FreeTanhLibrary` (§2.2) + compliance gate +
   `gm_override` path (GLN, §11).
3. `topology.py`: `small_world_graph` + `repeat_edges` + remap.
4. `differential_stage.py`: COO KCL + VCA + freeze variants + Heun loop +
   shared-sense bank (`readout_sense_*`, crossbar `W`, `raw_vref_sense`).
5. `io_mapper.py`: Null + sparse/fanout + `OutputAffine`.
6. `stage_transfer.py`: residual-relu-tanh.
7. `kirchhoff_net.py`: `KirchhoffNetWithIO` (§3–§6).
8. `build_net_from_config()` wiring boundary/temporal/readout (§5–§6) +
   shared `GLNRails` module (§11).
9. Train with AdamW + Heun BPTT; count params; add BO over §9 flags.

Gotchas: equal stage widths required for boundary/any-OTA-tail/drive;
any OTA-tail readout contradicts `residual_tanh` decoder/`grouped_readout`;
freeze flags are no-ops without their family; `small_world_k` even and
`< num_hidden`; VCA needs ≥1 gated family; GLN readout family needs
shared/shared-x2 and GLN needs `tanh_free`; `prune_stage` rejects
shared-sense; post_readout is identity at init — verify epoch-0
bit-equality when adding it (same for GLN: `Q=0` → `gm=gm0`).

---

## 16. NARMA experiment architecture (probes are eval-only by default)

This section describes the *apparatus* only — task, twins, drives,
instruments, entry points. Measured numbers, verdicts, and decisions
live in `docs/narma-progress-summary.md`; the active brief is the
handoff linked at the top of this file.

### 16.1 Task and data pipeline (`narma_experiment.py`)

- NARMA-10 recurrence (`narma()`): uniform input `u ∈ [0, 0.5]`,
  `y[n+1] = 0.3·y[n] + 0.05·y[n]·Σ_{i=0..9} y[n-i] + 1.5·u[n-9]·u[n]
  + 0.1` (NARMA-20 variant uses `u_max = 0.2`). First `order` samples
  are generator washout and discarded.
- Train streams: `_gen_narma_train_streams(order, seed, n_streams, n)`
  with per-stream seed `seed + b*7919`. Test stream: `_gen_narma_test`
  = 1000 samples at `seed + 10000` (fresh seed, never trained on).
- Drive scaling: `scale_input_to_rails` maps `[0, u_max]` to rail volts
  — bipolar (default) `→ [-x_max, +x_max]` so boundary OTAs sit at
  mid-rail; `_scale_drive` then applies the E4 drive-scale factor.
- Baselines in-harness: `RidgeRegressor` (tapped-delay ridge, l2 1e-2),
  `ESN` (25-node tanh reservoir, spectral radius 0.9, leak 1.0, ridge
  readout l2 1e-2), `MLPRegressor`, `LSTMRegressor`.

### 16.2 Digital twins (`narma_linear_controls.twin_node_tanh`, `post_twin_gates.py`)

Two standalone maps stand in for fabric dynamics (no stage/cell
mutation, no training — pure forward simulation + offline ridge fits):

- Discrete twin: `x' = (1-a)·x + a·tanh(W_mix@x + W_in@d)`, `W_mix`
  Gaussian spectral-radius-normalized, `W_in` Gaussian, 25 nodes.
- Heun flow twin: `dx/dt = -λ·x + λ·tanh(W@x + W_in@d)` with the drive
  held constant inside each per-sample window (`t_span = 1.0` per
  NARMA step, fixed-dt rule `dt ≤ 0.125`). Pinned Gate-1 operating
  point: radius 1.0, `λT = 1.0`, drive RMS 0.2.
- Drive harnesses: `taps8` = 8-tap causal delay bank (tap 0 = current
  input), columns RMS-normalized to the drive scale; `scalar` =
  single-column `u·scale`. Same streams/seeds across all legs.

### 16.3 Aux-core drive probe (`aux_core_probe.py`)

Strict-feedforward input stage replacing the static tap bank as main
drive (no main→aux path; fabric integration is gated on a passing
verdict and never started speculatively):

- Flavors: `echo_state` (8-node ESN, radius 0.95, leak ∈ {0.3, 1.0}),
  `shift_register` (exact chain `x[t][i] = gain^i·hi·u[t-i]`, gain ∈
  {0.7, 0.9}, head injection 1.0), `slow_feature` (8 decoupled leaky
  integrators, τ log-spread over [1, 40]), `residual_distilled` (echo
  core selected by fit to a target-blind tap-minus-main residual
  teacher built with offline SVD/ridge only).
- Legs: `aux_only` (8-wide drive), `aux_scalar` (8 + scalar column),
  `aux_concat` (8-wide drive, readout over `[main(25); aux(8)]`).
  Drive-power parity: every leg scaled to drive RMS == input scale.
- Device-budget accounting per row: `aux_recurrent + aux_fan_in +
  main_fan_in` vs reference `hidden_dim × n_taps` (200 at default
  dims). D2 variant: sparse main fan-out (k ∈ {3, 6} nonzeros per
  drive column, Gate-3 builder reuse). D3 reference: `tapsvd` = SVD
  top-4 PCs of the tap bank driving the main core at parity.
- Pre-registered pass/fail verdict rules per flavor (tap-subspace
  copy flag, void/dead guards, scalar-gain and taps-match paths);
  see the spec, not this README, for thresholds and outcomes.

### 16.4 Instruments (no new metrics; shared with the twin harness)

Ridge NRMSE/R² (l2 1e-2 + bias, washout 200), washout-corrected
per-delay memory-capacity curve + total (max delay 20), participation
ratio of the washed state covariance, Jacobian eigenvalue spectra at
observed transitions (analytic Heun composition, autograd-checked),
tanh-engagement guard (median `|W@x + W_in@d|`), in-harness ESN parity
on identical streams. Probe outputs land in the leg output dir next
to (never inside) training tables.

### 16.5 Entry points

- `post_twin_gates.py {heun,sparsity,mismatch,w_in,all}` — Heun
  flow-twin sweep + sparsity/mismatch/W_in gates + buildable-spec
  emission; `--max-corners` grid cap, `--dt` fixed-dt guard.
- `aux_core_probe.py {sweep,smoke}` — aux sweep with `--flavors`,
  `--legs`, `--aux-seeds`, `--data-seed`, `--fanout-kinds`,
  `--max-corners`.
- Audits: `audit_post_twin_gates.py`, `audit_twin_standalone.py`,
  `audit_aux_core.py` (run with `python -B`; all must print their
  `*_OK` trailer).
- Specs: `docs/specs/aux-core-probe.md` (aux contract),
  `docs/specs/twin-probe-spec.md`, `docs/specs/flow-sparse-gates.md`;
  verdicts/decisions: `docs/narma-progress-summary.md`.


## Repository layout, branches and checkouts (Nov 2026 refactor)

**Branches** (all share the `archive/full-snapshot-2026-11` history; `simplified` is the pre-snapshot GitHub branch):

| Branch | Checkout on ssr0 | Purpose |
|---|---|---|
| `hw-refactor` | `~/Documents/kirchhoffnet_refactor` | Main work: simplification and the Verilog-A / Cadence refactor |
| `narma-experiments` | `~/Documents/Narma_experiments` | All NARMA code, docs and batch scripts, `dynamic_leak` goes here |
| `deq-experiments` | `~/Documents/DEQ_experiments` | DEQ solver (restored from `7b34b60^`), see `docs/DEQ_BRANCH_NOTES.md` |
| `archive/full-snapshot-2026-11` | (none) | Snapshot of every script, doc and test that was previously git-ignored, plus the `ctle_kirchhoff_bo.py` adjoint/dopri5 variant (not the paper HR-KNet) |

The two experiment checkouts are separate clones: their `origin` is GitHub (`https://github.com/ZiZizou/kirchhoffnet_realistic.git`) and the remote `local` points at `~/Documents/kirchhoffnet_refactor`. Nothing has been pushed to GitHub.

**Nested repo:** `kirchhoffnet/` is a clone of <https://github.com/zhengqigao/kirchhoffnet.git> (HEAD `7024072aa115099c3ff781bc7292d33c9cfcf5f0`). It is git-ignored and not vendored; clone it again if needed.

**Logs:** `*.out` and `*.log` are git-ignored. Key paper logs are kept in `docs/logs/` (paper run `run_train_knet_single_shot_distillation_2.sh-59014723.out`, its sibling run, and `trial_0019_log.txt`). Audit reports are in `docs/analysis/`.

**Assets:** `assets/teachers/trial_0019/` is the PlainMLP teacher (W96, L3) used by the paper run. `assets/baselines/fixed_mlp_knet/` is the loadable `fixed_mlp_knet` CTLE checkpoint (golden baseline). Both are committed, via `!assets/**` exceptions in `.gitignore`. The CTLE dataset `augmented-cvae-ctle/` is an ignored on-disk asset.

**Shell tip:** the ssr0 login shell is tcsh. `2>&1` and `2>/dev/null` fail there with "Ambiguous output redirect". Wrap commands in `bash -c '...'` or pipe an LF-only script to `bash -s`.
