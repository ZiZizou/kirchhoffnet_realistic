# Spec: core-vca-feature

Scope: feature

# Core VCA Expansion + Boundary/Readout VCA Init Fix

## Context

Low-rank input-driven VCA gating in `DifferentialStage` has a URI of regression: gate = sigmoid(z) with z init near 0 gives gate≈0.5 at epoch 0 (a −6 dB gain cut on the entire I/O path). Measured regression: friedman2 baseline best val 6.65e-5 → VCA 1.11e-4; housing 0.1159 → 0.1329. The optimizer must simultaneously learn input-dependent modulation and climb out of the init-hole through a multiplicative gradient path coupling W and v_e.

This spec defines (a) the fix that makes VCA epoch-0 bit-identical to VCA-off, and (b) the extension of VCA to core (hidden) edges, which is a well-motivated placement on static tasks: under freeze_read, core tanh currents are computed once per sample, so the gate folds into the existing precompute with zero extra Heun-loop cost. The resulting per-node bias becomes b_j(u) = sum over edges — literally a one-hidden-layer sigmoid network from inputs to per-node bias, real function-class addition on friedman2.

The goal: apply VCA with initial conditions (input) and learn the input-dependence once the init is correct.

## Requirements

### R1 — Identity at initialization (non-negotiable gate)

The low-rank gate per edge is `gate_e = 2 · σ(z_e)` with `z_e = (u @ vca_W) @ v_e.T`. At init `vca_W` must be all-zero so `z_e = 0` for every edge and every family. `gate` init `= 1.0` → epoch-0 forward/backward must be bit-identical (float precision, ≤1e-12) to the VCA-off baseline. Embeddings (`vca_v_*`) stay random ~N(0, VCA['scale_init']).

This is the correct null hypothesis: any deviation from baseline is then attributable to learned modulation, not to an init-time gain cut.

### R2. Shared VCA bus

One set of `vca_rank` global control buses (`vca_W`, shape [in_dim, rank]) per stage, shared by all gated edge families (boundary, readout, core). Each family taps with its own `v_c` embeddings. Gradients to `vca_W` sum across all families — this is expected and desired (correct identity).

### R3. Core-edge VCA parameter

- Per stage, add `vca_v_core : [n_core_edges, rank]` where `n_core_edges = src.numel()`. Random init ~N(0, scale_init) per R1 (embeddings stay random).
- With W all-zero at init: `dL/dW` = u^T ⊗ v_e is nonzero at step 0 for every family whose embeddings are random. For a core-only run (boundary/readout off), `vca_W` still receives gradient because `vcd_v_core` is random.
- Grad check: `vca_W.grad` nonzero at step 0; `vca_v_core.grad` becomes nonzero after a few optimizer steps (it is ~0 at step 0 by design — dz/dv_e = u@W = 0).

### R4. Gate computation hoisted (never inside a Heun sub-step)

`u` is constant per sample: compute the core gate once per sample per stage at stage entry in ALL forward paths:

- `_forward_heun` (before the Heun loop)
- `_forward_heun_sequence` (per-sample inside loop)
- `forward_equilibrium` (before the DEQ fixed-point `phi`)

Result stored in a NON-PERSISTENT buffer `_gate_core_cached` (shape [B, n_core_edges]) so `torch.compile`'d `rhs`/`_heun_steps` treats it as a graph input (parameter-like) rather than a recompiling constant. Do NOT read it as a plain Python attribute inside compiled code. Any code path that would compute the gate inside a Heun sub-step is forbidden.

### 3. freeze_read ON integration (production path)

In each freeze_read precompute site, multiply the frozen tanh current by the cached core gate before scatter-add:

`i_edge = frozen_tanh_currents * static_edge_gate * gate_core_cached` (broadcast [B, E_core])

`i_edge_const` becomes an input-dependent per-node bias vector; downstream Heun loop, cached scatter reuse unchanged. All three sites: `_forward_heun`, `_forward_heun_sequence`, `forward_equilibrium`.

### 4. freeze_read OFF integration

In `rhs()`, when `i_edge_const` is None and core VCA enabled: multiply the recomputed core tanh current and resistive shunt by the cached gate (edge_gate semantics: IOPS scaling of the entire non-internal I–V curve per sample).

### 5. Shunt gating (flag, default FALSE, evaluate last)

`--vca-gate-shunt`: `i_res = g_resistive · (V_src − V_dst) · gate_core_cached`. Makes `M(u)` input-dependent (routing). Passivity preserved (gain in (0,2) is dissipative). Interacts with matrix-exponential integrator precompute. Default OFF.

### E6. Static gate interaction

On core edges, VCA MULTIPLIES the existing sigmoid(z_logits) gate (do not stack a second independent gate). Log both gates' distributions during training; if one collapses to compensate, merge into a single effective gate in a follow-up refactor.

### E7. Separate-core-bus ablation

`--vca-separate-core-bus`: gives core edges their own `vca_W_core` projection instead of the shared `x` bus. Default OFF (shared). Parameter cost of shared: 200 per stage at rank 2, 100 core edges.

### CLI flags

- `--vca-core` — explicitly enable core-edge gating
- `--vca-gate-shunt` — default off (step 5)
- `--vca-separate-core-bus` — default off (step E7)
- Relax the boundary/readout-requirement validation so `--vca` works with core edges only (run C).

### Optimizer

Structural LR group (4× base, already used in friedman2 baseline `struct_lr_scale: 4.0`): route ALL VCA params (`vca_W`, `vca_v_core`, `vca_v_boundary`, `vca_v_readout`, `vca_W_core`) into the struct bucket alongside `*.z_logits` in `make_optimizer`. VCA routing-structure, not dynamics. Update `_update_norms` bucketing to be consistent.

### Exploitability (sanity harness, before any training run)

1. **Identity**: zero-init + 2g gate → forward/backward at step 0 matches VCA-off baseline to float precision.
2. **Gradient**: `vca_W.grad` nonzero after one backward; `vca_v_core.grad` nonzero after one step.
3. **Cache freshness**: `_gate_core_cached` recomputed every stage entry / new batch. Stale cache across samples = the bug most likely to silently tank results.

### DiVA / logging

- Distribution of `gate_core` per edge across validation samples (collapse = mechanism unused).
- Per-node `b_j(u)` variance across samples (input-dependent operating-point measure).

### Experiment grid (friedman2, 3 seeds each, same protocol)

| Run | Boundary VCA | Core VCA | Shunt | Purpose |
|-----|--------------|----------|-------|---------|
| S | OFF | OFF | — | Baseline (~6.65e-5) |
| B | fixed init | OFF | — | init res observed VCA |
| C | { | OFF | core VCA alone — clean test |
| D | fixed init | ERR | — | full VCA |
| E | OFF | ON | on | input-dependent mixing (routing) |

- Same validation protocol per run, 3 seeds each, mean ± std.
- C > A ⇒ core VCA helps; C ≈ A but E > A ⇒ routing was the mechanism, not new scaling; nothing beats A ⇒ ceiling at transfer/readout nonlinearities (redirects to stage count / depth scaling experiment).

## Out of scope

- Merging static abstract + VCA gates (follow-up). — compute the matrix-exponential integrator change for shunt gating; the per-sample precompute survives but "precompute once per stage" does not. — evolving-dynamics hypothesis experiment (fabric freeze_read OFF / non-freeze) remains unrun and is out of scope here.

## Acceptance checklist

1. friedman2 baseline (run A) reproduces ≈6.65e-5 with the grid protocol.
2. Identity sanity check green: no-VCA run vs VCA-on (share-states) identical at step 0.
3. Gate distributions show some input-dependence in core-VCA runs (not collapsed to constant).
4. Report converges to a mechanism verdict per the mitigation flow, and notes the strategic architecture discussion: Unicode single-chip hypothesis: nonlinear I/O + linear core + smart biasing.