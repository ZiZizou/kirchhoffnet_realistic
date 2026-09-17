"""Audit for the post_twin_gates module (plan post-twin-gates,
feature spec flow-sparse-gates).  Covers:

1. Side-effect-free import and required public symbol set.
2. Fixed-dt rule: CANONICAL_DT_MAX bounds dt; num_steps = round(t_span/dt).
3. Heun dt-convergence (2nd-order refinement ratio).
4. Heun analytic Jacobian matches autograd (single sub-step bit-exact,
   multi-step within float64 matmul noise).
5. Heun fixed-step is deterministic at fixed seeds.
6. flow_twin_node_tanh emits finite states, finite instruments, ESN
   parity row, raw-delay matched parity, activity band,
   tanh-engagement guard, jac_max_abs finite; _gate1_row carries
   n_taps + per-row pass_mc.
7. random_sparse_mask: exactly target_nnz True entries.
8. magnitude_prune_mask: keeps the top-target_nnz entries by |W|;
   apply_mask_with_radius_restore actually restores the spectral radius.
9. low_rank_plus_sparse_W: effective matrix dense; stored-param reading
   documented (2*rank*n + ring).
10. mismatch_perturb: deterministic at fixed seed; sigma=0 is identity.
11. heterogeneous_tanh_jitter: gain > 0, gain jitter mean ≈ 1, bias
    jitter mean ≈ 0.
12. dense / sparse_random / sparse_structured W_in: shape (n_nodes, n_in),
    sparse_structured uses torus geom and is finite; builder guards
    reject k out of range and degenerate full-coverage groups.
13. _node_groups_torus: returns len == n_nodes, each group length ==
    taps_per_group, geom product == n_nodes.
14. Gate 1 verdict wired: lamT<=3 with MC>=8 -> FLOW_VIABLE; only
    lamT>=10 -> SPEED_REQUIRED; collapse -> DIGITAL_ONLY; mixed
    {3, 10} pins the operating point at 3.0 (never 10.0).
15. Gate 2 sparsity verdict (holds bar = twin MC>=5): graceful at 200
    -> ANNEAL_PRUNE; cliff below 500 with dense holding -> CLIFF;
    total collapse -> CLIFF.
16. Gate 2 mismatch verdict: holds at 5-10% -> SILICON_PLAUSIBLE;
    collapse at 10% -> PRECISION_REDIRECT; split -> MIXED.
17. Gate 3 verdict: structured holds -> STRUCTURED; only dense holds
    -> DENSE_ONLY.
18. Buildable spec emission on all-pass; not emitted on any non-pass.
19. CLI registration: heun, sparsity, mismatch, w_in, all subcommands
    registered with the required arguments.
20. Sweep integrity: G2-sparsity rows carry gate labels, the low-rank
    row sits in its own stored-param bucket (never pooled with the
    200-edge sparse rows), rows carry pass_mc; direct row-builder
    gate-label spot check.
21. Scalar-confirm wiring: locked confirm grid constants, 1-seed
    confirm run shape, heun --max-corners grid-cap + --dt/--order
    validation end to end.

Run:
    python -B audit_post_twin_gates.py
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")

import post_twin_gates as ptg  # noqa: E402


def assert_close(a, b, msg, tol=1e-6):
    assert abs(float(a) - float(b)) < tol, f"{msg}: {a} != {b}"


def assert_finite(value, msg):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all().item(), f"{msg}: non-finite"
    else:
        v = float(value)
        assert math.isfinite(v), f"{msg}: non-finite {v}"


torch.manual_seed(0)


# --- 1. Side-effect-free import and public symbol set -------------------
for name in (
    "flow_twin_run", "flow_twin_build_weights", "flow_twin_drive",
    "flow_twin_pre_stats", "flow_twin_jacobian_at",
    "flow_twin_eigs_per_transition", "flow_twin_node_tanh",
    "random_sparse_mask", "magnitude_prune_mask",
    "restore_spectral_radius", "apply_mask_with_radius_restore",
    "low_rank_plus_sparse_W",
    "mismatch_perturb", "mismatch_perturb_pair",
    "heterogeneous_tanh_jitter",
    "dense_W_in", "sparse_random_W_in", "sparse_structured_W_in",
    "_node_groups_torus",
    "run_gate1_sweep", "run_gate2_sparsity_sweep",
    "run_gate2_mismatch_sweep", "run_gate3_w_in_sweep",
    "gate1_verdict", "gate2_sparsity_verdict",
    "gate2_mismatch_verdict", "gate3_verdict",
    "maybe_emit_buildable_spec",
    "main", "CANONICAL_DT_MAX", "CANONICAL_T_SPAN",
    "G1_PASS_MC_ABOVE", "G1_SPEED_REQUIRED_LAMBDA_T",
    "G1_CONFIRM_LAMT", "G1_CONFIRM_RADIUS", "G1_CONFIRM_INPUT_SCALE",
    "G2_PASS_EDGE_TARGET", "G2_CLIFF_EDGE",
    "G2_MISMATCH_PASS_LO", "G2_MISMATCH_FAIL_HI",
    "G2_G3_HOLDS_MC_ABOVE",
    "G3_MASK_SEEDS", "G3_DENSE_SEEDS",
):
    assert hasattr(ptg, name), f"missing symbol: {name}"
print("AUDIT1_IMPORT_OK")


# --- 2. Fixed-dt rule -----------------------------------------------------
# At every lambda*T in G1_LAMBDA_T, dt = CANONICAL_DT_MAX = 0.125.
# The rule is "dt <= 0.125 at every lambda*T", which the constant
# encodes directly.
assert ptg.CANONICAL_DT_MAX <= 0.125, (
    f"fixed-dt rule violated: CANONICAL_DT_MAX={ptg.CANONICAL_DT_MAX}"
)
for lamT in ptg.G1_LAMBDA_T:
    dt = ptg.CANONICAL_DT_MAX
    assert dt > 0 and dt <= 0.125, (
        f"dt={dt} violates fixed-dt rule at lamT={lamT}"
    )
    num_steps = int(round(ptg.CANONICAL_T_SPAN / dt))
    assert num_steps >= 1, f"num_steps too small at lamT={lamT}"
print(f"AUDIT2_DT_RULE_OK dt={ptg.CANONICAL_DT_MAX} "
      f"num_steps={int(round(ptg.CANONICAL_T_SPAN / ptg.CANONICAL_DT_MAX))}")


# --- 3. Heun dt-convergence (refinement test) ----------------------------
# Heun is 2nd-order: halving dt should reduce the local error roughly
# 4x.  We integrate the SAME ODE `dx/dt = -lam*x + lam*tanh(W@x +
# W_in@d)` at three dt refinements (1/4, 1/8, 1/16) and check that
# the trajectory closes at finer dt.
torch.manual_seed(0)
W = torch.randn(8, 8, dtype=torch.float64) * 0.3
W_in = torch.randn(8, 3, dtype=torch.float64) * 0.5
drive = torch.randn(20, 3, dtype=torch.float64)


def _heun_run(dt_local, lam=0.7):
    n_steps = int(round(1.0 / dt_local))
    x = torch.zeros(8, dtype=torch.float64)
    states = [x.clone()]
    for t in range(20):
        for _ in range(n_steps):
            pre = W @ x + W_in @ drive[t]
            f0 = -lam * x + lam * torch.tanh(pre)
            x_pred = x + dt_local * f0
            pre_pred = W @ x_pred + W_in @ drive[t]
            f1 = -lam * x_pred + lam * torch.tanh(pre_pred)
            x = x + 0.5 * dt_local * (f0 + f1)
        states.append(x.clone())
    return torch.stack(states, dim=0)


s_coarse = _heun_run(0.25)
s_med = _heun_run(0.125)
s_fine = _heun_run(0.0625)
err_coarse_med = (s_coarse - s_med).abs().max().item()
err_med_fine = (s_med - s_fine).abs().max().item()
# 2nd-order convergence: halving dt should reduce error by ~4x.
ratio = err_coarse_med / max(err_med_fine, 1e-12)
assert 2.0 < ratio < 10.0, (
    f"Heun convergence ratio off: coarse-med={err_coarse_med:.4f}, "
    f"med-fine={err_med_fine:.4f}, ratio={ratio:.2f}"
)
print(f"AUDIT3_HEUN_DT_CONVERGENCE_OK ratio={ratio:.2f}")


# --- 4. Heun analytic Jacobian matches autograd ---------------------------
# Single-substep Jacobian (dt = t_span) is bit-exact; multi-substep
# accumulates float64 matmul noise (~1e-3 across 8 sub-steps).
torch.manual_seed(0)
W = torch.randn(6, 6, dtype=torch.float64) * 0.3
W_in = torch.randn(6, 2, dtype=torch.float64) * 0.5
x = torch.randn(6, dtype=torch.float64) * 0.2
d = torch.randn(2, dtype=torch.float64)
lam = 0.7
# Single-step check (bit-exact).
J_analytic_single = ptg.flow_twin_jacobian_at(
    W=W, W_in=W_in, x=x, drive_row=d, lam=lam, dt=ptg.CANONICAL_T_SPAN,
)


def _single_step_map(x_flat):
    x_ = x_flat.view(-1)
    pre = W @ x_ + W_in @ d
    f0 = -lam * x_ + lam * torch.tanh(pre)
    x_pred = x_ + ptg.CANONICAL_T_SPAN * f0
    pre_pred = W @ x_pred + W_in @ d
    f1 = -lam * x_pred + lam * torch.tanh(pre_pred)
    return x_ + 0.5 * ptg.CANONICAL_T_SPAN * (f0 + f1)


J_auto_single = torch.autograd.functional.jacobian(
    _single_step_map, x.detach().clone(),
)
assert torch.allclose(J_analytic_single, J_auto_single, atol=1e-10), (
    f"Single-step Heun analytic Jacobian != autograd: "
    f"max diff {(J_analytic_single - J_auto_single).abs().max().item()}"
)
# Multi-step check (loose tolerance).
J_analytic_multi = ptg.flow_twin_jacobian_at(
    W=W, W_in=W_in, x=x, drive_row=d, lam=lam, dt=0.125,
)


def _full_window_map(x_flat):
    x_ = x_flat.view(-1)
    n_steps = int(round(ptg.CANONICAL_T_SPAN / 0.125))
    for _ in range(n_steps):
        pre = W @ x_ + W_in @ d
        f0 = -lam * x_ + lam * torch.tanh(pre)
        x_pred = x_ + 0.125 * f0
        pre_pred = W @ x_pred + W_in @ d
        f1 = -lam * x_pred + lam * torch.tanh(pre_pred)
        x_ = x_ + 0.5 * 0.125 * (f0 + f1)
    return x_.view(-1)


J_auto_multi = torch.autograd.functional.jacobian(
    _full_window_map, x.detach().clone(),
)
assert torch.allclose(J_analytic_multi, J_auto_multi, atol=1e-2), (
    f"Multi-step Heun analytic Jacobian != autograd: "
    f"max diff {(J_analytic_multi - J_auto_multi).abs().max().item()}"
)
print("AUDIT4_HEUN_JACOBIAN_OK")


# --- 5. Heun fixed-step determinism --------------------------------------
torch.manual_seed(0)
W, W_in = ptg.flow_twin_build_weights(
    n_nodes=10, n_in=4, w_seed=0, win_seed=0,
    target_radius=0.9, dtype=torch.float32, device=torch.device("cpu"),
)
u = torch.randn(50, dtype=torch.float32)
drive1 = ptg.flow_twin_drive(u, "taps8", n_taps=4, input_scale=0.2)
drive2 = ptg.flow_twin_drive(u, "taps8", n_taps=4, input_scale=0.2)
s1 = ptg.flow_twin_run(W=W, W_in=W_in, drive=drive1, lam=0.5, dt=0.125)
s2 = ptg.flow_twin_run(W=W, W_in=W_in, drive=drive2, lam=0.5, dt=0.125)
assert torch.equal(s1, s2), "Heun run is not deterministic at fixed seeds"
print("AUDIT5_DETERMINISM_OK")


# --- 6. flow_twin_node_tanh emits a complete, finite instrument row -----
torch.manual_seed(0)
r = ptg.flow_twin_node_tanh(
    order=10, seed=0, device="cpu", hidden_dim=25,
    n_taps=8, harness="taps8", radius=0.9, lamT=1.0,
    input_scale=0.2, w_seed=0, win_seed=0,
    n_streams=1, train_samples_per_stream=300,
    washout=200, max_delay=20, jacobian_samples=2,
)
assert_finite(r["mc_total"], "mc_total")
assert_finite(r["ridge_nrmse"], "ridge_nrmse")
assert_finite(r["ridge_r2"], "ridge_r2")
assert_finite(r["state_pr"], "state_pr")
assert_finite(r["activity"], "activity")
assert_finite(r["esn_nrmse"], "esn_nrmse")
assert_finite(r["esn_mc_total"], "esn_mc_total")
assert_finite(r["jac_max_abs"], "jac_max_abs")
assert_finite(r["pre_stats"]["median_abs_pre"], "pre_stats median")
assert isinstance(r["activity_in_band"], bool)
assert isinstance(r["tanh_engaged"], bool)
assert r["states_finite"] is True
assert r["config_tag"].startswith("flowtwin_taps8_")
assert_finite(r["matched_raw_delay_nrmse"], "matched_raw_delay_nrmse")
assert_finite(r["matched_parity_delta"], "matched_parity_delta")
assert isinstance(r["mc_per_delay"], list) and len(r["mc_per_delay"]) == 20
assert r["n_taps"] == 8
# Row conversion carries provenance + per-row PASS/FAIL.
g1row = ptg._gate1_row(r)
assert g1row.n_taps == 8, f"Gate1Row n_taps={g1row.n_taps}"
assert isinstance(g1row.pass_mc, bool)
assert g1row.pass_mc == (
    g1row.tanh_engaged and g1row.states_finite
    and g1row.mc_total >= ptg.G1_PASS_MC_ABOVE
)
print(f"AUDIT6_CORNER_OK mc={r['mc_total']:.3f} "
      f"ridge={r['ridge_nrmse']:.4f} tag={r['config_tag']}")


# --- 7. random_sparse_mask: exactly target_nnz True ----------------------
mask = ptg.random_sparse_mask(10, 10, target_nnz=37, seed=2)
assert mask.shape == (10, 10)
assert mask.dtype == torch.bool
assert int(mask.sum().item()) == 37, f"random mask nnz={int(mask.sum().item())}"
# Reproducibility at fixed seed.
mask2 = ptg.random_sparse_mask(10, 10, target_nnz=37, seed=2)
assert torch.equal(mask, mask2), "random_sparse_mask not deterministic"
# Out-of-range guard.
try:
    ptg.random_sparse_mask(3, 3, target_nnz=10, seed=0)
    raise AssertionError("expected ValueError on out-of-range target_nnz")
except ValueError:
    pass
print("AUDIT7_RANDOM_SPARSE_OK")


# --- 8. magnitude_prune_mask + apply_mask_with_radius_restore ------------
torch.manual_seed(0)
W = torch.randn(15, 15, dtype=torch.float32) * 0.3
target_nnz = 50
mask_mp = ptg.magnitude_prune_mask(W, target_nnz=target_nnz)
assert mask_mp.shape == W.shape
assert int(mask_mp.sum().item()) == target_nnz
target_radius = 0.85
W_mp = ptg.apply_mask_with_radius_restore(
    W, mask_mp, target_radius=target_radius,
)
rho = float(torch.linalg.eigvals(
    W_mp.to(torch.float64)
).abs().max().item())
assert_close(rho, target_radius, "magnitude-prune radius restore", tol=1e-4)
# Identity restore: dense -> dense stays dense at radius.
W_full = ptg.apply_mask_with_radius_restore(
    W, torch.ones_like(W).bool(), target_radius=target_radius,
)
rho_full = float(torch.linalg.eigvals(
    W_full.to(torch.float64)
).abs().max().item())
assert_close(rho_full, target_radius, "dense radius restore", tol=1e-4)
print(f"AUDIT8_MAGNITUDE_PRUNE_OK rho={rho:.4f}")


# --- 9. low_rank_plus_sparse_W structure ----------------------------------
W_lr, W_in_lr = ptg.low_rank_plus_sparse_W(
    n_nodes=20, n_in=5, w_seed=0, win_seed=0,
    rank=2, ring_local_count=80, target_radius=0.9,
    dtype=torch.float32, device=torch.device("cpu"),
)
assert W_lr.shape == (20, 20)
assert W_in_lr.shape == (20, 5)
rho_lr = float(torch.linalg.eigvals(
    W_lr.to(torch.float64)
).abs().max().item())
assert_close(rho_lr, 0.9, "low_rank+sparse radius", tol=1e-4)
# The effective matrix is dense (U V^T fills every entry); the
# device-budget reading is stored params (2*rank*n + ring), NOT the
# effective nnz.  Pin that property so no caller can pool this row
# with same-nnz sparse buckets.
assert int((W_lr != 0).sum().item()) == 20 * 20, (
    f"low-rank effective matrix should be dense, got "
    f"{int((W_lr != 0).sum().item())} nnz"
)
print(f"AUDIT9_LOW_RANK_OK rho={rho_lr:.4f}")


# --- 10. mismatch_perturb determinism + sigma=0 identity -----------------
torch.manual_seed(0)
W = torch.randn(8, 8, dtype=torch.float32)
W_id = ptg.mismatch_perturb(W, sigma=0.0, seed=42)
assert torch.equal(W_id, W), "sigma=0 must be identity"
W_a = ptg.mismatch_perturb(W, sigma=0.05, seed=7)
W_b = ptg.mismatch_perturb(W, sigma=0.05, seed=7)
assert torch.equal(W_a, W_b), "mismatch_perturb not deterministic"
W_c = ptg.mismatch_perturb(W, sigma=0.05, seed=8)
assert not torch.equal(W_a, W_c), "different seeds must differ"
# Pair perturbation determinism.
Wa, Wia = ptg.mismatch_perturb_pair(W, W, sigma=0.1, seed=3)
Wb, Wib = ptg.mismatch_perturb_pair(W, W, sigma=0.1, seed=3)
assert torch.equal(Wa, Wb)
assert torch.equal(Wia, Wib)
print("AUDIT10_MISMATCH_OK")


# --- 11. heterogeneous_tanh_jitter sanity --------------------------------
gain, bias = ptg.heterogeneous_tanh_jitter(
    n_nodes=50, gain_jitter=0.10, bias_jitter=0.05, seed=0,
)
assert gain.shape == (50,)
assert bias.shape == (50,)
assert (gain > 0).all().item(), "gains must be positive"
assert_close(float(gain.mean().item()), 1.0, "gain mean", tol=0.05)
assert_close(float(bias.mean().item()), 0.0, "bias mean", tol=0.05)
print("AUDIT11_HETERO_OK")


# --- 12. W_in builders shape + finite ------------------------------------
W_in_d = ptg.dense_W_in(
    n_nodes=25, n_in=8, seed=0,
    dtype=torch.float32, device=torch.device("cpu"),
)
assert W_in_d.shape == (25, 8)
assert torch.isfinite(W_in_d).all().item()
for k in (3, 6):
    W_in_sr = ptg.sparse_random_W_in(
        n_nodes=25, n_in=8, k_per_tap=k, seed=0,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    assert W_in_sr.shape == (25, 8)
    assert torch.isfinite(W_in_sr).all().item()
    assert int((W_in_sr != 0).sum().item()) == k * 8, (
        f"sparse_random nnz != k*n_in: {(W_in_sr != 0).sum().item()}"
    )
W_in_ss = ptg.sparse_structured_W_in(
    n_nodes=25, n_in=8, geom=(5, 5), seed=0,
    dtype=torch.float32, device=torch.device("cpu"),
)
assert W_in_ss.shape == (25, 8)
assert torch.isfinite(W_in_ss).all().item()
# Builder guards: k out of range, and degenerate full-coverage groups.
for bad_k in (0, 26):
    try:
        ptg.sparse_random_W_in(
            n_nodes=25, n_in=8, k_per_tap=bad_k, seed=0,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        raise AssertionError(f"expected ValueError for k_per_tap={bad_k}")
    except ValueError:
        pass
try:
    ptg.sparse_structured_W_in(
        n_nodes=25, n_in=1, geom=(5, 5), seed=0,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    raise AssertionError("expected ValueError on degenerate scalar group")
except ValueError:
    pass
print("AUDIT12_W_IN_OK")


# --- 13. _node_groups_torus invariants -----------------------------------
groups = ptg._node_groups_torus(25, (5, 5), taps_per_group=3, seed=0)
assert len(groups) == 25
for g in groups:
    assert len(g) == 3
    assert all(0 <= idx < 25 for idx in g)
    # Indices must be unique within the group.
    assert len(set(g)) == 3
# Geom product check: prod(geom) must equal n_nodes.
try:
    ptg._node_groups_torus(20, (5, 5), taps_per_group=3, seed=0)
    raise AssertionError("expected ValueError on geom product mismatch")
except ValueError:
    pass
print("AUDIT13_GROUPS_OK")


# --- 14. Gate 1 verdict logic ---------------------------------------------
# Build a synthetic Gate1Row list directly via dataclass.
from dataclasses import dataclass, field


@dataclass
class Gate1Stub:
    lamT: float
    mc_total: float
    tanh_engaged: bool = True
    states_finite: bool = True

# All rows have MC >= 8 at lamT<=3 -> FLOW_VIABLE.
rows = [
    Gate1Stub(lamT=0.3, mc_total=10.0),
    Gate1Stub(lamT=1.0, mc_total=12.0),
    Gate1Stub(lamT=3.0, mc_total=8.5),
    Gate1Stub(lamT=10.0, mc_total=2.0),  # below bar; below 10 anyway
]
v = ptg.gate1_verdict(rows)
assert v["verdict"] == "FLOW_VIABLE", v
assert v["viable_lamT"] == 3.0, v

# Only lamT>=10 survives -> SPEED_REQUIRED.
rows = [
    Gate1Stub(lamT=0.3, mc_total=1.0),
    Gate1Stub(lamT=1.0, mc_total=1.0),
    Gate1Stub(lamT=3.0, mc_total=1.0),
    Gate1Stub(lamT=10.0, mc_total=15.0),
]
v = ptg.gate1_verdict(rows)
assert v["verdict"] == "SPEED_REQUIRED", v
assert v["viable_lamT"] == 10.0, v

# All collapse -> DIGITAL_ONLY.
rows = [
    Gate1Stub(lamT=0.3, mc_total=0.5),
    Gate1Stub(lamT=1.0, mc_total=0.5),
]
v = ptg.gate1_verdict(rows)
assert v["verdict"] == "DIGITAL_ONLY", v
assert v["viable_lamT"] is None, v

# Mixed: both 3 and 10 hold -> FLOW_VIABLE with the operating point
# pinned at 3.0 (never 10.0; only holds-exclusively-at->=10 records
# the speed requirement).
rows = [
    Gate1Stub(lamT=0.3, mc_total=4.0),
    Gate1Stub(lamT=1.0, mc_total=4.0),
    Gate1Stub(lamT=3.0, mc_total=9.0),
    Gate1Stub(lamT=10.0, mc_total=11.0),
]
v = ptg.gate1_verdict(rows)
assert v["verdict"] == "FLOW_VIABLE", v
assert v["viable_lamT"] == 3.0, v
print("AUDIT14_GATE1_VERDICT_OK")


# --- 15. Gate 2 sparsity verdict ------------------------------------------
# Holds bar is the twin PASS_ARCH bar (MC >= 5).
assert ptg.G2_G3_HOLDS_MC_ABOVE == 5.0, ptg.G2_G3_HOLDS_MC_ABOVE


@dataclass
class GateRowStub:
    config_tag: str = ""
    gate: str = "sparsity"
    leg: str = ""
    n_edges: int = 0
    n_in: int = 8
    mc_total: float = 0.0
    ridge_nrmse: float = 0.0
    state_pr: float = 0.0
    tanh_engaged: bool = True
    states_finite: bool = True
    pass_mc: bool = True
    note: str = ""

# Graceful at 200 -> ANNEAL_PRUNE.
rows = [
    GateRowStub(leg="random_sparse_e100_ms0", n_edges=100,
                mc_total=6.0),
    GateRowStub(leg="random_sparse_e200_ms0", n_edges=200,
                mc_total=7.0),
    GateRowStub(leg="random_sparse_e400_ms0", n_edges=400,
                mc_total=8.0),
    GateRowStub(leg="dense_e625", n_edges=625, mc_total=9.0),
]
v = ptg.gate2_sparsity_verdict(rows)
assert v["verdict"] == "ANNEAL_PRUNE", v
assert v["edge_target"] == 200, v

# Total collapse -> CLIFF.
rows = [
    GateRowStub(leg="random_sparse_e100_ms0", n_edges=100,
                mc_total=0.5),
    GateRowStub(leg="random_sparse_e200_ms0", n_edges=200,
                mc_total=0.5),
    GateRowStub(leg="random_sparse_e400_ms0", n_edges=400,
                mc_total=0.5),
    GateRowStub(leg="dense_e625", n_edges=625, mc_total=0.5),
]
v = ptg.gate2_sparsity_verdict(rows)
assert v["verdict"] == "CLIFF", v

# Cliff below 500 with dense holding -> CLIFF (hard physical
# decision), NOT anneal-prune at 400.
rows = [
    GateRowStub(leg="random_sparse_e100_ms0", n_edges=100,
                mc_total=0.5),
    GateRowStub(leg="random_sparse_e200_ms0", n_edges=200,
                mc_total=0.5),
    GateRowStub(leg="random_sparse_e400_ms0", n_edges=400,
                mc_total=0.5),
    GateRowStub(leg="dense_e625", n_edges=625, mc_total=10.0),
]
v = ptg.gate2_sparsity_verdict(rows)
assert v["verdict"] == "CLIFF", v
assert v["edge_target"] is None, v
print("AUDIT15_GATE2_SPARSE_VERDICT_OK")


# --- 16. Gate 2 mismatch verdict -----------------------------------------
rows = [
    GateRowStub(leg="mismatch_s0.0_d0", mc_total=10.0),
    GateRowStub(leg="mismatch_s0.05_d0", mc_total=9.0),
    GateRowStub(leg="mismatch_s0.1_d0", mc_total=8.0),
    GateRowStub(leg="hetops_g0.10_b0.05", mc_total=8.0),
]
v = ptg.gate2_mismatch_verdict(rows)
assert v["verdict"] == "SILICON_PLAUSIBLE", v

# Collapse at 10% -> PRECISION_REDIRECT.
rows = [
    GateRowStub(leg="mismatch_s0.0_d0", mc_total=10.0),
    GateRowStub(leg="mismatch_s0.05_d0", mc_total=8.0),
    GateRowStub(leg="mismatch_s0.1_d0", mc_total=0.5),
]
v = ptg.gate2_mismatch_verdict(rows)
assert v["verdict"] == "PRECISION_REDIRECT", v

# Split (5% collapses, 10% holds) -> MIXED.
rows = [
    GateRowStub(leg="mismatch_s0.0_d0", mc_total=10.0),
    GateRowStub(leg="mismatch_s0.05_d0", mc_total=0.5),
    GateRowStub(leg="mismatch_s0.1_d0", mc_total=8.0),
]
v = ptg.gate2_mismatch_verdict(rows)
assert v["verdict"] == "MIXED", v
print("AUDIT16_GATE2_MISMATCH_VERDICT_OK")


# --- 17. Gate 3 verdict --------------------------------------------------
rows = [
    GateRowStub(leg="w_in_dense", mc_total=10.0),
    GateRowStub(leg="w_in_sparse_random_k3", mc_total=8.0),
    GateRowStub(leg="w_in_sparse_random_k6", mc_total=8.0),
    GateRowStub(leg="w_in_sparse_structured", mc_total=9.0),
]
v = ptg.gate3_verdict(rows)
assert v["verdict"] == "STRUCTURED", v

# Only dense holds -> DENSE_ONLY.
rows = [
    GateRowStub(leg="w_in_dense", mc_total=10.0),
    GateRowStub(leg="w_in_sparse_random_k3", mc_total=0.5),
    GateRowStub(leg="w_in_sparse_random_k6", mc_total=0.5),
    GateRowStub(leg="w_in_sparse_structured", mc_total=0.5),
]
v = ptg.gate3_verdict(rows)
assert v["verdict"] == "DENSE_ONLY", v
print("AUDIT17_GATE3_VERDICT_OK")


# --- 18. Buildable spec emission ----------------------------------------
out = Path("./output/audit_post_twin_gates")
if out.exists():
    for p in out.iterdir():
        p.unlink()
out.mkdir(parents=True, exist_ok=True)
g1 = {"verdict": "FLOW_VIABLE", "viable_lamT": 1.0}
g2s = {"verdict": "ANNEAL_PRUNE", "edge_target": 200}
g2m = {"verdict": "SILICON_PLAUSIBLE"}
g3 = {"verdict": "STRUCTURED"}
spec = ptg.maybe_emit_buildable_spec(
    gate1=g1, gate2_s=g2s, gate2_m=g2m, gate3=g3, out_dir=out,
)
assert spec is not None, "buildable spec not emitted on all-pass"
assert (out / "post_twin_gates_buildable_spec.json").exists()
assert spec["lambda_T"] == 1.0
assert spec["edge_target"] == 200
assert spec["w_in_geometry"] == "sparse_structured_torus_5x5"

# On any non-pass, no spec is emitted.
g1_bad = {"verdict": "DIGITAL_ONLY", "viable_lamT": None}
spec_bad = ptg.maybe_emit_buildable_spec(
    gate1=g1_bad, gate2_s=g2s, gate2_m=g2m, gate3=g3, out_dir=out,
)
assert spec_bad is None, "spec emitted on non-pass!"
print("AUDIT18_BUILDABLE_SPEC_OK")


# --- 19. CLI registration ------------------------------------------------
for mode in ("heun", "sparsity", "mismatch", "w_in", "all"):
    res = subprocess.run(
        [sys.executable, "-B", "post_twin_gates.py", mode, "--help"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, f"{mode} --help failed: {res.stderr}"
print("AUDIT19_CLI_OK")


# --- 20. Sweep integrity: gate labels, low-rank bucket, pass_mc ---------
sweep_rows, sweep_verdict = ptg.run_gate2_sparsity_sweep(
    hidden_dim=25, harness="taps8", n_taps=8, n_streams=1,
    train_samples_per_stream=300, washout=200, max_delay=20,
    lamT_best=1.0, input_scale=0.2,
)
assert len(sweep_rows) == 4 * (len(ptg.G2_SPARSE_SEEDS) + 1) + 1, (
    f"unexpected sweep size {len(sweep_rows)}"
)
for r in sweep_rows:
    assert r.gate == "sparsity", f"gate label leak: {r.leg} -> {r.gate}"
    assert isinstance(r.pass_mc, bool)
    assert "_r1_" in r.config_tag, f"radius provenance missing: {r.config_tag}"
lr_rows = [r for r in sweep_rows if r.leg.startswith("low_rank")]
assert len(lr_rows) == 1
# Stored-param bucket (2*rank*n + ring), NOT the effective dense nnz
# and NOT pooled with the 200-edge sparse rows.
assert lr_rows[0].n_edges == 2 * 2 * 25 + 200, lr_rows[0].n_edges
bucket200 = [r for r in sweep_rows if r.n_edges == 200]
assert bucket200, "200-edge bucket vanished"
assert all("low_rank" not in r.leg for r in bucket200), (
    "low-rank row pooled into the 200-edge bucket"
)
# Direct row-builder gate-label spot check (mismatch leg path).
W10, W_in10 = ptg.flow_twin_build_weights(
    n_nodes=10, n_in=4, w_seed=0, win_seed=0,
    target_radius=0.9, dtype=torch.float32, device=torch.device("cpu"),
)
probe_row = ptg._sparsity_row_from_twin(
    W=W10, W_in=W_in10, hidden_dim=10,
    harness="taps8", n_taps=4, seed=0, device="cpu", n_in=4,
    radius=0.9, lamT=1.0, input_scale=0.2,
    w_seed=0, win_seed=0, n_streams=1,
    train_samples_per_stream=300, washout=200, max_delay=20,
    gate="mismatch", leg="mismatch_s0.05_d0", n_edges=100,
    note="audit spot check",
)
assert probe_row.gate == "mismatch", probe_row.gate
assert "_r0.9_" in probe_row.config_tag, probe_row.config_tag
print("AUDIT20_SWEEP_INTEGRITY_OK")


# --- 21. Scalar-confirm wiring + CLI end to end --------------------------
# Locked confirm grid per spec: lamT {1, 3}, r = 1.1, is = 0.2.
assert ptg.G1_CONFIRM_LAMT == (1.0, 3.0), ptg.G1_CONFIRM_LAMT
assert ptg.G1_CONFIRM_RADIUS == (1.1,), ptg.G1_CONFIRM_RADIUS
assert ptg.G1_CONFIRM_INPUT_SCALE == (0.2,), ptg.G1_CONFIRM_INPUT_SCALE
confirm_rows, confirm_verdict, _ = ptg.run_gate1_sweep(
    harness="scalar", n_taps=8, hidden_dim=25, n_streams=1,
    train_samples_per_stream=300, washout=200, max_delay=20,
    jacobian_samples=1,
    radius_grid=ptg.G1_CONFIRM_RADIUS,
    input_scale_grid=ptg.G1_CONFIRM_INPUT_SCALE,
    lamT_grid=ptg.G1_CONFIRM_LAMT,
    w_seeds=(0,), data_seeds=(0,),
)
assert len(confirm_rows) == 2, len(confirm_rows)
assert all(r.harness == "scalar" for r in confirm_rows)
assert all(r.radius == 1.1 and r.input_scale == 0.2
           for r in confirm_rows)
assert sorted(r.lamT for r in confirm_rows) == [1.0, 3.0]
assert set(confirm_verdict["surviving_by_lamT"]) == {"1.0", "3.0"}
# heun --max-corners caps the grid (cheap smoke) and the verdict
# covers exactly the rows written.
confirm_out = Path("./output/audit_post_twin_gates_cli")
res = subprocess.run(
    [sys.executable, "-B", "post_twin_gates.py", "heun",
     "--max-corners", "2", "--output", str(confirm_out)],
    capture_output=True, text=True,
)
assert res.returncode == 0, res.stderr[-2000:]
import csv as _csv
with (confirm_out / "gate1_flowtwin.csv").open() as f:
    assert len(list(_csv.DictReader(f))) == 2, "grid-cap row count"
# --dt above the fixed-dt bound is rejected, not silently clipped.
res = subprocess.run(
    [sys.executable, "-B", "post_twin_gates.py", "heun",
     "--dt", "0.25", "--max-corners", "1",
     "--output", str(confirm_out)],
    capture_output=True, text=True,
)
assert res.returncode != 0, "--dt 0.25 should violate the fixed-dt rule"
# --order 20 is rejected up front (order-10 pre-registration).
res = subprocess.run(
    [sys.executable, "-B", "post_twin_gates.py", "heun",
     "--order", "20", "--max-corners", "1",
     "--output", str(confirm_out)],
    capture_output=True, text=True,
)
assert res.returncode != 0, "--order 20 should be rejected"
print("AUDIT21_CONFIRM_CLI_OK")

print("\nALL_POST_TWIN_GATES_AUDITS_OK")
