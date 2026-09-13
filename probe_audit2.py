"""Audit #2 for narma_linear_controls: invariants for the linear controls.

Covers: import side-effect-freeness, frozen gate thresholds, the
install/restore roundtrip (gates, rails, leak, rhs, no attribute leaks,
cell_library init defaults UNTOUCHED), dense-W construction, the
spectral tuner loop, the memory_capacity delay-convention pin
(delay line must score MC_1 ~= 1.0), C0/R0 helper wiring on tiny
streams, the C2/C3 restore-flag plumbing, and CLI registration.

c1b-revision amendments:
  - new C1b install/restore roundtrip (QR-orthogonal W, uniform leak)
  - new C1b-real install/restore roundtrip (resistive core, no rhs override)
  - new c1b-protocol gate thresholds (C1B_PASS_MC_ABOVE=5.0, J-band [0.97, 0.99])
  - new standardized-state PR helper
  - new per-delay MC with SVD fallback (ill-conditioned lags never
    produce absurd negative MC)
  - new tuned-base sidecar roundtrip (W + G_in .pt + JSON pointer)

Run:
    python -B probe_audit2.py
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, ".")

import narma_linear_controls as nlc
import narma_experiment as ne

torch.manual_seed(0)


def assert_close(a, b, msg, tol=1e-6):
    assert abs(float(a) - float(b)) < tol, f"{msg}: {a} != {b}"


def build_net(hidden_dim=9):
    net, t_span, num_steps = ne._build_fabric_net(
        order=10, seed=0, freeze_read=False,
        t_span=1.0, num_steps=8, cell_library="tanh_free",
        core_refresh_interval=0, leak_constant=None,
        compile_sequence=False, hidden_dim=hidden_dim,
    )
    return net, t_span, num_steps


def tiny_stream(n=250):
    u_raw, y_raw = ne._gen_narma_train_streams(
        order=10, seed=0, n_streams=1, n=n,
    )
    u = ne._scale_drive(u_raw[0], bipolar=True, order=10, input_scale=1.0)
    return u, y_raw[0]


# --- 1. Side-effect-free import -----------------------------------------
for name in (
    "r0_reconciliation", "c0_esn_calibration", "c1_linear_reservoir",
    "c2_c3_c4_bisection", "install_linear_reservoir_v2",
    "restore_linear_reservoir_v2", "tune_to_target_radius",
    "measure_spectral_radius", "main",
    # c1b-revision additions
    "c1b_orthogonal_reservoir", "c1b_real_resistive",
    "c3_gm_grid_sweep", "install_linear_reservoir_c1b",
    "install_linear_reservoir_c1b_real",
    "restore_linear_reservoir_c1b_real", "tune_c1b_to_j_band",
    "tune_c1b_real_to_j_band", "save_tuned_base", "load_tuned_base",
    "C1B_REAL_UNIFORM_MIN", "C1B_REAL_STABILITY_CAP",
    "C1B_REAL_MIN_SCALE", "C1B_REAL_RAIL_BELOW",
    "reinstall_tuned_c1b", "_qr_orthogonal_W", "_make_c1b_rhs",
    "_resistive_G", "_set_resistive_G",
    "_standardized_state_pr", "_per_delay_mc", "_safe_abs_eigvals",
    "C1B_PASS_MC_ABOVE", "C1B_PASS_RAIL_BELOW",
    "C1B_SPECTRAL_TARGET_LOW", "C1B_SPECTRAL_TARGET_HIGH",
    "C1B_LEAK_VALUE", "C1B_EXPECTED_MC_LOW", "C1B_EXPECTED_MC_HIGH",
    "C3_GM_GRID", "C3_MC_COLLAPSE_RATIO",
    "C1B_SIDECAR_NAME", "C1B_SIDECAR_JSON",
):
    assert hasattr(nlc, name), f"missing {name}"
print("AUDIT2_IMPORT_OK")

# --- 2. Frozen gate thresholds ------------------------------------------
assert_close(nlc.R0_MATCHED_PARITY_TOL, 0.03, "R0 parity tol")
assert_close(nlc.C1_PASS_MC_ABOVE, 3.0, "C1 MC gate")
assert_close(nlc.C1_PASS_PR_MIN, 6.0, "C1 PR gate")
assert_close(nlc.C1_SPECTRAL_TARGET_LOW, 0.93, "spectral low")
assert_close(nlc.C1_SPECTRAL_TARGET_HIGH, 0.97, "spectral high")
assert nlc.CANONICAL_HIDDEN == 49, "C1 width must be hidden-49"
assert_close(nlc.CANONICAL_T_SPAN, 1.0, "canonical t_span")
assert nlc.CANONICAL_NUM_STEPS == 8, "canonical num_steps"
# c1b-revision additions.
assert_close(nlc.C1B_PASS_MC_ABOVE, 5.0, "C1b MC gate")
assert_close(nlc.C1B_PASS_RAIL_BELOW, 0.05, "C1b rail gate")
assert_close(nlc.C1B_SPECTRAL_TARGET_LOW, 0.97, "C1b spectral low")
assert_close(nlc.C1B_SPECTRAL_TARGET_HIGH, 0.99, "C1b spectral high")
assert_close(nlc.C1B_LEAK_VALUE, 0.05, "C1b uniform leak value")
assert_close(nlc.C1B_REAL_UNIFORM_MIN, 0.90, "C1b-real uniformity floor")
assert_close(nlc.C1B_REAL_STABILITY_CAP, 0.995, "C1b-real stability cap")
assert_close(nlc.C1B_REAL_MIN_SCALE, 1e-3, "C1b-real min G scale")
assert_close(nlc.C1B_EXPECTED_MC_LOW, 5.0, "C1b pre-registered MC low")
assert_close(nlc.C1B_EXPECTED_MC_HIGH, 15.0, "C1b pre-registered MC high")
assert_close(nlc.C3_MC_COLLAPSE_RATIO, 0.5, "C3 collapse ratio")
assert tuple(nlc.C3_GM_GRID) == (-8.0, -5.0, -2.0, 0.0, 1.5, 3.0), "C3 grid"
assert nlc.C1B_SIDECAR_NAME == "c1b_tuned_base.pt", "C1b sidecar .pt name"
assert nlc.C1B_SIDECAR_JSON == "c1b_tuned_base.json", "C1b sidecar json name"
print("AUDIT2_THRESHOLDS_OK")

# --- 3. memory_capacity delay convention --------------------------------
# A noiseless 1-step delay line must score MC_1 ~= 1.0 (pins the
# k-step-back fix; the swapped convention scored 0.022).
T, N = 1000, 5
u_dl = torch.rand(T)
states_dl = torch.zeros(T, N)
states_dl[1:, 0] = u_dl[:-1]
mc_per, mc_total = ne.memory_capacity(states_dl, u_dl, max_delay=5)
assert_close(mc_per[0], 1.0, "delay-line MC_1", tol=0.02)
assert mc_total < 1.5, f"delay-line total MC should be ~1, got {mc_total}"
print(f"AUDIT2_MC_CONVENTION_OK mc1={mc_per[0]:.4f} total={mc_total:.4f}")

# --- 3b. _per_delay_mc SVD fallback (c1b-revision amendment) -------------
# An ill-conditioned regression (here: degenerate constant column) must
# not return absurd negative MC values; the SVD fallback clamps the lag
# to a small non-negative value.
T, N = 800, 6
u_dl = torch.rand(T)
states_dl = torch.zeros(T, N)
states_dl[1:, 0] = u_dl[:-1]
# Add a degenerate constant column so cond(XtX) explodes.
states_dl[:, 1] = 1.0
mc_per_fb, mc_total_fb = nlc._per_delay_mc(
    states_dl, u_dl, washout=200, max_delay=10, use_svd_fallback=True,
)
assert all(v >= -1e-6 for v in mc_per_fb if math.isfinite(v)), (
    f"SVD fallback produced negative MC values: {mc_per_fb}"
)
assert mc_total_fb >= 0, (
    f"SVD fallback total MC non-negative, got {mc_total_fb}"
)
# Disable fallback reproduces the legacy contract (may be negative if
# the underlying X is degenerate); this assertion guards the toggle.
mc_per_nofb, _ = nlc._per_delay_mc(
    states_dl, u_dl, washout=200, max_delay=10, use_svd_fallback=False,
)
assert len(mc_per_nofb) == 10, "per-delay length"
print(f"AUDIT2_MC_SVD_FALLBACK_OK total={mc_total_fb:.4f}")

# --- 3c. _standardized_state_pr is finite on small well-spread inputs ---
torch.manual_seed(0)
sample = torch.randn(200, 5)
std_pr = nlc._standardized_state_pr(sample)
assert math.isfinite(std_pr), f"standardized PR non-finite: {std_pr}"
assert 1.0 <= std_pr <= 5.0, f"standardized PR out of range: {std_pr}"
print(f"AUDIT2_STD_PR_OK pr={std_pr:.2f}")

# --- 3d. QR-orthogonal W construction has unit spectral radius ---------
W_qr = nlc._qr_orthogonal_W(seed=42, dim=10, target_radius=1.0)
assert W_qr.shape == (10, 10), f"QR W shape: {W_qr.shape}"
eigs = torch.linalg.eigvals(W_qr).abs()
assert_close(float(eigs.max().item()), 1.0, "QR radius=1", tol=1e-4)
W_qr_05 = nlc._qr_orthogonal_W(seed=42, dim=10, target_radius=0.5)
eigs_05 = torch.linalg.eigvals(W_qr_05).abs()
assert_close(float(eigs_05.max().item()), 0.5, "QR radius=0.5", tol=1e-4)
print(f"AUDIT2_QR_W_OK")

# --- 4. install/restore roundtrip ---------------------------------------
net, _, _ = build_net(hidden_dim=9)
stage = net.core.stages[0]
orig_gm = stage.cell_lib.gm_raw.detach().clone()
orig_isat = stage.cell_lib.isat_raw.detach().clone()
orig_b_gm = stage.boundary_cell_lib.gm_raw.detach().clone()
orig_z = stage.z_logits.detach().clone()
orig_bz = stage.boundary_z_logits.detach().clone()
orig_leak = stage.raw_leak.detach().clone()
orig_xmax = float(stage.x_max)
orig_clip = float(stage.clip_current)
orig_mode = str(stage.leak_mode)
orig_rhs = stage.rhs
u, y = tiny_stream()

saved = nlc.install_linear_reservoir_v2(
    net, G_edge_seed=0, G_in_seed=0, leak_seed=0,
    target_x_max=20.0, clip_current=0.0,
    drive_rms_target=nlc._drive_rms(u), u_seq_for_rms=u,
)
# Gates open, rails far, clip zeroed, W dense square.
assert_close(float(stage.z_logits.mean().item()), 12.0, "core gates open")
assert_close(float(stage.boundary_z_logits.mean().item()), 12.0, "boundary gates open")
assert_close(float(stage.x_max), 20.0, "far rail")
assert_close(float(stage.clip_current), 0.0, "clip zeroed")
W = getattr(stage, "_lin_G_edge", None)
assert W is not None and W.shape == (10, 10), f"W shape {None if W is None else W.shape}"
assert torch.isfinite(W).all(), "W finite"
assert stage.boundary_cell_lib._lin_G_in.shape[0] == 9, "G_in per boundary edge"
assert stage._lin_restore_boundary is False and stage._lin_restore_tanh is False
# cell_library init defaults untouched by install.
assert torch.equal(stage.cell_lib.gm_raw, orig_gm), "gm_raw mutated by install"
assert torch.equal(stage.cell_lib.isat_raw, orig_isat), "isat_raw mutated by install"
assert torch.equal(stage.boundary_cell_lib.gm_raw, orig_b_gm), "boundary gm mutated"
print("AUDIT2_INSTALL_OK")

# Restore returns everything.
nlc.restore_linear_reservoir_v2(net, saved)
assert torch.equal(stage.z_logits, orig_z), "z_logits not restored"
assert torch.equal(stage.boundary_z_logits, orig_bz), "boundary gates not restored"
assert torch.equal(stage.raw_leak, orig_leak), "raw_leak not restored"
assert_close(float(stage.x_max), orig_xmax, "x_max not restored")
assert_close(float(stage.clip_current), orig_clip, "clip not restored"
)
assert str(stage.leak_mode) == orig_mode, "leak_mode not restored"
assert not hasattr(stage, "_lin_G_edge"), "_lin_G_edge leaked"
assert not hasattr(stage, "_orig_rhs"), "_orig_rhs leaked"
assert not hasattr(stage, "_lin_restore_tanh"), "restore flag leaked"
# rhs is the original bound method again.
assert hasattr(stage.rhs, "__self__"), "rhs not restored to bound method"
# Fresh build reproduces init raws -> cell_library.py defaults preserved.
net_fresh, _, _ = build_net(hidden_dim=9)
assert torch.equal(net_fresh.core.stages[0].cell_lib.gm_raw, orig_gm), \
    "cell_library defaults changed"
print("AUDIT2_RESTORE_OK")

# --- 4b. C1b install/restore roundtrip (c1b-revision) --------------------
net_c1b, _, _ = build_net(hidden_dim=9)
stage_c1b = net_c1b.core.stages[0]
orig_gm_c1b = stage_c1b.cell_lib.gm_raw.detach().clone()
orig_isat_c1b = stage_c1b.cell_lib.isat_raw.detach().clone()
orig_leak_c1b = stage_c1b.raw_leak.detach().clone()
orig_xmax_c1b = float(stage_c1b.x_max)
orig_clip_c1b = float(stage_c1b.clip_current)
orig_z_c1b = stage_c1b.z_logits.detach().clone()

saved_c1b = nlc.install_linear_reservoir_c1b(
    net_c1b, w_seed=0, g_in_seed=0,
    target_radius=nlc.C1B_LEAK_VALUE,
    leak_value=nlc.C1B_LEAK_VALUE,
    target_x_max=20.0, clip_current=0.0,
    drive_rms_target=nlc._drive_rms(u), u_seq_for_rms=u,
)
# Gates open, rails far, clip zeroed, W orthogonal (QR-from-seed-Ginibre).
assert_close(float(stage_c1b.z_logits.mean().item()), 12.0, "C1b core gates open")
assert_close(
    float(stage_c1b.boundary_z_logits.mean().item()), 12.0,
    "C1b boundary gates open",
)
assert_close(float(stage_c1b.x_max), 20.0, "C1b far rail")
assert_close(float(stage_c1b.clip_current), 0.0, "C1b clip zeroed")
W_c1b = getattr(stage_c1b, "_lin_G_edge", None)
assert W_c1b is not None and W_c1b.shape == (10, 10), \
    f"C1b W shape {None if W_c1b is None else W_c1b.shape}"
assert torch.isfinite(W_c1b).all(), "C1b W finite"
# W is scaled-orthogonal: max |eig| == target_radius (QR-of-Ginibre
# is orthogonal before scaling, so |eig|=1; after scaling by 0.05 the
# spectral radius lands at 0.05 +/- numerical noise).
eigs_c1b = torch.linalg.eigvals(W_c1b).abs()
err_radius = abs(float(eigs_c1b.max().item()) - nlc.C1B_LEAK_VALUE)
assert err_radius < 1e-4, f"C1b W radius off: {eigs_c1b.max().item()}"
# Uniform leak: raw_leak is constant across all nodes.
assert_close(
    float(stage_c1b.raw_leak.std().item()), 0.0, "C1b leak uniform",
)
# cell_library defaults untouched by C1b install.
assert torch.equal(
    stage_c1b.cell_lib.gm_raw, orig_gm_c1b,
), "C1b install mutated gm_raw"
assert torch.equal(
    stage_c1b.cell_lib.isat_raw, orig_isat_c1b,
), "C1b install mutated isat_raw"
print("AUDIT2_C1B_INSTALL_OK")

# Restore roundtrip returns the original state.
nlc.restore_linear_reservoir_v2(net_c1b, saved_c1b)
assert torch.equal(stage_c1b.z_logits, orig_z_c1b), "C1b z_logits not restored"
assert torch.equal(stage_c1b.raw_leak, orig_leak_c1b), "C1b raw_leak not restored"
assert_close(float(stage_c1b.x_max), orig_xmax_c1b, "C1b x_max not restored")
assert_close(
    float(stage_c1b.clip_current), orig_clip_c1b, "C1b clip not restored",
)
assert not hasattr(stage_c1b, "_lin_G_edge"), "C1b _lin_G_edge leaked"
print("AUDIT2_C1B_RESTORE_OK")

# --- 4c. C1b-real install/restore roundtrip (c1b-revision) --------------
net_c1br, _, _ = build_net(hidden_dim=9)
stage_c1br = net_c1br.core.stages[0]
orig_gm_c1br = stage_c1br.cell_lib.gm_raw.detach().clone()
orig_isat_c1br = stage_c1br.cell_lib.isat_raw.detach().clone()
orig_b_gm_c1br = stage_c1br.boundary_cell_lib.gm_raw.detach().clone()
orig_b_isat_c1br = stage_c1br.boundary_cell_lib.isat_raw.detach().clone()
orig_z_c1br = stage_c1br.z_logits.detach().clone()

saved_c1br = nlc.install_linear_reservoir_c1b_real(
    net_c1br, w_seed=0, g_in_seed=0,
    leak_value=nlc.C1B_LEAK_VALUE,
    boundary_drive_scale=0.5,
    gm_raw_fill=-10.0, isat_raw_fill=-10.0,
    target_x_max=20.0, clip_current=0.0,
    drive_rms_target=nlc._drive_rms(u), u_seq_for_rms=u,
)
# gm/isat fills applied (small-gm regime).
gm_after = stage_c1br.cell_lib.gm_raw.detach().clone()
isat_after = stage_c1br.cell_lib.isat_raw.detach().clone()
assert_close(float(gm_after.mean().item()), -10.0, "C1b-real gm_raw filled")
assert_close(float(isat_after.mean().item()), -10.0, "C1b-real isat_raw filled")
# Gates open, rails far.
assert_close(float(stage_c1br.z_logits.mean().item()), 12.0, "C1b-real core gates open")
assert_close(float(stage_c1br.x_max), 20.0, "C1b-real far rail")
# Uniform leak.
assert_close(
    float(stage_c1br.raw_leak.std().item()), 0.0, "C1b-real leak uniform",
)
print("AUDIT2_C1B_REAL_INSTALL_OK")

nlc.restore_linear_reservoir_c1b_real(net_c1br, saved_c1br)
assert torch.equal(stage_c1br.cell_lib.gm_raw, orig_gm_c1br), \
    "C1b-real gm_raw not restored"
assert torch.equal(stage_c1br.cell_lib.isat_raw, orig_isat_c1br), \
    "C1b-real isat_raw not restored"
assert torch.equal(stage_c1br.z_logits, orig_z_c1br), \
    "C1b-real z_logits not restored"
assert torch.equal(stage_c1br.boundary_cell_lib.gm_raw, orig_b_gm_c1br), \
    "C1b-real boundary gm not restored"
assert torch.equal(stage_c1br.boundary_cell_lib.isat_raw, orig_b_isat_c1br), \
    "C1b-real boundary isat not restored"
print("AUDIT2_C1B_REAL_RESTORE_OK")

# --- 4c2. Rung-1 install/restore roundtrip (knet-gated-memory) -----------
# Leak mapping: esn_leak=1 -> -ln(1e-3)/t_span; esn_leak=0.3 -> -ln(0.7).
assert_close(
    nlc.rung1_leak_from_esn(1.0, 1.0), -math.log(1e-3),
    "rung1 leak map esn_leak=1",
)
assert_close(
    nlc.rung1_leak_from_esn(0.3, 1.0), -math.log(0.7),
    "rung1 leak map esn_leak=0.3",
)
try:
    nlc.rung1_leak_from_esn(0.0, 1.0)
    raise AssertionError("rung1 leak map accepted esn_leak=0")
except ValueError:
    pass
net_r1, _, _ = build_net(hidden_dim=9)
stage_r1 = net_r1.core.stages[0]
orig_gm_r1 = stage_r1.cell_lib.gm_raw.detach().clone()
orig_gr_r1 = stage_r1.cell_lib.g_resistive_raw.detach().clone()
orig_lm_r1 = str(stage_r1.leak_mode)
saved_r1 = nlc.install_native_linear_rung1(
    net_r1, gm_raw_fill=-8.0, isat_raw_fill=-2.0,
    g_resistive_fill=-20.0, leak_value=0.36,
)
assert_close(float(stage_r1.cell_lib.gm_raw.mean().item()), -8.0, "rung1 gm_raw filled")
assert_close(float(stage_r1.cell_lib.isat_raw.mean().item()), -2.0, "rung1 isat_raw filled")
assert_close(
    float(stage_r1.cell_lib.g_resistive_raw.mean().item()), -20.0,
    "rung1 g_resistive_raw filled (shunt killed)",
)
assert_close(
    float(stage_r1.boundary_cell_lib.gm_raw.mean().item()), -8.0,
    "rung1 boundary gm_raw filled",
)
assert stage_r1.leak_mode == "non-programmable", "rung1 leak non-programmable"
assert_close(float(stage_r1.leak_constant), 0.36, "rung1 leak_constant")
print("AUDIT2_RUNG1_INSTALL_OK")

nlc.restore_native_linear_rung1(net_r1, saved_r1)
assert torch.equal(stage_r1.cell_lib.gm_raw, orig_gm_r1), \
    "rung1 gm_raw not restored"
assert torch.equal(stage_r1.cell_lib.g_resistive_raw, orig_gr_r1), \
    "rung1 g_resistive_raw not restored"
assert str(stage_r1.leak_mode) == orig_lm_r1, \
    "rung1 leak_mode not restored"
print("AUDIT2_RUNG1_RESTORE_OK")

# --- 4d. C1b restore_tanh flag changes dynamics (C3 sweep contract) ----
# Regression: the C3 sweep fills gm_raw on a C1b base, which is dead
# unless the rhs honors _lin_restore_tanh.
net_c1b_f, _, _ = build_net(hidden_dim=9)
saved_c1b_f = nlc.install_linear_reservoir_c1b(
    net_c1b_f, w_seed=0, g_in_seed=0,
    target_radius=nlc.C1B_LEAK_VALUE,
    leak_value=nlc.C1B_LEAK_VALUE,
    target_x_max=20.0, clip_current=0.0,
    drive_rms_target=nlc._drive_rms(u), u_seq_for_rms=u,
)
try:
    stage_c1b_f = net_c1b_f.core.stages[0]
    x0 = u.new_zeros(1, 10)
    with torch.no_grad():
        base_f = stage_c1b_f._forward_heun_sequence(
            x0=x0, t_span=1.0, num_steps=8, u_seq=u[:50],
        )
    stage_c1b_f._lin_restore_tanh = True
    with torch.no_grad():
        tanh_f = stage_c1b_f._forward_heun_sequence(
            x0=x0, t_span=1.0, num_steps=8, u_seq=u[:50],
        )
    assert not torch.allclose(base_f, tanh_f), \
        "C1b _lin_restore_tanh flag does not change dynamics (C3 dead-fill)"
    print("AUDIT2_C1B_RESTORE_TANH_FLAG_OK")
finally:
    nlc.restore_linear_reservoir_v2(net_c1b_f, saved_c1b_f)

# --- 4e. C1b-real tuner moves real G (hang-contract regression) ---------
# The tuner must rescale cell_lib.g_resistive_raw itself; a tuner that
# edits a dead attribute loops to max_iters (the local smoke hang).
net_c1br_t, _, _ = build_net(hidden_dim=9)
stage_c1br_t = net_c1br_t.core.stages[0]
orig_gres_t = stage_c1br_t.cell_lib.g_resistive_raw.detach().clone()
saved_c1br_t = nlc.install_linear_reservoir_c1b_real(
    net_c1br_t, w_seed=0, g_in_seed=0,
    leak_value=nlc.C1B_LEAK_VALUE,
    boundary_drive_scale=0.5,
    gm_raw_fill=-10.0, isat_raw_fill=-10.0,
    target_x_max=20.0, clip_current=0.0,
    drive_rms_target=nlc._drive_rms(u), u_seq_for_rms=u,
)
try:
    g_before = nlc._resistive_G(stage_c1br_t.cell_lib).detach().clone()
    r_t, h_t = nlc.tune_c1b_real_to_j_band(
        net_c1br_t, u, t_span=1.0, num_steps=8,
        washout=200, n_samples=1, device="cpu", max_iters=2,
    )
    assert len(h_t) >= 1 and math.isfinite(r_t), "C1b-real tuner no history"
    g_after = nlc._resistive_G(stage_c1br_t.cell_lib).detach().clone()
    if len(h_t) > 1:
        # Ran more than one iteration => the tuner must have moved G.
        assert not torch.allclose(g_before, g_after), \
            "C1b-real tuner did not move g_resistive_raw"
    print(f"AUDIT2_C1B_REAL_TUNER_OK radius={r_t:.4f} iters={len(h_t)}")
finally:
    nlc.restore_linear_reservoir_c1b_real(net_c1br_t, saved_c1br_t)
    assert torch.equal(
        stage_c1br_t.cell_lib.g_resistive_raw, orig_gres_t,
    ), "C1b-real g_resistive_raw not restored"

# --- 5. Spectral tuner loop ---------------------------------------------
net, _, _ = build_net(hidden_dim=9)
stage = net.core.stages[0]
saved = nlc.install_linear_reservoir_v2(
    net, G_edge_seed=0, G_in_seed=0, leak_seed=0,
    target_x_max=20.0, clip_current=0.0,
    drive_rms_target=nlc._drive_rms(u), u_seq_for_rms=u,
)
W_before = stage._lin_G_edge.detach().clone()
radius, history = nlc.tune_to_target_radius(
    net, u, t_span=1.0, num_steps=8,
    washout=200, n_samples=1, device="cpu", max_iters=2,
)
assert 1 <= len(history) <= 2, f"tuner history length {len(history)}"
assert math.isfinite(radius), f"tuner radius non-finite: {radius}"
assert all(math.isfinite(h["max_abs"]) for h in history), "tuner NaN leak"
print(f"AUDIT2_TUNER_OK radius={radius:.4f} iters={len(history)}")
nlc.restore_linear_reservoir_v2(net, saved)

# --- 5b. C1b spectral tuner runs (c1b-revision) -------------------------
saved_c1b_t = nlc.install_linear_reservoir_c1b(
    net, w_seed=0, g_in_seed=0,
    target_radius=nlc.C1B_LEAK_VALUE,
    leak_value=nlc.C1B_LEAK_VALUE,
    target_x_max=20.0, clip_current=0.0,
    drive_rms_target=nlc._drive_rms(u), u_seq_for_rms=u,
)
radius_c1b, history_c1b = nlc.tune_c1b_to_j_band(
    net, u, t_span=1.0, num_steps=8,
    washout=200, n_samples=1, device="cpu", max_iters=2,
)
assert len(history_c1b) >= 1, f"C1b tuner history: {len(history_c1b)}"
assert math.isfinite(radius_c1b), f"C1b tuner radius non-finite: {radius_c1b}"
print(f"AUDIT2_C1B_TUNER_OK radius={radius_c1b:.4f} iters={len(history_c1b)}")
nlc.restore_linear_reservoir_v2(net, saved_c1b_t)

# --- 6. C0 tiny ----------------------------------------------------------
row, meta = nlc.c0_esn_calibration(
    order=10, seed=0, n_reservoir=9, n_streams=1,
    train_samples_per_stream=250, washout=200,
    jacobian_samples=1, device="cpu",
)
assert math.isfinite(row.mc_total) and row.mc_total > 0, \
    f"C0 MC non-positive: {row.mc_total}"
assert math.isfinite(row.state_pr), "C0 PR non-finite"
assert math.isfinite(row.jac_max_abs), "C0 Jacobian non-finite"
assert "rail" in row.note and "N/A" in row.note, "C0 rail N/A note missing"
assert len(row.mc_per_delay) == 20, "C0 per-delay length"
print(f"AUDIT2_C0_OK mc={row.mc_total:.3f} pr={row.state_pr:.2f} "
      f"jac={row.jac_max_abs:.4f}")

# --- 7. R0 helpers tiny --------------------------------------------------
raw = nlc._raw_delay_ridge(u, y, n_taps=20, washout=200)
assert math.isfinite(raw["nrmse"]), "raw-delay Ridge non-finite"
assert raw["n_features"] == 21, "raw-delay feature count"
assert_close(nlc.R0_MATCHED_PARITY_TOL, 0.03, "parity tol refrozen")
print(f"AUDIT2_R0_HELPERS_OK raw_ridge={raw['nrmse']:.4f}")

# --- 8. Restore-flag plumbing (C2/C3 change dynamics) --------------------
saved = nlc.install_linear_reservoir_v2(
    net, G_edge_seed=0, G_in_seed=0, leak_seed=0,
    target_x_max=20.0, clip_current=0.0,
    drive_rms_target=nlc._drive_rms(u), u_seq_for_rms=u,
)
try:
    x0 = u.new_zeros(1, 10)
    with torch.no_grad():
        base = stage._forward_heun_sequence(
            x0=x0, t_span=1.0, num_steps=8, u_seq=u[:50],
        )
    stage._lin_restore_boundary = True
    with torch.no_grad():
        c2 = stage._forward_heun_sequence(
            x0=x0, t_span=1.0, num_steps=8, u_seq=u[:50],
        )
    assert not torch.allclose(base, c2), "C2 flag does not change dynamics"
    stage._lin_restore_boundary = False
    stage._lin_restore_tanh = True
    with torch.no_grad():
        c3 = stage._forward_heun_sequence(
            x0=x0, t_span=1.0, num_steps=8, u_seq=u[:50],
        )
    assert not torch.allclose(base, c3), "C3 flag does not change dynamics"
    print("AUDIT2_BISECT_FLAGS_OK")
finally:
    nlc.restore_linear_reservoir_v2(net, saved)

# --- 9. CLI registration -------------------------------------------------
res = subprocess.run(
    [sys.executable, "-B", "narma_linear_controls.py", "--help"],
    capture_output=True, text=True,
)
assert res.returncode == 0, res.stderr
for mode in ("r0", "c0", "c1", "c2c3c4", "c1b", "c1b_real", "c3_sweep", "rung1"):
    assert mode in res.stdout, f"CLI mode {mode} missing"
res = subprocess.run(
    [sys.executable, "-B", "narma_linear_controls.py", "c1", "--help"],
    capture_output=True, text=True,
)
assert res.returncode == 0 and "--hidden-dim" in res.stdout, "c1 CLI args"
res = subprocess.run(
    [sys.executable, "-B", "narma_linear_controls.py", "c2c3c4", "--help"],
    capture_output=True, text=True,
)
assert res.returncode == 0 and "--c3-gm-init" in res.stdout, "c2c3c4 CLI args"
# c1b / c1b_real / c3_sweep CLI args.
res = subprocess.run(
    [sys.executable, "-B", "narma_linear_controls.py", "c1b", "--help"],
    capture_output=True, text=True,
)
assert res.returncode == 0, res.stderr
for arg in ("--w-seed", "--j-low", "--j-high", "--leak-value", "--no-sidecar"):
    assert arg in res.stdout, f"c1b CLI arg {arg} missing"
res = subprocess.run(
    [sys.executable, "-B", "narma_linear_controls.py", "c1b_real", "--help"],
    capture_output=True, text=True,
)
assert res.returncode == 0, res.stderr
for arg in ("--gm-raw-fill", "--boundary-drive-scale", "--w-seed",
              "--j-uniform-min", "--j-stability-cap"):
    assert arg in res.stdout, f"c1b_real CLI arg {arg} missing"
res = subprocess.run(
    [sys.executable, "-B", "narma_linear_controls.py", "c3_sweep", "--help"],
    capture_output=True, text=True,
)
assert res.returncode == 0, res.stderr
assert "--c1b-sidecar-dir" in res.stdout, "c3_sweep CLI arg missing"
res = subprocess.run(
    [sys.executable, "-B", "narma_linear_controls.py", "rung1", "--help"],
    capture_output=True, text=True,
)
assert res.returncode == 0, res.stderr
for arg in ("--esn-leak", "--esn-input-scaling", "--gm-raw-fill",
            "--isat-raw-fill", "--g-resistive-fill", "--drive-scale"):
    assert arg in res.stdout, f"rung1 CLI arg {arg} missing"
print("AUDIT2_CLI_OK")

# --- 10. Sidecar roundtrip (c1b-revision) --------------------------------
# Save and reload a synthetic tuned base.  The audit checks the .pt
# payload shape and the JSON pointer schema; it does NOT exercise the
# full C1b install (which requires the spectral tuner; see smoke test).
with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    W_test = torch.eye(10) * 0.05
    G_in_test = torch.randn(9)
    payload = nlc.save_tuned_base(
        W=W_test, G_in=G_in_test, leak_value=0.05,
        w_seed=0, hidden_dim=9, hidden_dim_with_output=10,
        spectral_target_low=0.97, spectral_target_high=0.99,
        n_tuning_iterations=4, final_spectral_radius=0.98,
        n_params=42, c1b_metrics={"mc_total": 5.5, "nrmse": 0.6},
        sidecar_dir=tmp,
    )
    assert (tmp / nlc.C1B_SIDECAR_NAME).exists(), "sidecar .pt missing"
    assert (tmp / nlc.C1B_SIDECAR_JSON).exists(), "sidecar .json missing"
    loaded = nlc.load_tuned_base(tmp)
    assert torch.allclose(loaded["W"], W_test), "sidecar W roundtrip"
    assert torch.allclose(loaded["G_in"], G_in_test), "sidecar G_in roundtrip"
    assert loaded["meta"]["schema_version"] == 1, "sidecar schema_version"
    assert loaded["meta"]["leak_value"] == 0.05, "sidecar leak_value"
    assert loaded["meta"]["final_spectral_radius"] == 0.98, "sidecar radius"
    assert loaded["meta"]["c1b_metrics"]["mc_total"] == 5.5, "sidecar metrics"
print("AUDIT2_SIDECAR_ROUNDTRIP_OK")

# --- 11. Sidecar missing .pt raises (audit contract) --------------------
with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    # Write only the JSON; load should fail with FileNotFoundError on .pt.
    (tmp / nlc.C1B_SIDECAR_JSON).write_text(
        json.dumps({"schema_version": 1, "sidecar_pt_path": "missing.pt"}),
    )
    try:
        nlc.load_tuned_base(tmp)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("load_tuned_base did not raise on missing .pt")
print("AUDIT2_SIDECAR_MISSING_PT_OK")

print("\nALL_AUDITS_OK")
