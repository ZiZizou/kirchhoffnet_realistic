"""Audit #T for the regime-transplant implementation (plan narma-regime-transplant).

Covers: coupled-fill byte-identity, positional isat pairing (duplicate-gm
corners keep distinct Isat), isat length-mismatch guard, the ``raw:<val>``
programmable-leak token, anchor pre-fill readout exclusion + strict missing-
key failure, row-tag/resume-tag round-trip for decoupled corners, the
``pre_fill_overrides_after=True`` path (was UnboundLocalError), the
4-outcome canary verdict table, the 54-corner grid count, and the anchor
mutual-exclusivity CLI guard.

Run:
    python -B probe_audit_transplant.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, ".")

import narma_advisor_probes as npr
import narma_experiment as ne

torch.manual_seed(0)

ANCHOR_CKPT = Path("outputs_alliance_can/narma_refresh2/fabric_refresh_k2_seed0.pt")


def build_net(**kw):
    args = dict(order=10, seed=0, freeze_read=False, t_span=1.0, num_steps=8,
                cell_library="tanh_free", core_refresh_interval=0,
                leak_constant=None, compile_sequence=False)
    args.update(kw)
    net, _, _ = ne._build_fabric_net(**args)
    return net


def assert_close(a, b, msg, tol=1e-6):
    assert abs(float(a) - float(b)) < tol, f"{msg}: {a} != {b}"


# --- T1. Coupled-fill byte-identity ---------------------------------------
# isat=None with per_lib=None must fill gm_raw and isat_raw identically on
# every library (legacy path untouched).
net = build_net()
npr.apply_gain_override(net, gm_init=-3.5, isat_init=None,
                        leak_mode="slow-fixed")
for lib_name in ("cell_lib", "boundary_cell_lib", "output_ode_cell_lib"):
    lib = getattr(net.core.stages[0], lib_name)
    assert torch.equal(lib.gm_raw, lib.isat_raw), \
        f"T1 coupled fill diverged on {lib_name}"
    assert_close(lib.gm_raw.mean().item(), -3.5, f"T1 fill value {lib_name}")
print("AUDIT-T1_COUPLED_IDENTITY_OK")

# --- T2. Positional isat pairing with duplicate gm -------------------------
# gain=(-3.5,-3.5) + isat=(None,-2) must yield TWO corners with distinct
# Isat (a gm->isat dict would collapse them into one).
corners = npr._select_corner_subset(
    (0.2,), ("raw:-0.5",), (-3.5, -3.5), None,
    isat_grid=(None, -2.0),
)
assert len(corners) == 2, f"T2 expected 2 corners, got {len(corners)}"
assert corners[0][2] == -3.5 and corners[0][3] is None, f"T2 corner0 {corners[0]}"
assert corners[1][2] == -3.5 and corners[1][3] == -2.0, f"T2 corner1 {corners[1]}"
print("AUDIT-T2_POSITIONAL_ISAT_OK")

# --- T3. isat length mismatch guards ---------------------------------------
try:
    npr._select_corner_subset((0.2,), ("raw:-0.5",), (-3.5,), None,
                              isat_grid=(None, -2.0))
except ValueError:
    print("AUDIT-T3_ISAT_LENGTH_GUARD_OK")
else:
    raise AssertionError("T3: length mismatch did not raise")

# --- T4. raw:<val> programmable-leak token ---------------------------------
net = build_net()
npr.apply_gain_override(net, gm_init=-3.5, leak_mode="raw:-0.5")
stage = net.core.stages[0]
assert stage.leak_mode == "programmable", \
    f"T4 raw: token must set programmable, got {stage.leak_mode}"
assert_close(stage.raw_leak.mean().item(), -0.5, "T4 raw_leak fill")
# Effective leak is softplus(-0.5) ~= 0.474 (heun leak_floor=0.0); the bare
# numeric token instead sets leak_constant directly (different mechanism).
leak_eff = torch.nn.functional.softplus(
    torch.tensor(-0.5)).item()
assert abs(leak_eff - 0.474) < 0.01, f"T4 softplus sanity {leak_eff}"
net2 = build_net()
npr.apply_gain_override(net2, gm_init=-3.5, leak_mode=-0.5)
assert net2.core.stages[0].leak_mode == "non-programmable", \
    "T4 bare numeric must stay non-programmable"
print("AUDIT-T4_RAW_LEAK_TOKEN_OK")

# --- T5. Anchor pre-fill: readout excluded, dynamics copied, strict --------
assert ANCHOR_CKPT.exists(), f"anchor ckpt missing: {ANCHOR_CKPT}"
net = build_net(core_refresh_interval=2)
before_mapper = {k: v.detach().clone()
                 for k, v in net.named_parameters()
                 if k.startswith("output_mapper.")}
before_src = net.core.stages[0].src.detach().clone()
pre = npr._make_anchor_pre_fill(str(ANCHOR_CKPT), "cpu")
pre(net)
ckpt_state = torch.load(ANCHOR_CKPT, map_location="cpu",
                        weights_only=False)["model_state"]
for k in ("core.stages.0.cell_lib.gm_raw",
          "core.stages.0.boundary_cell_lib.isat_raw",
          "core.stages.0.raw_leak"):
    own = dict(net.named_parameters())[k]
    assert torch.equal(own, ckpt_state[k].to(own.dtype)), \
        f"T5 dynamics not copied for {k}"
for k, v in before_mapper.items():
    assert torch.equal(dict(net.named_parameters())[k], v), \
        f"T5 readout overwritten for {k} (must be ridge-refit, not copied)"
assert torch.equal(net.core.stages[0].src, before_src), \
    "T5 topology index buffer must come from the fresh build"
print("AUDIT-T5_ANCHOR_READOUT_EXCLUDED_OK")

# --- T5b. Anchor strict: wrong hidden_dim hard-fails ------------------------
net_bad = build_net(hidden_dim=36, core_refresh_interval=2)
try:
    pre(net_bad)
except ValueError:
    print("AUDIT-T5B_ANCHOR_STRICT_OK")
else:
    raise AssertionError("T5b: incompatible build did not hard-fail")

# --- T6. Row tag == resume tag for decoupled corners ------------------------
tag_row = npr._e0_config_tag(
    order=10, seed=0, device="cpu", hidden_dim=25, refresh=0,
    t_span=1.0, num_steps=8, gm_init=-3.5, leak_mode="raw:-0.5",
    drive=0.2, washout=200, jacobian_samples=1,
    boundary_fan_out=None, isat_init=-2.0,
    per_lib_overrides={"cell_lib": {"gm": -3.5, "isat": -2.0}},
)
tag_plain = npr._e0_config_tag(
    order=10, seed=0, device="cpu", hidden_dim=25, refresh=0,
    t_span=1.0, num_steps=8, gm_init=-3.5, leak_mode="raw:-0.5",
    drive=0.2, washout=200, jacobian_samples=1,
)
assert tag_row != tag_plain and "_isat" in tag_row and "_plib" in tag_row, \
    f"T6 decoupled tag missing suffixes: {tag_row}"
print("AUDIT-T6_TAG_SUFFIX_OK")

# --- T7. pre_fill_overrides_after=True path (was UnboundLocalError) ---------
with tempfile.TemporaryDirectory() as td:
    rows = npr.e0_sweep(
        order=10, seed=0, device="cpu",
        gain_grid=(-3.5,), leak_grid=("raw:-0.5",), drive_grid=(0.2,),
        n_streams=1, train_samples_per_stream=600,
        washout=200, jacobian_samples=1,
        t_span=1.0, num_steps=8, hidden_dim=25,
        pre_fill=npr._make_anchor_pre_fill(str(ANCHOR_CKPT), "cpu"),
        pre_fill_overrides_after=True,
        refresh=2,
        progress_json=Path(td) / "prog.json",
    )
    assert len(rows) == 1 and rows[0].ridge_nrmse == rows[0].ridge_nrmse, \
        "T7 override-on-top sweep failed"
print("AUDIT-T7_OVERRIDES_AFTER_OK")


def _dummy_row(**kw):
    base = dict(
        config_tag="dummy", hidden_dim=25, n_params=1,
        gm_init=-3.5, isat_init=-2.0, leak_mode="raw:-0.5", drive=0.2,
        per_lib_overrides_json="", boundary_fan_out="",
        ridge_nrmse=0.5, ridge_r2=0.5, mc_total_washout_corrected=5.0,
        state_pr=7.0, jac_max_abs=0.9, jac_min_abs=0.1, jac_mean_abs=0.5,
        jac_rank_proxy=2.0, rail_frac=0.01, sat_max_ratio=0.5,
        gate_zero_ridge_nrmse=0.6, gate_zero_ridge_r2=0.4,
        gate_zero_mc_total=4.0, gate_zero_state_pr=6.0,
        gate_zero_jac_max_abs=0.9, gate_zero_jac_min_abs=0.1,
        gate_zero_jac_mean_abs=0.5, gate_zero_jac_rank_proxy=2.0,
        gate_zero_rail_frac=0.01, gate_zero_sat_max_ratio=0.5,
        rail_disqualified=False, pass_init_regime=True, pass_rail=True,
        pass_all=True, gate_zero_rail_disqualified=False,
        gate_zero_pass_init_regime=True, gate_zero_pass_rail=True,
        gate_zero_pass_all=True,
    )
    base.update(kw)
    return npr.E0SweepRow(**base)


# --- T8. Canary 4-outcome table ---------------------------------------------
# (a) calibrated anchor + passing canary -> EXPAND
v = npr.canary_verdict(0.64, _dummy_row())
assert v["outcome"] == "EXPAND-TO-FULL-GRID", f"T8a {v}"
# (b) calibrated anchor + failing canary, no flip -> WOUNDED
v = npr.canary_verdict(0.64, _dummy_row(
    mc_total_washout_corrected=0.3, ridge_nrmse=0.70,
    gate_zero_ridge_nrmse=0.70, gate_zero_mc_total=0.3))
assert v["outcome"] == "TRANSPLANT-WOUNDED", f"T8b {v}"
# (c) anchor >= 0.72 -> BLIND regardless of canary
v = npr.canary_verdict(0.73, _dummy_row())
assert v["outcome"] == "INSTRUMENT-BLIND", f"T8c {v}"
# (d) calibrated anchor + gate-zero flip without pass -> CORE-USEFUL
v = npr.canary_verdict(0.64, _dummy_row(
    mc_total_washout_corrected=0.3, ridge_nrmse=0.70,
    gate_zero_ridge_nrmse=0.90, gate_zero_mc_total=0.05))
assert v["outcome"] == "CORE-USEFUL-NO-PASS", f"T8d {v}"
assert v["gate_zero_flip"] is True, "T8d flip flag"
# (e) off-nominal anchor -> unregistered zone
v = npr.canary_verdict(0.69, _dummy_row())
assert v["outcome"] == "ANCHOR-OFF-NOMINAL", f"T8e {v}"
print("AUDIT-T8_VERDICT_TABLE_OK")

# --- T9. Full grid counts 54 -------------------------------------------------
assert npr.transplant_corner_count() == 54, \
    f"T9 grid count {npr.transplant_corner_count()}"
print("AUDIT-T9_GRID_COUNT_OK")

# --- T10. Anchor CLI exclusivity ----------------------------------------------
# Note: the '=' form is required for values starting with '-' (standard
# argparse: a bare '--isat-grid -2,...' never reaches the guard).
try:
    npr.main(["e0", "--order", "10", "--seed", "0",
              "--anchor-ckpt", str(ANCHOR_CKPT),
              "--isat-grid=-2.0,-2.0,-2.0,-2.0",
              "--output", tempfile.mkdtemp()])
except SystemExit as e:
    assert e.code != 0, "T10 exclusivity must fail non-zero"
    print("AUDIT-T10_ANCHOR_EXCLUSIVITY_OK")
else:
    raise AssertionError("T10: anchor+isat combo did not fail")

print("ALL_TRANSPLANT_AUDITS_OK")
