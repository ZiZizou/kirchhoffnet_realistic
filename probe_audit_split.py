"""Audit #S for the split-leg implementation (plan canary-core-split).

Covers: arm fills land on the core lib only (boundary/output untouched),
node_diagnostics keys + determinism + Pearson sanity, s_raw seed-identity
across same-seed rebuilds, gate-zero z_logits restore, split CLI help
smoke, and a thin end-to-end run_split_leg (500 samples, CPU).

Run:
    python -B probe_audit_split.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")

import narma_advisor_probes as npr
import narma_experiment as ne

torch.manual_seed(0)


def build_base(**kw):
    args = dict(order=10, seed=0, t_span=1.0, num_steps=8, hidden_dim=25)
    args.update(kw)
    return npr._build_split_base_net(**args)


def assert_close(a, b, msg, tol=1e-6):
    assert abs(float(a) - float(b)) < tol, f"{msg}: {a} != {b}"


# --- S1. Arm fills land on core only ----------------------------------------
net = build_base()
stage = net.core.stages[0]
bnd_before = stage.boundary_cell_lib.gm_raw.detach().clone()
npr.apply_arm_fill(net, "A")
g_eff = torch.nn.functional.softplus(
    stage.cell_lib.g_resistive_raw.detach()).mean().item()
assert g_eff < 1e-6, f"S1 arm A G not killed: {g_eff}"
assert torch.equal(stage.boundary_cell_lib.gm_raw, bnd_before), \
    "S1 arm A touched boundary lib"
assert_close(stage.cell_lib.gm_raw.mean().item(), -3.5, "S1 arm A gm moved")
print("AUDIT-S1_ARM_A_FILL_OK")

net = build_base()
stage = net.core.stages[0]
g_before = stage.cell_lib.g_resistive_raw.detach().clone()
bnd_before = stage.boundary_cell_lib.gm_raw.detach().clone()
npr.apply_arm_fill(net, "B")
assert_close(stage.cell_lib.gm_raw.mean().item(), -20.0, "S1 arm B gm")
assert_close(stage.cell_lib.isat_raw.mean().item(), -20.0, "S1 arm B isat")
assert torch.equal(stage.cell_lib.g_resistive_raw, g_before), \
    "S1 arm B touched shunt"
assert torch.equal(stage.boundary_cell_lib.gm_raw, bnd_before), \
    "S1 arm B touched boundary lib"
try:
    npr.apply_arm_fill(net, "Z")
except ValueError:
    pass
else:
    raise AssertionError("S1 bad arm did not raise")
print("AUDIT-S1_ARM_B_FILL_OK")

# --- S2. node_diagnostics keys + determinism + Pearson ----------------------
net = build_base()
u_raw, y_raw = ne._gen_narma_train_streams(10, 0, 1, 500)
u = ne._scale_drive(u_raw[0], bipolar=True, order=10, input_scale=0.2)
states = npr._forward_states(net, u)
d1 = npr.node_diagnostics(net, states, u, hidden_dim=25)
d2 = npr.node_diagnostics(net, states, u, hidden_dim=25)
for key in ("max_abs_ratio", "shunt_G_sum", "sign_balance",
            "boundary_current", "s_raw_sign",
            "corr_shunt_vs_max", "corr_sign_vs_max",
            "corr_boundary_vs_max"):
    assert key in d1, f"S2 missing key {key}"
assert len(d1["max_abs_ratio"]) == 25, "S2 node count"
assert d1 == d2, "S2 diagnostics not deterministic"
assert_close(npr._pearson(torch.tensor([1.0, 2.0, 3.0]),
                          torch.tensor([1.0, 2.0, 3.0])), 1.0, "S2 pearson")
import math
assert math.isnan(npr._pearson(torch.tensor([1.0, 1.0]),
                               torch.tensor([1.0, 2.0]))), "S2 const NaN"
print("AUDIT-S2_NODE_DIAG_OK")

# --- S3. s_raw seed-identity across rebuilds --------------------------------
a = build_base().core.stages[0].cell_lib.s_raw.detach()
b = build_base().core.stages[0].cell_lib.s_raw.detach()
assert torch.equal(a, b), "S3 same-seed rebuilds differ in s_raw"
assert a.abs().max().item() < 1.0, "S3 s_raw not small-random init"
print("AUDIT-S3_SRAW_SEED_IDENTITY_OK")

# --- S4. gate-zero restores z_logits ----------------------------------------
net = build_base()
stage = net.core.stages[0]
z_before = stage.z_logits.detach().clone()
_ = npr._gate_zero_eval_forward(net, u)
assert torch.equal(stage.z_logits.detach(), z_before), \
    "S4 gate-zero did not restore z_logits"
print("AUDIT-S4_GATEZERO_RESTORE_OK")

# --- S5. split CLI help smoke ------------------------------------------------
try:
    npr.main(["split", "--help"])
except SystemExit as e:
    assert e.code == 0, f"S5 split --help exit {e.code}"
else:
    raise AssertionError("S5 split --help did not exit")
print("AUDIT-S5_SPLIT_CLI_HELP_OK")

# --- S6. thin end-to-end run_split_leg --------------------------------------
res = npr.run_split_leg(order=10, seed=0, device="cpu",
                        train_samples=500, jacobian_samples=1, n_streams=1)
for key in ("base", "arm_A", "arm_B", "arm_C"):
    assert key in res, f"S6 missing {key}"
    assert math.isfinite(res[key]["score"]["ridge_nrmse"]), \
        f"S6 {key} ridge non-finite"
    assert len(res[key]["node"]["max_abs_ratio"]) == 25, f"S6 {key} nodes"
print(f"S6 base ridge={res['base']['score']['ridge_nrmse']:.4f} "
      f"MC={res['base']['score']['mc_total']:.2f} "
      f"rail={100.0 * res['base']['score']['rail_frac']:.1f}%")
print("AUDIT-S6_SPLIT_E2E_OK")

print("ALL_SPLIT_AUDITS_OK")
