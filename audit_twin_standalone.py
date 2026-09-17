"""Standalone audit for the twin_node_tanh probe (plan twin-node-tanh,
feature spec twin-probe-spec). Runs only the twin-related checks without
touching the pre-existing C1b/rung1 sequence-test crash unrelated to
this work.

Run from repo root: .\\venv\\Scripts\\python.exe -B audit_twin_standalone.py
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")

import narma_linear_controls as nlc  # noqa: E402
import narma_experiment as ne  # noqa: E402

torch.manual_seed(0)


def assert_close(a, b, msg, tol=1e-6):
    assert abs(float(a) - float(b)) < tol, f"{msg}: {a} != {b}"


def tiny_stream(n=200):
    u_raw, y_raw = ne._gen_narma_train_streams(
        order=10, seed=0, n_streams=1, n=n,
    )
    u = ne._scale_drive(u_raw[0], bipolar=True, order=10, input_scale=1.0)
    return u, y_raw[0]


# --- 12. Twin probe API surface -----------------------------------------
for name in (
    "twin_node_tanh", "twin_delay_bank", "_twin_build_weights",
    "_twin_run", "_twin_jacobian_at", "_twin_eigs_per_transition",
    "_twin_pre_stats", "TWIN_HARNESSES", "TWIN_DEFAULT_N_TAPS",
    "TWIN_PRE_LINEAR_REGIME_MEDIAN", "TWIN_ACTIVITY_LOW",
    "TWIN_ACTIVITY_HIGH",
):
    assert hasattr(nlc, name), f"missing twin symbol: {name}"
assert nlc.TWIN_HARNESSES == ("taps8", "scalar"), "TWIN_HARNESSES order"
print("AUDIT2_TWIN_API_OK")

# --- 13. Twin delay-bank shape and lag semantics -------------------------
u_dl = torch.arange(20, dtype=torch.float32)
db = nlc.twin_delay_bank(u_dl, n_taps=4)
assert db.shape == (20, 4), f"delay-bank shape {db.shape}"
assert torch.equal(db[:, 0], u_dl), "delay-bank tap 0 == u"
assert db[0, 1].item() == 0.0 and db[1, 1].item() == 0.0, (
    "delay-bank tap 1 pre-pad"
)
assert db[3, 1].item() == u_dl[2].item(), "delay-bank tap 1 causal"
assert db[3, -1].item() == u_dl[0].item(), "delay-bank tap n-1 causal"
assert torch.equal(
    db[:, -1], torch.cat([torch.zeros(3), u_dl[:-3]]),
), "delay-bank tap n-1 full column"
print("AUDIT2_TWIN_DELAY_BANK_OK")

# --- 14. Twin W_mix spectral-radius normalization -------------------------
torch.manual_seed(0)
W_norm, W_in = nlc._twin_build_weights(
    n_nodes=8, n_in=3, w_seed=0, win_seed=0,
    target_radius=0.7, dtype=torch.float32, device=torch.device("cpu"),
)
eigs = torch.linalg.eigvals(W_norm.to(torch.float64)).abs()
assert_close(float(eigs.max().item()), 0.7, "twin W_mix radius=0.7", tol=1e-4)
assert W_norm.shape == (8, 8), f"twin W shape {W_norm.shape}"
assert W_in.shape == (8, 3), f"twin W_in shape {W_in.shape}"
print("AUDIT2_TWIN_RADIUS_NORM_OK")

# --- 15. Twin deterministic at fixed seeds -------------------------------
u, _ = tiny_stream()
drive1 = nlc._twin_drive(u, "scalar", n_taps=8, input_scale=1.0)
drive2 = nlc._twin_drive(u, "scalar", n_taps=8, input_scale=1.0)
assert torch.equal(drive1, drive2), "twin scalar drive deterministic"
W, W_in = nlc._twin_build_weights(
    n_nodes=10, n_in=1, w_seed=7, win_seed=3,
    target_radius=0.9, dtype=torch.float32, device=torch.device("cpu"),
)
s1 = nlc._twin_run(W=W, W_in=W_in, drive=drive1, a=1.0)
s2 = nlc._twin_run(W=W, W_in=W_in, drive=drive2, a=1.0)
assert torch.equal(s1, s2), "twin run deterministic at fixed seeds"
assert s1.shape == (drive1.shape[0], 10), f"twin state shape {s1.shape}"
assert torch.isfinite(s1).all(), "twin states non-finite"
print("AUDIT2_TWIN_DETERMINISM_OK")

# --- 16. Twin Jacobian analytic form vs autograd (single-step F) ---------
torch.manual_seed(0)
W_j = torch.randn(6, 6, dtype=torch.float64) * 0.3
W_in_j = torch.randn(6, 2, dtype=torch.float64) * 0.5
x_j = torch.randn(6, dtype=torch.float64) * 0.2
d_j = torch.randn(2, dtype=torch.float64)
a_twin = 0.7
J_analytic = nlc._twin_jacobian_at(
    W=W_j, W_in=W_in_j, x=x_j, drive_row=d_j, a=a_twin,
)


def _single_step_map(x_flat):
    x_ = x_flat.view(-1)
    pre = W_j @ x_ + W_in_j @ d_j
    x_new = (1.0 - a_twin) * x_ + a_twin * torch.tanh(pre)
    return x_new.view(-1)


J_auto = torch.autograd.functional.jacobian(
    _single_step_map, x_j.detach().clone(),
)
assert torch.allclose(J_analytic, J_auto, atol=1e-6), (
    "twin analytic Jacobian != autograd Jacobian"
)
print("AUDIT2_TWIN_JACOBIAN_OK")

# --- 17. Twin standalone purity: no stage/cell-library mutation ---------
# Purity check: twin_node_tanh() must NOT attach _lin_* attributes to
# the module, must NOT touch the fabric's stage/cell_lib state, and must
# return a structurally valid report dict.
before_keys = set(nlc.__dict__.keys())
report = nlc.twin_node_tanh(
    order=10, seed=0, device="cpu", hidden_dim=10,
    harness="scalar", radius=0.9, leak_a=1.0, input_scale=1.0,
    n_streams=1, train_samples_per_stream=200, washout=50,
    jacobian_samples=2,
)
after_keys = set(nlc.__dict__.keys())
new_module_attrs = after_keys - before_keys
# Twin should not have polluted the module with stage-mutation helpers.
for forbidden in ("_lin_G_edge", "_lin_G_in", "_lin_restore_tanh"):
    assert forbidden not in new_module_attrs, (
        f"twin attached module attr {forbidden}"
    )
# Pillar public symbols must still be present.
for sym in ("install_linear_reservoir_v2", "c1_linear_reservoir",
            "twin_node_tanh"):
    assert sym in nlc.__dict__, f"twin dropped public symbol {sym}"
assert report["harness"] == "scalar", "scalar harness path"
assert report["instrument"].state_pr > 0, "twin state_pr non-zero"
# Activity-band flag wired (twin-probe-spec: tanh-units screen).
assert isinstance(report["activity_in_band"], bool), "activity_in_band type"
assert list(report["activity_band"]) == [
    nlc.TWIN_ACTIVITY_LOW, nlc.TWIN_ACTIVITY_HIGH,
], "activity_band values"
print("AUDIT2_TWIN_PURITY_OK")

# --- 18. Twin CLI args registered ----------------------------------------
res = subprocess.run(
    [sys.executable, "-B", "narma_linear_controls.py", "twin", "--help"],
    capture_output=True, text=True,
)
assert res.returncode == 0, res.stderr
for arg in (
    "--harness", "--n-taps", "--radius", "--leak-a", "--input-scale",
    "--w-seed", "--win-seed", "--max-delay", "--jacobian-samples",
):
    assert arg in res.stdout, f"twin CLI arg {arg} missing"
print("AUDIT2_TWIN_CLI_OK")

print("\nALL_TWIN_AUDITS_OK")
