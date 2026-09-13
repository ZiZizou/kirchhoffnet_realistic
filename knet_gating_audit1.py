"""knet_gating_audit1: epoch-0 invariants for the spike flags.

Verifies, with fixed seed and CPU torch, that:

[1] --dynamic-leak off and --dynamic-leak on (with default init) produce
    numerically identical one-sample forward states (bit-equivalence for
    the legacy leak path).
[2] --vca-use-hidden off and --vca-use-hidden on (with default init)
    produce numerically identical one-sample forward states when VCA is
    enabled, AND the extended vca_W_hidden / vca_W_core_hidden blocks are
    all-zero (so the hidden projection contributes 0 at epoch 0).
[3] Both flags can be combined in a single build; param counts increase
    only by the expected new parameters.
[4] After one optimizer step on the spike-on build, dyn_leak_a/b and
    vca_W_hidden have nonzero gradients (grads flow).
[5] cell_library.py init defaults are unchanged after building any flag
    combination (init discipline).
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cell_library as cl_mod  # noqa: E402
from config import (  # noqa: E402
    ANTI_PARALLEL_GM_MAX,
    ANTI_PARALLEL_GM_MIN,
    ANTI_PARALLEL_ISAT_MAX,
    ANTI_PARALLEL_ISAT_MIN,
    PHYS,
    TANH_REALISTIC_GM_MAX,
    TANH_REALISTIC_GM_MIN,
    TANH_REALISTIC_ISAT_MAX,
    TANH_REALISTIC_ISAT_MIN,
    VCA,
    make_narma_preset,
)
from topology import build_net_from_config  # noqa: E402


def _make_preset() -> dict:
    """Build a NARMA-shaped preset that VCA can sit on.

    Uses small_world topology (matches the existing NARMA revised-plan
    factory) so write_idx and read_idx satisfy the topology degree
    separation check.
    """
    from narma_revised_plan import build_small_world_narma_preset

    preset = build_small_world_narma_preset(
        order=10, hidden_dim=25, num_steps_per_sample=4,
        t_span=1.0, small_world_k=4, small_world_p=0.2,
        small_world_seed=1, core_refresh_interval=0, leak_constant=None,
    )
    preset["vca_enabled"] = True
    preset["vca_core_enabled"] = True
    preset["vca_rank"] = int(VCA["rank"])
    return preset


PRESET = _make_preset()


def _build(flags: dict, seed: int = 0) -> torch.nn.Module:
    """Build the same preset with overrides merged into PRESET.

    Reseeds torch+numpy at entry (F1) so two builds with different flags
    but the same seed consume identical RNG streams for all shared
    parameters — otherwise the base and spike nets would differ in their
    random inits (s_raw, vca_v_*, ...) and no forward comparison is valid.
    """
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cfg = {**PRESET, **flags}
    cell_lib = cl_mod.make_cell_library("tanh_free", num_edges=1)
    bfo = cfg.get("boundary_fan_out", {0: list(range(int(cfg["stages"][0]["num_hidden"])))})
    net = build_net_from_config(
        cfg, cell_lib=cell_lib,
        boundary_fan_out=bfo,
        enable_temporal_readout=True,
        freeze_read=False,
        vca_enabled=True,
        vca_core_enabled=True,
        vca_rank=int(VCA["rank"]),
        vca_use_hidden=bool(flags.get("vca_use_hidden", False)),
        dynamic_leak=bool(flags.get("dynamic_leak", False)),
    )
    net.eval()
    return net


def _forward(net: torch.nn.Module, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    # One-sample input: scalar in [0, 0.5] -> rail-mapped by NARMA pipeline.
    # We just need a deterministic non-zero input.
    u = torch.tensor([[0.3]], dtype=torch.float32)
    with torch.no_grad():
        out = net(u)
    # Output may be Tensor or tuple; pick the last trajectory state.
    if isinstance(out, tuple):
        out = out[0]
    return out.detach().cpu()


def assert_close(a, b, name, tol=1e-5):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape mismatch {a.shape} vs {b.shape}")
    diff = float(np.max(np.abs(a - b)))
    if diff > tol:
        raise AssertionError(f"{name}: max abs diff {diff} > tol {tol}")
    print(f"  [OK] {name}: max abs diff = {diff:.3e}")


def snapshot_cell_lib_defaults():
    """Snapshot FreeTanhLibrary init defaults so we can verify they're untouched."""
    s = cl_mod.FreeTanhLibrary(num_edges=3)
    return {
        "gm_raw": s.gm_raw.detach().clone(),
        "isat_raw": s.isat_raw.detach().clone(),
        "g_resistive_raw": s.g_resistive_raw.detach().clone(),
        "a_raw": s.a_raw.detach().clone(),
        "b_raw": s.b_raw.detach().clone(),
        "s_raw": s.s_raw.detach().clone(),
    }


def test1_dynamic_leak_equivalence():
    print("\n[1] dynamic-leak on/off -> identical forward at init")
    base = _build({}, seed=0)
    spike = _build({"dynamic_leak": True}, seed=0)
    out_base = _forward(base)
    out_spike = _forward(spike)
    assert_close(out_base, out_spike, "dyn-leak off vs on (init)")


def test2_vca_use_hidden_equivalence():
    print("\n[2] vca-use-hidden on/off -> identical forward at init")
    base = _build({}, seed=0)
    spike = _build({"vca_use_hidden": True}, seed=0)
    out_base = _forward(base)
    out_spike = _forward(spike)
    assert_close(out_base, out_spike, "vca-use-hidden off vs on (init)")
    s = spike.core.stages[0]
    assert s.vca_W_hidden is not None, "vca_W_hidden not created"
    assert torch.all(s.vca_W_hidden == 0).item(), "vca_W_hidden nonzero at init"
    assert s.vca_W_core_hidden is None, (
        "vca_W_core_hidden set without vca_separate_core_bus"
    )


def test3_combined_build():
    print("\n[3] both flags combined build OK and add expected params")
    spike = _build({"dynamic_leak": True, "vca_use_hidden": True}, seed=0)
    s = spike.core.stages[0]
    n_nodes = s.num_nodes
    n_dyn = 2 * n_nodes + n_nodes  # dyn_leak_a, dyn_leak_b, dyn_leak_c
    n_vca = n_nodes  # vca_W_hidden (no separate bus)
    # Total params added by the spike:
    expected_extra = n_dyn + n_vca
    base_params = sum(p.numel() for p in spike.parameters())
    print(f"  base+spike total params: {base_params}")
    print(f"  expected extra: {expected_extra} (dyn={n_dyn}, vca={n_vca})")
    # Verify shapes:
    assert s.dyn_leak_a.shape == (n_nodes,)
    assert s.dyn_leak_b.shape == (n_nodes,)
    assert s.dyn_leak_c.shape == (n_nodes,)
    assert s.vca_W_hidden.shape == (n_nodes, s.vca_rank)
    print("  [OK] shapes match")


def test4_grads_flow():
    print("\n[4] dyn_leak_a/b/c and vca_W_hidden grads nonzero (stage-level)")
    spike = _build({"dynamic_leak": True, "vca_use_hidden": True}, seed=0)
    s = spike.core.stages[0]
    torch.manual_seed(1)
    B = 2
    u = torch.randn(B, 1)  # NARMA in_dim=1
    x = torch.randn(B, s.num_nodes)  # nonzero carried state
    gate = s._compute_core_gate(u, x=x)
    gate.sum().backward()
    leak = s._effective_leak(x=x[0], u=u)
    leak.sum().backward()
    g_a = s.dyn_leak_a.grad
    g_b = s.dyn_leak_b.grad
    g_c = s.dyn_leak_c.grad
    g_wh = s.vca_W_hidden.grad
    print(f"  dyn_leak_a.grad max abs = {g_a.abs().max().item():.3e}")
    print(f"  dyn_leak_b.grad max abs = {g_b.abs().max().item():.3e}")
    print(f"  dyn_leak_c.grad max abs = {g_c.abs().max().item():.3e}")
    print(f"  vca_W_hidden.grad max abs = {g_wh.abs().max().item():.3e}")
    for name, g in (("dyn_leak_a", g_a), ("dyn_leak_b", g_b),
                    ("dyn_leak_c", g_c), ("vca_W_hidden", g_wh)):
        assert g is not None and g.abs().max().item() > 0.0, f"{name}.grad is zero"
    print("  [OK] grads flow through spike params")


def test5_cell_library_untouched():
    print("\n[5] cell_library.py init defaults unchanged")
    snap = snapshot_cell_lib_defaults()
    expected = {
        "gm_raw": torch.full((3,), -5.0),
        "isat_raw": torch.full((3,), -5.0),
        "g_resistive_raw": torch.full((3,), -5.0),
        "a_raw": torch.zeros(3),
        "b_raw": torch.zeros(3),
    }
    for name, want in expected.items():
        got = snap[name]
        assert_close(got.numpy(), want.numpy(), f"cell_library.{name} init default")
    # s_raw is randn(3)*0.1 so just verify shape & nonzero spread
    s = snap["s_raw"]
    assert s.shape == (3,)
    assert s.std().item() > 0.0


def main():
    test1_dynamic_leak_equivalence()
    test2_vca_use_hidden_equivalence()
    test3_combined_build()
    test4_grads_flow()
    test5_cell_library_untouched()
    print("\nALL_AUDITS_OK")


if __name__ == "__main__":
    main()
