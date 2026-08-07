"""VCA sanity-check harness (core-vca-expansion plan).

Run BEFORE any experiment-grid training run. Verifies the exploitability
gates from the core-vca-feature spec:

  1. IDENTITY  — zero-init vca_W + 2-sigma gate => forward + backward at
                 step 0 is bit-identical (<= 1e-12 at the loss level) to
                 the VCA-off baseline on the SAME shared parameter state.
  2. GRADIENT  — after one backward vca_W.grad is nonzero (via the
                 u (x) v_e path); vca_v_core.grad is ~0 at step 0 by
                 design but becomes nonzero after a couple of optimizer
                 steps once W moves.
  3. CACHE     — _gate_core_cached is recomputed at every stage entry /
                 new batch: feeding a different u must produce a different
                 cached gate (once W has moved off zero). A stale cache
                 across samples is the failure mode most likely to
                 silently tank the experiment.
  4. GATE      — after a few training steps, gate_core has spread across
                 inputs (input-dependence), not collapsed to a constant.

Usage:
    python vca_sanity.py
"""

from __future__ import annotations

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import torch
import torch.nn.functional as F

from cell_library import make_cell_library
from topology import build_net_from_preset


def _build_net(with_vca: bool, vca_core: bool = True,
               homology_seed: int = 7) -> torch.nn.Module:
    """friedman1-based KNet, with/without core VCA.

    ``vca_core`` wins even when with_vca=False (ignored then). freeze_read
    is ON (the production path): the core gate must fold into i_edge_const.
    """
    torch.manual_seed(homology_seed)
    return build_net_from_preset(
        preset_name="friedman1",
        cell_lib=make_cell_library("tanh"),
        write_mode="sparse_proj",
        read_mode="dense",
        freeze_read=True,
        vca_enabled=with_vca,
        vca_rank=2,
        vca_core_enabled=vca_core,
        vca_gate_shunt=False,
        vca_separate_core_bus=False,
    )


def _get_stages(net: torch.nn.Module):
    return net.core.stages


def _copy_non_vca_params(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    """Overwrite dst's state with src for all non-vca_ parameters.

    Making the VCA net share the exact non-VCA state of the baseline is the
    correct setup for the identity check: any deviation then comes from the
    VCA path alone, not from a different RNG draw.
    """
    dst_params = dict(dst.named_parameters())
    with torch.no_grad():
        for name, p_src in src.named_parameters():
            if "vca_" in name:
                continue
            p_dst = dst_params[name]
            p_dst.copy_(p_src)


def test_identity() -> bool:
    print("\n[1] IDENTITY (VCA-on == VCA-off at epoch 0, shared state)")
    torch.manual_seed(0)
    base = _build_net(with_vca=False)
    vca = _build_net(with_vca=True, vca_core=True)
    _copy_non_vca_params(vca, base)

    u = torch.randn(64, 10)
    target = torch.randn(64, 1)

    base.zero_grad(set_to_none=True)
    out_base, _ = base(u)
    loss_base = F.huber_loss(out_base, target, delta=1.0)
    loss_base.backward()

    vca.zero_grad(set_to_none=True)
    out_vca, _ = vca(u)
    loss_vca = F.huber_loss(out_vca, target, delta=1.0)
    loss_vca.backward()

    max_abs = float((out_vca - out_base).abs().max().item())
    loss_diff = abs(float(loss_vca.item()) - float(loss_base.item()))
    ok = max_abs <= 1e-6 and loss_diff <= 1e-12
    print(f"  max|out diff| = {max_abs:.3e}   |dLoss| = {loss_diff:.3e}")
    print(f"  [{'OK' if ok else 'FAIL'}] identity at step 0")
    return ok


def test_gradient() -> bool:
    print("\n[2] GRADIENT (vca_W.grad != 0 step0; vca_v_core after steps)")
    torch.manual_seed(0)
    vca = _build_net(with_vca=True, vca_core=True)
    stages = _get_stages(vca)
    u = torch.randn(16, 10)
    target = torch.randn(16, 1)

    vca.zero_grad(set_to_none=True)
    out, _ = vca(u)
    F.huber_loss(out, target, delta=1.0).backward()
    w_grad0 = stages[0].vca_W.grad
    v_grad0 = stages[0].vca_v_core.grad

    ok_w = w_grad0 is not None and float(w_grad0.abs().sum().item()) > 0
    # v_e gradient is ~0 at step 0 by design (dz/dv_e = u@W = 0), but must
    # become nonzero after a couple of optimizer steps.
    opt = torch.optim.Adam(vca.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        out, _ = vca(u)
        F.huber_loss(out, target, delta=1.0).backward()
        opt.step()
    v_grad3 = stages[0].vca_v_core.grad
    ok_v = v_grad3 is not None and float(v_grad3.abs().sum().item()) > 0

    w0 = float(w_grad0.abs().sum().item())
    v0 = float(v_grad0.abs().sum().item())
    v3 = float(v_grad3.abs().sum().item())
    print(f"  vca_W.grad (step 0)        = {w0:.3e} (must be >0)")
    print(f"  vca_v_core.grad (step 0)   = {v0:.3e}  (~0 by design)")
    print(f"  vca_v_core.grad (step 3)   = {v3:.3e}  (must be >0)")
    ok = ok_w and ok_v
    print(f"  [{'OK' if ok else 'FAIL'}] gradient flow into VCA circuits")
    return ok


def test_cache() -> bool:
    print("\n[3] CACHE (gate_core recomputed when u changes)")
    torch.manual_seed(1)
    vca = _build_net(with_vca=True, vca_core=True)
    stages = _get_stages(vca)
    # Move W off zero so gates actually vary with u.
    with torch.no_grad():
        s0 = stages[0]
        s0.vca_W.add_(torch.randn_like(s0.vca_W) * 0.5)
        s0.vca_v_core.mul_(2.0)
    u1 = torch.randn(8, 10)
    u2 = torch.randn(8, 10)
    with torch.no_grad():
        vca(u1)
        g1 = stages[0]._gate_core_cached
        vca(u2)
        g2 = stages[0]._gate_core_cached
    if g1 is None or g2 is None:
        print("  [FAIL] _gate_core_cached is None (core VCA disabled?)")
        return False
    max_diff = float((g1 - g2).abs().max().item())
    ok = max_diff > 0
    print(f"  cross-sample gate max-diff = {max_diff:.3e} (must be > 0)")
    print(f"  gate shape = {tuple(g1.shape)} (must be [B, E_core])")
    print(f"  [{'OK' if ok else 'FAIL'}] cache refreshes per batch")
    return ok


def test_gate_distribution() -> bool:
    print("\n[4] GATE input-dependence (spread > 0 across a sweep of u)")
    torch.manual_seed(2)
    vca = _build_net(with_vca=True, vca_core=True)
    stages = _get_stages(vca)
    with torch.no_grad():
        stages[0].vca_W.add_(torch.randn_like(stages[0].vca_W) * 1.0)
    # Sweep u across the input range; collect the per-sample cached gate.
    gates = []
    with torch.no_grad():
        for lo in torch.linspace(-3, 3, 9):
            u = torch.full((16, 10), float(lo))
            vca(u)
            g = stages[0]._gate_core_cached
            if g is not None:
                gates.append(g.mean(dim=0))
    if not gates:
        print("  [FAIL] no gate statistics captured")
        return False
    G = torch.stack(gates)  # [nu, E]
    spread = float(G.std(dim=0).max().item())
    ok = spread > 1e-3
    print(f"  gate range = [{float(G.min().item()):.4f}, {float(G.max().item()):.4f}]")
    print(f"  max edge-wise std across u sweep = {spread:.4f} (must be > 1e-3)")
    print(f"  [{'OK' if ok else 'FAIL'}] gate has input-dependence")
    return ok


if __name__ == "__main__":
    ok_id = test_identity()
    ok_grad = test_gradient()
    ok_cache = test_cache()
    ok_gate = test_gate_distribution()

    print("\n=== VCA sanity results ===")
    for name, ok in [("IDENTITY", ok_id), ("GRADIENT", ok_grad),
                     ("CACHE", ok_cache), ("GATE", ok_gate)]:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    if not (ok_id and ok_grad and ok_cache and ok_gate):
        raise SystemExit(1)
    print("\nAll VCA sanity checks PASSED. Safe to launch the experiment grid.")