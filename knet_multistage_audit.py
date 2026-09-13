"""knet_multistage_audit: Phase-4 multi-stage plumbing invariants.

Verifies on CPU with fixed seeds:

[1] Window split: stage_times sum to t_span; stage_steps each >= 1 and
    sum to num_steps.
[2] Transfer identity at init: residual-relu-tanh with default init is
    the identity map (max abs diff 0).
[3] Pipeline drive: stage 1 has boundary edges, stages > 1 have none
    (boundary_first_stage_only).
[4] Single-stage equivalence: KirchhoffNet.forward_sequence on a 1-stage
    net matches DifferentialStage._forward_heun_sequence exactly.
[5] 2-stage grad flow: one backward reaches stage-1 cell params and
    transfer params (both nonzero).
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import narma_experiment as ne  # noqa: E402


def assert_close(a, b, name, tol=1e-5):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape mismatch {a.shape} vs {b.shape}")
    diff = float(np.max(np.abs(a - b)))
    if diff > tol:
        raise AssertionError(f"{name}: max abs diff {diff} > tol {tol}")
    print(f"  [OK] {name}: max abs diff = {diff:.3e}")


def _build(num_stages: int, seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net, t_span, num_steps = ne._build_fabric_net(
        10, seed, False, t_span=1.0, num_steps=8,
        cell_library="tanh_free", core_refresh_interval=0,
        leak_constant=None, compile_sequence=False,
        num_stages=num_stages, interstage_activation="residual-relu-tanh",
    )
    net.eval()
    return net


def test1_window_split():
    print("\n[1] t_span/steps split across 3 stages")
    net = _build(3)
    times = list(net.core.stage_times)
    steps = list(net.core.stage_steps)
    assert abs(sum(times) - 1.0) < 1e-9, f"times sum {sum(times)}"
    assert all(s >= 1 for s in steps), f"steps {steps}"
    assert sum(steps) == 8, f"steps sum {sum(steps)}"
    print(f"  [OK] times={times} steps={steps}")


def test2_transfer_identity():
    print("\n[2] residual-relu-tanh transfer is identity at init")
    net = _build(2)
    tr = net.core.transfers[0]
    torch.manual_seed(0)
    x = torch.randn(4, tr.in_nodes)
    with torch.no_grad():
        y = tr(x)
    assert_close(x.numpy(), y.numpy(), "transfer init identity", tol=1e-6)


def test3_pipeline_drive():
    print("\n[3] boundary drive on stage 1 only")
    net = _build(2)
    s0, s1 = net.core.stages[0], net.core.stages[1]
    assert bool(getattr(s0, "_has_boundary", False)), "stage 1 missing boundary"
    assert not bool(getattr(s1, "_has_boundary", False)), "stage 2 has boundary"
    n0 = int(s0.boundary_src.numel())
    print(f"  [OK] stage1 boundary edges={n0}, stage2 has none")


def test4_single_stage_equivalence():
    print("\n[4] forward_sequence == _forward_heun_sequence for 1 stage")
    net = _build(1)
    stage = net.core.stages[0]
    torch.manual_seed(1)
    B, T = 2, 5
    x0 = torch.randn(B, stage.num_nodes)
    u_seq = torch.randn(B, T, 1) * 0.5
    with torch.no_grad():
        a = stage._forward_heun_sequence(
            x0=x0, t_span=1.0, num_steps=8, u_seq=u_seq,
        )
        b = net.core.forward_sequence(x0, u_seq)
    assert_close(a.numpy(), b.numpy(), "1-stage helper equivalence")


def test5_two_stage_grads():
    print("\n[5] 2-stage backward reaches stage-1 cells + transfer")
    net = _build(2)
    net.train()
    torch.manual_seed(2)
    B, T = 1, 3
    x0 = torch.zeros(B, net.core.stages[0].num_nodes)
    u_seq = torch.randn(B, T, 1) * 0.5
    out = net.core.forward_sequence(x0, u_seq)
    out.pow(2).sum().backward()
    g_cell = net.core.stages[0].cell_lib.gm_raw.grad
    tr = net.core.transfers[0]
    g_w1 = tr.residual_w1.grad
    print(f"  stage0 gm_raw.grad max abs = {g_cell.abs().max().item():.3e}")
    print(f"  transfer w1.grad max abs = {g_w1.abs().max().item():.3e}")
    assert g_cell is not None and g_cell.abs().max().item() > 0.0
    assert g_w1 is not None and g_w1.abs().max().item() > 0.0
    print("  [OK] grads reach both stage-1 cells and transfer")


def main():
    test1_window_split()
    test2_transfer_identity()
    test3_pipeline_drive()
    test4_single_stage_equivalence()
    test5_two_stage_grads()
    print("\nALL_AUDITS_OK")


if __name__ == "__main__":
    main()
