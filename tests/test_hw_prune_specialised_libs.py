"""Partial pruning must preserve the stage-owned specialised cell libraries.

W8b reproduction: ``topology.prune_stage`` rebuilt a ``DifferentialStage``
without the boundary fan-out / output-ODE (ota_mesh) cell libraries, their
gate logits, their VCA taps, the output-ODE accumulator nodes, the VCA
projections and the leak mode.  On the Friedman-2 golden net a *partial*
prune therefore dropped the boundary + readout families (~40% of the model
parameters) and changed the forward pass.

The keep-all fast path (commit e22af1d) hides the bug because it returns a
deep copy of the untouched core.
"""

import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "golden"))
import make_golden as mg  # noqa: E402
from topology import prune_network  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _build(hw=None):
    """Build the Friedman-2 golden net (float64, seeded, eval mode)."""
    torch.set_num_threads(2)
    mg.seed_all(4242)
    with mg.f64():
        net = mg._build_net(mg.FRIEDMAN2_NET, hw=hw).double().eval()
    return net


def _hw_preset(name):
    """Build an ``HWConfig`` from ``hw/presets/<name>.json``."""
    import argparse

    from hw import parse_hw_args
    from hw.cli import hw_config_from_args

    p = argparse.ArgumentParser()
    parse_hw_args(p)
    return hw_config_from_args(p.parse_args(["--hw-preset", name]))


def _core_edge_param_cost(stage) -> int:
    """Number of parameters carried by ONE core edge of ``stage``.

    ``z_logits`` + every per-edge cell-library raw parameter (including the
    hw-specific ``r_raw`` of ``hw_ratio_ab``) + the per-edge VCA core tap.
    """
    n_edges = max(1, int(stage.num_edges()))
    cost = 1  # z_logits
    for p in stage.cell_lib.parameters():
        cost += p.numel() // n_edges
    if getattr(stage, "vca_v_core", None) is not None:
        cost += stage.vca_v_core.numel() // n_edges
    return cost


def _family_snapshot(core):
    """Snapshot every specialised-family parameter of ``core`` by name.

    The per-edge CORE VCA taps (``vca_v_core`` / ``vca_b_core``) are
    excluded: they are re-indexed to the surviving core edges, so they are
    checked separately (row-by-row against the surviving edges).
    """
    out = {}
    for name, p in core.named_parameters():
        if "vca_v_core" in name or "vca_b_core" in name:
            continue
        if any(f in name for f in ("boundary", "output_ode", "vca", "readout_sense")):
            out[name] = p.detach().clone()
    return out


def _gate_edges_to_zero(core, removed):
    """Force the listed core edges of ``core`` to gate exactly 0 (sigma(-inf))."""
    with torch.no_grad():
        for stage_idx, edges in removed.items():
            z = core.stages[stage_idx].z_logits
            for e in edges:
                z.data[e] = float("-inf")


# ---------------------------------------------------------------------------
# the reproduction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "hw_name", [None, "phase3a_core"], ids=["no-hw", "phase3a-core"],
)
def test_partial_prune_keeps_specialised_libs_params_and_forward(hw_name):
    """A partial prune keeps the boundary/readout families bit-for-bit.

    (a) the boundary and output-ODE cell libraries (and their gate logits and
        VCA taps) survive with exactly the pre-prune values,
    (b) the parameter count drops by exactly the removed core edges' share,
    (c) the pruned forward equals the unpruned forward with the removed
        edges' gates forced to exactly 0.
    """
    hw = None if hw_name is None else _hw_preset(hw_name)
    net = _build(hw)
    core = net.core
    x = mg._friedman2_xy(16, 23)[0]

    # Two core edges of stage 0 and one of stage 1 are pruned.
    removed = {0: [0, 1], 1: [5]}
    n_removed = sum(len(e) for e in removed.values())

    # Reference: the *unpruned* net with the removed edges gated to exactly 0.
    ref = copy.deepcopy(net)
    _gate_edges_to_zero(ref.core, removed)
    _gate_edges_to_zero(core, removed)
    with torch.no_grad():
        before = mg.model_forward(ref, x)

    old_count = sum(p.numel() for p in core.parameters())
    old_families = _family_snapshot(core)

    pruned, _remaps = prune_network(core, edge_threshold=0.01)

    # (a) the specialised libraries are present and copied exactly ---------
    assert len(pruned.stages) == len(core.stages)
    for i, (old_s, new_s) in enumerate(zip(core.stages, pruned.stages)):
        assert new_s._has_boundary, f"stage {i}: boundary family was dropped"
        assert new_s._has_output_ode, f"stage {i}: output-ODE family was dropped"
        assert new_s.boundary_cell_lib is not None
        assert new_s.output_ode_cell_lib is not None
        # buffers (indices) are carried over
        assert new_s.boundary_src.tolist() == old_s.boundary_src.tolist()
        assert new_s.boundary_dst.tolist() == old_s.boundary_dst.tolist()
        assert new_s.output_ode_src.tolist() == old_s.output_ode_src.tolist()
        assert new_s.output_ode_dst.tolist() == old_s.output_ode_dst.tolist()
        # gate logits are carried over
        torch.testing.assert_close(
            new_s.boundary_z_logits, old_s.boundary_z_logits, rtol=0, atol=0)
        torch.testing.assert_close(
            new_s.output_ode_z_logits, old_s.output_ode_z_logits, rtol=0, atol=0)
        # the VCA projections/taps survive
        assert new_s.vca_enabled
        torch.testing.assert_close(new_s.vca_W, old_s.vca_W, rtol=0, atol=0)
        torch.testing.assert_close(
            new_s.vca_W_core, old_s.vca_W_core, rtol=0, atol=0)
        torch.testing.assert_close(
            new_s.vca_v_boundary, old_s.vca_v_boundary, rtol=0, atol=0)
        torch.testing.assert_close(
            new_s.vca_v_readout, old_s.vca_v_readout, rtol=0, atol=0)
        # the per-edge core VCA taps are re-indexed to the surviving edges
        kept = [
            e for e in range(int(old_s.num_edges()))
            if float(old_s.edge_gates().detach()[e]) > 0.01
        ]
        assert old_s.vca_v_core is not None
        assert tuple(new_s.vca_v_core.shape) == (
            len(kept), int(old_s.vca_v_core.shape[1]))
        torch.testing.assert_close(
            new_s.vca_v_core, old_s.vca_v_core[kept], rtol=0, atol=0)
        # the accumulator tail survives
        assert new_s.num_nodes == old_s.num_nodes

    new_families = _family_snapshot(pruned)
    for name, ref_val in old_families.items():
        assert name in new_families, f"family parameter {name} was dropped"
        torch.testing.assert_close(
            new_families[name], ref_val, rtol=0, atol=0,
            msg=lambda m, name=name: f"family parameter {name} changed: {m}",
        )

    # (b) the parameter count drops by exactly the removed edges' share -----
    new_count = sum(p.numel() for p in pruned.parameters())
    expected_drop = sum(
        _core_edge_param_cost(core.stages[si]) for si, edges in removed.items()
        for _ in edges
    )
    assert n_removed == 3
    assert new_count == old_count - expected_drop, (
        f"param count {new_count} != {old_count} - {expected_drop}"
    )

    # (c) the forward pass is unchanged ------------------------------------
    net.core = pruned
    with torch.no_grad():
        after = mg.model_forward(net, x)
    torch.testing.assert_close(after, before, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("hw_name", ["phase3a_core", "phase3a_full"])
def test_partial_prune_preserves_hardware_edge_params_and_trainability(hw_name):
    hw = _hw_preset(hw_name)
    net = _build(hw)
    core = net.core
    x = mg._friedman2_xy(16, 23)[0]
    removed = {0: [0, 1], 1: [5]}
    # The golden fixture does not opt into learnable clip sharpness. Install
    # the same dead compliance parameter that the production builder creates.
    if hw_name == "phase3a_full":
        for stage in core.stages:
            stage._learnable_clip_sharpness = True
            stage.clip_sharpness_raw = torch.nn.Parameter(
                torch.tensor(0.25, dtype=stage.z_logits.dtype), requires_grad=False)
    ref = copy.deepcopy(net)
    _gate_edges_to_zero(ref.core, removed)
    _gate_edges_to_zero(core, removed)
    with torch.no_grad():
        before = mg.model_forward(ref, x)
    snapshots = {}
    for si, stage in enumerate(core.stages):
        lib = stage.cell_lib
        snapshots[si] = {n: (getattr(lib, n).detach().clone(), getattr(lib, n).requires_grad)
                         for n in ("a_raw", "b_raw", "r_raw", "cap_a_raw", "cap_b_raw") if hasattr(lib, n)}
    clip_flags = {n: p.requires_grad for n, p in core.named_parameters()
                  if n.endswith("clip_sharpness_raw")}
    pruned, _ = prune_network(core, edge_threshold=0.01)
    for si, stage in enumerate(pruned.stages):
        old = core.stages[si]
        kept = [e for e in range(int(old.num_edges()))
                if float(old.edge_gates().detach()[e]) > 0.01]
        for name, (values, flag) in snapshots[si].items():
            p = getattr(stage.cell_lib, name)
            torch.testing.assert_close(p, values[kept], rtol=0, atol=0)
            assert p.requires_grad == flag
        for name, flag in ((n, f) for n, f in clip_flags.items()):
            if name.startswith(f"stages.{si}."):
                assert dict(pruned.named_parameters())[name].requires_grad == flag
    if hw_name == "phase3a_core":
        for stage in pruned.stages:
            assert not stage.cell_lib.a_raw.requires_grad
            assert not stage.cell_lib.b_raw.requires_grad
    if hw_name == "phase3a_full":
        assert clip_flags and all(not f for f in clip_flags.values())
        assert all(not dict(pruned.named_parameters())[n].requires_grad for n in clip_flags)
    net.core = pruned
    with torch.no_grad():
        after = mg.model_forward(net, x)
    torch.testing.assert_close(after, before, rtol=1e-6, atol=1e-7)
