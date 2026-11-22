"""Compliance (3.4) coverage: every current family must reach the rhs.

Phase 3b W5 step 3.  The 3.4 leg-wise output compliance replaces the input
compliance gate and the soft clip.  The W5 4a bug was that the boundary and
the ota_mesh families computed their rolled-off currents but never scattered
them into ``acc``: under ``hw_compliance`` the network input (boundary OTAs)
and the temporal readout were disconnected.

This module checks every family of ``DifferentialStage.rhs`` separately, in the
``freeze_read`` on/off and the ``freeze_boundary`` on/off paths (and the
``freeze_temporal_read`` on/off path of the readout families):

* the *core* edges (the tanh part of the OTA current),
* the *boundary* fan-out OTAs,
* the *shared-sense* OTAs + the crossbar accumulator,
* the *ota_mesh* (output-ODE) readout,
* the parallel *resistive shunts* (a conductor: they must NOT be rolled off),
* the *leak* (a conductance to ground: it must NOT be rolled off).

Method: at ``x = 0`` (mid-rail) the 3.4 roll-off factor is exactly 1 --
``leg_factors`` normalises the P and N leg factors by their value at
``x_n = 0`` -- so the compliance path and the plain path must give the *same*
rhs at ``x = 0``.  The families are isolated by zeroing the edge gate of the
other families (the gate multiplies the current of its family), so a family
that is computed but not scattered shows up as a mismatch of exactly its own
contribution.

The OTA input offset (``theta_raw``) is set to a non-zero value in the test:
with the init value 0 every tanh current is exactly 0 at ``x = 0`` and the
comparison would be vacuous.

Known pre-existing inconsistency (NOT touched here: it is baked into the
golden data): with ``freeze_read=False`` the *plain* path counts the parallel
resistive shunt of the core family TWICE (``cell_lib(...)`` already contains
it and the ``_has_resistive`` block of ``rhs`` adds it again), while the
frozen path and the compliance path count it once.  The resistive-shunt test
therefore checks the compliance path against the analytic shunt current.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "golden"))
sys.path.insert(0, os.path.dirname(HERE))

import make_golden as mg  # noqa: E402
from hw import HWConfig  # noqa: E402
from hw.constraints import HWConstraints  # noqa: E402

D = torch.float64
B = 6
THETA = 0.5            # non-zero OTA input offset (see the module docstring)

# Flags every comparison needs:
#  * hw_no_alpha drops the legacy ``ALPHA * u`` term (the 3.4 legs have none),
#  * hw_vref_symmetric puts the sense Vref rail at 0 V at the init raw (the
#    sense legs roll off at that rail).
BASE = dict(hw_no_alpha=True, hw_vref_symmetric=True)

# The families of ``rhs`` and the edge gate that multiplies their current.
FAMILIES = {
    "TOY_NET": ("core", "boundary", "sense"),
    "FRIEDMAN2_NET": ("core", "boundary", "mesh"),
}
GATE_ATTR = {
    "core": "z_logits",
    "boundary": "boundary_z_logits",
    "sense": "readout_sense_z_logits",
    "mesh": "output_ode_z_logits",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _build(spec, cons=None, **build_over):
    """Build a net (float64) with the given hardware constraints."""
    hw = HWConfig(constraints=HWConstraints(**cons)) if cons else None
    spec = dict(spec)
    build = dict(spec["build"])
    build.update(build_over)
    spec["build"] = build
    with mg.f64():
        mg.seed_all(int(spec["seed"]))
        net = mg._build_net(spec, hw=hw).double()
    return net


def _set_theta(net, value=THETA):
    """Give every OTA a non-zero input offset (the init value is 0)."""
    with torch.no_grad():
        for st in net.core.stages:
            for lib in (st.cell_lib, getattr(st, "boundary_cell_lib", None),
                        getattr(st, "output_ode_cell_lib", None),
                        getattr(st, "readout_sense_cell_lib", None)):
                theta = getattr(lib, "theta_raw", None) if lib is not None else None
                if isinstance(theta, torch.Tensor):
                    theta.fill_(value)


def _gate_snapshot(net):
    """Snapshot every edge gate of the net (so the gates can be restored)."""
    out = []
    for st in net.core.stages:
        for attr in GATE_ATTR.values():
            p = getattr(st, attr, None)
            if isinstance(p, torch.Tensor):
                out.append((p, p.detach().clone()))
    return out


def _kill_all_gates(net):
    """Zero every edge gate: no current family is left."""
    with torch.no_grad():
        for st in net.core.stages:
            for attr in GATE_ATTR.values():
                p = getattr(st, attr, None)
                if isinstance(p, torch.Tensor):
                    p.fill_(-30.0)


def _restore_gates(snapshot):
    with torch.no_grad():
        for p, v in snapshot:
            p.copy_(v)


def _kill_gates(net, keep):
    """Zero the edge gate of every family except ``keep``.

    The gate multiplies the current of its family, so the other families then
    contribute ~1e-13 and the rhs comparison isolates ``keep``.
    """
    with torch.no_grad():
        for st in net.core.stages:
            for fam, attr in GATE_ATTR.items():
                if fam == keep:
                    continue
                p = getattr(st, attr, None)
                if isinstance(p, torch.Tensor):
                    p.fill_(-30.0)


def _rhs(stage, u, x, freeze_read, freeze_boundary, freeze_temporal_read):
    """The stage rhs at ``x`` with the frozen constants taken at the same ``x``."""
    gate = stage.edge_gain("core", u)
    i_edge = stage._compute_i_edge_const(x, gate) if freeze_read else None
    i_bnd = stage._compute_frozen_boundary(u, x) if freeze_boundary else None
    i_out = stage._compute_frozen_readout(u, x) if freeze_temporal_read else None
    return stage.rhs(x, u=u, i_edge_const=i_edge, i_boundary_const=i_bnd,
                     i_readout_const=i_out, vca_gate_core=gate)


def _inputs():
    """A network input inside the rails (the input compliance gate is 1 there)."""
    g = torch.Generator().manual_seed(3)
    return (torch.rand(B, 4, generator=g, dtype=D) * 2.0 - 1.0)


# ---------------------------------------------------------------------------
# the per-family comparison
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spec_name", sorted(FAMILIES))
@pytest.mark.parametrize("freeze_read", [False, True])
@pytest.mark.parametrize("freeze_boundary", [False, True])
@pytest.mark.parametrize("freeze_temporal_read", [False, True])
def test_every_family_matches_the_plain_path_at_x0(spec_name, freeze_read,
                                                   freeze_boundary, freeze_temporal_read):
    """At ``x = 0`` the roll-off factor is 1: the compliance rhs == the plain rhs."""
    spec = getattr(mg, spec_name)
    plain = _build(spec, BASE)
    comp = _build(spec, dict(BASE, hw_compliance=True))
    comp.load_state_dict(plain.state_dict())
    _set_theta(plain)
    _set_theta(comp)
    u = _inputs()
    snap_p = _gate_snapshot(plain)
    snap_c = _gate_snapshot(comp)
    for family in FAMILIES[spec_name]:
        # restore the gates: they are killed family by family (cumulatively)
        _restore_gates(snap_p)
        _restore_gates(snap_c)
        _kill_gates(plain, family)
        _kill_gates(comp, family)
        for sp, sc in zip(plain.core.stages, comp.core.stages):
            x = torch.zeros(B, sp.num_nodes, dtype=D)
            f_p = _rhs(sp, u, x, freeze_read, freeze_boundary, freeze_temporal_read)
            f_c = _rhs(sc, u, x, freeze_read, freeze_boundary, freeze_temporal_read)
            # the family must really contribute, else the comparison is vacuous
            assert float(f_c.abs().max()) > 0.0, (spec_name, family)
            torch.testing.assert_close(f_c, f_p, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("spec_name", sorted(FAMILIES))
def test_the_compliance_path_actually_rolls_the_legs_off(spec_name):
    """Sanity: near the rail the compliance path differs from the plain path."""
    spec = getattr(mg, spec_name)
    plain = _build(spec, BASE)
    comp = _build(spec, dict(BASE, hw_compliance=True))
    comp.load_state_dict(plain.state_dict())
    _set_theta(plain)
    _set_theta(comp)
    u = _inputs()
    sp, sc = plain.core.stages[0], comp.core.stages[0]
    x = torch.zeros(B, sp.num_nodes, dtype=D)
    x[:, :] = 0.0
    # push the nodes close to the hard rail: the outward leg must turn off
    x = x + 4.5 * torch.ones_like(x)
    f_p = _rhs(sp, u, x, False, False, False)
    f_c = _rhs(sc, u, x, False, False, False)
    assert float((f_c - f_p).abs().max()) > 0.0


# ---------------------------------------------------------------------------
# the resistive shunt and the leak are conductors: they are NOT rolled off
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spec_name", sorted(FAMILIES))
@pytest.mark.parametrize("freeze_read", [False, True])
def test_resistive_shunt_enters_the_rhs_once_and_is_not_rolled_off(spec_name, freeze_read):
    """The parallel resistive shunt is a conductor: no rail factor, counted once.

    The shunt contribution is isolated by switching it off (``G -> 0``) and on
    again and differencing the two rhs values; the result must equal the
    analytic ``G * (x_src - x_dst)`` scatter (no roll-off factor).
    """
    spec = getattr(mg, spec_name)
    comp = _build(spec, dict(BASE, hw_compliance=True))
    _set_theta(comp)
    u = _inputs()
    for st in comp.core.stages:
        g_raw = getattr(st.cell_lib, "g_resistive_raw", None)
        if not isinstance(g_raw, torch.Tensor):
            continue
        # a state with a voltage difference across the core edges
        x = torch.zeros(B, st.num_nodes, dtype=D)
        x[:, :] = torch.linspace(0.0, 0.6, st.num_nodes, dtype=D)
        with torch.no_grad():
            g_raw.fill_(-30.0)                     # G = softplus(-30) ~ 0
            off = _rhs(st, u, x, freeze_read, False, False)
            g_raw.fill_(0.5)                       # G = softplus(0.5) = 1.07
            on = _rhs(st, u, x, freeze_read, False, False)
        xs, xd = x[:, st.src], x[:, st.dst]
        mask = torch.sigmoid(st.z_logits).unsqueeze(0)
        i_res = st.cell_lib.resistive_current(xs, xd) * mask
        expect = torch.zeros_like(x)
        expect.index_add_(1, st.dst, i_res.float().double())
        expect.index_add_(1, st.src, -i_res.float().double())
        expect = expect / st.c_eff
        got = (on - off)
        torch.testing.assert_close(got, expect, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("spec_name", sorted(FAMILIES))
@pytest.mark.parametrize("freeze_read", [False, True])
def test_leak_is_present_and_identical_in_both_paths(spec_name, freeze_read):
    """The leak is a conductance to ground: it is never rolled off."""
    spec = getattr(mg, spec_name)
    plain = _build(spec, BASE)
    comp = _build(spec, dict(BASE, hw_compliance=True))
    comp.load_state_dict(plain.state_dict())
    _set_theta(plain)
    _set_theta(comp)
    u = _inputs()
    _kill_all_gates(plain)
    _kill_all_gates(comp)
    for sp, sc in zip(plain.core.stages, comp.core.stages):
        # every node at the same (mid-rail) voltage: no resistive current, the
        # leak is the only current left
        x = torch.full((B, sp.num_nodes), 0.5, dtype=D)
        f_p = _rhs(sp, u, x, freeze_read, False, False)
        f_c = _rhs(sc, u, x, freeze_read, False, False)
        leak = sp._effective_leak().unsqueeze(0).to(D) * x
        torch.testing.assert_close(f_p, -leak / sp.c_eff, rtol=1e-6, atol=1e-12)
        torch.testing.assert_close(f_c, -leak / sp.c_eff, rtol=1e-6, atol=1e-12)
