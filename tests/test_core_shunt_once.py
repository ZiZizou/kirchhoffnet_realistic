"""Regression test: the core resistive shunt enters the rhs EXACTLY ONCE.

Bug fixed in commit 7cda290: with freeze_read=False the dynamic core path of
DifferentialStage.rhs called cell_lib(...) (FreeTanhLibrary.forward =
i_tanh + i_res) and the explicit _has_resistive block added the shunt AGAIN,
so the core shunt entered the KCL sum twice (2.000x on TOY_NET).  The frozen
path (freeze_read=True), the boundary/sense/mesh paths and the hw_compliance
path counted it once.

Checked here:
  1. with freeze_read=False the core shunt contribution to the rhs equals the
     analytic G*(x_src - x_dst) scatter (edge-gated) EXACTLY ONCE;
  2. the core VCA gate reaches the shunt only with vca_gate_shunt=True;
  3. freeze_read=True and False give the SAME rhs at the stage-entry state
     (and differ away from it).
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

D = torch.float64
B = 5
G_OFF = -30.0      # softplus(-30) ~ 0 -> shunt off
G_ON = 0.5         # softplus(0.5) = 1.07
SPECS = ["TOY_NET", "FRIEDMAN2_NET"]


def _build(spec_name, freeze_read, vca_gate_shunt=False):
    spec = dict(getattr(mg, spec_name))
    spec["build"] = dict(spec["build"])
    spec["build"]["freeze_read"] = freeze_read
    spec["build"]["vca_gate_shunt"] = vca_gate_shunt
    with mg.f64():
        mg.seed_all(int(spec["seed"]))
        return mg._build_net(spec).double()


def _perturb_vca(net, std=0.5, seed=7):
    """The core VCA gate is exactly 1.0 at init; make it non-trivial."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in net.named_parameters():
            if "vca_W" in name:
                p.add_(std * torch.randn(p.shape, generator=g, dtype=p.dtype))


def _inputs():
    g = torch.Generator().manual_seed(3)
    return torch.rand(B, 4, generator=g, dtype=D) * 2.0 - 1.0


def _state(stage, lo=0.0, hi=0.6):
    x = torch.zeros(B, stage.num_nodes, dtype=D)
    x[:, :] = torch.linspace(lo, hi, stage.num_nodes, dtype=D)
    return x


def _rhs(stage, u, x, gate, i_edge_const=None):
    return stage.rhs(x, u=u, i_edge_const=i_edge_const, i_boundary_const=None,
                     i_readout_const=None, vca_gate_core=gate)


def _analytic_shunt(stage, x, gate, vca_gate_shunt):
    """The single-count analytic shunt: G*(x_src - x_dst), edge-gated.

    Returns ``(scattered [B, N], max |per-edge current|)``.  The second value
    is the tolerance scale: the KCL scatter cancels, so a node current can be
    far smaller than the individual edge terms (which the rhs accumulates in
    float32).
    """
    i_res = stage.cell_lib.resistive_current(x[:, stage.src], x[:, stage.dst])
    i_res = i_res * torch.sigmoid(stage.z_logits).unsqueeze(0)
    if gate is not None and vca_gate_shunt:
        i_res = i_res * gate
    mag = float(i_res.detach().abs().max())
    acc = torch.zeros_like(x, dtype=D)
    acc.index_add_(1, stage.dst, i_res)
    if not stage.read_only_source:
        acc.index_add_(1, stage.src, -i_res)
    return acc / stage.c_eff, mag


def _shunt_contribution(stage, u, x, gate):
    """rhs(G on) - rhs(G off): the shunt contribution."""
    g_raw = stage.cell_lib.g_resistive_raw
    with torch.no_grad():
        g_raw.fill_(G_OFF)
        off = _rhs(stage, u, x, gate)
        g_raw.fill_(G_ON)
        on = _rhs(stage, u, x, gate)
    return on - off


@pytest.mark.parametrize("spec_name", SPECS)
def test_core_shunt_enters_the_rhs_exactly_once(spec_name):
    """The core shunt is counted ONCE (it used to be counted twice)."""
    net = _build(spec_name, freeze_read=False)
    _perturb_vca(net)
    u = _inputs()
    for st in net.core.stages:
        x = _state(st)
        gate = st.edge_gain("core", u)
        got = _shunt_contribution(st, u, x, gate)
        expect, mag = _analytic_shunt(st, x, gate, False)
        assert mag > 0.0
        torch.testing.assert_close(got, expect, rtol=1e-6, atol=1e-6 * mag)


@pytest.mark.parametrize("spec_name", SPECS)
def test_core_shunt_ignores_the_core_vca_gate_unless_requested(spec_name):
    """--vca-gate-shunt is the only way the core VCA gate reaches the shunt."""
    u = _inputs()
    net_plain = _build(spec_name, freeze_read=False, vca_gate_shunt=False)
    net_gated = _build(spec_name, freeze_read=False, vca_gate_shunt=True)
    _perturb_vca(net_plain)
    _perturb_vca(net_gated)
    for st_p, st_g in zip(net_plain.core.stages, net_gated.core.stages):
        st_g.load_state_dict(st_p.state_dict())
        x = _state(st_p)
        gate = st_p.edge_gain("core", u)
        assert gate is not None
        assert float((gate - 1.0).detach().abs().max()) > 0.0
        plain = _shunt_contribution(st_p, u, x, gate)
        gated = _shunt_contribution(st_g, u, x, gate)
        exp_plain, mag_p = _analytic_shunt(st_p, x, gate, False)
        exp_gated, mag_g = _analytic_shunt(st_g, x, gate, True)
        torch.testing.assert_close(plain, exp_plain, rtol=1e-6, atol=1e-6 * mag_p)
        torch.testing.assert_close(gated, exp_gated, rtol=1e-6, atol=1e-6 * mag_g)
        assert float((gated - plain).detach().abs().max()) > 0.0


@pytest.mark.parametrize("spec_name", SPECS)
def test_freeze_read_on_and_off_agree_at_the_entry_state(spec_name):
    """At the stage-entry state the frozen and the dynamic rhs are the same."""
    net = _build(spec_name, freeze_read=False)
    u = _inputs()
    for st in net.core.stages:
        x0 = _state(st)
        gate = st.edge_gain("core", u)
        with torch.no_grad():
            dyn = _rhs(st, u, x0, gate)
            frozen = _rhs(st, u, x0, gate,
                          i_edge_const=st._compute_i_edge_const(x0, gate))
        torch.testing.assert_close(frozen, dyn, rtol=1e-9, atol=1e-12)
        x1 = _state(st, lo=0.1, hi=0.9)
        with torch.no_grad():
            dyn1 = _rhs(st, u, x1, gate)
            frozen1 = _rhs(st, u, x1, gate,
                           i_edge_const=st._compute_i_edge_const(x0, gate))
        assert float((frozen1 - dyn1).abs().max()) > 0.0