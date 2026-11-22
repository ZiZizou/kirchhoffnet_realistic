"""Stage-level hardware hooks (Phase 3a 3.4 - 3.9, workstream W2c).

Every flag is checked alone on the toy network of ``tests/golden/make_golden.py``
(``TOY_NET``): the forward/backward stays finite and the invariant of the flag
holds.  The flags-off path is covered by ``tests/test_hw_off_is_golden.py``.
"""

import math
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "golden"))
sys.path.insert(0, os.path.dirname(HERE))

import make_golden as mg  # noqa: E402
from cell_library import FreeTanhLibrary, make_cell_library  # noqa: E402
from differential_stage import DifferentialStage, FrozenLegs  # noqa: E402
from hw import HWConfig  # noqa: E402
from hw.constraints import HWConstraints, leg_factors  # noqa: E402
from topology import build_net_from_config  # noqa: E402

D = torch.float64


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def build_toy(hw=None, spec=None):
    """Build the toy net the way ``make_golden`` does, with an optional ``hw``."""
    spec = dict(spec or mg.TOY_NET)
    cfg = mg._build_cfg(spec)
    with mg.f64():
        mg.seed_all(int(spec["seed"]))
        cl = {k: v for k, v in spec["cell_lib"].items() if k != "name"}
        cell_lib = make_cell_library(spec["cell_lib"]["name"], **cl, hw=hw)
        kw = dict(spec["build"])
        kw["boundary_fan_out"] = mg._int_keys(kw["boundary_fan_out"])
        net = build_net_from_config(cfg, cell_lib=cell_lib, hw=hw, **kw)
    return net.double()


def hw_of(**kwargs) -> HWConfig:
    return HWConfig(constraints=HWConstraints(**kwargs))


def toy_forward_backward(hw):
    """Forward + backward of the toy net; returns (y, grads dict, params without grad)."""
    spec = dict(mg.TOY_NET)
    net = build_toy(hw, spec)
    x = mg.make_inputs(spec)
    net.train()
    net.zero_grad(set_to_none=True)
    y = mg.model_forward(net, x)
    loss = y.pow(2).mean()
    loss.backward()
    grads = {n: p.grad for n, p in net.named_parameters()}
    no_grad = sorted(n for n, g in grads.items() if g is None)
    return y, grads, no_grad


def cell(hw=None, num_edges=4, bias_enabled=True):
    lib = FreeTanhLibrary(num_edges=num_edges, bias_enabled=bias_enabled, hw=hw)
    return lib.double()


def _stage(hw=None, num_nodes=2, src=(0,), dst=(1,), freeze_read=False,
           leak_constant=0.0, isat_max=10.0, gm_max=5.0):
    """A minimal stage: one core edge ``src -> dst`` and one boundary drive."""
    n_edges = len(src)
    lib = make_cell_library("tanh_free", num_edges=n_edges, gm_max=gm_max,
                             isat_max=isat_max, hw=hw)
    stage = DifferentialStage(
        num_nodes=num_nodes,
        src=list(src), dst=list(dst), cell_lib=lib,
        leak_mode="non-programmable", leak_constant=leak_constant,
        freeze_read=freeze_read, hw=hw,
    )
    return stage.double()


def _boundary_stage(hw=None, num_nodes=3, isat_max=10.0, gm_max=5.0):
    """A stage with a core edge 0 -> 1 and a boundary drive into node 2."""
    lib = make_cell_library("tanh_free", num_edges=1, gm_max=gm_max,
                             isat_max=isat_max, hw=hw)
    blib = make_cell_library("tanh_free", num_edges=1, gm_max=gm_max,
                              isat_max=isat_max, hw=hw)
    stage = DifferentialStage(
        num_nodes=num_nodes,
        src=[0], dst=[1], cell_lib=lib,
        leak_mode="non-programmable", leak_constant=0.0,
        boundary_src=[0], boundary_dst=[2], boundary_cell_lib=blib,
        hw=hw,
    )
    return stage.double()


# ---------------------------------------------------------------------------
# 3.4 output compliance (leg-wise)
# ---------------------------------------------------------------------------
def test_compliance_legs_at_vcm_are_exact():
    """At x = 0 the two legs are exactly Isat*(1 +- tanh u)/2."""
    hw = hw_of(hw_compliance=True)
    lib = cell(hw=hw, num_edges=3)
    x_src = torch.tensor([[0.5, -0.2, 1.0]], dtype=D)
    x_dst = torch.zeros(1, 3, dtype=D)
    i_pos, i_neg = lib.forward_legs(x_src=x_src, x_dst=x_dst)
    # reference: u of the legacy cell
    A = torch.nn.functional.softplus(lib.a_raw).unsqueeze(0)
    B = torch.nn.functional.softplus(lib.b_raw).unsqueeze(0)
    s = torch.sign(lib.s_raw).unsqueeze(0)
    gm = lib.current_gm().unsqueeze(0)
    theta = lib.theta_raw.unsqueeze(0)
    u = (s * (A * x_src - B * x_dst) + theta) * gm
    isat = lib.isat_values().unsqueeze(0)
    torch.testing.assert_close(i_pos, isat * (1.0 + torch.tanh(u)) / 2.0,
                               rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(i_neg, isat * (1.0 - torch.tanh(u)) / 2.0,
                               rtol=1e-12, atol=1e-12)
    # the legs add up to Isat (the OTA always sinks Isat through the pair)
    torch.testing.assert_close(i_pos + i_neg, isat.expand_as(i_pos),
                               rtol=1e-12, atol=1e-12)


def test_compliance_skips_the_input_compliance_gate():
    """The input gate sigma((x_max-|V|)/0.02) is replaced by the leg compliance."""
    hw = hw_of(hw_compliance=True)
    lib = cell(hw=hw, num_edges=2)
    x_src = torch.tensor([[10.0, -12.0]], dtype=D)
    x_dst = torch.tensor([[0.0, 0.0]], dtype=D)
    gated = lib.forward_tanh(x_src=x_src, x_dst=x_dst, x_max=3.0)
    core = lib._tanh_core(x_src, x_dst).expand_as(gated)
    torch.testing.assert_close(gated, core, rtol=1e-12, atol=1e-12)
    # the legacy (flag off) path does gate
    plain = cell(hw=None, num_edges=2)
    legacy = plain.forward_tanh(x_src=x_src, x_dst=x_dst, x_max=3.0)
    assert float(legacy.abs().max()) < 1e-6 * max(1.0, float(core.abs().max()))


def test_compliance_zeroes_the_soft_clip():
    """The soft rail clip is replaced by the hard rails of the legs."""
    net = build_toy(hw_of(hw_compliance=True))
    st = net.core.stages[0]
    x = torch.full((1, st.num_nodes), 50.0, dtype=D)
    assert float(st.clip_term(x).abs().max()) == 0.0
    plain = build_toy(None)
    st0 = plain.core.stages[0]
    assert float(st0.clip_term(x).abs().max()) > 0.0


def test_compliance_confines_a_strongly_driven_node_to_the_hard_rail():
    """A long Heun integration of a driven node never leaves +-1.05*x_hard."""
    hw = hw_of(hw_compliance=True)
    st = _boundary_stage(hw)
    x_hard = hw.units.x_hard
    u = torch.tensor([[20.0]], dtype=D)
    x0 = torch.zeros(1, 3, dtype=D)
    x, traj = st(x0, t_span=2.0, num_steps=800, store_trajectory=True, u=u)
    assert torch.isfinite(x).all()
    assert float(traj.abs().max()) <= 1.05 * x_hard


def _make_strong(lib):
    """Push the cell parameters into saturation (a large A, gm and Isat)."""
    with torch.no_grad():
        lib.a_raw.fill_(3.0)
        lib.gm_raw.fill_(3.0)
        lib.isat_raw.fill_(3.0)
    return lib


def test_compliance_freeze_read_keeps_the_rail_factors_live():
    """The frozen quantity is the leg current; c_top / c_bot are evaluated live."""
    hw = hw_of(hw_compliance=True)
    st = _stage(hw, freeze_read=True)
    _make_strong(st.cell_lib)
    x0 = torch.tensor([[2.0, 0.0]], dtype=D)
    legs = st._compute_i_edge_const(x0)
    assert isinstance(legs, FrozenLegs)
    k0 = st.rhs(x0, i_edge_const=legs)
    # the same frozen legs, but the destination node sits at the hard rail it
    # is pushed towards: the leg that pushes outward is off there, so the
    # current into the node collapses (the sign of the OTA current depends on
    # the switch bit s).
    i0 = float(k0[0, 1])
    x1 = x0.clone()
    x1[0, 1] = math.copysign(hw.units.x_hard, i0)
    k1 = st.rhs(x1, i_edge_const=legs)
    assert abs(i0) > 0.1
    assert abs(float(k1[0, 1])) < 0.01 * abs(i0)


def test_compliance_forward_backward_is_finite():
    y, grads, _ = toy_forward_backward(hw_of(hw_compliance=True))
    assert torch.isfinite(y).all()
    assert all(torch.isfinite(g).all() for g in grads.values() if g is not None)


# ---------------------------------------------------------------------------
# 3.4 interstage transfer clamp
# ---------------------------------------------------------------------------
def test_clamp_transfer_bounds_the_write_with_a_straight_through_gradient():
    from stage_transfer import StageTransfer
    hw = hw_of(hw_clamp_transfer=True)
    x_hard = hw.units.x_hard
    t = StageTransfer(4, 6, activation="residual-relu-tanh", hw=hw).double()
    with torch.no_grad():
        t.residual_w1.fill_(2.0)
    x = torch.tensor([[-9.0, -1.0, 2.0, 7.5]], dtype=D, requires_grad=True)
    y = t(x)
    assert float(y.abs().max()) <= x_hard          # exact, no ulp overshoot
    assert float(y[0, 0]) == -x_hard and float(y[0, 3]) == x_hard
    ref = StageTransfer(4, 6, activation="residual-relu-tanh").double()
    ref.load_state_dict(t.state_dict())
    y_ref = ref(x)
    inside = y_ref.abs() < x_hard
    torch.testing.assert_close(y[inside], y_ref[inside], rtol=0, atol=0)
    # straight-through: the gradient is that of the unclamped transfer
    g, = torch.autograd.grad(y.sum(), x)
    g_ref, = torch.autograd.grad(y_ref.sum(), x)
    torch.testing.assert_close(g, g_ref, rtol=0, atol=0)
    assert float(g[0, 0]) != 0.0


def test_clamp_transfer_is_threaded_through_the_builder():
    hw = hw_of(hw_clamp_transfer=True)
    net = build_toy(hw)
    assert len(net.core.transfers) >= 1
    assert all(tr.hw is hw for tr in net.core.transfers)
    # the post-readout transfer acts on the OutputAffine output, not a node voltage
    assert getattr(net.post_readout_transfer, "hw", None) is None
    tr = net.core.transfers[0]
    x = torch.full((2, tr.in_nodes), 40.0, dtype=D)
    assert float(tr(x).abs().max()) <= hw.units.x_hard
    plain = build_toy(None)
    assert float(plain.core.transfers[0](x).abs().max()) > hw.units.x_hard


def test_clamp_transfer_forward_backward_is_finite():
    y, grads, _ = toy_forward_backward(hw_of(hw_clamp_transfer=True))
    assert torch.isfinite(y).all()
    assert all(torch.isfinite(g).all() for g in grads.values() if g is not None)
