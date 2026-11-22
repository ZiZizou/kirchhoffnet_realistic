"""Hardware-constraint hooks in the cell library and in the built network (W2 steps 2-4).

Every wired flag is checked alone: the forward/backward stays finite and the
invariant of the flag holds.  The toy network is the one of
``tests/golden/make_golden.py`` (``TOY_NET``).
"""

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "golden"))
sys.path.insert(0, os.path.dirname(HERE))

import make_golden as mg  # noqa: E402
from cell_library import FreeTanhLibrary, make_cell_library  # noqa: E402
from hw import HWConfig  # noqa: E402
from hw.constraints import HWConstraints, THETA_MAX  # noqa: E402
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


# ---------------------------------------------------------------------------
# 3.1 bounded A/B (ratio form)
# ---------------------------------------------------------------------------
def test_ratio_ab_sum_is_two_and_the_pair_is_normalised_with_the_slope_bound():
    lib = cell(hw_of(hw_ratio_ab=True, hw_slope_bound=True))
    with torch.no_grad():
        lib.r_raw.normal_(0.0, 3.0)
    A, B = lib.ratio_ab_values()
    torch.testing.assert_close(A + B, torch.full_like(A, 2.0), rtol=0, atol=1e-12)
    r = A / B
    assert float(r.min()) >= 0.25 - 1e-9 and float(r.max()) <= 4.0 + 1e-9
    # with the slope bound the common gain is folded into the slope: max = 1
    Ae, Be = lib.ab_values()
    torch.testing.assert_close(torch.maximum(Ae, Be), torch.ones_like(Ae), rtol=0, atol=1e-12)


def test_ratio_ab_at_init_is_one_and_one():
    lib = cell(hw_of(hw_ratio_ab=True))
    A, B = lib.ratio_ab_values()
    torch.testing.assert_close(A, torch.ones_like(A), rtol=0, atol=1e-12)
    torch.testing.assert_close(B, torch.ones_like(B), rtol=0, atol=1e-12)


def test_ratio_ab_forward_backward_is_finite():
    y, grads, _ = toy_forward_backward(hw_of(hw_ratio_ab=True))
    assert torch.isfinite(y).all()
    assert all(torch.isfinite(g).all() for g in grads.values() if g is not None)
    # the ratio parameter receives a gradient
    assert any("r_raw" in n for n, g in grads.items() if g is not None)


# ---------------------------------------------------------------------------
# 3.2 / 3.3 slope bound and IC coupling
# ---------------------------------------------------------------------------
def test_slope_bound_keeps_the_per_input_slope_under_slope_max():
    hw = hw_of(hw_ratio_ab=True, hw_slope_bound=True)
    lib = cell(hw=hw)
    with torch.no_grad():
        lib.gm_raw.normal_(0.0, 5.0)
        lib.r_raw.normal_(0.0, 3.0)
    A, B = lib.ab_values()
    gm = lib.gm_eff_values()
    assert float((gm * torch.maximum(A, B)).max()) <= hw.units.slope_max * (1 + 1e-9)


def test_slope_bound_is_bounded_also_without_the_ratio_flag():
    hw = hw_of(hw_slope_bound=True)
    lib = cell(hw=hw)
    with torch.no_grad():
        lib.gm_raw.normal_(0.0, 5.0)
    gm = lib.gm_eff_values()
    assert float(gm.max()) <= hw.units.slope_max * (1 + 1e-9)
    assert float(gm.min()) >= hw.units.slope_max * (1.0 / 16.0) * (1 - 1e-9)


def test_ic_coupling_scales_the_slope_down_and_needs_the_slope_bound():
    hw = hw_of(hw_slope_bound=True)
    lib = cell(hw=hw)
    with torch.no_grad():
        lib.isat_raw.fill_(5.0)          # strong inversion -> f(IC) < 1
    gm_ic = lib.gm_eff_values()
    hw2 = hw_of(hw_slope_bound=True, hw_ic_coupling=True)
    lib2 = cell(hw=hw2)
    with torch.no_grad():
        lib2.isat_raw.fill_(5.0)
    gm = lib2.gm_eff_values()
    assert float(gm.max()) < float(gm_ic.max())
    with pytest.raises(ValueError):
        HWConstraints(hw_ic_coupling=True)


def test_slope_bound_forward_backward_is_finite():
    y, grads, _ = toy_forward_backward(hw_of(hw_slope_bound=True, hw_ic_coupling=True))
    assert torch.isfinite(y).all()
    assert all(torch.isfinite(g).all() for g in grads.values() if g is not None)


# ---------------------------------------------------------------------------
# 3.5 no-ALPHA / straight-through floor
# ---------------------------------------------------------------------------
def test_no_alpha_drops_the_linear_term():
    lib = cell(hw_of(hw_no_alpha=True))
    x_src = torch.tensor([[0.5, -0.2, 1.0, 0.0]], dtype=D)
    x_dst = torch.tensor([[0.1, 0.3, -0.4, 0.2]], dtype=D)
    got = lib._tanh_core(x_src, x_dst)
    # reference: tanh(u) without the ALPHA*u term
    A = torch.nn.functional.softplus(lib.a_raw).unsqueeze(0)
    B = torch.nn.functional.softplus(lib.b_raw).unsqueeze(0)
    s = torch.sign(lib.s_raw)
    s_ste = s + lib.s_raw - lib.s_raw.detach()
    gm = lib.current_gm().unsqueeze(0)
    theta = lib.theta_raw.unsqueeze(0)
    u = (s_ste.unsqueeze(0) * (A * x_src - B * x_dst) + theta) * gm
    expect = lib.isat_values().unsqueeze(0) * torch.tanh(u)
    torch.testing.assert_close(got, expect, rtol=1e-12, atol=1e-12)


def test_alpha_bwd_keeps_the_forward_value_and_floors_the_gradient():
    alpha = 0.05
    lib = cell(hw_of(hw_no_alpha=True, hw_alpha_bwd=alpha))
    x_src = torch.tensor([[0.5, -0.2, 1.0, 0.0]], dtype=D)
    x_dst = torch.tensor([[0.1, 0.3, -0.4, 0.2]], dtype=D)
    x_src.requires_grad_(True)
    got = lib._tanh_core(x_src, x_dst)
    # forward value: exactly tanh(u)
    A = torch.nn.functional.softplus(lib.a_raw).unsqueeze(0)
    B = torch.nn.functional.softplus(lib.b_raw).unsqueeze(0)
    s = torch.sign(lib.s_raw)
    s_ste = s + lib.s_raw - lib.s_raw.detach()
    gm = lib.current_gm().unsqueeze(0)
    theta = lib.theta_raw.unsqueeze(0)
    u = (s_ste.unsqueeze(0) * (A * x_src - B * x_dst) + theta) * gm
    expect = lib.isat_values().unsqueeze(0) * torch.tanh(u)
    torch.testing.assert_close(got, expect, rtol=1e-12, atol=1e-12)
    # backward: d/du = 1 - tanh^2 + alpha  (the straight-through floor)
    got.sum().backward()
    expect_grad = (lib.isat_values().unsqueeze(0)
                   * (1.0 - torch.tanh(u) ** 2 + alpha)
                   * gm * s_ste.unsqueeze(0) * A)
    torch.testing.assert_close(x_src.grad, expect_grad, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# 3.6 theta bound
# ---------------------------------------------------------------------------
def test_theta_bound_limits_the_offset():
    lib = cell(hw_of(hw_theta_bound=True))
    with torch.no_grad():
        lib.theta_raw.normal_(0.0, 4.0)
    theta = lib.theta_values()
    assert float(theta.abs().max()) <= THETA_MAX * (1 + 1e-9)


def test_theta_bound_forward_backward_is_finite():
    y, grads, _ = toy_forward_backward(hw_of(hw_theta_bound=True))
    assert torch.isfinite(y).all()
    assert all(torch.isfinite(g).all() for g in grads.values() if g is not None)
