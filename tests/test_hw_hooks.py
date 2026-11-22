"""Hardware-constraint hooks in the cell library and in the built network (W2 steps 2-4).

Every wired flag is checked alone: the forward/backward stays finite and the
invariant of the flag holds.  The toy network is the one of
``tests/golden/make_golden.py`` (``TOY_NET``).
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
from hw import HWConfig  # noqa: E402
from hw.constraints import (  # noqa: E402
    HWConstraints,
    ISAT_HW_MAX,
    IsatScale,
    MISMATCH_DEFAULT,
    MismatchState,
    THETA_MAX,
    mismatch_draw,
)
from gating.vca import isat_eff_scale, vca_edge_gain  # noqa: E402
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
# 3.9 mismatch
# ---------------------------------------------------------------------------
def test_mismatch_changes_the_training_forward_but_not_the_eval_one():
    sig = dict(MISMATCH_DEFAULT)
    spec = dict(mg.TOY_NET)
    x = mg.make_inputs(spec)

    plain = build_toy(None, spec)
    plain.train()
    with torch.no_grad():
        y_plain = mg.model_forward(plain, x)

    noisy = build_toy(hw_of(hw_mismatch_sigma=sig), spec)
    noisy.train()
    with torch.no_grad():
        y_train = mg.model_forward(noisy, x)
    assert float((y_train - y_plain).abs().max()) > 0.0

    noisy.eval()
    with torch.no_grad():
        y_eval = mg.model_forward(noisy, x)
    torch.testing.assert_close(y_eval, y_plain, rtol=0, atol=0)

    noisy_eval = build_toy(hw_of(hw_mismatch_sigma=sig, hw_mismatch_eval=True), spec)
    noisy_eval.eval()
    with torch.no_grad():
        y_eval_on = mg.model_forward(noisy_eval, x)
    assert float((y_eval_on - y_plain).abs().max()) > 0.0


def test_mismatch_is_the_same_for_the_same_slot_in_every_stage():
    sig = dict(MISMATCH_DEFAULT)
    net = build_toy(hw_of(hw_mismatch_sigma=sig))
    net.train()
    spec = dict(mg.TOY_NET)
    with torch.no_grad():
        mg.model_forward(net, mg.make_inputs(spec))     # one forward -> one draw set

    ref = torch.zeros(4, dtype=D)
    f0 = net.core.stages[0].cell_lib._mismatch_factor("isat", 4, ref)
    f1 = net.core.stages[1].cell_lib._mismatch_factor("isat", 4, ref)
    assert f0 is not None and f1 is not None
    torch.testing.assert_close(f1, f0, rtol=0, atol=0)


def test_mismatch_multiplier_mean_matches_the_lognormal():
    sigma = 0.05
    g = torch.Generator().manual_seed(7)
    m = mismatch_draw((1000,), sigma, g)
    assert abs(float(m.mean()) - math.exp(sigma ** 2 / 2.0)) <= 0.01 * math.exp(sigma ** 2 / 2.0)


def test_mismatch_state_is_off_in_eval_and_shared_by_the_stages():
    st = MismatchState(dict(MISMATCH_DEFAULT), eval_on=False, seed=0)
    assert st.enabled(training=True) and not st.enabled(training=False)
    st2 = MismatchState(dict(MISMATCH_DEFAULT), eval_on=True, seed=0)
    assert st2.enabled(training=False)
    f1 = st.get("core", "isat", 4)
    f2 = st.get("core", "isat", 4)
    torch.testing.assert_close(f2, f1, rtol=0, atol=0)
    st.new_batch()
    f3 = st.get("core", "isat", 4)
    assert not torch.equal(f3, f1)


def test_mismatch_forward_backward_is_finite():
    y, grads, _ = toy_forward_backward(hw_of(hw_mismatch_sigma=dict(MISMATCH_DEFAULT)))
    assert torch.isfinite(y).all()
    assert all(torch.isfinite(g).all() for g in grads.values() if g is not None)


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


# ---------------------------------------------------------------------------
# 3.10 GLN tail mode, VCA bits, Isat_eff ceiling
# ---------------------------------------------------------------------------
def test_gln_tail_mode_multiplies_isat_and_leaves_the_slope_static():
    lib = cell(hw_of(hw_gln_mode="tail"))
    x_src = torch.tensor([[0.5, -0.2, 1.0, 0.0]], dtype=D)
    x_dst = torch.tensor([[0.1, 0.3, -0.4, 0.2]], dtype=D)
    scale = IsatScale(torch.full((1, 4), 2.0, dtype=D))
    got = lib._tanh_core(x_src, x_dst, gm_override=scale)
    # reference: the static gm, Isat multiplied by the GLN tail factor
    A = torch.nn.functional.softplus(lib.a_raw).unsqueeze(0)
    B = torch.nn.functional.softplus(lib.b_raw).unsqueeze(0)
    s = torch.sign(lib.s_raw)
    s_ste = s + lib.s_raw - lib.s_raw.detach()
    gm = lib.current_gm().unsqueeze(0)
    theta = lib.theta_raw.unsqueeze(0)
    u = (s_ste.unsqueeze(0) * (A * x_src - B * x_dst) + theta) * gm
    expect = (2.0 * lib.isat_values().unsqueeze(0)) * (torch.tanh(u) + lib.ALPHA * u)
    torch.testing.assert_close(got, expect, rtol=1e-12, atol=1e-12)


def test_gln_tail_multiplier_is_one_at_init_and_bounded_by_delta_max():
    net = build_toy(hw_of(hw_gln_mode="tail"))
    st = net.core.stages[0]
    u = torch.rand(4, 4, generator=torch.Generator().manual_seed(5), dtype=D)
    ov = st.edge_gm_scale("boundary", st.boundary_cell_lib, st._gln_rails_z(u))
    assert isinstance(ov, IsatScale)
    # the GLN mix is zero at init -> the tail multiplier is exactly 1
    torch.testing.assert_close(ov.m, torch.ones_like(ov.m), rtol=0, atol=0)
    # after a perturbation it stays inside exp(+-delta_max)
    with torch.no_grad():
        for mix in st.gln_rails.families.values():
            mix.Q.normal_(0.0, 0.5)
    ov = st.edge_gm_scale("boundary", st.boundary_cell_lib, st._gln_rails_z(u))
    dmax = math.exp(st.hw_constraints.hw_gln_delta_max)
    assert float(ov.m.min()) >= 1.0 / dmax - 1e-9
    assert float(ov.m.max()) <= dmax + 1e-9


def test_gln_tail_mode_forward_backward_is_finite():
    y, grads, _ = toy_forward_backward(hw_of(hw_gln_mode="tail"))
    assert torch.isfinite(y).all()
    assert all(torch.isfinite(g).all() for g in grads.values() if g is not None)


def test_vca_bits_quantise_the_logit_with_a_straight_through_gradient():
    g = torch.Generator().manual_seed(1)
    u = torch.randn(8, 4, generator=g, dtype=D)
    W = torch.randn(4, 3, generator=g, dtype=D)
    v_e = torch.randn(6, 3, generator=g, dtype=D)
    plain = vca_edge_gain(u, W, v_e)
    assert len(torch.unique(plain)) > 2
    for bits in (4, 6):
        q = vca_edge_gain(u, W, v_e, None, bits=bits)
        # at most 2**bits distinct gains (the zeta grid)
        assert len(torch.unique(q)) <= 2 ** bits
        assert len(torch.unique(q)) < len(torch.unique(plain))
    # STE: the gradient of the quantised logit is 1
    from hw.constraints import vca_quantize
    z = torch.tensor([0.3], dtype=D, requires_grad=True)
    vca_quantize(z, 4).sum().backward()
    assert float(z.grad) == pytest.approx(1.0, rel=1e-9)


def test_vca_bits_forward_backward_is_finite():
    y, grads, _ = toy_forward_backward(hw_of(hw_vca_bits=4))
    assert torch.isfinite(y).all()
    assert all(torch.isfinite(g).all() for g in grads.values() if g is not None)


def test_isat_eff_clamp_keeps_the_effective_current_under_the_ceiling():
    isat = torch.tensor([100.0, 1.0], dtype=D)
    sigma_z = torch.tensor([1.0, 1.0], dtype=D)
    scale = isat_eff_scale(isat, sigma_z)
    eff = isat * scale
    assert float(eff.max()) <= ISAT_HW_MAX + 1e-9
    # nothing to clip -> the scale is 1
    torch.testing.assert_close(isat_eff_scale(torch.tensor([1.0]), sigma_z[:1]),
                               torch.ones(1, dtype=D), rtol=0, atol=0)


def test_isat_eff_clamp_forward_backward_is_finite():
    y, grads, _ = toy_forward_backward(hw_of(hw_isat_eff_clamp=True))
    assert torch.isfinite(y).all()
    assert all(torch.isfinite(g).all() for g in grads.values() if g is not None)


# ---------------------------------------------------------------------------
# every wired flag, alone
# ---------------------------------------------------------------------------
# (flag name, extra kwargs, parameters that are dead under that flag)
WIRED_FLAGS = [
    ("hw_ratio_ab", {"hw_ratio_ab": True}, ("a_raw", "b_raw")),
    ("hw_slope_bound", {"hw_slope_bound": True}, ()),
    ("hw_ic_coupling", {"hw_slope_bound": True, "hw_ic_coupling": True}, ()),
    ("hw_no_alpha", {"hw_no_alpha": True}, ()),
    ("hw_alpha_bwd", {"hw_no_alpha": True, "hw_alpha_bwd": 0.05}, ()),
    ("hw_theta_bound", {"hw_theta_bound": True}, ()),
    ("hw_mismatch_sigma", {"hw_mismatch_sigma": dict(MISMATCH_DEFAULT)}, ()),
    ("hw_mismatch_eval", {"hw_mismatch_eval": True}, ()),
    ("hw_vca_bits", {"hw_vca_bits": 4}, ()),
    ("hw_isat_eff_clamp", {"hw_isat_eff_clamp": True}, ()),
    ("hw_gln_mode", {"hw_gln_mode": "tail"}, ()),
]


@pytest.mark.parametrize("name,kwargs,dead", WIRED_FLAGS)
def test_each_wired_flag_alone_gives_a_finite_forward_and_backward(name, kwargs, dead):
    y, grads, no_grad = toy_forward_backward(hw_of(**kwargs))
    assert torch.isfinite(y).all()
    assert all(torch.isfinite(g).all() for g in grads.values() if g is not None)
    # only the documented parameters may stay without a gradient
    for n in no_grad:
        assert any(d in n for d in dead), f"{name}: unexpected dead parameter {n}"


def test_the_ratio_form_and_the_slope_bound_hold_together():
    """A + B = 2 under hw_ratio_ab and gm_eff*max(A_hat, B_hat) <= slope_max."""
    hw = hw_of(hw_ratio_ab=True, hw_slope_bound=True)
    lib = cell(hw=hw)
    with torch.no_grad():
        lib.r_raw.normal_(0.0, 5.0)
        lib.gm_raw.normal_(0.0, 5.0)
    A_r, B_r = lib.ratio_ab_values()
    torch.testing.assert_close(A_r + B_r, torch.full_like(A_r, 2.0), rtol=0, atol=1e-12)
    A, B = lib.ab_values()
    gm = lib.gm_eff_values()
    assert float((gm * torch.maximum(A, B)).max()) <= hw.units.slope_max * (1 + 1e-9)
