"""Pure-function unit tests of hw/constraints.py (Phase 3b 2.4(a)) and HWConfig / CLI plumbing."""

import argparse
import json
import math

import pytest
import torch

import hw as hwpkg
from hw import HWConfig, hw_config_from_args, hw_or_none, parse_hw_args
from hw import constraints as C
from hw.units import HWUnits

D = torch.float64
U = HWUnits()


def _rand(*shape, lo=-10.0, hi=10.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(*shape, generator=g, dtype=D) * (hi - lo) + lo)


# ---------------------------------------------------------------------------
# 3.1 ratio A/B
# ---------------------------------------------------------------------------
def test_ratio_ab_sum_and_bounds():
    r_raw = _rand(2000)
    A, B = C.ratio_ab(r_raw, 4.0)
    torch.testing.assert_close(A + B, torch.full_like(A, 2.0), rtol=0, atol=1e-12)
    r = A / B
    assert float(r.min()) >= 0.25 - 1e-9 and float(r.max()) <= 4.0 + 1e-9
    assert float(A.min()) >= 0.4 - 1e-9 and float(A.max()) <= 1.6 + 1e-9


def test_ratio_ab_identity_and_grad():
    A, B = C.ratio_ab(torch.zeros(3, dtype=D))
    assert torch.equal(A, torch.ones_like(A)) and torch.equal(B, torch.ones_like(B))
    for v in (-10.0, 10.0):
        x = torch.full((4,), v, dtype=D, requires_grad=True)
        A, B = C.ratio_ab(x)
        (A.sum() + 2 * B.sum()).backward()
        assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# 3.2 slope bound / 3.3 IC
# ---------------------------------------------------------------------------
def test_slope_bound_invariant():
    g_raw = _rand(1000, seed=1)
    A = torch.nn.functional.softplus(_rand(1000, seed=2, lo=-8, hi=8))
    B = torch.nn.functional.softplus(_rand(1000, seed=3, lo=-8, hi=8))
    gm, Ah, Bh = C.slope_bound(g_raw, U, A=A, B=B)
    assert float((gm * torch.maximum(Ah, Bh)).max()) <= U.slope_max * (1 + 1e-9)
    torch.testing.assert_close(torch.maximum(Ah, Bh), torch.ones_like(Ah))
    g = C.attenuation(g_raw)
    assert float(g.min()) >= 1.0 / 16 - 1e-12 and float(g.max()) <= 1.0
    s = torch.linspace(-12, 12, 200, dtype=D)
    assert bool((C.attenuation(s)[1:] >= C.attenuation(s)[:-1]).all())
    gm_s, _, _ = C.slope_bound(s, U)
    assert bool((gm_s[1:] >= gm_s[:-1]).all())


def test_ic_factor():
    assert float(C.ic_factor(torch.tensor(1e-9, dtype=D), U)) == pytest.approx(1.0, abs=1e-9)
    isat = torch.linspace(0.0, 40.0, 100, dtype=D)
    f = C.ic_factor(isat, U)
    assert bool((f[1:] < f[:-1]).all())
    ic = isat * U.iu / (2 * U.i_spec)
    torch.testing.assert_close(f, 2.0 / (1.0 + torch.sqrt(1.0 + 4.0 * ic)))
    # 3a 3.3 numbers: I_spec = 2 uA: Isat 0.6 -> 0.99, Isat 16 -> 0.86
    assert float(C.ic_factor(torch.tensor(0.6, dtype=D), U)) == pytest.approx(0.99, abs=0.01)
    assert float(C.ic_factor(torch.tensor(16.0, dtype=D), U)) == pytest.approx(0.86, abs=0.02)


# ---------------------------------------------------------------------------
# 3.4 compliance
# ---------------------------------------------------------------------------
def test_compliance_at_vcm_is_exact():
    isat = torch.tensor([0.5, 3.0, 12.0], dtype=D)
    t = torch.tanh(torch.tensor([-1.3, 0.2, 2.5], dtype=D))
    i_p, i_n = isat * (1 + t) / 2, isat * (1 - t) / 2
    z = torch.zeros(3, dtype=D)
    into_d, into_s = C.compliance_legs(i_p, i_n, z, z, U)
    torch.testing.assert_close(into_d, isat * t, rtol=0, atol=1e-9)
    torch.testing.assert_close(into_s, -isat * t, rtol=0, atol=1e-9)
    fp, fn = C.leg_factors(z, U)
    assert float((fp - 1).abs().max()) < 1e-9 and float((fn - 1).abs().max()) < 1e-9


def test_compliance_outward_leg_dies_at_the_rail():
    xh = torch.tensor(U.x_hard, dtype=D)
    ftop_p, fbot_p = C.leg_factors(xh, U, lambda_v=0.0)     # at +x_hard: P leg outward
    ftop_n, fbot_n = C.leg_factors(-xh, U, lambda_v=0.0)    # at -x_hard: N leg outward
    assert float(ftop_p) < 1e-3 and float(fbot_n) < 1e-3
    assert float(fbot_p) > 0.999 and float(ftop_n) > 0.999
    # inside |x| <= 3.5 the roll-off is below 1 % (3a 3.4 quotes 0.3 % at 26 C; U_T at 85 C gives 0.65 %)
    for x in (3.5, -3.5, 0.0, 2.0):
        a, b = C.leg_factors(torch.tensor(x, dtype=D), U, lambda_v=0.0)
        assert float(a) > 0.99 and float(b) > 0.99


def test_compliance_heun_single_driven_node_never_exceeds_rail():
    for sign in (1.0, -1.0):
        x = torch.zeros(1, dtype=D)
        dt, leak = 0.0666, 0.0486
        isat = torch.tensor([10.0], dtype=D)
        i_p, i_n = (isat, torch.zeros_like(isat)) if sign > 0 else (torch.zeros_like(isat), isat)

        def f(xx):
            d, _ = C.compliance_legs(i_p, i_n, xx, None, U)
            return d - leak * xx

        peak = 0.0
        for _ in range(400):
            k1 = f(x)
            k2 = f(x + dt * k1)
            x = x + 0.5 * dt * (k1 + k2)
            peak = max(peak, float(x.abs()))
        assert peak <= U.x_hard * 1.02


def test_g_out_is_lambda_vu_isat():
    isat = torch.tensor([2.0], dtype=D)
    x = torch.tensor([0.7], dtype=D)
    gout = C.g_out_term(isat, x, U)
    assert float(gout) == pytest.approx(-0.15 * 0.07 * 2.0 * 0.7)
    # at rest (t = 0) the legs give the same conductance
    i_p = i_n = isat / 2
    d, _ = C.compliance_legs(i_p, i_n, x, None, U, rail=False)
    assert float(d) == pytest.approx(float(gout), rel=1e-9)


# ---------------------------------------------------------------------------
# 3.5 STE
# ---------------------------------------------------------------------------
def test_tanh_ste_forward_and_backward():
    u = _rand(100, lo=-8, hi=8, seed=5).requires_grad_()
    assert torch.equal(C.tanh_ste(u, 0.0), torch.tanh(u))
    assert torch.equal(C.tanh_ste(u, 0.02).detach(), torch.tanh(u).detach())
    for ab in (0.0, 0.01, 0.05):
        u1 = torch.tensor([12.0, -12.0], dtype=D, requires_grad=True)
        C.tanh_ste(u1, ab).sum().backward()
        # 1 - tanh(12)^2 = 1.5e-10
        assert float((u1.grad - ab).abs().max()) < 1e-9
    u2 = torch.tensor([6.0], dtype=D, requires_grad=True)
    C.tanh_ste(u2, 0.01).sum().backward()
    assert float(u2.grad) == pytest.approx(0.01 + 1 - math.tanh(6.0) ** 2, abs=1e-12)
    u3 = _rand(50, lo=-3, hi=3, seed=6).requires_grad_()
    C.tanh_ste(u3, 0.0).sum().backward()
    torch.testing.assert_close(u3.grad, 1 - torch.tanh(u3.detach()) ** 2)


# ---------------------------------------------------------------------------
# quantiser, Vref, theta, crossbar
# ---------------------------------------------------------------------------
def test_quantize_ste():
    x = torch.linspace(-4.5, 4.5, 4001, dtype=D, requires_grad=True)
    q = C.quantize_ste(x, 6, -4.0, 4.0)
    assert torch.unique(q.detach()).numel() <= 64
    assert float(q.detach().min()) >= -4.0 and float(q.detach().max()) <= 4.0
    q.sum().backward()
    inside = (x.detach().abs() <= 4.0)
    assert bool((x.grad[inside] == 1.0).all()) and bool((x.grad[~inside] == 0.0).all())
    assert torch.equal(C.quantize_ste(x.detach(), None, -4, 4), x.detach())
    step = 8.0 / 63
    assert float((q.detach() - x.detach().clamp(-4, 4)).abs().max()) <= step / 2 + 1e-12


def test_vref_map():
    raw = torch.linspace(-3, 3, 2000, dtype=D)
    v = C.vref_map(raw, 4.0, bits=6, symmetric=True)
    assert torch.unique(v).numel() <= 64
    assert float(v.min()) >= -4 and float(v.max()) <= 4
    assert float(C.vref_map(torch.zeros(1, dtype=D), 4.0)) == 0.0
    leg = C.vref_map(raw, 4.0, symmetric=False)
    torch.testing.assert_close(leg, torch.sigmoid(raw) * 4.0)
    r = torch.tensor([0.3], dtype=D, requires_grad=True)
    C.vref_map(r, 4.0, bits=6).sum().backward()
    assert float(r.grad) == pytest.approx(4.0 * (1 - math.tanh(0.3) ** 2))   # STE: identity on the code


def test_theta_bound():
    th = C.theta_bound(_rand(500, seed=7))
    assert float(th.abs().max()) <= 1.0


def test_xbar_map():
    W = _rand(7, 22, lo=-9, hi=9, seed=8)
    Wb = C.xbar_map(W, bound=True)
    assert float(Wb.abs().max()) <= C.W_MAX
    assert torch.equal(C.xbar_map(W), W)
    bits = 5
    step = 2 * C.W_MAX / (2 ** bits - 1)
    xs = _rand(10, 22, lo=-1, hi=1, seed=9)
    ref = xs @ C.xbar_map(W, bound=True).T
    prods = {}
    for mode in C.XBAR_MODES:
        Wq = C.xbar_map(W, mode, bits, bound=True)
        assert float(Wq.abs().max()) <= C.W_MAX + 1e-12
        prods[mode] = xs @ Wq.T
        assert float((prods[mode] - ref).abs().max()) <= xs.abs().sum(1).max() * step + 1e-9
    # offset-binary has the same code grid as sign-magnitude
    torch.testing.assert_close(prods["offset_binary"], prods["signed"], rtol=0, atol=1e-9)
    assert float((prods["differential"] - prods["signed"]).abs().max()) <= xs.abs().sum(1).max() * step + 1e-9
    Wg = torch.randn(3, 4, dtype=D, requires_grad=True)
    C.xbar_map(Wg, "differential", bits).sum().backward()
    assert torch.isfinite(Wg.grad).all()
    with pytest.raises(ValueError):
        C.xbar_map(W, "nope")


# ---------------------------------------------------------------------------
# c_node
# ---------------------------------------------------------------------------
def _paper_like_graph():
    n_h, n_acc = 11, 7
    src = [i for i in range(n_h)] + [i for i in range(n_h)]
    dst = [(i + 1) % n_h for i in range(n_h)] + [(i + 2) % n_h for i in range(n_h)]
    return dict(num_nodes=n_h + n_acc, src=src, dst=dst, boundary_dst=list(range(8)),
                sense_src=[i for i in range(n_h) for _ in range(2)], acc_start=n_h, n_xbar_cols=22)


def test_c_node_uniform_is_ones():
    g = _paper_like_graph()
    c = C.c_node_vector(g["num_nodes"], g["src"], g["dst"], U, "uniform", boundary_dst=g["boundary_dst"],
                        sense_src=g["sense_src"], acc_start=g["acc_start"], n_xbar_cols=22)
    assert c.shape == (1, g["num_nodes"])
    assert torch.equal(c, torch.ones_like(c))


def test_c_node_physical_matches_terminal_formula():
    g = _paper_like_graph()
    kw = dict(boundary_dst=g["boundary_dst"], sense_src=g["sense_src"], acc_start=g["acc_start"],
              n_xbar_cols=22)
    cpar, start = C.node_parasitics(g["num_nodes"], g["src"], g["dst"], **kw)
    assert start == 11
    # accumulator: 22 legs + wire = 26 fF; hidden mean ~22 fF (3a 3.8)
    assert float(cpar[11:].mean()) == pytest.approx(26e-15, rel=1e-9)
    assert float(cpar[:11].mean()) == pytest.approx(22e-15, rel=0.02)
    c = C.c_node_vector(g["num_nodes"], g["src"], g["dst"], U, "physical", **kw)
    assert c.shape == (1, g["num_nodes"])
    # per-node values: (fixed + C_par)/C_u with fixed = C_u - class-mean parasitic
    fixed_h = U.cu - float(cpar[:11].mean())
    torch.testing.assert_close(c[0, :11].double(), ((fixed_h + cpar[:11]) / U.cu).float().double(),
                               rtol=1e-6, atol=1e-6)
    assert float(c[0, :11].mean()) == pytest.approx(1.0, abs=1e-6)
    assert float(c[0, :11].std()) > 0.0     # hidden nodes differ (boundary / sense terminals)
    # differential columns double the accumulator legs and the fixed cap vanishes
    cpar_d, _ = C.node_parasitics(g["num_nodes"], g["src"], g["dst"], xbar_mode="differential", **kw)
    assert float(cpar_d[11:].mean()) == pytest.approx(48e-15, rel=1e-9)


def test_c_node_uniform_raises_when_parasitic_exceeds_cu():
    with pytest.raises(ValueError):
        C.c_node_vector(3, [0], [1], HWUnits(cu=3e-15), "uniform")
    with pytest.raises(ValueError):
        C.c_node_vector(3, [0], [1], U, "nope")


# ---------------------------------------------------------------------------
# power
# ---------------------------------------------------------------------------
def test_power_penalty_hand_built():
    isat = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=D, requires_grad=True)
    z = torch.tensor([[0.5, 0.5], [1.0, 0.25]], dtype=D)
    g = torch.tensor([[1.0, 2.0], [1.0, 1.0]], dtype=D)
    p = C.power_penalty(isat, z, g, U)
    tot = (1 * 0.5 * 1 + 2 * 0.5 * 2) + (3 * 1.0 + 4 * 0.25)
    assert float(p) == pytest.approx(3.0 * U.vdd * U.iu * tot / 2.0, rel=1e-12)
    p.backward()
    assert bool((isat.grad > 0).all())
    # list-of-stages form (different edge counts per stage) and g_vca = None
    p2 = C.power_penalty([torch.tensor([1.0, 2.0], dtype=D), torch.tensor([3.0], dtype=D)],
                         [torch.tensor([1.0, 1.0], dtype=D), torch.tensor([1.0], dtype=D)], None, U)
    assert float(p2) == pytest.approx(3.0 * U.vdd * U.iu * 6.0 / 2.0, rel=1e-12)


# ---------------------------------------------------------------------------
# mismatch
# ---------------------------------------------------------------------------
def test_mismatch_mean_matches_lognormal():
    for sigma in (0.05, 0.10):
        m = C.mismatch_draw((200000,), sigma, torch.Generator().manual_seed(0))
        assert float(m.mean()) == pytest.approx(math.exp(sigma ** 2 / 2), rel=0.01)
    m = C.mismatch_draw((1000,), 0.05, torch.Generator().manual_seed(1))
    assert float(m.mean()) == pytest.approx(math.exp(0.05 ** 2 / 2), rel=0.01)
    a = C.mismatch_draw((5,), 0.1, additive=True, gen=torch.Generator().manual_seed(2))
    assert float(a.abs().max()) > 0 and float(C.apply_mismatch(torch.zeros(5, dtype=D), a, True).std()) > 0
    assert C.apply_mismatch(torch.ones(2), None).tolist() == [1.0, 1.0]


def test_mismatch_state_shared_and_eval_gated():
    sig = {"isat": 0.05, "g": 0.1, "a": 0.01, "b": 0.01, "theta": 0.1}
    mm = C.MismatchState(sig, eval_on=False, seed=3)
    s0 = mm.get("core", "isat", 6)
    s1 = mm.get("core", "isat", 6)               # "another stage": same slots -> same epsilon
    assert torch.equal(s0, s1)
    assert torch.equal(mm.get("core", "isat", 4), s0[:4])
    s2 = mm.get("core", "isat", 8)               # larger family extends, prefix unchanged
    assert torch.equal(s2[:6], s0)
    assert not torch.equal(mm.get("boundary", "isat", 6), s0)
    mm.new_batch()
    assert not torch.equal(mm.get("core", "isat", 6), s0)
    # off at eval unless hw_mismatch_eval
    assert mm.get("core", "isat", 6, training=False) is None
    mm_e = C.MismatchState(sig, eval_on=True, seed=3)
    assert mm_e.get("core", "isat", 6, training=False) is not None
    # zero sigma -> None
    assert C.MismatchState({"isat": 0.0}).get("core", "isat", 3) is None


# ---------------------------------------------------------------------------
# GLN tail / Isat clamp / VCA bits
# ---------------------------------------------------------------------------
def test_gln_tail_and_isat_clamp():
    d = torch.linspace(-6, 6, 101, dtype=D)
    m = C.gln_tail(d, math.log(4.0))
    assert float(m.min()) == pytest.approx(0.25) and float(m.max()) == pytest.approx(4.0)
    assert float(C.gln_tail(torch.zeros(1, dtype=D), 1.0)) == 1.0
    isat = _rand(300, lo=0, hi=20, seed=10)
    e = C.isat_eff_clamp(isat, torch.full_like(isat, 0.9), torch.full_like(isat, 1.8))
    assert float(e.max()) <= C.ISAT_HW_MAX and float(e.min()) >= 0.0
    lo = C.isat_eff_clamp(torch.tensor([1.0], dtype=D), torch.tensor([0.5], dtype=D))
    assert float(lo) == pytest.approx(0.5)
    z = _rand(100, lo=-9, hi=9, seed=11)
    assert torch.unique(C.vca_quantize(z, 5)).numel() <= 32


# ---------------------------------------------------------------------------
# HWConstraints flags
# ---------------------------------------------------------------------------
def test_flag_names_and_defaults_are_3a_3_11():
    c = C.HWConstraints()
    want = {
        "hw_ratio_ab": False, "hw_slope_bound": False, "hw_ic_coupling": False,
        "hw_compliance": False, "hw_clamp_transfer": False, "hw_no_alpha": False, "hw_alpha_bwd": 0.0,
        "hw_vref_symmetric": False, "hw_vref_bits": None, "hw_theta_bound": False,
        "hw_xbar_bound": False, "hw_xbar_bits": None, "hw_xbar_mode": "signed", "hw_c_node": None,
        "hw_power_lambda": 0.0, "hw_mismatch_sigma": None, "hw_mismatch_eval": False,
        "hw_vca_bits": None, "hw_isat_eff_clamp": False, "hw_gln_mode": None,
        "hw_gln_delta_max": math.log(4.0), "hw_check_stiffness": False,
    }
    assert c.to_dict() == want
    assert c.is_off()
    for k, v in (("hw_slope_bound", True), ("hw_c_node", "uniform"), ("hw_vref_bits", 6),
                 ("hw_power_lambda", 1e-3), ("hw_mismatch_sigma", {"isat": 0.05})):
        assert not C.HWConstraints(**{k: v}).is_off()


def test_flag_validation_and_from_dict():
    with pytest.raises(ValueError):
        C.HWConstraints(hw_xbar_mode="x")
    with pytest.raises(ValueError):
        C.HWConstraints(hw_c_node="x")
    with pytest.raises(ValueError):
        C.HWConstraints(hw_gln_mode="x")
    with pytest.raises(ValueError):
        C.HWConstraints(hw_vref_bits=0)
    with pytest.raises(ValueError):
        C.HWConstraints(hw_mismatch_sigma={"bogus": 1.0})
    with pytest.raises(KeyError):
        C.HWConstraints.from_dict({"not_a_flag": True})
    assert C.HWConstraints.from_dict({"slope_bound": True}).hw_slope_bound     # short key accepted


# ---------------------------------------------------------------------------
# HWConfig
# ---------------------------------------------------------------------------
def test_hwconfig_default_is_off_and_roundtrips(tmp_path):
    cfg = HWConfig()
    assert cfg.is_off() and hw_or_none(cfg) is None
    cfg2 = HWConfig(constraints=C.HWConstraints(hw_slope_bound=True, hw_mismatch_sigma={"isat": 0.05}),
                    units=HWUnits(vu=0.1), timemux={"enable": False, "c_hold_ff": 100.0})
    assert not cfg2.is_off() and hw_or_none(cfg2) is cfg2
    s = cfg2.to_json()
    back = HWConfig.from_json(s)
    assert back == cfg2 and back.hash() == cfg2.hash()
    p = tmp_path / "c.json"
    cfg2.to_json(str(p))
    assert HWConfig.from_json(str(p)) == cfg2
    assert cfg.hash() != cfg2.hash()
    assert HWConfig.from_json("{}").is_off()
    # timemux is opaque; disabled timemux keeps is_off() True
    assert HWConfig(timemux={"enable": False}).is_off()
    assert not HWConfig(timemux={"enable": True}).is_off()


def test_hwconfig_unknown_keys_raise():
    with pytest.raises(KeyError):
        HWConfig.from_dict({"bogus": {}})
    with pytest.raises(KeyError):
        HWConfig.from_dict({"units": {"vuu": 1}})
    with pytest.raises(KeyError):
        HWConfig.from_dict({"constraints": {"hw_nope": 1}})


def _parser():
    p = argparse.ArgumentParser()
    parse_hw_args(p)
    return p


def test_cli_defaults_off_and_flags():
    a = _parser().parse_args([])
    assert hw_config_from_args(a).is_off()
    a = _parser().parse_args(["--hw-slope-bound", "--hw-c-node", "physical", "--hw-gln-mode", "tail",
                              "--hw-vref-bits", "6", "--hw-alpha-bwd", "0.01",
                              "--hw-mismatch-sigma", "isat=0.05,g=0.1", "--hw-noise-eval",
                              "--hw-compliance", "false"])
    cfg = hw_config_from_args(a)
    c = cfg.constraints
    assert c.hw_slope_bound and c.hw_c_node == "physical" and c.hw_gln_mode == "tail"
    assert c.hw_vref_bits == 6 and c.hw_alpha_bwd == 0.01 and c.hw_mismatch_eval
    assert c.hw_mismatch_sigma == {"isat": 0.05, "g": 0.1} and c.hw_compliance is False


def test_cli_precedence_cli_over_config_over_preset(tmp_path, monkeypatch):
    pdir = tmp_path / "presets"
    pdir.mkdir()
    (pdir / "mine.json").write_text(json.dumps({
        "units": {"vu": 0.09}, "constraints": {"hw_ratio_ab": True, "hw_slope_bound": True,
                                               "hw_vref_bits": 4},
        "timemux": {"enable": False}}))
    monkeypatch.setattr(hwpkg, "PRESET_DIR", str(pdir))
    cfgf = tmp_path / "c.json"
    cfgf.write_text(json.dumps({"constraints": {"hw_slope_bound": False, "hw_vref_bits": 5},
                                "units": {"x_max": 3.0}}))
    a = _parser().parse_args(["--hw-preset", "mine"])
    c = hw_config_from_args(a)
    assert c.constraints.hw_ratio_ab and c.constraints.hw_slope_bound and c.units.vu == 0.09
    a = _parser().parse_args(["--hw-preset", "mine", "--hw-config", str(cfgf)])
    c = hw_config_from_args(a)
    assert c.constraints.hw_ratio_ab and not c.constraints.hw_slope_bound
    assert c.constraints.hw_vref_bits == 5 and c.units.x_max == 3.0 and c.units.vu == 0.09
    a = _parser().parse_args(["--hw-preset", "mine", "--hw-config", str(cfgf), "--hw-vref-bits", "7",
                              "--hw-slope-bound"])
    c = hw_config_from_args(a)
    assert c.constraints.hw_vref_bits == 7 and c.constraints.hw_slope_bound
    with pytest.raises(FileNotFoundError):
        hw_config_from_args(_parser().parse_args(["--hw-preset", "does_not_exist"]))
