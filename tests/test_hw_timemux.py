"""Unit tests for the time-multiplexing model (Phase 3b section 2.4(a)).

Covers the rows of the "time-mux capture", "settle" and "transfer modes"
entries of ``kn_hwdocs/phase3b_fabric_impl_retrain.md`` section 2.4(a):

* zero config -> identity
* kT/C std over 20000 draws equals ``ktc_rms_units`` within 3 %
* droop sign fixed per node
* STE quantiser gradient is 1
* settle limits
* transfer ``full`` == ``stage_transfer.StageTransfer`` on random parameters
* ``kick`` -> ``full`` as ``t_x -> 0``

Run:  venv/bin/python -m pytest tests/test_hw_timemux.py -q
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests", "golden")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stage_transfer import StageTransfer  # noqa: E402
from hw.timemux import (  # noqa: E402
    XFER_MODES,
    StageBoundary,
    TimeMuxConfig,
    quantize_ste,
    settle_factor,
    transfer_apply,
    transfer_full,
    transfer_kick,
    transfer_no_relu,
    transfer_no_tanh,
    transfer_quant,
    transfer_shared_vth,
)

DTYPE = torch.float64


# --------------------------------------------------------------------------
# a tiny stand-in for hw/units.py (the module must work before it exists)
# --------------------------------------------------------------------------

class _Units:
    """Duck-typed stand-in for ``hw.units.HWUnits`` (Phase 3a section 2.5)."""

    vdd = 0.8
    vcm = 0.4
    vu = 0.07
    cu = 50e-15
    temp_c = 85.0
    n_slope = 1.3
    x_max = 4.0

    @property
    def x_hard(self) -> float:
        return self.vdd / 2.0 / self.vu

    @property
    def ktc_rms_units(self) -> float:
        k_b, q_e = 1.380649e-23, 1.602176634e-19
        return math.sqrt(k_b * (self.temp_c + 273.15) / self.cu) / self.vu


UNITS = _Units()


def _boundary(cfg, n_nodes=8, **kwargs):
    """A boundary with the default unit quantities."""
    kwargs.setdefault("units", UNITS)
    return StageBoundary(cfg, n_nodes=n_nodes, **kwargs)


def _only_noise_cfg(**overrides):
    """Config with kT/C noise as the only active effect."""
    cfg = dict(
        enable=True,
        inject_mv=0.0,
        inject_gain_err=0.0,
        droop_mv=0.0,
        ktc=True,
    )
    cfg.update(overrides)
    return TimeMuxConfig(**cfg)


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def test_config_defaults_match_the_doc():
    cfg = TimeMuxConfig()
    assert cfg.enable is False
    assert cfg.c_hold_ff == 100.0
    assert cfg.inject_mv == 1.0
    assert cfg.inject_gain_err == 0.0
    assert cfg.droop_mv == 1.0
    assert cfg.ktc is True
    assert cfg.settle is False
    assert cfg.c_int_ff == 3.0
    assert cfg.isat_floor == 0.0
    assert cfg.xfer_mode == "full"
    assert cfg.param_bits is None
    assert cfg.is_off() is True


def test_config_roundtrip():
    cfg = TimeMuxConfig(enable=True, xfer_mode="kick", param_bits=6)
    d = cfg.to_dict()
    assert d["enable"] is True
    assert d["xfer_mode"] == "kick"
    assert d["param_bits"] == 6
    again = TimeMuxConfig.from_dict(d)
    assert again.to_dict() == d
    # unspecified keys keep their default
    partial = TimeMuxConfig.from_dict({"enable": True})
    assert partial.enable is True
    assert partial.xfer_mode == "full"
    assert partial.c_hold_ff == 100.0


def test_config_rejects_unknown_xfer_mode():
    with pytest.raises(ValueError):
        TimeMuxConfig(xfer_mode="nope")
    for mode in XFER_MODES:
        assert TimeMuxConfig(xfer_mode=mode).xfer_mode == mode


# --------------------------------------------------------------------------
# capture: identity
# --------------------------------------------------------------------------

def test_zero_config_is_exact_identity():
    b = _boundary(TimeMuxConfig(), n_nodes=8)
    x = torch.randn(6, 8, dtype=DTYPE)
    x_hold, x_to = b.capture(x, 0)
    assert torch.equal(x_hold, x)
    assert torch.equal(x_to, x)


def test_disabled_boundary_is_identity_in_train_and_eval():
    b = _boundary(TimeMuxConfig(), n_nodes=8)
    x = torch.randn(6, 8, dtype=DTYPE)
    b.train()
    assert torch.equal(b.capture(x, 0)[0], x)
    b.eval()
    assert torch.equal(b.capture(x, 0)[0], x)


def test_enabled_but_all_effects_zero_is_identity():
    cfg = TimeMuxConfig(
        enable=True, inject_mv=0.0, inject_gain_err=0.0, droop_mv=0.0, ktc=False
    )
    b = _boundary(cfg, n_nodes=8)
    b.train()
    x = torch.randn(6, 8, dtype=DTYPE)
    x_hold, x_to = b.capture(x, 0)
    assert torch.equal(x_hold, x)
    assert torch.equal(x_to, x)


def test_boundary_adds_no_state_dict_entries():
    """The off path must not change state_dict().keys() (Phase 3b 2.4(b))."""
    b = _boundary(TimeMuxConfig(enable=True), n_nodes=8)
    assert b.state_dict() == {}


# --------------------------------------------------------------------------
# capture: kT/C noise
# --------------------------------------------------------------------------

def test_ktc_std_over_20000_draws_within_3_percent():
    n_nodes = 18
    b = _boundary(_only_noise_cfg(), n_nodes=n_nodes)
    b.train()
    gen = torch.Generator().manual_seed(1234)
    x = torch.zeros(20000, n_nodes, dtype=DTYPE)
    x_hold, x_to = b.capture(x, 0, generator=gen)
    assert torch.equal(x_hold, x_to)
    std = float(x_to.std())
    expected = UNITS.ktc_rms_units
    assert abs(std - expected) / expected < 0.03, (std, expected)


def test_ktc_noise_is_off_in_eval_unless_noise_eval():
    gen = torch.Generator().manual_seed(7)
    x = torch.zeros(64, 8, dtype=DTYPE)

    b = _boundary(_only_noise_cfg(), n_nodes=8)
    b.eval()
    out = b.capture(x, 0, generator=gen)[1]
    assert torch.equal(out, x)

    b2 = _boundary(_only_noise_cfg(), n_nodes=8, noise_eval=True)
    b2.eval()
    out2 = b2.capture(x, 0, generator=gen)[1]
    assert not torch.equal(out2, x)
    assert abs(float(out2.std()) - UNITS.ktc_rms_units) / UNITS.ktc_rms_units < 0.15


def test_ktc_noise_is_deterministic_given_a_generator():
    x = torch.zeros(16, 8, dtype=DTYPE)
    b = _boundary(_only_noise_cfg(), n_nodes=8)
    b.train()
    g1 = torch.Generator().manual_seed(99)
    g2 = torch.Generator().manual_seed(99)
    assert torch.equal(b.capture(x, 0, generator=g1)[1], b.capture(x, 0, generator=g2)[1])


# --------------------------------------------------------------------------
# capture: injection, gain error, droop
# --------------------------------------------------------------------------

def test_injection_offset_has_a_fixed_sign_per_node():
    cfg = TimeMuxConfig(
        enable=True, inject_mv=1.0, inject_gain_err=0.0, droop_mv=0.0, ktc=False
    )
    b = _boundary(cfg, n_nodes=8)
    x = torch.randn(32, 8, dtype=DTYPE)
    _, x_to = b.capture(x, 0)
    offset = x_to - x
    # the same offset for every row -> a per-node constant
    assert torch.allclose(offset, offset[0].expand_as(offset), atol=1e-12, rtol=0.0)
    expected = cfg.inject_mv * 1e-3 / UNITS.vu
    assert torch.allclose(offset.abs(), torch.full_like(offset, expected))
    assert set(torch.sign(offset[0]).unique().tolist()) <= {-1.0, 1.0}


def test_droop_sign_is_fixed_per_node_and_applied_to_x_hold_only():
    cfg = TimeMuxConfig(
        enable=True, inject_mv=0.0, inject_gain_err=0.0, droop_mv=1.0, ktc=False
    )
    b = _boundary(cfg, n_nodes=8)
    x = torch.randn(32, 8, dtype=DTYPE)
    x_hold, x_to = b.capture(x, 0)
    droop = x_hold - x_to
    # fixed per node: identical for every row, non-zero, and of one sign
    assert torch.allclose(droop, droop[0].expand_as(droop), atol=1e-12, rtol=0.0)
    assert torch.all(droop != 0.0)
    signs = torch.sign(droop[0])
    assert set(signs.unique().tolist()) <= {-1.0, 1.0}
    expected = cfg.droop_mv * 1e-3 / UNITS.vu
    assert torch.allclose(droop.abs(), torch.full_like(droop, expected))
    # the transfer input is untouched by the droop
    assert torch.equal(x_to, x)


def test_droop_direction_is_stable_across_calls():
    cfg = TimeMuxConfig(
        enable=True, inject_mv=0.0, inject_gain_err=0.0, droop_mv=1.0, ktc=False
    )
    b = _boundary(cfg, n_nodes=8)
    x1 = torch.randn(4, 8, dtype=DTYPE)
    x2 = -torch.randn(4, 8, dtype=DTYPE)
    d1 = b.capture(x1, 0)[0] - b.capture(x1, 0)[1]
    d2 = b.capture(x2, 0)[0] - b.capture(x2, 0)[1]
    assert torch.allclose(d1[0], d2[0], atol=1e-12, rtol=0.0)


def test_gain_error_scales_the_signal():
    cfg = TimeMuxConfig(
        enable=True, inject_mv=0.0, inject_gain_err=0.002, droop_mv=0.0, ktc=False
    )
    b = _boundary(cfg, n_nodes=8)
    x = torch.randn(16, 8, dtype=DTYPE)
    _, x_to = b.capture(x, 0)
    assert torch.allclose(x_to, x * (1.0 + 0.002))


def test_effects_are_applied_in_the_documented_order():
    """kT/C -> injection -> gain error -> droop -> quantisation."""
    cfg = TimeMuxConfig(
        enable=True, inject_mv=1.0, inject_gain_err=0.01, droop_mv=1.0, ktc=False
    )
    b = _boundary(cfg, n_nodes=8)
    x = torch.randn(4, 8, dtype=DTYPE)
    x_hold, x_to = b.capture(x, 0)
    inject = b.inject_sign * b.inject_units
    gain = (x + inject) * (1.0 + 0.01)
    assert torch.allclose(x_to, gain)
    assert torch.allclose(x_hold, gain + b.droop_sign * b.droop_units)


# --------------------------------------------------------------------------
# quantiser
# --------------------------------------------------------------------------

def test_quantize_ste_gradient_is_one():
    x = torch.randn(32, 8, dtype=DTYPE, requires_grad=True)
    q = quantize_ste(x, 8, -UNITS.x_hard, UNITS.x_hard)
    q.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_quantize_ste_gradient_is_one_also_outside_the_range():
    x = (torch.randn(32, 8, dtype=DTYPE) * 100.0).requires_grad_(True)
    q = quantize_ste(x, 8, -UNITS.x_hard, UNITS.x_hard)
    q.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_quantize_ste_lands_on_the_grid():
    x = torch.randn(256, 8, dtype=DTYPE) * 3.0
    lo, hi, bits = -UNITS.x_hard, UNITS.x_hard, 8
    q = quantize_ste(x, bits, lo, hi)
    step = (hi - lo) / (2 ** bits - 1)
    codes = (q - lo) / step
    assert torch.allclose(codes, torch.round(codes), atol=1e-9)
    assert torch.all(q <= hi + 1e-12) and torch.all(q >= lo - 1e-12)


def test_quantize_ste_is_a_noop_for_zero_bits():
    x = torch.randn(8, 8, dtype=DTYPE)
    assert torch.equal(quantize_ste(x, 0, -1.0, 1.0), x)
    assert torch.equal(quantize_ste(x, None, -1.0, 1.0), x)


def test_boundary_optional_node_quantisation():
    cfg = TimeMuxConfig(enable=True, inject_mv=0.0, droop_mv=0.0, ktc=False)
    b = _boundary(cfg, n_nodes=8, quant_bits=8)
    x = torch.randn(16, 8, dtype=DTYPE) * 2.0
    x_hold, x_to = b.capture(x, 0)
    step = 2.0 * UNITS.x_hard / (2 ** 8 - 1)
    codes = (x_to + UNITS.x_hard) / step
    assert torch.allclose(codes, torch.round(codes), atol=1e-9)


# --------------------------------------------------------------------------
# bias settling
# --------------------------------------------------------------------------

def test_settle_with_equal_currents_is_the_new_current():
    i_new = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
    out = StageBoundary.settle_core(i_new, i_new.clone(), 0.03, 0.6658)
    assert torch.allclose(out, i_new)


def test_settle_with_zero_previous_current():
    tau_e, t_stage = 0.03, 0.6658
    f = settle_factor(tau_e, t_stage)
    i_new = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
    out = StageBoundary.settle_core(i_new, torch.zeros_like(i_new), tau_e, t_stage)
    assert torch.allclose(out, i_new * (1.0 - f))
    # the factor of the doc: (tau_e/T) * (1 - exp(-T/tau_e))
    assert abs(f - (tau_e / t_stage) * (1.0 - math.exp(-t_stage / tau_e))) < 1e-12


def test_settle_limits():
    i_new = torch.tensor([1.0], dtype=DTYPE)
    i_prev = torch.tensor([4.0], dtype=DTYPE)
    # fast settling -> the new current
    fast = StageBoundary.settle_core(i_new, i_prev, 1e-9, 1.0)
    assert torch.allclose(fast, i_new, atol=1e-6)
    # no settling at all -> the old current survives
    slow = StageBoundary.settle_core(i_new, i_prev, math.inf, 1.0)
    assert torch.allclose(slow, i_prev)


def test_settle_factor_is_monotone_in_tau_e():
    ts = 0.6658
    fs = [settle_factor(t, ts) for t in (0.001, 0.01, 0.03, 0.1, 1.0)]
    assert all(a < b for a, b in zip(fs, fs[1:]))
    assert 0.0 < fs[0] < fs[-1] < 1.0


def test_tau_e_from_kappa_int_over_isat():
    b = _boundary(TimeMuxConfig(enable=True, c_int_ff=3.0, isat_floor=0.5), n_nodes=8)
    kappa = b.kappa_int
    assert kappa > 0.0
    assert abs(b.tau_e_units(1.0) - kappa) < 1e-12
    # the Isat floor bounds tau_e
    assert b.tau_e_units(0.1) == pytest.approx(b.tau_e_units(0.5))
    assert b.tau_e_units(0.0) == pytest.approx(kappa / 0.5)
    # without a floor an off OTA never settles
    b0 = _boundary(TimeMuxConfig(enable=True, isat_floor=0.0), n_nodes=8)
    assert b0.tau_e_units(0.0) == math.inf


# --------------------------------------------------------------------------
# transfer modes
# --------------------------------------------------------------------------

def _random_transfer_params(n_nodes, seed=0):
    gen = torch.Generator().manual_seed(seed)
    w1 = torch.randn(n_nodes, generator=gen, dtype=DTYPE)
    w2 = torch.randn(n_nodes, generator=gen, dtype=DTYPE) * 0.4
    w3 = torch.randn(n_nodes, generator=gen, dtype=DTYPE) * 0.5
    vth = torch.randn(n_nodes, generator=gen, dtype=DTYPE) * 0.3
    return w1, w2, w3, vth


def test_transfer_full_equals_stage_transfer():
    n_nodes = 7
    w1, w2, w3, vth = _random_transfer_params(n_nodes, seed=3)
    st = StageTransfer(in_nodes=n_nodes, out_nodes=n_nodes,
                        activation="residual-relu-tanh").double()
    with torch.no_grad():
        st.residual_w1.copy_(w1)
        st.residual_w2.copy_(w2)
        st.residual_w3.copy_(w3)
        st.residual_vth.copy_(vth)
    x = torch.randn(5, n_nodes, dtype=DTYPE)
    ref = st(x)
    got = transfer_full(x, w1, w2, w3, vth)
    assert torch.allclose(ref, got, rtol=1e-12, atol=1e-12)


def test_transfer_full_matches_the_formula():
    n_nodes = 4
    w1, w2, w3, vth = _random_transfer_params(n_nodes, seed=5)
    x = torch.randn(3, n_nodes, dtype=DTYPE)
    expected = (
        w1 * x
        + w2 * torch.tanh(x)
        + w3 * torch.clamp(x - vth, min=0.0)
    )
    assert torch.allclose(transfer_full(x, w1, w2, w3, vth), expected)


def test_shared_vth_uses_one_vth_per_boundary():
    n_nodes = 8
    w1, w2, w3, vth = _random_transfer_params(n_nodes, seed=11)
    x = torch.randn(3, n_nodes, dtype=DTYPE)
    shared = float(vth.mean())
    # a shared (scalar) vth reproduces the full transfer
    assert torch.allclose(
        transfer_shared_vth(x, w1, w2, w3, shared),
        transfer_full(x, w1, w2, w3, torch.full((n_nodes,), shared, dtype=DTYPE)),
    )
    # a per-node vector is collapsed onto the single shared DAC value
    assert torch.allclose(
        transfer_shared_vth(x, w1, w2, w3, vth),
        transfer_full(x, w1, w2, w3, torch.full((n_nodes,), shared, dtype=DTYPE)),
    )
    # six boundaries carry six vth values in total (one per boundary)
    n_boundaries = 6
    vths = [torch.tensor(0.1 * (i + 1), dtype=DTYPE) for i in range(n_boundaries)]
    assert len({float(v) for v in vths}) == n_boundaries


def test_kick_equals_full_as_t_x_goes_to_zero():
    n_nodes = 6
    w1, w2, w3, vth = _random_transfer_params(n_nodes, seed=17)
    x = torch.randn(4, n_nodes, dtype=DTYPE)
    full = transfer_full(x, w1, w2, w3, vth)
    # the idealised pulse realises the whole delta -> numerically identical
    assert torch.allclose(transfer_kick(x, w1, w2, w3, vth, t_x=0.0), full,
                          rtol=1e-12, atol=1e-12)
    assert torch.allclose(transfer_kick(x, w1, w2, w3, vth, t_x=3e-9), full,
                          rtol=1e-12, atol=1e-12)
    # a finite-current penalty scales the delivered delta
    partial = transfer_kick(x, w1, w2, w3, vth, t_x=3e-9, pulse_fraction=0.5)
    delta = (w1 - 1.0) * x + w2 * torch.tanh(x) + w3 * torch.clamp(x - vth, min=0.0)
    assert torch.allclose(partial, x + 0.5 * delta)


def test_no_tanh_and_no_relu_drop_the_right_term():
    n_nodes = 5
    w1, w2, w3, vth = _random_transfer_params(n_nodes, seed=23)
    x = torch.randn(3, n_nodes, dtype=DTYPE)
    assert torch.allclose(
        transfer_no_tanh(x, w1, w2, w3, vth),
        w1 * x + w3 * torch.clamp(x - vth, min=0.0),
    )
    assert torch.allclose(
        transfer_no_relu(x, w1, w2, w3, vth),
        w1 * x + w2 * torch.tanh(x),
    )


def test_transfer_quant_uses_ste_on_the_parameters():
    n_nodes = 5
    w1, w2, w3, vth = _random_transfer_params(n_nodes, seed=29)
    x = torch.randn(3, n_nodes, dtype=DTYPE)
    q = transfer_quant(x, w1, w2, w3, vth, bits=5)
    qw1 = quantize_ste(w1, 5, -2.0, 2.0)
    qw2 = quantize_ste(w2, 5, -2.0, 2.0)
    qw3 = quantize_ste(w3, 5, -2.0, 2.0)
    qvth = quantize_ste(vth, 5, -2.0, 2.0)
    assert torch.allclose(q, transfer_full(x, qw1, qw2, qw3, qvth))
    step = 4.0 / (2 ** 5 - 1)
    codes = (qw1 + 2.0) / step
    assert torch.allclose(codes, torch.round(codes), atol=1e-9)


def test_transfer_apply_dispatch():
    n_nodes = 4
    w1, w2, w3, vth = _random_transfer_params(n_nodes, seed=31)
    x = torch.randn(2, n_nodes, dtype=DTYPE)
    for mode in XFER_MODES:
        out = transfer_apply(x, w1, w2, w3, vth, mode=mode)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
    with pytest.raises(ValueError):
        transfer_apply(x, w1, w2, w3, vth, mode="nope")


def test_boundary_apply_transfer_follows_the_config():
    n_nodes = 4
    w1, w2, w3, vth = _random_transfer_params(n_nodes, seed=37)
    x = torch.randn(2, n_nodes, dtype=DTYPE)
    b = _boundary(TimeMuxConfig(enable=True, xfer_mode="no_tanh"), n_nodes=n_nodes)
    assert torch.allclose(
        b.apply_transfer(x, w1, w2, w3, vth), transfer_no_tanh(x, w1, w2, w3, vth)
    )
    b2 = _boundary(TimeMuxConfig(enable=True, xfer_mode="quant", param_bits=5),
                   n_nodes=n_nodes)
    assert torch.allclose(
        b2.apply_transfer(x, w1, w2, w3, vth),
        transfer_quant(x, w1, w2, w3, vth, bits=5),
    )


# --------------------------------------------------------------------------
# presets (hw/presets/*.json)
# --------------------------------------------------------------------------

PRESET_DIR = os.path.join(REPO_ROOT, "hw", "presets")

# The flag names of Phase 3a section 3.11 - a preset may not use anything else.
ALLOWED_CONSTRAINT_FLAGS = {
    "hw_ratio_ab",
    "hw_slope_bound",
    "hw_ic_coupling",
    "hw_compliance",
    "hw_clamp_transfer",
    "hw_no_alpha",
    "hw_alpha_bwd",
    "hw_vref_symmetric",
    "hw_vref_bits",
    "hw_theta_bound",
    "hw_xbar_bound",
    "hw_xbar_bits",
    "hw_xbar_mode",
    "hw_c_node",
    "hw_power_lambda",
    "hw_mismatch_sigma",
    "hw_mismatch_eval",
    "hw_vca_bits",
    "hw_isat_eff_clamp",
    "hw_gln_mode",
    "hw_gln_delta_max",
    "hw_check_stiffness",
}


@pytest.mark.parametrize("name", ["phase3a_core", "phase3a_full", "phase3b_tm"])
def test_presets_have_the_expected_shape(name):
    path = os.path.join(PRESET_DIR, f"{name}.json")
    assert os.path.isfile(path), path
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert set(data) == {"units", "constraints", "timemux"}
    assert isinstance(data["units"], dict)
    assert isinstance(data["constraints"], dict)
    assert isinstance(data["timemux"], dict)
    # only flag names of Phase 3a section 3.11
    unknown = set(data["constraints"]) - ALLOWED_CONSTRAINT_FLAGS
    assert not unknown, unknown
    # the timemux section must be loadable
    cfg = TimeMuxConfig.from_dict(data["timemux"])
    assert cfg.xfer_mode in XFER_MODES


def test_preset_contents():
    def load(name):
        with open(os.path.join(PRESET_DIR, f"{name}.json"), "r", encoding="utf-8") as fh:
            return json.load(fh)

    core = load("phase3a_core")["constraints"]
    assert core["hw_ratio_ab"] is True
    assert core["hw_slope_bound"] is True
    assert core["hw_ic_coupling"] is True
    assert core["hw_no_alpha"] is True

    full = load("phase3a_full")["constraints"]
    for key, value in core.items():
        assert full[key] == value
    assert full["hw_compliance"] is True
    assert full["hw_c_node"] == "physical"
    assert full["hw_gln_mode"] == "tail"
    assert full["hw_vref_symmetric"] is True
    assert full["hw_vref_bits"] is not None
    assert full["hw_xbar_bound"] is True
    assert full["hw_xbar_bits"] is not None

    tm = load("phase3b_tm")
    assert tm["timemux"]["enable"] is True
    assert tm["timemux"]["xfer_mode"] in XFER_MODES
    # the constraints of phase3b_tm are those of phase3a_full
    assert tm["constraints"] == full

    # phase3a_core / phase3a_full do not switch the time-multiplexing on
    assert load("phase3a_core")["timemux"].get("enable", False) is False
    assert load("phase3a_full")["timemux"].get("enable", False) is False
