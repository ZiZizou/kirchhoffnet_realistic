"""HWUnits: Phase 3a section 2.4/2.5 default values and conversions."""

import dataclasses
import math

import pytest

from hw.units import HWUnits


def test_defaults_match_phase3a():
    u = HWUnits()
    assert (u.vu, u.cu, u.t_total_phys, u.t_total_units) == (0.07, 50e-15, 350e-9, 4.660583)
    assert (u.n_slope, u.temp_c, u.x_max, u.vdd, u.vcm) == (1.3, 85.0, 4.0, 0.8, 0.4)
    assert u.t_ovh_per_boundary == 0.0
    assert u.tu == pytest.approx(75.10e-9, rel=1e-3)
    assert u.iu == pytest.approx(46.6e-9, rel=2e-3)
    assert u.gu == pytest.approx(0.6658e-6, rel=1e-3)
    assert u.slope_max == pytest.approx(0.872, abs=1e-3)
    assert u.x_hard == pytest.approx(5.714, abs=1e-3)
    assert u.x_swing_v == pytest.approx(0.28)
    assert u.iu == pytest.approx(u.gu * u.vu, rel=1e-12)


def test_ktc_and_power():
    u = HWUnits()
    assert u.ktc_rms_units * u.vu == pytest.approx(math.sqrt(1.380649e-23 * 358.15 / 50e-15))
    assert u.power_w(104.0) == pytest.approx(104.0 * u.iu * 0.8)


def test_conversions_round_trip():
    u = HWUnits()
    for x in (-5.7, -1.0, 0.0, 0.3, 4.0):
        assert u.v_to_x(u.x_to_v(x)) == pytest.approx(x)
    assert u.x_to_v(0.0) == pytest.approx(0.4)
    assert u.t_to_s(u.t_total_units) == pytest.approx(350e-9)
    assert u.isat_to_amps(1.0) == pytest.approx(u.iu)
    assert u.g_to_siemens(1.0) == pytest.approx(u.gu)


def test_tu_effective_subtracts_overhead():
    u = HWUnits()
    assert u.tu_effective == u.tu
    v = HWUnits(t_ovh_per_boundary=5e-9)
    assert v.tu_effective == pytest.approx((350e-9 - 6 * 5e-9) / 4.660583)
    assert v.tu_effective < v.tu


def test_frozen():
    u = HWUnits()
    with pytest.raises(dataclasses.FrozenInstanceError):
        u.vu = 0.1
    assert HWUnits(vu=0.1).slope_max > u.slope_max
