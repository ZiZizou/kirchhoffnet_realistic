"""``HWConfig`` owns a real ``TimeMuxConfig`` (Phase 3b W4, step 1).

The time-multiplexing section of an ``HWConfig`` is no longer an opaque
dict: it is parsed into the ``hw.timemux.TimeMuxConfig`` dataclass, it
round-trips through JSON, its hash is stable and every preset of
``hw/presets/*.json`` loads and validates.

Run:  venv/bin/python -m pytest tests/test_hw_config.py -q
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hw import (  # noqa: E402
    HWConfig,
    hw_config_from_args,
    hw_or_none,
    load_preset,
    parse_hw_args,
)
from hw.timemux import StageBoundary, TimeMuxConfig  # noqa: E402


PRESETS = ["phase3a_core", "phase3a_full", "phase3b_tm"]


def _parse(argv):
    parser = parse_hw_args(argparse.ArgumentParser())
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", PRESETS)
def test_every_preset_loads_and_validates(name):
    data = load_preset(name)
    assert set(data) == {"units", "constraints", "timemux"}
    cfg = HWConfig.from_dict(data)
    # the constraints validate (HWConstraints.__post_init__ -> validate)
    assert cfg.constraints is not None
    # the timemux section is a real TimeMuxConfig (or None when absent)
    assert cfg.timemux is None or isinstance(cfg.timemux, TimeMuxConfig)
    if cfg.timemux is not None:
        assert cfg.timemux.xfer_mode in ("full", "shared_vth", "quant", "kick",
                                         "no_tanh", "no_relu")
    # the config round-trips
    again = HWConfig.from_dict(cfg.to_dict())
    assert again.to_dict() == cfg.to_dict()
    assert again.hash() == cfg.hash()
    # and it hashes to 12 hex digits
    h = cfg.hash()
    assert len(h) == 12 and all(c in "0123456789abcdef" for c in h)


@pytest.mark.parametrize("name", ["phase3a_full", "phase3b_tm"])
def test_presets_clamp_the_transfer(name):
    cfg = HWConfig.from_dict(load_preset(name))
    assert cfg.constraints.hw_clamp_transfer is True


def test_only_the_time_mux_preset_switches_the_fabric_on():
    assert HWConfig.from_dict(load_preset("phase3a_core")).timemux is None
    assert HWConfig.from_dict(load_preset("phase3a_full")).timemux is None
    tm = HWConfig.from_dict(load_preset("phase3b_tm")).timemux
    assert isinstance(tm, TimeMuxConfig)
    assert tm.enable is True


def test_presets_load_through_the_cli():
    for name in PRESETS:
        args = _parse(["--hw-preset", name])
        cfg = hw_config_from_args(args)
        assert HWConfig.from_dict(load_preset(name)).to_dict() == cfg.to_dict()


# --------------------------------------------------------------------------
# the timemux section
# --------------------------------------------------------------------------

def test_timemux_section_is_parsed_into_a_time_mux_config():
    cfg = HWConfig.from_dict({
        "constraints": {},
        "timemux": {"enable": True, "xfer_mode": "shared_vth", "inject_mv": 2.0},
    })
    assert isinstance(cfg.timemux, TimeMuxConfig)
    assert cfg.timemux.enable is True
    assert cfg.timemux.xfer_mode == "shared_vth"
    assert cfg.timemux.inject_mv == 2.0
    # the untouched fields keep their defaults
    assert cfg.timemux.c_hold_ff == 100.0
    assert cfg.timemux.param_bits is None


def test_an_empty_timemux_section_means_off():
    cfg = HWConfig.from_dict({"timemux": {}})
    assert cfg.timemux is None
    assert cfg.is_off() is True
    assert hw_or_none(cfg) is None


def test_is_off_follows_the_time_mux_switch():
    off = HWConfig.from_dict({"timemux": {"enable": False}})
    assert off.is_off() is True
    on = HWConfig.from_dict({"timemux": {"enable": True}})
    assert on.is_off() is False
    assert hw_or_none(on) is on


def test_is_off_ignores_a_disabled_but_non_default_timemux():
    """``enable=False`` means the model does nothing: still \"off\"."""
    cfg = HWConfig.from_dict({"timemux": {"enable": False, "inject_mv": 5.0}})
    assert cfg.is_off() is True


def test_timemux_round_trips_through_json(tmp_path):
    cfg = HWConfig.from_dict({"timemux": {"enable": True, "droop_mv": 2.5}})
    path = str(tmp_path / "hw.json")
    cfg.to_json(path)
    with open(path) as fh:
        raw = json.load(fh)
    assert raw["timemux"]["enable"] is True
    assert raw["timemux"]["droop_mv"] == 2.5
    again = HWConfig.from_json(path)
    assert again.to_dict() == cfg.to_dict()
    assert again.hash() == cfg.hash()


def test_hash_is_stable_and_discriminating():
    a = HWConfig.from_dict({"timemux": {"enable": True, "inject_mv": 1.0}})
    b = HWConfig.from_dict({"timemux": {"enable": True, "inject_mv": 1.0}})
    c = HWConfig.from_dict({"timemux": {"enable": True, "inject_mv": 1.5}})
    assert a.hash() == b.hash()
    assert a.hash() != c.hash()
    # the hash does not depend on the dict ordering
    d = HWConfig.from_dict({"timemux": {"inject_mv": 1.0, "enable": True}})
    assert d.hash() == a.hash()


def test_a_time_mux_config_object_is_accepted():
    tm = TimeMuxConfig(enable=True, xfer_mode="kick")
    cfg = HWConfig.from_dict({"timemux": tm})
    assert isinstance(cfg.timemux, TimeMuxConfig)
    assert cfg.timemux.xfer_mode == "kick"
    assert cfg.timemux is not tm  # a copy, not the same object


# --------------------------------------------------------------------------
# the node quantiser bits
# --------------------------------------------------------------------------

def test_node_quant_bits_default_is_none():
    assert TimeMuxConfig().node_quant_bits is None
    assert HWConfig().timemux is None


def test_node_quant_bits_is_exposed_on_the_config():
    cfg = HWConfig.from_dict({"timemux": {"enable": True, "node_quant_bits": 8}})
    assert cfg.timemux.node_quant_bits == 8
    assert cfg.to_dict()["timemux"]["node_quant_bits"] == 8


def test_node_quant_bits_must_be_a_positive_int():
    with pytest.raises(ValueError):
        TimeMuxConfig(node_quant_bits=0)
    with pytest.raises(ValueError):
        TimeMuxConfig(node_quant_bits=-3)
    with pytest.raises(ValueError):
        HWConfig.from_dict({"timemux": {"node_quant_bits": 0}})


def test_the_boundary_takes_its_quantiser_width_from_the_config():
    cfg = TimeMuxConfig(enable=True, node_quant_bits=8)
    b = StageBoundary(cfg, n_nodes=6)
    assert b.quant_bits == 8
    # an explicit argument still wins
    b2 = StageBoundary(cfg, n_nodes=6, quant_bits=4)
    assert b2.quant_bits == 4
    # the default config has no quantiser
    assert StageBoundary(TimeMuxConfig(), n_nodes=6).quant_bits is None


# --------------------------------------------------------------------------
# the CLI
# --------------------------------------------------------------------------

def test_cli_builds_the_time_mux_config():
    args = _parse([
        "--hw-tm",
        "--hw-tm-xfer-mode", "shared_vth",
        "--hw-tm-param-bits", "6",
        "--hw-tm-c-hold-ff", "50",
        "--hw-tm-node-quant-bits", "8",
    ])
    cfg = hw_config_from_args(args)
    assert isinstance(cfg.timemux, TimeMuxConfig)
    assert cfg.timemux.enable is True
    assert cfg.timemux.xfer_mode == "shared_vth"
    assert cfg.timemux.param_bits == 6
    assert cfg.timemux.c_hold_ff == 50.0
    assert cfg.timemux.node_quant_bits == 8
    assert cfg.is_off() is False


def test_cli_flags_override_the_preset():
    args = _parse(["--hw-preset", "phase3b_tm", "--hw-tm-xfer-mode", "no_tanh"])
    cfg = hw_config_from_args(args)
    assert cfg.timemux.enable is True          # from the preset
    assert cfg.timemux.xfer_mode == "no_tanh"  # from the CLI
    assert cfg.constraints.hw_ab_mode == "cap_divider"
    assert cfg.constraints.hw_ab_bits == 3 and cfg.constraints.hw_ab_rmax == 2.0


def test_new_ab_cli_flags():
    cfg = hw_config_from_args(_parse([
        "--hw-ab-mode", "cap_divider", "--hw-ab-bits", "4", "--hw-ab-rmax", "3",
        "--hw-slope-bound",
    ]))
    assert cfg.constraints.hw_ab_mode == "cap_divider"
    assert cfg.constraints.hw_ab_bits == 4 and cfg.constraints.hw_ab_rmax == 3


def test_no_flag_leaves_the_config_off():
    cfg = hw_config_from_args(_parse([]))
    assert cfg.timemux is None
    assert cfg.is_off() is True


def test_hw_tm_false_turns_the_fabric_off():
    cfg = hw_config_from_args(_parse(["--hw-tm", "false"]))
    assert cfg.timemux is not None
    assert cfg.timemux.enable is False
    assert cfg.is_off() is True
