"""hw/cli.py: the entry-script hardware CLI glue (Phase 3b 2.2, W4 step 3).

The unknown-flag guard matters because the entry scripts parse with
``parse_known_args``: without it a mistyped ``--hw-*`` flag is silently dropped
and a run trains the *legacy* fabric while its command line claims hardware.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys

import pytest

from hw import HWConfig, hw_or_none
from hw.cli import (add_entry_hw_args, check_unknown_hw_args, hw_passthrough_argv,
                    parse_known_args_hw_guarded, resolve_degrade_threshold,
                    setup_hw_config)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _entry_parser():
    """A stand-in for ``_bo_parser`` of the dagger script."""
    p = argparse.ArgumentParser(prog="entry", add_help=False)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", type=str, default=None)
    return add_entry_hw_args(p)


# --------------------------------------------------------------------------- guard
@pytest.mark.parametrize("argv", [
    ["--hw-slope-bnd"],                      # typo of --hw-slope-bound
    ["--seed", "3", "--hw-cnode", "physical"],  # (a unique *prefix* is argparse abbreviation)
    ["--hw-tm-foo=3"],                        # = form
    ["--zig-degrade-threshold", "0.2"],       # old spelling of --zig-degrade-thr
    ["--hw-from-a-newer-branch", "--output", "x"],
])
def test_guard_rejects_unknown_hw_flags(argv):
    with pytest.raises(SystemExit) as ei:
        parse_known_args_hw_guarded(_entry_parser(), argv)
    msg = str(ei.value.code)
    bad = [a.split("=")[0] for a in argv if a.startswith(("--hw-", "--zig-"))]
    for b in bad:
        assert b in msg
    assert "silently dropped" in msg and "--hw-slope-bound" in msg


def test_guard_keeps_other_unknown_args():
    args, unknown = parse_known_args_hw_guarded(
        _entry_parser(), ["--seed", "1", "--some-legacy-flag", "7", "-x"])
    assert args.seed == 1 and unknown == ["--some-legacy-flag", "7", "-x"]
    # a value that merely *looks* like a flag value is not a flag
    check_unknown_hw_args(["hw-slope", "x--hw-"])


def test_known_hw_flags_parse():
    args, unknown = parse_known_args_hw_guarded(_entry_parser(), [
        "--hw-preset", "phase3a_core", "--hw-slope-bound", "--hw-c-node", "physical",
        "--hw-tm", "--hw-tm-xfer-mode", "shared_vth", "--zig-degrade-thr", "0.2"])
    assert unknown == []
    assert args.hw_slope_bound is True and args.hw_c_node == "physical"
    assert args.hw_tm is True and args.zig_degrade_thr == 0.2


def test_bad_value_of_a_known_flag_still_errors():
    # argparse errors (bad choice / type) are SystemExit too, never swallowed
    with pytest.raises(SystemExit):
        parse_known_args_hw_guarded(_entry_parser(), ["--hw-tm-xfer-mode", "nope"])


# --------------------------------------------------------------------------- threshold
def test_degrade_threshold_override():
    p = _entry_parser()
    a, _ = parse_known_args_hw_guarded(p, [])
    assert resolve_degrade_threshold(a, 0.10) == 0.10
    a, _ = parse_known_args_hw_guarded(p, ["--zig-degrade-thr", "0.25"])
    assert resolve_degrade_threshold(a, 0.10) == 0.25
    a, _ = parse_known_args_hw_guarded(p, ["--zig-degrade-thr", "0"])
    with pytest.raises(SystemExit):
        resolve_degrade_threshold(a, 0.10)


# --------------------------------------------------------------------------- setup
def test_setup_hw_config_logs_and_saves(tmp_path, caplog):
    a, _ = parse_known_args_hw_guarded(_entry_parser(), ["--hw-preset", "phase3a_core"])
    with caplog.at_level(logging.INFO, logger="t"):
        hw = setup_hw_config(a, output_dir=str(tmp_path / "out"), logger=logging.getLogger("t"))
    assert not hw.is_off()
    saved = HWConfig.from_json(str(tmp_path / "out" / "hw_config.json"))
    assert saved == hw and saved.hash() == hw.hash()
    assert hw.hash() in caplog.text and '"hw_slope_bound": true' in caplog.text


def test_setup_hw_config_default_is_off(tmp_path):
    a, _ = parse_known_args_hw_guarded(_entry_parser(), [])
    hw = setup_hw_config(a, output_dir=str(tmp_path))
    assert hw.is_off() and hw_or_none(hw) is None
    assert os.path.isfile(tmp_path / "hw_config.json")


def test_setup_hw_config_bad_preset_raises():
    a, _ = parse_known_args_hw_guarded(_entry_parser(), ["--hw-preset", "no_such_preset"])
    with pytest.raises(FileNotFoundError):
        setup_hw_config(a)


def test_passthrough_argv(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text("{}")
    a, _ = parse_known_args_hw_guarded(_entry_parser(), [
        "--hw-config", str(cfg), "--hw-preset", "phase3b_tm", "--zig-degrade-thr", "0.15"])
    argv = hw_passthrough_argv(a)
    assert argv == ["--hw-config", str(cfg), "--hw-preset", "phase3b_tm",
                    "--zig-degrade-thr", "0.15"]
    # what the driver passes, the entry parser accepts
    b, unknown = parse_known_args_hw_guarded(_entry_parser(), argv)
    assert unknown == [] and hw_passthrough_argv(b) == argv
    assert hw_passthrough_argv(parse_known_args_hw_guarded(_entry_parser(), [])[0]) == []


# --------------------------------------------------------------------------- wiring
def test_dagger_script_uses_the_guard():
    """Static check: the heavy dagger script parses via the guarded helper."""
    with open(os.path.join(ROOT, "dagger-nuance-distillation-kirchhoffnet.py")) as fh:
        src = fh.read()
    assert "add_entry_hw_args(_bo_parser)" in src
    assert "parse_known_args_hw_guarded(_bo_parser)" in src
    assert not re.search(r"_bo_parser\.parse_known_args\(", src)
    assert "hw=self.hw" in src and "hw=KN_HW" in src
    assert "hw_power_penalty" in src
