"""fixed-mlp-distillation-kirchhoffnet.py builds its KNet student with the HWConfig."""

import importlib.util
import os
import sys

import numpy as np

from hw import HWConfig
from hw.cli import hw_config_from_args

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "fixed_mlp_distill_hw", os.path.join(ROOT, "fixed-mlp-distillation-kirchhoffnet.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(mod, monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["fixed", "--dataset-dir", "/nonexistent",
                                      "--output", "/nonexistent"] + extra)
    return mod.parse_args()


LO = np.array([-4.0, 0.0, -1.0, 0.0], dtype=np.float32)
HI = np.array([-1.0, 3.0, 3.0, 3.0], dtype=np.float32)


def test_fixed_mlp_student_gets_hw(monkeypatch):
    mod = _load()
    a = _args(mod, monkeypatch, ["--hw-preset", "phase3a_core", "--zig-degrade-thr", "0.2"])
    assert a.zig_degrade_threshold == 0.2       # alias of --zig-degrade-threshold
    a.hw_resolved = hw_config_from_args(a).to_dict()
    net = mod.build_student(a, LO, HI).net
    hw = net.core.stages[0].hw
    assert hw is not None and hw == HWConfig.from_dict(a.hw_resolved)


def test_fixed_mlp_student_default_is_legacy(monkeypatch):
    mod = _load()
    a = _args(mod, monkeypatch, [])
    a.hw_resolved = hw_config_from_args(a).to_dict()
    assert mod.build_student(a, LO, HI).net.core.stages[0].hw is None
