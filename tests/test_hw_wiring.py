"""The hardware configuration reaches every device of a built network (W2 step 1).

``hw`` (an ``hw.HWConfig`` or ``None``) is threaded
``topology.build_net_from_config`` -> ``DifferentialStage`` -> ``FreeTanhLibrary``.
``None`` (or an all-default ``HWConfig``) must leave the forward pass and the
state_dict untouched.
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
from topology import build_net_from_config  # noqa: E402


def _build(spec, hw=None):
    """Build the toy net the way ``make_golden`` does, with an optional ``hw``."""
    cfg = mg._build_cfg(spec)
    with mg.f64():
        mg.seed_all(int(spec["seed"]))
        cl = {k: v for k, v in spec["cell_lib"].items() if k != "name"}
        cell_lib = make_cell_library(spec["cell_lib"]["name"], **cl, hw=hw)
        kw = dict(spec["build"])
        kw["boundary_fan_out"] = mg._int_keys(kw["boundary_fan_out"])
        net = build_net_from_config(cfg, cell_lib=cell_lib, hw=hw, **kw)
    return net.double()


def test_hw_is_threaded_to_every_stage_and_cell_library():
    hw = HWConfig()
    net = _build(dict(mg.TOY_NET), hw=hw)
    assert len(net.core.stages) >= 2
    for st in net.core.stages:
        assert st.hw is hw
        assert st.cell_lib.hw is hw
        assert st.boundary_cell_lib is None or st.boundary_cell_lib.hw is hw
        assert st.readout_sense_cell_lib is None or st.readout_sense_cell_lib.hw is hw
        assert st.output_ode_cell_lib is None or st.output_ode_cell_lib.hw is hw


def test_hw_none_is_the_default_path():
    lib = make_cell_library("tanh_free", num_edges=4)
    assert lib.hw is None
    assert lib.hw_constraints is None
    net = _build(dict(mg.TOY_NET))
    for st in net.core.stages:
        assert st.hw is None
        assert st.cell_lib.hw is None


def test_all_default_hw_config_is_the_same_as_none():
    """``HWConfig()`` has every flag off: the forward must be bit-identical."""
    spec = dict(mg.TOY_NET)
    net_none = _build(spec, hw=None)
    net_default = _build(spec, hw=HWConfig())
    assert sorted(net_none.state_dict()) == sorted(net_default.state_dict())
    x = mg.make_inputs(spec)
    with torch.no_grad():
        y0 = mg.model_forward(net_none, x)
        y1 = mg.model_forward(net_default, x)
    torch.testing.assert_close(y1, y0, rtol=0, atol=0)


def test_hw_is_not_a_parameter_or_buffer():
    """The hardware config must not appear in the state_dict (goldens stay loadable)."""
    hw = HWConfig()
    lib = FreeTanhLibrary(num_edges=3, hw=hw)
    assert not any("hw" in k for k in lib.state_dict())
    assert lib.hw_constraints is hw.constraints
