"""The time-multiplexed boundary inside ``KirchhoffNet.forward`` (Phase 3b W4).

The hook (Phase 3b section 2.3) inserts a sample-and-hold boundary between a
stage call and its interstage transfer::

    x_hold, x = boundary.capture(x, i)   # the hold cap droops, the node does not
    x        = transfer(x)               # the live state of stage i+1
    # stage i+1 reads x_hold through its FROZEN core read (freeze_read)

These tests cover:

* the fabric off (``enable=False``) is bit-identical to the legacy path,
* the boundary does not change the ``state_dict``,
* kT/C noise varies in train mode and is deterministic in eval,
* the injection / the droop shift the output by the expected order,
* the frozen core read uses ``x_hold`` and nothing else does,
* the ``xfer_mode`` helpers replace the transfer module.

Run:  venv/bin/python -m pytest tests/test_hw_timemux_hook.py -q
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _p in (os.path.join(REPO_ROOT, "tests", "golden"), REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_golden as mg  # noqa: E402
from cell_library import make_cell_library  # noqa: E402
from hw import HWConfig  # noqa: E402
from hw.timemux import TimeMuxConfig, transfer_no_tanh  # noqa: E402
from topology import build_net_from_config  # noqa: E402


def _build(hw=None):
    """The toy net of ``make_golden`` (2 stages, a residual-relu-tanh transfer)."""
    spec = dict(mg.TOY_NET)
    cfg = mg._build_cfg(spec)
    with mg.f64():
        mg.seed_all(int(spec["seed"]))
        cl = {k: v for k, v in spec["cell_lib"].items() if k != "name"}
        cell_lib = make_cell_library(spec["cell_lib"]["name"], **cl, hw=hw)
        kw = dict(spec["build"])
        kw["boundary_fan_out"] = mg._int_keys(kw["boundary_fan_out"])
        net = build_net_from_config(cfg, cell_lib=cell_lib, hw=hw, **kw)
    return net.double()


def _tm(**over):
    """A time-mux config with only the requested effect switched on."""
    over.setdefault("ktc", False)
    over.setdefault("inject_mv", 0.0)
    over.setdefault("droop_mv", 0.0)
    return TimeMuxConfig(enable=True, **over)


def _forward(net, x):
    with torch.no_grad():
        return mg.model_forward(net, x)


# --------------------------------------------------------------------------
# off / on
# --------------------------------------------------------------------------

def test_enable_false_is_the_legacy_path():
    net_off = _build(HWConfig(timemux=TimeMuxConfig(enable=False)))
    net_none = _build(None)
    assert net_off.core.boundaries is None
    x = mg.make_inputs(dict(mg.TOY_NET))
    y_off = _forward(net_off, x)
    y_none = _forward(net_none, x)
    assert torch.equal(y_off, y_none)


def test_the_boundary_does_not_change_the_state_dict():
    net_none = _build(None)
    net_tm = _build(HWConfig(timemux=_tm()))
    assert net_tm.core.boundaries is not None
    assert sorted(net_tm.state_dict()) == sorted(net_none.state_dict())
    for k in net_none.state_dict():
        assert torch.equal(net_tm.state_dict()[k], net_none.state_dict()[k]), k


def test_the_boundary_is_built_once_per_transfer():
    net = _build(HWConfig(timemux=_tm()))
    b = net.core.boundaries
    assert len(b) == len(net.core.transfers) == len(net.core.stages) - 1
    assert b[0].n_nodes == net.core.stages[0].num_nodes
    assert b[0].config.enable is True


# --------------------------------------------------------------------------
# noise
# --------------------------------------------------------------------------

def test_ktc_noise_varies_in_train_mode_and_is_deterministic_in_eval():
    net = _build(HWConfig(timemux=_tm(ktc=True)))
    x = mg.make_inputs(dict(mg.TOY_NET))

    net.train()
    a = _forward(net, x)
    b = _forward(net, x)
    assert not torch.equal(a, b), "the kT/C noise must change the train-mode output"
    assert (a - b).abs().max().item() > 0.0

    net.eval()
    c = _forward(net, x)
    d = _forward(net, x)
    assert torch.equal(c, d), "the eval-mode output must be deterministic"


def test_eval_mode_has_no_noise_at_all():
    net = _build(HWConfig(timemux=_tm(ktc=True)))
    x = mg.make_inputs(dict(mg.TOY_NET))
    net.eval()
    y = _forward(net, x)
    # the same net with the noise switched off gives exactly this
    net2 = _build(HWConfig(timemux=_tm(ktc=False)))
    net2.eval()
    y2 = _forward(net2, x)
    assert torch.equal(y, y2)


# --------------------------------------------------------------------------
# the deterministic errors
# --------------------------------------------------------------------------

def test_the_injection_shifts_the_output_by_the_expected_order():
    x = mg.make_inputs(dict(mg.TOY_NET))
    ref = _forward(_build(None), x)
    d1 = (_forward(_build(HWConfig(timemux=_tm(inject_mv=1.0))), x) - ref).abs().max().item()
    d2 = (_forward(_build(HWConfig(timemux=_tm(inject_mv=2.0))), x) - ref).abs().max().item()
    # 1 mV on a 0.07 V/unit scale is 0.0143 units: a small but visible shift
    assert 1e-3 < d1 < 1e-1, d1
    # and it is linear in the injected voltage
    assert 1.5 * d1 < d2 < 2.5 * d1, (d1, d2)


def test_the_droop_shifts_the_output_by_the_expected_order():
    x = mg.make_inputs(dict(mg.TOY_NET))
    ref = _forward(_build(None), x)
    d1 = (_forward(_build(HWConfig(timemux=_tm(droop_mv=1.0))), x) - ref).abs().max().item()
    d10 = (_forward(_build(HWConfig(timemux=_tm(droop_mv=10.0))), x) - ref).abs().max().item()
    # the droop only reaches the frozen core read: a much smaller effect than
    # the injection (which also shifts the live state of the next stage)
    assert 0.0 < d1 < 1e-5, d1
    # linear in the droop voltage
    assert 5.0 * d1 < d10 < 20.0 * d1, (d1, d10)


def test_the_injection_and_the_droop_do_not_cancel():
    x = mg.make_inputs(dict(mg.TOY_NET))
    ref = _forward(_build(None), x)
    inj = _forward(_build(HWConfig(timemux=_tm(inject_mv=1.0))), x)
    droop = _forward(_build(HWConfig(timemux=_tm(droop_mv=1.0))), x)
    both = _forward(_build(HWConfig(timemux=_tm(inject_mv=1.0, droop_mv=1.0))), x)
    assert not torch.equal(both, inj)
    assert not torch.equal(both, droop)


# --------------------------------------------------------------------------
# where the hold value is used
# --------------------------------------------------------------------------

def test_the_frozen_core_read_uses_x_hold_and_nothing_else():
    net = _build(HWConfig(timemux=_tm(inject_mv=1.0, droop_mv=1.0)))
    stage = net.core.stages[1]

    seen = []
    orig_compute = stage._compute_i_edge_const

    def spy_compute(x0, gate):
        seen.append(x0.detach().clone())
        return orig_compute(x0, gate)

    stage._compute_i_edge_const = spy_compute

    calls = []
    orig_forward = stage.forward

    def spy_forward(*args, **kwargs):
        calls.append((kwargs.get("x0"), kwargs.get("x_hold")))
        return orig_forward(*args, **kwargs)

    stage.forward = spy_forward

    x = mg.make_inputs(dict(mg.TOY_NET))
    _forward(net, x)

    assert len(calls) == 1
    x0, x_hold = calls[0]
    assert x_hold is not None, "the frozen core read must get the hold value"
    assert not torch.equal(x_hold, x0)
    # the frozen core current is computed from the hold value ...
    assert len(seen) == 1
    assert torch.equal(seen[0], x_hold)
    # ... and the hold value is the drooped version of the live state
    assert not torch.equal(x_hold, x0)


def test_the_boundary_capture_happens_once_per_boundary():
    net = _build(HWConfig(timemux=_tm()))
    calls = []
    for i, b in enumerate(net.core.boundaries):
        orig = b.capture

        def spy(x_pre, stage_idx=0, generator=None, _i=i, _orig=orig):
            calls.append((_i, stage_idx, x_pre.detach().clone()))
            return _orig(x_pre, stage_idx, generator)

        b.capture = spy

    x = mg.make_inputs(dict(mg.TOY_NET))
    _forward(net, x)
    assert len(calls) == len(net.core.boundaries)
    assert [c[1] for c in calls] == list(range(len(net.core.boundaries)))


def test_no_capture_when_the_fabric_is_off():
    net = _build(HWConfig(timemux=TimeMuxConfig(enable=False)))
    assert net.core.boundaries is None


# --------------------------------------------------------------------------
# the transfer variants
# --------------------------------------------------------------------------

def test_the_xfer_mode_helpers_replace_the_transfer_module():
    net = _build(HWConfig(timemux=_tm(xfer_mode="no_tanh")))
    tr = net.core.transfers[0]
    x = torch.randn(5, tr.in_nodes, dtype=torch.float64)
    got = net.core._apply_transfer(0, x)
    want = transfer_no_tanh(x, tr.residual_w1, tr.residual_w2, tr.residual_w3,
                            tr.residual_vth)
    assert torch.allclose(got, want)


def test_the_full_mode_uses_the_transfer_module():
    net = _build(HWConfig(timemux=_tm(xfer_mode="full")))
    tr = net.core.transfers[0]
    x = torch.randn(5, tr.in_nodes, dtype=torch.float64)
    assert torch.equal(net.core._apply_transfer(0, x), tr(x))


def test_a_non_residual_transfer_falls_back_to_the_module():
    """The xfer_mode helpers need the residual parameters of the transfer."""
    from stage_transfer import StageTransfer
    net = _build(HWConfig(timemux=_tm(xfer_mode="no_tanh")))
    old = net.core.transfers[0]
    tr = StageTransfer(old.in_nodes, old.out_nodes, activation="none").double()
    net.core.transfers[0] = tr
    x = torch.randn(5, tr.in_nodes, dtype=torch.float64)
    with pytest.warns(UserWarning):
        got = net.core._apply_transfer(0, x)
    assert torch.equal(got, tr(x))
