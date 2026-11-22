"""Flags off == golden (Phase 3b 2.4(b)).

Every golden case of ``tests/test_golden.py`` is rebuilt TWICE through the
SAME build path (``tests/golden/make_golden.py``):

  * with ``hw=None``      -> the legacy path, and
  * with ``hw=HWConfig()``-> the hardware layer is present but every flag is
    at its default (all off).

Both must reproduce the golden numbers (forward output, per-stage states,
loss, per-parameter gradients, and the 30-step Adam trajectory of the two
trajectory cases) at the golden tolerances, and the state_dict key set must
be unchanged: the hardware layer adds no parameter and no buffer.

The golden files are NOT regenerated (see ``tests/golden/README.md``).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "golden"), HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_golden as mg  # noqa: E402
import test_golden as tg  # noqa: E402  (the golden helpers: IGNORED_KEYS, load, spec_of)

from hw import HWConfig  # noqa: E402
from hw.timemux import TimeMuxConfig  # noqa: E402

RTOL, ATOL = tg.RTOL, tg.ATOL
TRAJ_RTOL, TRAJ_ATOL = tg.TRAJ_RTOL, tg.TRAJ_ATOL

CASES = tg.CASES
TRAJ_CASES = tg.TRAJ_CASES

# the ways of asking for "no hardware constraints"
MODES = {
    "none": None,
    "all_defaults": HWConfig(),
    # the time-multiplexed fabric exists but is switched off: still the
    # legacy path (the boundary is never built)
    "timemux_disabled": HWConfig(timemux=TimeMuxConfig(enable=False)),
}


def _ign(name: str) -> bool:
    return tg._ign(name)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("mode", sorted(MODES))
def test_hw_off_reproduces_the_golden(case, mode):
    rec = tg.load(case)
    res = mg.run_case(tg.spec_of(rec), golden_sd=rec["state_dict"],
                      ignored=tg.IGNORED_KEYS, hw=MODES[mode])

    # ---- state_dict: the same keys, the same shapes/dtypes ----
    model = res["model"]
    got = sorted(k for k in model.state_dict() if not _ign(k))
    want = sorted(k for k in rec["sorted_keys"] if not _ign(k))
    assert got == want, set(got) ^ set(want)
    for k in want:
        assert tuple(model.state_dict()[k].shape) == tuple(rec["state_dict"][k].shape), k

    # ---- forward ----
    assert torch.equal(res["inputs"], rec["inputs"])
    assert res["y"].dtype == torch.float64
    torch.testing.assert_close(res["y"], rec["y"], rtol=RTOL, atol=ATOL)

    # ---- per-stage states ----
    assert len(res["stage_states"]) == len(rec["stage_states"])
    for i, (a, b) in enumerate(zip(res["stage_states"], rec["stage_states"])):
        torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL,
                                   msg=lambda m, i=i: f"stage {i}: {m}")

    # ---- loss, gradients and the no-gradient set ----
    torch.testing.assert_close(res["loss"], rec["loss"], rtol=RTOL, atol=ATOL)
    got_g = {k: v for k, v in res["grads"].items() if not _ign(k)}
    want_g = {k: v for k, v in rec["grads"].items() if not _ign(k)}
    assert sorted(got_g) == sorted(want_g), set(got_g) ^ set(want_g)
    for k in want_g:
        torch.testing.assert_close(got_g[k], want_g[k], rtol=RTOL, atol=ATOL,
                                   msg=lambda m, k=k: f"grad {k}: {m}")
    assert sorted(n for n in res["no_grad_names"] if not _ign(n)) == \
        sorted(n for n in rec["no_grad_names"] if not _ign(n))


@pytest.mark.parametrize("case", TRAJ_CASES)
@pytest.mark.parametrize("mode", sorted(MODES))
def test_hw_off_reproduces_the_golden_trajectory(case, mode):
    rec = tg.load(case)
    tr = mg.run_trajectory(tg.spec_of(rec), rec["state_dict"],
                           ignored=tg.IGNORED_KEYS, hw=MODES[mode])
    t = rec["traj"]
    assert torch.equal(tr["X"], t["X"]) and torch.equal(tr["Y"], t["Y"])
    torch.testing.assert_close(tr["losses"], t["losses"], rtol=TRAJ_RTOL, atol=TRAJ_ATOL)
    for k, v in t["final_sd"].items():
        if _ign(k) or not v.is_floating_point():
            continue
        torch.testing.assert_close(tr["final_sd"][k], v, rtol=TRAJ_RTOL, atol=TRAJ_ATOL,
                                   msg=lambda m, k=k: f"final param {k}: {m}")
    assert t["losses"].numel() == mg.TRAJ_STEPS


def test_the_modes_agree_with_each_other():
    """Every way of asking for "no hardware constraints" is bit for bit the same."""
    case = CASES[0]
    rec = tg.load(case)
    a = mg.run_case(tg.spec_of(rec), golden_sd=rec["state_dict"],
                    ignored=tg.IGNORED_KEYS, hw=None)
    for name, mode in sorted(MODES.items()):
        if mode is None:
            continue
        b = mg.run_case(tg.spec_of(rec), golden_sd=rec["state_dict"],
                        ignored=tg.IGNORED_KEYS, hw=mode)
        assert torch.equal(a["y"], b["y"]), name
        assert torch.equal(a["loss"], b["loss"]), name
        assert sorted(a["model"].state_dict()) == sorted(b["model"].state_dict()), name
        for k in a["grads"]:
            assert torch.equal(a["grads"][k], b["grads"][k]), (name, k)


def test_all_default_hw_config_is_off():
    """``HWConfig()`` must be recognised as "no constraint" (the builder rule)."""
    from hw import hw_or_none
    assert HWConfig().is_off()
    assert hw_or_none(HWConfig()) is None
    assert hw_or_none(None) is None
