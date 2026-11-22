"""No dead trainable parameters under the hardware flags (Phase 3b W5 step 2).

A trainable parameter that never receives a gradient is *dead*: it does not
learn, it inflates the trainable-parameter count (the BO budget) and it hides
hook bugs -- the W5 4a bug was exactly that: under ``hw_compliance`` the
boundary and the ota_mesh families computed their currents but never
scattered them, so the network input was disconnected and its parameters got
no gradient.

The check runs a few optimiser steps on a random batch and then looks at the
gradient of every trainable parameter.  The warm-up steps matter: at the init
several parameters have a zero gradient *only* because of a zero
initialisation (the VCA bus matrices are zero, so the gain does not depend on
the taps; the GLN mixers are zero, ...).  They are alive after the first
update, so they are not dead.

ALLOWLIST (the legitimately grad-free parameters):

``core.stages.0.cell_lib.{a_raw,b_raw,r_raw,s_raw}``
    The first stage reads the (discharged) hold capacitor: with ``freeze_read``
    the core OTA of stage 0 is evaluated at the stage-entry state, which is
    exactly 0 V (every hidden node starts at 0).  The tanh argument is then
    ``u = s * (A * x_src - B * x_dst) * gm + theta = theta * gm``: it does not
    depend on the differential-gain weights ``A``/``B`` (nor on ``r`` under
    ``hw_ratio_ab``) nor on the sign ``s``, so their gradient is exactly zero.
    This is a property of the frozen-read architecture (Phase 3b 2.3), not of
    a hardware flag: it is the same with the hardware model off.
"""

from __future__ import annotations

import os
import re
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "golden"))
sys.path.insert(0, os.path.dirname(HERE))

import make_golden as mg  # noqa: E402
from hw import HWConfig, HWConstraints, hw_or_none, load_preset  # noqa: E402

D = torch.float64
WARMUP_STEPS = 3
LR = 1e-2

# Every flag of hw/constraints.HWConstraints that changes the model (the
# diagnostics-only flags are skipped).
SINGLE_FLAGS = [
    ("hw_ratio_ab", True),
    ("hw_slope_bound", True),
    ("hw_ic_coupling", True),            # needs hw_slope_bound
    ("hw_compliance", True),
    ("hw_clamp_transfer", True),
    ("hw_no_alpha", True),
    ("hw_alpha_bwd", 0.01),
    ("hw_vref_symmetric", True),
    ("hw_vref_bits", 6),
    ("hw_theta_bound", True),
    ("hw_xbar_bound", True),
    ("hw_xbar_bits", 5),
    ("hw_xbar_mode", "offset_binary"),
    ("hw_xbar_mode", "differential"),
    ("hw_c_node", "uniform"),
    ("hw_c_node", "physical"),
    ("hw_power_lambda", 1e-3),
    ("hw_mismatch_sigma", {"isat": 0.05, "g": 0.1, "a": 0.01, "b": 0.01, "theta": 0.1}),
    ("hw_mismatch_eval", True),
    ("hw_vca_bits", 6),
    ("hw_isat_eff_clamp", True),
    ("hw_gln_mode", "tail"),
    ("hw_gln_mode", "slope"),
]

# The legitimately grad-free parameters (see the module docstring).
ALLOWLIST = (
    re.compile(r"^core\.stages\.0\.cell_lib\.(a_raw|b_raw|r_raw|s_raw)$"),
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _build(spec, hw=None):
    with mg.f64():
        mg.seed_all(int(spec["seed"]))
        return mg._build_net(spec, hw=hw).double()


def _batch(spec, n=32, seed=7):
    """A random batch (the trajectory data of the task, shuffled)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, int(spec["in_dim"]), generator=g, dtype=D) * 2.0 - 1.0
    return x


def _dead_params(net, x, steps=WARMUP_STEPS):
    """Train ``steps`` steps on ``x`` and return the trainable params with no gradient."""
    params = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    opt = torch.optim.Adam([p for _, p in params], lr=LR)
    net.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = mg.model_forward(net, x).pow(2).mean()
        loss.backward()
        opt.step()
    opt.zero_grad(set_to_none=True)
    loss = mg.model_forward(net, x).pow(2).mean()
    loss.backward()
    dead = []
    for n, p in params:
        if p.grad is None or float(p.grad.abs().max()) == 0.0:
            dead.append(n)
    return dead


def _allowed(dead):
    return [n for n in dead if not any(rx.match(n) for rx in ALLOWLIST)]


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("preset", ["phase3a_core", "phase3a_full", "phase3b_tm"])
@pytest.mark.parametrize("spec_name", ["TOY_NET", "PAPER_NET"])
def test_no_dead_params_under_the_presets(preset, spec_name):
    """Every trainable parameter learns under every hardware preset."""
    spec = getattr(mg, spec_name)
    hw = hw_or_none(HWConfig.from_dict(load_preset(preset)))
    net = _build(spec, hw)
    dead = _dead_params(net, _batch(spec))
    assert _allowed(dead) == [], dead
    # the allowlisted ones are really there (the check is not vacuous)
    assert dead, "expected the stage-0 allowlist to be non-empty"


@pytest.mark.parametrize("flag,value", SINGLE_FLAGS)
def test_no_dead_params_under_a_single_flag(flag, value):
    """Every trainable parameter learns under every single hardware flag."""
    spec = mg.TOY_NET
    kw = {flag: value}
    if flag == "hw_ic_coupling":
        kw["hw_slope_bound"] = True       # required by HWConstraints.validate
    hw = HWConfig(constraints=HWConstraints(**kw))
    net = _build(spec, hw)
    dead = _dead_params(net, _batch(spec))
    assert _allowed(dead) == [], dead


def test_the_hardware_model_off_is_not_affected():
    """The baseline (hardware model off) has the same allowlist."""
    spec = mg.TOY_NET
    net = _build(spec, None)
    dead = _dead_params(net, _batch(spec))
    assert _allowed(dead) == [], dead
    assert dead == ["core.stages.0.cell_lib.a_raw",
                    "core.stages.0.cell_lib.b_raw",
                    "core.stages.0.cell_lib.s_raw"]


# ---------------------------------------------------------------------------
# the explicit fix: the soft clip is dead under hw_compliance
# ---------------------------------------------------------------------------
def test_clip_sharpness_is_frozen_under_compliance():
    """Under hw_compliance the soft clip is zero, so its sharpness is dead.

    It is frozen (requires_grad=False) but stays in the state_dict, exactly
    like the dead a_raw / b_raw of hw_ratio_ab.
    """
    hw = HWConfig(constraints=HWConstraints(hw_compliance=True))
    net = _build(mg.TOY_NET, hw)
    ps = [(n, p) for n, p in net.named_parameters() if n.endswith("clip_sharpness_raw")]
    assert ps, "the toy net has a learnable clip sharpness"
    assert all(not p.requires_grad for _, p in ps)
    sd = net.state_dict()
    assert all(n in sd for n, _ in ps)
    # strict loading of a legacy (flags-off) checkpoint still works
    plain = _build(mg.TOY_NET, None)
    net.load_state_dict(plain.state_dict(), strict=True)
    assert all(not p.requires_grad for _n, p in net.named_parameters()
               if _n.endswith("clip_sharpness_raw"))
    # the flags-off path is untouched
    assert all(p.requires_grad for _n, p in plain.named_parameters()
               if _n.endswith("clip_sharpness_raw"))


def test_bo_param_count_drops_the_frozen_clip_sharpness():
    """The BO budget counts trainable parameters only: the frozen ones drop out."""
    import bo_param_sampling as bps
    kw = dict(in_dim=4, out_dim=1, fanout_count=2, use_robust_input=False,
              dagger=False, readout_mode="shared_sense", readout_senses=1,
              learnable_clip=True)
    n_off = bps.knet_param_count(10, 4, 4, 2, hw=HWConfig(), **kw)
    n_comp = bps.knet_param_count(
        10, 4, 4, 2, hw=HWConfig(constraints=HWConstraints(hw_compliance=True)), **kw)
    # one clip_sharpness_raw per stage (4 stages in the BO build)
    assert n_comp == n_off - 4, (n_off, n_comp)
