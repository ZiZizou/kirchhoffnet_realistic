"""Small CPU learning validation of the hardware model (Phase 3b 2.4(c), W5 step 4).

Marked ``slow`` (excluded from the default run by ``pytest.ini``)::

    venv/bin/python -m pytest -m slow tests/test_hw_learning.py -q

Tasks (``tests/golden/make_golden.py``): ``FRIEDMAN2_NET`` (4 -> 1, 2 stages,
ota_mesh readout) and ``TOY_NET`` (4 -> 3, shared-sense + GLN + clip + VCA).
Presets: ``none`` (unconstrained reference), ``phase3a_core``, ``phase3a_full``,
``phase3b_tm``.  Adam, 400 full-batch steps on n = 256 training rows; held-out
MSE on 256 other rows of the same task; 3 init seeds; float64.

Acceptance (2.4(c), thresholds as proposed in the spec):

1. no NaN/Inf in loss or gradients; every trainable parameter gets a finite
   gradient on at least one step;
2. final train loss <= 0.25 x initial (every task, preset, seed);
3. constrained held-out MSE (mean over seeds) <= 1.5x / 2.0x / 2.5x the
   unconstrained one for phase3a_core / phase3a_full / phase3b_tm; the
   comparison counts only if the unconstrained Friedman-2 R^2 >= 0.8;
4. post-train invariants on the held-out set: gm_eff*max(A,B) <= slope_max,
   |x| <= 1.05 x_hard (hidden and accumulator nodes, every Heun state),
   max(dt*rho(J)) <= 2, quantised Vref / crossbar values on their code grid;
5. seed spread: max/min final constrained train loss <= 2.

``HW_LEARN_OUT=<dir>`` writes one JSON per (task, preset, seed) run and reuses
an existing one only when its result schema, steps, and learning rate match,
so runs can be spread over several processes (``-k``) and cross-run checks
are evaluated afterwards.
"""

from __future__ import annotations

import functools
import json
import math
import os
import time

import pytest
import torch
import torch.nn as nn

import make_golden as mg
from hw import HWConfig, hw_or_none, load_preset
from hw.audit import edge_stats, rail_stats, run_capture, stiffness_stats
from hw.constraints import STIFFNESS_LIMIT, W_MAX
from hw.units import HWUnits

pytestmark = pytest.mark.slow

TASKS = {"friedman2": mg.FRIEDMAN2_NET, "toy_mimo": mg.TOY_NET}
PRESETS = ("none", "phase3a_core", "phase3a_full", "phase3b_tm")
CONSTRAINED = PRESETS[1:]
SEEDS = (0, 1, 2)
STEPS = int(os.environ.get("HW_LEARN_STEPS", 400))
LR = float(os.environ.get("HW_LEARN_LR", 1e-2))
N_TRAIN = N_HELD = 256
MSE_RATIO_MAX = {"phase3a_core": 1.5, "phase3a_full": 2.0, "phase3b_tm": 2.5}
LEARN_RATIO_MAX = 0.25
SEED_SPREAD_MAX = 2.0
R2_MIN = 0.8
OUT = os.environ.get("HW_LEARN_OUT")
RESULT_SCHEMA = 2  # bump when result semantics / recorded fields change
# Parameters that legitimately get no gradient under a flag (criterion 1).
# Under hw_compliance the soft clip term is zero (the leg roll-off is the rail),
# so the learnable clip sharpness is dead: it is FROZEN (requires_grad=False,
# Phase 3b W5 step 2), so it is not a trainable parameter at all and never
# shows up in ``no_grad_params`` (which only lists trainable parameters).
DOCUMENTED_DEAD: dict[str, tuple[str, ...]] = {}


def preset_hw(name: str, task: str | None = None):
    if name == "none":
        return None
    return hw_or_none(HWConfig.from_dict(load_preset(name)))


def task_data(spec):
    """512 rows of the task's trajectory distribution: 256 train / 256 held-out."""
    t = spec["trajectory"]
    n = N_TRAIN + N_HELD
    X, Y = (mg._friedman2_xy(n, t["seed"]) if t["data"] == "friedman2"
            else mg._toy_xy(n, t["seed"]))
    return X[:N_TRAIN], Y[:N_TRAIN], X[N_TRAIN:], Y[N_TRAIN:]


class _Wrap(nn.Module):
    """``model.net.core`` layout expected by hw.audit."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return mg.model_forward(self.net, x)


def _on_grid(v: torch.Tensor, lo: float, hi: float, bits: int) -> float:
    """Max distance (in codes) of ``v`` from the ``bits``-bit grid on [lo, hi]."""
    c = (v.double() - lo) / (hi - lo) * (2 ** bits - 1)
    return float((c - c.round()).abs().max()) if v.numel() else 0.0


def quant_check(net, hw) -> dict:
    out = {}
    if hw is None:
        return out
    c = hw.constraints
    for i, st in enumerate(net.core.stages):
        if not getattr(st, "_has_shared_readout", False):
            continue
        with torch.no_grad():
            if c.hw_vref_bits is not None:
                v = st.sense_vref().reshape(-1)
                lo = -st.x_max if c.hw_vref_symmetric else 0.0
                out[f"vref_s{i}"] = {"off_grid": _on_grid(v, lo, st.x_max, c.hw_vref_bits),
                                     "n_distinct": int(torch.unique(v).numel())}
            if c.hw_xbar_bits is not None and c.hw_xbar_mode == "signed":
                W = st.crossbar_weights().reshape(-1)
                out[f"xbar_s{i}"] = {"off_grid": _on_grid(W, -W_MAX, W_MAX, c.hw_xbar_bits),
                                     "n_distinct": int(torch.unique(W).numel())}
    return out


def _train(task: str, preset: str, seed: int) -> dict:
    spec = TASKS[task]
    hw = preset_hw(preset, task)
    units = hw.units if hw is not None else HWUnits()
    Xtr, Ytr, Xte, Yte = task_data(spec)
    t0 = time.time()
    with mg.f64():
        mg.seed_all(int(spec["seed"]) + 101 * seed)
        net = mg._build_net(spec, hw=hw).double()
        params = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
        opt = torch.optim.Adam([p for _, p in params], lr=LR)
        got_grad = set()
        losses, nonfinite = [], 0
        net.train()
        for _ in range(STEPS):
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((mg.model_forward(net, Xtr) - Ytr) ** 2)
            if not torch.isfinite(loss):
                nonfinite += 1
                losses.append(float("nan"))
                continue
            loss.backward()
            for n, p in params:
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        nonfinite += 1
                    else:
                        got_grad.add(n)
            opt.step()
            losses.append(float(loss.detach()))
        net.eval()
        with torch.no_grad():
            final_train = float(torch.mean((mg.model_forward(net, Xtr) - Ytr) ** 2))
            pred = mg.model_forward(net, Xte)
            held = float(torch.mean((pred - Yte) ** 2))
            r2 = float(1.0 - torch.sum((pred - Yte) ** 2) / torch.sum((Yte - Yte.mean(0)) ** 2))
        # post-train invariants on the held-out rows
        model = _Wrap(net)
        y, recs, finals = run_capture(model, Xte)
        edges = edge_stats(model, recs, units)
        rails = rail_stats(model, recs, finals, units)
        stiff = stiffness_stats(model, recs, n_rows=32)
    slope_max = max((r["slope"][1] for r in edges.values() if "slope" in r), default=float("nan"))
    rail_max = max(r["max_abs"] for r in rails)
    dt_rho = max(r["max_dt_rho"] for r in stiff)
    quant = quant_check(net, hw)
    return {
        "result_schema": RESULT_SCHEMA,
        "task": task, "preset": preset, "seed": seed, "steps": STEPS, "lr": LR,
        "overrides": {},
        "n_params": sum(p.numel() for _, p in params),
        "initial_loss": losses[0], "final_loss": final_train,
        "last_step_loss": losses[-1], "learn_ratio": final_train / losses[0],
        "held_mse": held, "held_r2": r2, "nonfinite": nonfinite,
        "no_grad_params": sorted(n for n, _ in params if n not in got_grad),
        "slope_max_seen": slope_max, "slope_limit": units.slope_max,
        "slope_by_family": {f: r["slope"][1] for f, r in edges.items() if "slope" in r},
        "rail_max_abs": rail_max, "rail_limit": 1.05 * units.x_hard,
        "max_dt_rho": dt_rho, "stiffness_limit": STIFFNESS_LIMIT,
        "quant": quant, "y_finite": bool(torch.isfinite(y).all()),
        "losses_every_50": losses[::50] + [losses[-1]],
        "seconds": time.time() - t0,
    }


@functools.lru_cache(maxsize=None)
def run(task: str, preset: str, seed: int) -> dict:
    path = None
    if OUT:
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, f"{task}__{preset}__s{seed}.json")
        if os.path.isfile(path):
            with open(path) as fh:
                r = json.load(fh)
            if (r.get("result_schema") == RESULT_SCHEMA
                    and r.get("steps") == STEPS and r.get("lr") == LR):
                return r
    torch.set_num_threads(int(os.environ.get("HW_LEARN_THREADS", 4)))
    r = _train(task, preset, seed)
    if path:
        with open(path, "w") as fh:
            json.dump(r, fh, indent=1)
    return r


ALL = [(t, p, s) for t in TASKS for p in PRESETS for s in SEEDS]


@pytest.mark.parametrize("preset", ["phase3a_full", "phase3b_tm"])
def test_friedman2_physical_c_node_builds_and_runs(preset):
    """FRIEDMAN2 (c_eff = 9.06) builds and runs with hw_c_node='physical'.

    Its ota_mesh accumulator carries 16 mesh gate+leg terminals = 52 fF of
    parasitic.  The node capacitance budget is ``c_eff * C_u = 453 fF``, so
    the parasitic fits (the old code compared it against C_u = 50 fF and
    wrongly rejected the net).
    """
    hw = hw_or_none(HWConfig.from_dict(load_preset(preset)))
    spec = TASKS["friedman2"]
    with mg.f64():
        net = mg._build_net(spec, hw=hw).double()
        x = task_data(spec)[0][:4]
        y = mg.model_forward(net, x)
        assert torch.isfinite(y).all()
        st = net.core.stages[0]
        c = st.node_capacitance(torch.zeros(1, st.num_nodes, dtype=torch.float64))
        assert float(c.mean()) == pytest.approx(st.c_eff, rel=1e-12)
        # the per-node capacitance is the class mean (c_eff * C_u) plus the
        # node's own parasitic deviation: every node stays positive
        assert float(c.min()) > 0.0


@pytest.mark.parametrize("task,preset,seed", ALL)
def test_finite_and_learns(task, preset, seed):
    r = run(task, preset, seed)
    assert r["nonfinite"] == 0 and r["y_finite"], r
    hw = preset_hw(preset, task)
    dead = tuple(s for flag, sfx in DOCUMENTED_DEAD.items()
                 if hw is not None and getattr(hw.constraints, flag) for s in sfx)
    unexpected = [n for n in r["no_grad_params"] if not n.endswith(dead or ("\0",))]
    assert unexpected == [], unexpected
    assert r["learn_ratio"] <= LEARN_RATIO_MAX, (
        f"final/initial train loss {r['learn_ratio']:.3f} > {LEARN_RATIO_MAX}")


@pytest.mark.parametrize("task,preset,seed", [a for a in ALL if a[1] != "none"])
def test_post_train_invariants(task, preset, seed):
    r = run(task, preset, seed)
    hw = preset_hw(preset, task)
    assert r["slope_max_seen"] <= r["slope_limit"] * (1 + 1e-9), r["slope_by_family"]
    # Rail compliance is only guaranteed by presets that actually enable the
    # compliance cell/clip. Core-only intentionally leaves this unconstrained.
    if hw is not None and hw.constraints.hw_compliance:
        assert r["rail_max_abs"] <= r["rail_limit"], r["rail_max_abs"]
    assert r["max_dt_rho"] <= r["stiffness_limit"], r["max_dt_rho"]
    for k, q in r["quant"].items():
        assert q["off_grid"] <= 1e-6, (k, q)


def test_unconstrained_friedman2_reference_is_good_enough():
    r2 = [run("friedman2", "none", s)["held_r2"] for s in SEEDS]
    assert min(r2) >= R2_MIN, f"unconstrained Friedman-2 held-out R^2 {r2} < {R2_MIN}: re-tune lr"


@pytest.mark.parametrize("task", list(TASKS))
@pytest.mark.parametrize("preset", CONSTRAINED)
def test_constrained_over_unconstrained_mse(task, preset):
    ref = sum(run(task, "none", s)["held_mse"] for s in SEEDS) / len(SEEDS)
    con = sum(run(task, preset, s)["held_mse"] for s in SEEDS) / len(SEEDS)
    assert math.isfinite(con) and con <= MSE_RATIO_MAX[preset] * ref, (
        f"{task}/{preset}: held-out MSE {con:.4g} = {con / ref:.2f}x unconstrained {ref:.4g} "
        f"(limit {MSE_RATIO_MAX[preset]}x)")


@pytest.mark.parametrize("task", list(TASKS))
@pytest.mark.parametrize("preset", CONSTRAINED)
def test_seed_spread(task, preset):
    f = [run(task, preset, s)["final_loss"] for s in SEEDS]
    assert max(f) <= SEED_SPREAD_MAX * min(f), f
