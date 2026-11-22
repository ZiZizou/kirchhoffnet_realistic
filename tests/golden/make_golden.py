"""Golden numerical baseline for the pre-surgery HR-KNet code (Phase 2, step 1).

Usage (from the repo root, with the refactor venv):

    venv/bin/python tests/golden/make_golden.py            # (re)write tests/golden/data/*.pt
    venv/bin/python tests/golden/make_golden.py --only paper

Every case is described by a JSON-serialisable ``spec`` (the build kwargs).
``make_model(spec, golden_sd)`` is the ONE code path that builds a model; the
generator and tests/test_golden.py both use it, so they cannot drift apart.

Design rules
------------
* float64 everywhere the model supports it (``f64()`` context sets the default
  dtype, then ``module.double()``).  Known float32 islands inside the model are
  listed in tests/golden/README.md (KCL scatter accumulators are hard-coded
  float32 in differential_stage.py).
* The golden state_dict (including the integer src/dst/boundary/readout
  buffers) is SAVED.  Tests never regenerate the init: they build a fresh model
  with a DIFFERENT torch seed and load the golden state_dict into it.
* Only the model-entry-points that surgery may touch are isolated in the
  small adapter functions ``_net_forward`` / ``_stage_states`` below.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import random
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
DATA_DIR = os.path.join(HERE, "data")
ASSET_DIR = os.path.join(REPO, "assets", "baselines", "fixed_mlp_knet")

BATCH = 64            # rows in every golden input batch
TRAJ_STEPS = 30       # Adam steps in the training trajectories
TRAJ_BS = 32
TRAJ_LR = 5e-3
PERTURB_STD = 0.5     # std of seeded noise added to raw params (non-trivial outputs)


# --------------------------------------------------------------------------
# dtype / seeding helpers
# --------------------------------------------------------------------------
@contextlib.contextmanager
def f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------------
# case specs (the "builder kwargs")
# --------------------------------------------------------------------------
FAN4 = {"0": [0, 4], "1": [1, 5], "2": [2, 6], "3": [3, 7]}

# Paper model: dagger-nuance-distillation-kirchhoffnet.py Phase-A, git 5c6a4f4,
# docs/logs/...-59014723.out + spice_failure_check.py:138-203 defaults.
PAPER_NET = {
    "kind": "net",
    "seed": 100,
    "in_dim": 4,
    "stage": {
        "num_stages": 7, "num_hidden": 11, "num_inputs": 4,
        "hidden_family": "small_world", "k": 2, "p": 0.2, "sw_seed": 1,
        "edge_repeats": 2,
        "t_span_total": 4.660583, "num_steps_per_stage": 10,   # round(70/7)
    },
    "cfg_extra": {"out_dim": 7, "write_mode": "sparse_proj",
                  "read_mode": "dense", "use_robust_input": False},
    "cell_lib": {"name": "tanh_free", "gm_max": 4.241970, "isat_max": 28.68855},
    "build": {
        "leak_mode": "non-programmable", "freeze_read": True,
        "interstage_activation": "residual-relu-tanh",
        "boundary_fan_out": FAN4, "enable_temporal_readout": True,
        "readout_mode": "shared_sense", "readout_senses_per_node": 2,  # shared-x2
        "learnable_clip_sharpness": True,
        "gln_rails": True, "gln_B": 4, "gln_rank": 2, "gln_alpha_init": 1.0,
        "gln_families": "boundary,readout",
        "x_max": 4.0,
        "vca_enabled": True, "vca_rank": 3, "vca_core_enabled": True,
        "vca_gate_shunt": False, "vca_separate_core_bus": True, "vca_bias": False,
    },
    "input": {"dist": "uniform", "lo": -1.5, "hi": 1.5, "seed": 11},
    "target_seed": 12,
    "expected_params": 5649,
}


def _variant(name: str, **build_overrides) -> dict:
    s = copy.deepcopy(PAPER_NET)
    s["build"].update(build_overrides)
    s.pop("expected_params", None)   # param count differs per branch
    s["variant_of"] = "paper"
    s["name"] = name
    return s


# Friedman-2 BO config (friedman2_knet_e1-style, audit §feature_surgery 1.2):
# 16-node small-world, temporal (ota_mesh) readout, VCA rank 2 core,
# C_eff=9.06, x_max=6.49, gm_max=21.2, isat_max=12.2, 4->1.
FRIEDMAN2_NET = {
    "kind": "net",
    "seed": 7,
    "in_dim": 4,
    "preset": {"name": "friedman2", "hidden_family": "small_world",
               "small_world_k": 4, "small_world_p": 0.3, "small_world_seed": 1},
    "stage": {"num_stages": 2, "edge_repeats": 2,
              "t_span_total": 7.0, "num_steps_total": 20},
    "cfg_extra": {},
    "cell_lib": {"name": "tanh_free", "gm_max": 21.2, "isat_max": 12.2},
    "build": {
        "leak_mode": "non-programmable", "freeze_read": True,
        "interstage_activation": "residual-relu-tanh",
        "boundary_fan_out": {"0": [0, 4], "1": [1, 5], "2": [2, 6], "3": [3, 7]},
        "enable_temporal_readout": True,
        "readout_mode": "ota_mesh", "readout_senses_per_node": 1,
        "vca_enabled": True, "vca_rank": 2, "vca_core_enabled": True,
        "vca_gate_shunt": False, "vca_separate_core_bus": True, "vca_bias": False,
        "x_max": 6.49, "c_eff": 9.06,
    },
    "input": {"dist": "friedman2", "seed": 21},
    "target_seed": 22,
    "trajectory": {"data": "friedman2", "seed": 23, "n": 256},
}

# Small 4-in/3-out synthetic toy: every readout/GLN/clip branch on, 2x8 nodes.
TOY_NET = {
    "kind": "net",
    "seed": 3,
    "in_dim": 4,
    "stage": {
        "num_stages": 2, "num_hidden": 8, "num_inputs": 4,
        "hidden_family": "small_world", "k": 2, "p": 0.2, "sw_seed": 1,
        "edge_repeats": 2, "t_span_total": 3.0, "num_steps_per_stage": 6,
    },
    "cfg_extra": {"out_dim": 3, "write_mode": "sparse_proj",
                  "read_mode": "dense", "use_robust_input": False},
    "cell_lib": {"name": "tanh_free", "gm_max": 5.0, "isat_max": 10.0},
    "build": {
        "leak_mode": "non-programmable", "freeze_read": True,
        "interstage_activation": "residual-relu-tanh",
        "boundary_fan_out": FAN4, "enable_temporal_readout": True,
        "readout_mode": "shared_sense", "readout_senses_per_node": 1,
        "learnable_clip_sharpness": True,
        "gln_rails": True, "gln_B": 4, "gln_rank": 2, "gln_alpha_init": 1.0,
        "gln_families": "boundary,readout",
        "x_max": 3.0,
        "vca_enabled": True, "vca_rank": 2, "vca_core_enabled": True,
        "vca_gate_shunt": False, "vca_separate_core_bus": True, "vca_bias": False,
    },
    "input": {"dist": "uniform", "lo": -2.0, "hi": 2.0, "seed": 31},
    "target_seed": 32,
    "trajectory": {"data": "toy_mimo", "seed": 33, "n": 256},
}

# fixed_mlp_knet: args stored in assets/baselines/fixed_mlp_knet/student_bundle.pt
FIXED_MLP = {
    "kind": "ctle",
    "seed": 0,
    "stage": {"num_stages": 4, "num_hidden": 14, "num_inputs": 4,
              "hidden_family": "small_world", "k": 4, "p": 0.2, "sw_seed": 1,
              "edge_repeats": 2,
              "t_span_total": 7.0, "num_steps_per_stage": 18},  # round(70/4)=18
    "cfg_extra": {"out_dim": 7, "write_mode": "sparse_proj",
                  "read_mode": "dense", "use_robust_input": False},
    "cell_lib": {"name": "tanh_free"},
    "build": {
        "leak_mode": "non-programmable", "freeze_read": True,
        "interstage_activation": "residual-relu-tanh",
        "boundary_fan_out": {"0": [2, 4], "1": [1, 3], "2": [12, 5], "3": [7, 9]},
        "enable_temporal_readout": True, "x_max": 4.0,
        "vca_enabled": True, "vca_rank": 2, "vca_core_enabled": True,
        "vca_gate_shunt": False, "vca_separate_core_bus": True, "vca_bias": False,
    },
    "input_rail": 4.0,
    "n_rows": BATCH,
    "target_seed": 42,
}


def build_cases() -> dict:
    cases = {
        "fixed_mlp_knet": FIXED_MLP,
        "paper": PAPER_NET,
        "paper_rawinit": dict(copy.deepcopy(PAPER_NET), perturb_std=0.0),
        "friedman2": FRIEDMAN2_NET,
        "toy_mimo": TOY_NET,
        # ---- branch variants on the paper config -------------------------
        "paper_freeze_read_off": _variant("paper_freeze_read_off", freeze_read=False),
        "paper_vca_off": _variant("paper_vca_off", vca_enabled=False),
        "paper_gln_off": _variant("paper_gln_off", gln_rails=False),
        "paper_gln_boundary_only": _variant("paper_gln_boundary_only",
                                            gln_families="boundary"),
        "paper_clip_off": _variant("paper_clip_off", learnable_clip_sharpness=False),
        "paper_readout_temporal": _variant("paper_readout_temporal",
                                           readout_mode="ota_mesh", gln_families="boundary",
                                           readout_senses_per_node=1),
        "paper_readout_shared": _variant("paper_readout_shared",
                                         readout_senses_per_node=1),
        "paper_vca_gate_shunt": _variant("paper_vca_gate_shunt", vca_gate_shunt=True),
        "paper_vca_bias": _variant("paper_vca_bias", vca_bias=True),
        "paper_vca_shared_core_bus": _variant("paper_vca_shared_core_bus",
                                              vca_separate_core_bus=False),
        "paper_vca_core_off": _variant("paper_vca_core_off", vca_core_enabled=False),
        "paper_interstage_none": _variant("paper_interstage_none",
                                          interstage_activation="none"),
    }
    return cases


# --------------------------------------------------------------------------
# model construction (single code path for generator and tests)
# --------------------------------------------------------------------------
def _int_keys(d):
    return {int(k): list(v) for k, v in d.items()}


def _build_cfg(spec: dict) -> dict:
    st = spec["stage"]
    n_stages = int(st["num_stages"])
    t_stage = float(st["t_span_total"]) / n_stages
    if "num_steps_per_stage" in st:
        steps = int(st["num_steps_per_stage"])
    else:
        steps = int(round(int(st["num_steps_total"]) / n_stages))
    if "preset" in spec:
        from config import make_friedman2_preset
        p = spec["preset"]
        assert p["name"] == "friedman2"
        cfg = make_friedman2_preset(
            hidden_family=p["hidden_family"], small_world_k=p["small_world_k"],
            small_world_p=p["small_world_p"], small_world_seed=p["small_world_seed"])
        stage = dict(cfg["stages"][0])
        stage.update({"edge_repeats": int(st["edge_repeats"]),
                      "t_span": t_stage, "num_steps": steps})
        cfg["stages"] = [dict(stage) for _ in range(n_stages)]
    else:
        stage = {
            "num_inputs": int(st["num_inputs"]), "num_hidden": int(st["num_hidden"]),
            "num_proj": 0, "num_outputs": 0,
            "hidden_family": st["hidden_family"],
            "hidden_kwargs": {"k": int(st["k"]), "p": float(st["p"]),
                              "seed": int(st["sw_seed"]), "bidirectional": False},
            "input_pattern": "all_to_all", "output_pattern": "all_to_all",
            "proj_pattern": "all_to_all", "edge_repeats": int(st["edge_repeats"]),
            "t_span": t_stage, "num_steps": steps,
        }
        cfg = {"stages": [dict(stage) for _ in range(n_stages)]}
    cfg.update(spec.get("cfg_extra", {}))
    return cfg


class CtleStudent(nn.Module):
    """Mirror of fixed-mlp-distillation-kirchhoffnet.KirchhoffStudent
    (same buffer names so the bundle state_dict loads strictly)."""

    def __init__(self, net, n_in: int, n_out: int, rail: float):
        super().__init__()
        self.input_rail = float(rail)
        self.register_buffer("input_log_min", torch.zeros(n_in))
        self.register_buffer("input_log_max", torch.ones(n_in))
        self.register_buffer("log_lo", torch.zeros(n_out))
        self.register_buffer("log_hi", torch.ones(n_out))
        self.net = net

    def scale_input(self, specs):
        logs = torch.log10(specs.clamp_min(1e-12))
        scaled = 2.0 * (logs - self.input_log_min.to(specs)) / (
            self.input_log_max.to(specs) - self.input_log_min.to(specs)
        ).clamp_min(1e-8) - 1.0
        return scaled.clamp(-self.input_rail, self.input_rail)

    def forward(self, specs):
        return _net_forward(self.net, self.scale_input(specs))


def _build_net(spec: dict, hw=None):
    from cell_library import make_cell_library
    from topology import build_net_from_config
    cfg = _build_cfg(spec)
    cl = {k: v for k, v in spec["cell_lib"].items() if k != "name"}
    cell_lib = make_cell_library(spec["cell_lib"]["name"], **cl, hw=hw)
    kw = dict(spec["build"])
    kw["boundary_fan_out"] = _int_keys(kw["boundary_fan_out"])
    return build_net_from_config(cfg, cell_lib=cell_lib, hw=hw, **kw)


def make_model(spec: dict, golden_sd: dict | None = None, hw=None):
    """Build (float64) model.  golden_sd=None: seeded random init (generator
    side).  golden_sd given: build with a DIFFERENT seed and load the golden
    state_dict (test side) - the init is never regenerated.

    ``hw`` is the hardware configuration (``hw.HWConfig``) or ``None``.  With
    every flag off it must not change a single number of the forward pass or
    of the state_dict (Phase 3b 2.4(b))."""
    seed = int(spec["seed"]) + (0 if golden_sd is None else 1000)
    with f64():
        seed_all(seed)
        net = _build_net(spec, hw=hw)
        if spec["kind"] == "ctle":
            model = CtleStudent(net, 4, spec["cfg_extra"]["out_dim"], spec["input_rail"])
        else:
            model = net
        model = model.double()
    if golden_sd is not None:
        load_golden(model, golden_sd)
    model.eval()
    return model


def load_golden(model, golden_sd, ignored=()):
    """Strict load of the golden state_dict, modulo ``ignored`` key patterns."""
    import fnmatch

    def is_ign(k):
        return any(fnmatch.fnmatch(k, pat) for pat in ignored)

    sd = {k: v for k, v in golden_sd.items() if not is_ign(k)}
    own = {k for k in model.state_dict() if not is_ign(k)}
    missing, unexpected = sorted(own - set(sd)), sorted(set(sd) - own)
    if missing or unexpected:
        raise KeyError(f"state_dict key mismatch: missing={missing[:10]} "
                       f"unexpected={unexpected[:10]}")
    model.load_state_dict(sd, strict=not ignored)


# ---- the only places that touch model call signatures ---------------------
def _net_forward(net, u):
    out = net(u, store_trajectory=False)
    return out[0] if isinstance(out, tuple) else out


def _core(model):
    net = model.net if isinstance(model, CtleStudent) else model
    return net.core


def _stage_states(model):
    return [s.detach().clone() for s in _core(model).last_stage_outputs]


def model_forward(model, x):
    if isinstance(model, CtleStudent):
        return model(x)
    return _net_forward(model, x)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
_F2_RANGES = [(0.0, 100.0), (40.0 * math.pi, 560.0 * math.pi), (0.0, 1.0), (1.0, 11.0)]


def _friedman2_xy(n: int, seed: int):
    g = _gen(seed)
    u = torch.rand(n, 4, generator=g, dtype=torch.float64)
    lo = torch.tensor([r[0] for r in _F2_RANGES], dtype=torch.float64)
    hi = torch.tensor([r[1] for r in _F2_RANGES], dtype=torch.float64)
    x = lo + u * (hi - lo)
    y = torch.sqrt(x[:, 0] ** 2 + (x[:, 1] * x[:, 2] - 1.0 / (x[:, 1] * x[:, 3])) ** 2)
    y = (y - y.mean()) / y.std()
    return u, y.unsqueeze(1)          # inputs min-max-style normalised to [0,1)


def _toy_xy(n: int, seed: int):
    g = _gen(seed)
    x = (torch.rand(n, 4, generator=g, dtype=torch.float64) * 2 - 1) * 2.0
    w1 = torch.randn(4, 6, generator=g, dtype=torch.float64)
    w2 = torch.randn(6, 3, generator=g, dtype=torch.float64)
    return x, torch.tanh(x @ w1) @ w2


def make_inputs(spec: dict):
    """Return (input batch, golden extra dict) for the forward/loss/grad check."""
    if spec["kind"] == "ctle":
        z = np.load(os.path.join(ASSET_DIR, "test_teacher_student_predictions.npz"))
        return torch.from_numpy(np.asarray(z["specs"][: spec["n_rows"]],
                                           dtype=np.float64)).clone()
    inp = spec["input"]
    if inp["dist"] == "friedman2":
        return _friedman2_xy(BATCH, inp["seed"])[0]
    g = _gen(inp["seed"])
    u = torch.rand(BATCH, spec["in_dim"], generator=g, dtype=torch.float64)
    return inp["lo"] + (inp["hi"] - inp["lo"]) * u


def make_target(spec: dict, y: torch.Tensor):
    return torch.randn(y.shape, generator=_gen(spec["target_seed"]), dtype=torch.float64)


def make_traj_data(spec: dict):
    t = spec["trajectory"]
    if t["data"] == "friedman2":
        return _friedman2_xy(t["n"], t["seed"])
    return _toy_xy(t["n"], t["seed"])


# --------------------------------------------------------------------------
# evaluation = what gets compared
# --------------------------------------------------------------------------
def evaluate(model, x, target):
    """forward + MSE loss + one backward.  Returns a dict of detached tensors."""
    model.zero_grad(set_to_none=True)
    y = model_forward(model, x)
    states = _stage_states(model)
    loss = torch.mean((y - target) ** 2)
    loss.backward()
    grads, no_grad = {}, []
    for n, p in model.named_parameters():
        if p.grad is None:
            no_grad.append(n)
        else:
            grads[n] = p.grad.detach().clone()
    return {"y": y.detach().clone(), "stage_states": states, "loss": loss.detach().clone(),
            "grads": grads, "no_grad_names": sorted(no_grad), "target": target}


def run_case(spec: dict, golden_sd: dict | None = None, ignored=(), hw=None):
    """Build model (golden_sd or seeded init), run forward/loss/backward."""
    model = make_model(spec, None if golden_sd is None else golden_sd, hw=hw) \
        if not ignored else _make_model_ignoring(spec, golden_sd, ignored, hw=hw)
    x = make_inputs(spec)
    with f64():
        with torch.no_grad():
            y0 = model_forward(model, x)
        target = make_target(spec, y0)
        res = evaluate(model, x, target)
    res["inputs"] = x
    res["model"] = model
    return res


def _make_model_ignoring(spec, golden_sd, ignored, hw=None):
    seed = int(spec["seed"]) + 1000
    with f64():
        seed_all(seed)
        net = _build_net(spec, hw=hw)
        model = (CtleStudent(net, 4, spec["cfg_extra"]["out_dim"], spec["input_rail"])
                 if spec["kind"] == "ctle" else net).double()
    load_golden(model, golden_sd, ignored)
    model.eval()
    return model


def run_trajectory(spec: dict, golden_sd: dict, ignored=(), hw=None):
    """30 seeded Adam steps in float64 from the golden init, fixed data order."""
    model = (make_model(spec, golden_sd, hw=hw) if not ignored
             else _make_model_ignoring(spec, golden_sd, ignored, hw=hw))
    X, Y = make_traj_data(spec)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=TRAJ_LR)
    losses = []
    with f64():
        for i in range(TRAJ_STEPS):
            j = (i * TRAJ_BS) % X.shape[0]
            xb, yb = X[j:j + TRAJ_BS], Y[j:j + TRAJ_BS]
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((model_forward(model, xb) - yb) ** 2)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
    return {"losses": torch.tensor(losses, dtype=torch.float64),
            "final_sd": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "X": X, "Y": Y}


# --------------------------------------------------------------------------
# golden generation
# --------------------------------------------------------------------------
def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _initial_sd(spec: dict) -> dict:
    """Seeded random init (generator side) or the trained bundle (fixed_mlp)."""
    if spec["kind"] != "ctle":
        m = make_model(spec, None)
        std = float(spec.get("perturb_std", PERTURB_STD))
        if std > 0:
            # Raw random init gives near-constant outputs (std ~1e-5), which
            # would make the golden check weak.  Add seeded Gaussian noise to
            # every float parameter so the dynamics are non-trivial.  The
            # perturbed state_dict is what gets SAVED; tests only load it.
            g = _gen(int(spec["seed"]) + 555)
            with torch.no_grad():
                for n, p in sorted(m.named_parameters(), key=lambda t: t[0]):
                    p.add_(std * torch.randn(p.shape, generator=g, dtype=p.dtype))
        return {k: v.detach().clone() for k, v in m.state_dict().items()}
    b = torch.load(os.path.join(ASSET_DIR, "student_bundle.pt"), map_location="cpu",
                   weights_only=False)
    with f64():
        seed_all(spec["seed"])
        net = _build_net(spec)
        model = CtleStudent(net, 4, spec["cfg_extra"]["out_dim"], spec["input_rail"])
    r = model.load_state_dict(b["model"], strict=True)       # strict at HEAD
    assert not r.missing_keys and not r.unexpected_keys, r
    model = model.double()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def make_one(name: str, spec: dict) -> dict:
    sd = _initial_sd(spec)
    res = run_case(spec, golden_sd=sd)           # SAME path the tests use
    n_params = sum(p.numel() for p in res["model"].parameters() if p.requires_grad)
    if "expected_params" in spec:
        assert n_params == spec["expected_params"], (name, n_params)
    rec = {
        "name": name, "spec": spec, "state_dict": sd,
        "sorted_keys": sorted(sd.keys()), "n_params": n_params,
        "inputs": res["inputs"], "target": res["target"], "y": res["y"],
        "stage_states": res["stage_states"], "loss": res["loss"],
        "grads": res["grads"], "no_grad_names": res["no_grad_names"],
        "meta": {"torch": torch.__version__, "git_head": _git_head(),
                 "default_dtype": "float64", "n_threads": torch.get_num_threads()},
    }
    if "trajectory" in spec:
        tr = run_trajectory(spec, sd)
        rec["traj"] = {"losses": tr["losses"], "final_sd": tr["final_sd"],
                       "X": tr["X"], "Y": tr["Y"],
                       "steps": TRAJ_STEPS, "bs": TRAJ_BS, "lr": TRAJ_LR}
    return rec


def golden_path(name: str) -> str:
    return os.path.join(DATA_DIR, f"{name}.pt")


# --------------------------------------------------------------------------
# post-surgery refresh (--refresh)
# --------------------------------------------------------------------------
# The golden files in data/ were generated BEFORE the surgery (tag
# ``pre-surgery``) and still carry the surgery-removed keys (``u_logits``,
# ``_drive_idx``, ``ref_dst``, ``_drive_mask``).  The model built at HEAD has
# no such keys any more, so the *init* of those goldens can no longer be
# reproduced by :func:`_initial_sd`: the seeded parameter noise is drawn per
# parameter in sorted-name order, so dropping one parameter shifts the whole
# noise stream and yields a DIFFERENT network.  ``--refresh`` therefore keeps
# the SAVED ``state_dict`` of an existing golden file and recomputes only the
# numeric outputs (forward, stage states, loss, gradients, trajectory) with
# it -- the very code path tests/test_golden.py uses -- so the golden keeps
# guarding the SAME model and the old-vs-new difference is exactly the
# intended change.
SURGERY_REMOVED_KEY_SUFFIXES = ("u_logits", "_drive_idx", "ref_dst", "_drive_mask")


def ignored_keys() -> tuple:
    # state_dict key suffixes the HEAD model no longer has (dropped by the surgery).
    return tuple(f"*{k}" for k in SURGERY_REMOVED_KEY_SUFFIXES)


def refresh_one(name: str) -> dict:
    # Recompute the numeric outputs of an existing golden, keeping its weights.
    path = golden_path(name)
    rec = torch.load(path, map_location="cpu", weights_only=False)
    spec = rec["spec"]
    cur = build_cases().get(name)
    if cur is not None and cur != spec:
        raise SystemExit(f"{name}: the stored spec differs from build_cases()")
    sd = rec["state_dict"]
    ign = ignored_keys()
    res = run_case(spec, golden_sd=sd, ignored=ign)
    rec["inputs"] = res["inputs"]
    rec["target"] = res["target"]
    rec["y"] = res["y"]
    rec["stage_states"] = res["stage_states"]
    rec["loss"] = res["loss"]
    rec["grads"] = res["grads"]
    # ``no_grad_names`` is NOT refreshed: it documents the saved (pre-surgery)
    # state_dict, whose surgery-removed parameters had no gradient.
    if "traj" in rec:
        tr = run_trajectory(spec, sd, ignored=ign)
        rec["traj"] = {"losses": tr["losses"], "final_sd": tr["final_sd"],
                       "X": tr["X"], "Y": tr["Y"],
                       "steps": TRAJ_STEPS, "bs": TRAJ_BS, "lr": TRAJ_LR}
    rec.setdefault("meta", {})["git_head"] = _git_head()
    rec["meta"]["n_threads"] = torch.get_num_threads()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--refresh", action="store_true",
                    help="recompute the numeric outputs of the selected goldens, "
                         "keeping their saved state_dict (post-surgery refresh)")
    a = ap.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    cases = build_cases()
    for name, spec in cases.items():
        if a.only and name not in a.only:
            continue
        rec = refresh_one(name) if a.refresh else make_one(name, spec)
        torch.save(rec, golden_path(name))
        print(f"{name:28s} params={rec['n_params']:6d} loss={float(rec['loss']):.12g} "
              f"size={os.path.getsize(golden_path(name))/1024:.0f} KiB"
              + (f" traj_last={float(rec['traj']['losses'][-1]):.6g}" if 'traj' in rec else ""))


if __name__ == "__main__":
    main()
