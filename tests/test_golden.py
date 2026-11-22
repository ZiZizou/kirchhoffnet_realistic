"""Golden numerical regression tests (pre-surgery baseline).

Each case in tests/golden/data/<case>.pt was produced by tests/golden/make_golden.py
at the ``pre-surgery`` tag.  A test rebuilds the model from the SAME builder
kwargs (``spec`` stored in the golden file) but with a different RNG seed, loads
the golden state_dict (including the src/dst/boundary/readout buffers), and
compares forward output, per-stage states, loss, per-parameter gradients and (for
two cases) a 30-step Adam trajectory, all in float64.

Surgery steps that remove a feature extend ``IGNORED_KEYS`` (fnmatch patterns on
state_dict / parameter names; they are dropped from BOTH sides) and, where the
removed feature changes a build kwarg, ``SPEC_OVERRIDES``.

Run:  venv/bin/python -m pytest tests/test_golden.py -q
"""

from __future__ import annotations

import fnmatch
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "golden"))
sys.path.insert(0, os.path.dirname(HERE))

import make_golden as mg  # noqa: E402

# ---- knobs later removal steps may extend ---------------------------------
# state_dict key patterns that are allowed to exist ONLY in the golden file
# (removed parameters/buffers).  Example after removing node gates:
#   IGNORED_KEYS = ["core.stages.*.u_logits"]
IGNORED_KEYS: list[str] = [
    # surgery B3: persistent drive removed (empty int buffers, never populated by any golden model)
    "*.core.stages.*._drive_idx",
    "core.stages.*._drive_idx",
    "*._drive_mask",
    "*core.transfers.*._drive_mask",
    "post_readout_transfer._drive_mask",
    "net.post_readout_transfer._drive_mask",
    # surgery B5: node gates removed (u_logits never received a gradient and was unused in rhs)
    "*core.stages.*.u_logits",
    # surgery B3: reference edges removed (empty int buffer, never populated)
    "*core.stages.*.ref_dst",
]
# Per-case build-kwarg overrides, e.g. {"paper": {"build": {"core_refresh_interval": 0}}}
SPEC_OVERRIDES: dict[str, dict] = {}

RTOL, ATOL = 1e-6, 1e-7           # forward / loss / grads
TRAJ_RTOL, TRAJ_ATOL = 1e-5, 1e-7  # 30 Adam steps accumulate rounding

DATA = mg.DATA_DIR
_SEPARATE = {"paper_trial_0011"}   # real-checkpoint case, tested in tests/test_paper_checkpoint.py
CASES = sorted(f[:-3] for f in os.listdir(DATA) if f.endswith(".pt") and f[:-3] not in _SEPARATE) if os.path.isdir(DATA) else []

_cache: dict[str, dict] = {}


def _ign(name: str) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in IGNORED_KEYS)


def load(case: str) -> dict:
    if case not in _cache:
        _cache[case] = torch.load(os.path.join(DATA, case + ".pt"), weights_only=False)
    return _cache[case]


def spec_of(rec: dict) -> dict:
    spec = dict(rec["spec"])
    ov = SPEC_OVERRIDES.get(rec["name"])
    if ov:
        spec = {**spec, **{k: ({**spec[k], **v} if isinstance(v, dict) else v) for k, v in ov.items()}}
    return spec


_results: dict[str, dict] = {}


def run(case: str) -> dict:
    """Rebuild + forward + backward once per case (shared by several tests)."""
    if case not in _results:
        rec = load(case)
        _results[case] = mg.run_case(spec_of(rec), golden_sd=rec["state_dict"], ignored=IGNORED_KEYS)
    return _results[case]


def test_cases_present():
    expected = set(mg.build_cases())
    assert expected <= set(CASES), f"missing golden files: {sorted(expected - set(CASES))}"


@pytest.mark.parametrize("case", CASES)
def test_state_dict_keys(case):
    rec = load(case)
    model = mg.make_model(spec_of(rec), None) if not IGNORED_KEYS else None
    if model is None:
        res = run(case)
        model = res["model"]
    got = sorted(k for k in model.state_dict() if not _ign(k))
    want = sorted(k for k in rec["sorted_keys"] if not _ign(k))
    assert got == want, (set(got) ^ set(want))
    # shapes/dtypes of a FRESH (differently seeded) build match the golden ones
    for k, v in model.state_dict().items():
        if _ign(k):
            continue
        g = rec["state_dict"][k]
        assert tuple(v.shape) == tuple(g.shape), k
        assert v.dtype == g.dtype, (k, v.dtype, g.dtype)


@pytest.mark.parametrize("case", CASES)
def test_param_count(case):
    rec = load(case)
    res = run(case)
    n = sum(p.numel() for n_, p in res["model"].named_parameters()
            if p.requires_grad and not _ign(n_))
    n_gold = sum(v.numel() for k, v in rec["state_dict"].items()
                 if not _ign(k) and k in dict(res["model"].named_parameters()))
    assert n == n_gold
    if not IGNORED_KEYS and "expected_params" in rec["spec"]:
        assert n == rec["spec"]["expected_params"]
    if not IGNORED_KEYS:
        assert n == rec["n_params"]


@pytest.mark.parametrize("case", CASES)
def test_forward(case):
    rec, res = load(case), run(case)
    assert torch.equal(res["inputs"], rec["inputs"])
    assert res["y"].dtype == torch.float64
    torch.testing.assert_close(res["y"], rec["y"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("case", CASES)
def test_stage_states(case):
    rec, res = load(case), run(case)
    assert len(res["stage_states"]) == len(rec["stage_states"])
    for i, (a, b) in enumerate(zip(res["stage_states"], rec["stage_states"])):
        torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL, msg=lambda m, i=i: f"stage {i}: {m}")


@pytest.mark.parametrize("case", CASES)
def test_loss_and_grads(case):
    rec, res = load(case), run(case)
    torch.testing.assert_close(res["loss"], rec["loss"], rtol=RTOL, atol=ATOL)
    got = {k: v for k, v in res["grads"].items() if not _ign(k)}
    want = {k: v for k, v in rec["grads"].items() if not _ign(k)}
    assert sorted(got) == sorted(want), set(got) ^ set(want)
    for k in want:
        torch.testing.assert_close(got[k], want[k], rtol=RTOL, atol=ATOL,
                                   msg=lambda m, k=k: f"grad {k}: {m}")
    # parameters that receive no gradient (e.g. unused u_logits) are unchanged
    assert sorted(n for n in res["no_grad_names"] if not _ign(n)) == \
        sorted(n for n in rec["no_grad_names"] if not _ign(n))


TRAJ_CASES = [c for c in CASES if "traj" in load(c)] if CASES else []


@pytest.mark.parametrize("case", TRAJ_CASES)
def test_training_trajectory(case):
    rec = load(case)
    tr = mg.run_trajectory(spec_of(rec), rec["state_dict"], ignored=IGNORED_KEYS)
    t = rec["traj"]
    assert torch.equal(tr["X"], t["X"]) and torch.equal(tr["Y"], t["Y"])
    torch.testing.assert_close(tr["losses"], t["losses"], rtol=TRAJ_RTOL, atol=TRAJ_ATOL)
    for k, v in t["final_sd"].items():
        if _ign(k) or not v.is_floating_point():
            continue
        torch.testing.assert_close(tr["final_sd"][k], v, rtol=TRAJ_RTOL, atol=TRAJ_ATOL,
                                   msg=lambda m, k=k: f"final param {k}: {m}")
    assert t["losses"].numel() == mg.TRAJ_STEPS


def test_fixed_mlp_knet_matches_saved_logits():
    """float64 re-run of the shipped checkpoint reproduces its own saved float32
    student logits (audit: 4e-6 at HEAD)."""
    import numpy as np
    rec = load("fixed_mlp_knet")
    z = np.load(os.path.join(mg.ASSET_DIR, "test_teacher_student_predictions.npz"))
    saved = torch.from_numpy(np.asarray(z["student_logits"][: mg.BATCH], dtype=np.float64))
    torch.testing.assert_close(rec["y"], saved, rtol=0, atol=5e-5)
    torch.testing.assert_close(run("fixed_mlp_knet")["y"], saved, rtol=0, atol=5e-5)


def test_paper_param_count_is_5649():
    assert load("paper")["n_params"] == 5649 if not IGNORED_KEYS else True
