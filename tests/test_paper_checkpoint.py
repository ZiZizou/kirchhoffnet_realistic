"""Paper checkpoint (trial_0011) tests: loader, golden case, and reproduction.

* loader: only the surgery-removed keys are dropped, rest loads strict, 5523 params.
* golden ``paper_trial_0011``: float64 forward / stage states / loss / grads of the
  REAL trained weights on 64 fixed canonical test rows (tests/golden/data).
* reproduction: scripts/eval_paper_checkpoint.py re-creates the logged
  ``[phaseA] final failure: val= 8.05% test= 8.15%`` (needs the canonical npz and
  the ZIG surrogate directory; skipped when absent).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(HERE, "golden")):
    if p not in sys.path:
        sys.path.insert(0, p)

import checkpoints as ck  # noqa: E402
import make_paper_trial_golden as mpg  # noqa: E402

RTOL, ATOL = 1e-6, 1e-7
GOLDEN = os.path.join(HERE, "golden", "data", "paper_trial_0011.pt")

_cache: dict = {}


def golden():
    if "g" not in _cache:
        _cache["g"] = torch.load(GOLDEN, weights_only=False)
    return _cache["g"]


def model64():
    if "m" not in _cache:
        _cache["m"] = ck.load_paper_checkpoint(dtype=torch.float64)
    return _cache["m"]


def result():
    if "r" not in _cache:
        _cache["r"] = mpg.run(model64(), golden()["inputs"])
    return _cache["r"]


def test_loader_drops_only_surgery_keys():
    m = model64()
    sd = torch.load(ck.DEFAULT_CKPT, map_location="cpu", weights_only=True)
    assert len(sd) == len(m.state_dict()) + len(m.dropped_keys)
    assert len(m.dropped_keys) == 28
    kinds = {"u_logits": 7, "_drive_idx": 7, "ref_dst": 7, "_drive_mask": 7}
    for suffix, n in kinds.items():
        got = [k for k in m.dropped_keys if k.endswith(suffix)]
        if suffix == "_drive_mask":
            assert len(got) == 7          # 6 transfers + post_readout_transfer
        else:
            assert len(got) == n, (suffix, got)
    # nothing but those four kinds was dropped
    assert all(k.endswith(tuple(kinds)) for k in m.dropped_keys)
    assert set(m.state_dict()) == set(sd) - set(m.dropped_keys)


def test_loader_param_count_is_5523():
    assert ck.trainable_param_count(model64()) == 5523 == golden()["n_params"]


def test_loader_rejects_other_mismatch(tmp_path):
    """A checkpoint with an extra unknown key must fail the loader's assert."""
    sd = torch.load(ck.DEFAULT_CKPT, map_location="cpu", weights_only=True)
    sd["net.core.stages.0.bogus"] = torch.zeros(1)
    bad = tmp_path / "bad.pt"
    torch.save(sd, bad)
    with pytest.raises(AssertionError):
        ck.load_paper_checkpoint(bad)


def test_golden_inputs_are_canonical_test_rows():
    g = golden()
    assert g["inputs"].shape == (64, 4) and g["inputs"].dtype == torch.float64
    import hashlib
    assert g["meta"]["ckpt_sha256"] == hashlib.sha256(open(ck.DEFAULT_CKPT, "rb").read()).hexdigest()


def test_golden_forward_and_states():
    g, r = golden(), result()
    assert r["y"].dtype == torch.float64
    torch.testing.assert_close(r["y"], g["y"], rtol=RTOL, atol=ATOL)
    assert len(r["stage_states"]) == len(g["stage_states"]) == 7
    for i, (a, b) in enumerate(zip(r["stage_states"], g["stage_states"])):
        torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL, msg=lambda m, i=i: f"stage {i}: {m}")


def test_golden_loss_and_grads():
    g, r = golden(), result()
    torch.testing.assert_close(r["loss"], g["loss"], rtol=RTOL, atol=ATOL)
    assert sorted(r["grads"]) == sorted(g["grads"])
    for k, v in g["grads"].items():
        torch.testing.assert_close(r["grads"][k], v, rtol=RTOL, atol=ATOL,
                                   msg=lambda m, k=k: f"grad {k}: {m}")
    assert r["no_grad_names"] == g["no_grad_names"] == ["log_hi", "log_lo"]


def test_float32_matches_float64_loosely():
    """The training/eval dtype (float32) agrees with the float64 golden outputs."""
    m32 = ck.load_paper_checkpoint(dtype=torch.float32)
    with torch.no_grad():
        y32 = m32(golden()["inputs"].float())
    torch.testing.assert_close(y32.double(), golden()["y"], rtol=0, atol=1e-3)


# ---- reproduction of the logged failure rates ------------------------------
import eval_paper_checkpoint as ev  # noqa: E402

HAVE_DATA = os.path.isfile(ev.DEFAULT_CANONICAL) and os.path.isdir(ev.DEFAULT_ZIG_DIR)


@pytest.mark.skipif(not HAVE_DATA, reason="canonical npz / ZIG surrogate dir not available")
def test_eval_script_reproduces_logged_failure_rates():
    """Log line: ``[phaseA] final failure: train= 7.67% val= 8.05% test= 8.15%``."""
    res = ev.evaluate(splits=("train", "val", "test"))
    assert res["_params"] == 5523
    assert res["val"]["n_failed"] == 161 and res["val"]["failure_rate"] == pytest.approx(0.0805, abs=1e-12)
    assert res["test"]["n_failed"] == 163 and res["test"]["failure_rate"] == pytest.approx(0.0815, abs=1e-12)
    assert round(res["train"]["failure_rate"] * 100, 2) == 7.67
    for s in ("train", "val", "test"):
        assert res[s]["invalid_rate"] == 0.0
