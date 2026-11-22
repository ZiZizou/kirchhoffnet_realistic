"""Golden case ``paper_trial_0011``: the REAL trained paper checkpoint.

    venv/bin/python tests/golden/make_paper_trial_golden.py

Unlike the other golden cases (seeded random init), the weights come from
assets/baselines/paper_trial_0011/phase_a_best.pt via
``checkpoints.load_paper_checkpoint`` (float64).  Inputs are the first 64 rows
of the canonical Phase-A TEST split.  The file stores inputs, a seeded random
target, float64 outputs, per-stage states, MSE loss and per-parameter grads.
No state_dict is stored (the checkpoint asset is the source of truth).

Writes tests/golden/data/paper_trial_0011.pt (small, committed; never
regenerate it to make a failing test pass).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from checkpoints import DEFAULT_CKPT, load_paper_checkpoint, trainable_param_count  # noqa: E402

CASE = "paper_trial_0011"
OUT = os.path.join(HERE, "data", CASE + ".pt")
CANONICAL = os.path.join(os.path.expanduser("~"),
                         "Documents/ASPDAC_2026/kirchhoff_redesign/ideal/models/canonical_ctle_phase_a_v2.npz")
N_ROWS = 64
TARGET_SEED = 12


def test_rows(canonical_path: str = CANONICAL, n: int = N_ROWS) -> torch.Tensor:
    z = np.load(canonical_path, allow_pickle=False)
    idx = np.asarray(z["test_idx"], dtype=np.int64)[:n]
    return torch.from_numpy(np.asarray(z["specs"][idx], dtype=np.float64)).clone()


def run(model, x: torch.Tensor):
    """forward + MSE loss vs a seeded random target + one backward (float64)."""
    g = torch.Generator()
    g.manual_seed(TARGET_SEED)
    model.zero_grad(set_to_none=True)
    y = model(x)
    states = [s.detach().clone() for s in model.net.core.last_stage_outputs]
    target = torch.randn(y.shape, generator=g, dtype=torch.float64)
    loss = torch.mean((y - target) ** 2)
    loss.backward()
    grads, no_grad = {}, []
    for n_, p in model.named_parameters():
        (no_grad.append(n_) if p.grad is None else grads.__setitem__(n_, p.grad.detach().clone()))
    return {"y": y.detach().clone(), "stage_states": states, "loss": loss.detach().clone(),
            "grads": grads, "no_grad_names": sorted(no_grad), "target": target}


def build_record() -> dict:
    model = load_paper_checkpoint(DEFAULT_CKPT, dtype=torch.float64)
    x = test_rows()
    res = run(model, x)
    sha = hashlib.sha256(open(DEFAULT_CKPT, "rb").read()).hexdigest()
    try:
        head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        head = "unknown"
    return {"name": CASE, "inputs": x, **res, "n_params": trainable_param_count(model),
            "dropped_keys": model.dropped_keys,
            "meta": {"torch": torch.__version__, "git_head": head, "default_dtype": "float64",
                     "ckpt_sha256": sha, "n_rows": N_ROWS, "target_seed": TARGET_SEED,
                     "rows": "canonical_ctle_phase_a_v2 test_idx[:64]"}}


if __name__ == "__main__":
    rec = build_record()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(rec, OUT)
    print(f"{CASE}: params={rec['n_params']} loss={float(rec['loss']):.12g} "
          f"y={tuple(rec['y'].shape)} grads={len(rec['grads'])} no_grad={rec['no_grad_names']} "
          f"size={os.path.getsize(OUT)/1024:.0f} KiB")
