#!/usr/bin/env python
"""Re-evaluate the PAPER checkpoint (trial_0011) on the Phase-A canonical splits.

Reproduces the training-time ``[phaseA] final failure`` line
(train 7.67% / val 8.05% / test 8.15% at the 10% degrade rule) without running
the training script.  Uses the same ``StudentEvaluator.identify_failures`` as
``ctle_dagger_common.run_phase_a_training`` (ZIG forward surrogate, validity
gate 0.5, ``ctle_metrics_config`` rule), CPU float32.

    python scripts/eval_paper_checkpoint.py [--splits val test] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from checkpoints import DEFAULT_CKPT, load_paper_checkpoint, trainable_param_count  # noqa: E402
from ctle_dagger_common import (HybridHurdleModel, StudentEvaluator,  # noqa: E402
                                load_canonical_flow_dataset)
from ctle_metrics_config import DEGRADE_REL_THRESHOLD  # noqa: E402

HOME = os.path.expanduser("~")
DEFAULT_CANONICAL = os.path.join(
    HOME, "Documents/ASPDAC_2026/kirchhoff_redesign/ideal/models/canonical_ctle_phase_a_v2.npz")
DEFAULT_ZIG_DIR = os.path.join(HOME, "Documents/improved-zig-nf-spline-pytorch-default-v2")
LOGGED = {"train": 0.0766875, "val": 0.0805, "test": 0.0815}   # phase_a_history.json


def load_zig(zig_dir: str, device="cpu"):
    """ZIG forward surrogate + scalers, exactly as ctle_dagger_common.setup()."""
    scaler_X = joblib.load(os.path.join(zig_dir, "hybrid_hurdle_scaler_X.pkl"))
    scaler_y_p = joblib.load(os.path.join(zig_dir, "hybrid_hurdle_scaler_y_power.pkl"))
    cfg = joblib.load(os.path.join(zig_dir, "hybrid_hurdle_config.pkl"))
    zig = HybridHurdleModel(dropout=0.0, per_target=bool(cfg.get("per_target_hurdle", False))).to(device)
    zig.load_state_dict(torch.load(os.path.join(zig_dir, "hybrid_hurdle_ctle_model.pt"), map_location=device))
    zig.eval()
    for p in zig.parameters():
        p.requires_grad = False
    return zig, scaler_X, scaler_y_p, cfg


def evaluate(ckpt=DEFAULT_CKPT, canonical_path=DEFAULT_CANONICAL, zig_dir=DEFAULT_ZIG_DIR,
             splits=("train", "val", "test"), threshold=DEGRADE_REL_THRESHOLD, device="cpu"):
    """Return ``{split: {failure_rate, n, boundary_failure_rate, interior_failure_rate}}``."""
    canonical = load_canonical_flow_dataset(canonical_path)
    model = load_paper_checkpoint(ckpt, device=device)
    zig, scaler_X, scaler_y_p, cfg = load_zig(zig_dir, device)
    ev = StudentEvaluator(model, scaler_X, zig, cfg["eye_scale_h"], cfg["eye_scale_w"],
                          cfg["eye_scale_j"], scaler_y_p, torch.device(device))
    out = {}
    for split in splits:
        idx = np.asarray(canonical[f"{split}_idx"], dtype=np.int64)
        specs = canonical["specs"][idx]
        mask, m = ev.identify_failures(specs, threshold=threshold)
        bm = m["is_boundary"]
        out[split] = {
            "failure_rate": float(mask.mean()),
            "n": int(len(idx)),
            "n_failed": int(mask.sum()),
            "boundary_failure_rate": float(m["boundary_failure"].sum() / max(1, bm.sum())),
            "interior_failure_rate": float(m["interior_failure"].sum() / max(1, (~bm).sum())),
            "invalid_rate": float(m["invalid_mask"].mean()),
        }
    out["_params"] = trainable_param_count(model)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--canonical-dataset", default=DEFAULT_CANONICAL)
    ap.add_argument("--zig-dir", default=DEFAULT_ZIG_DIR)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--threshold", type=float, default=DEGRADE_REL_THRESHOLD)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.time()
    res = evaluate(a.ckpt, a.canonical_dataset, a.zig_dir, a.splits, a.threshold)
    dt = time.time() - t0
    print(f"trainable params at HEAD: {res['_params']}")
    for s in a.splits:
        r = res[s]
        ref = LOGGED.get(s)
        print(f"{s:5s}: failure {r['failure_rate']*100:6.3f}% ({r['n_failed']}/{r['n']})  "
              f"boundary {r['boundary_failure_rate']*100:.2f}%  interior {r['interior_failure_rate']*100:.2f}%  "
              f"invalid {r['invalid_rate']*100:.2f}%  logged {ref*100:.3f}%" if ref is not None else s)
    print(f"[phaseA-eval] final failure: " + " ".join(
        f"{s}={res[s]['failure_rate']*100:5.2f}%" for s in a.splits) + f"  ({dt:.1f}s)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
