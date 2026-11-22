"""Dry-run KNet input preprocessing and clipping diagnostics.

This intentionally does not import the DAgger training script, load teacher
artifacts, construct a network, or start training.  It reproduces the KNet
log10/min-max preprocessing and empirical 5% padded bounds.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


CSV_FILES = [
    "dataset_log_part_1_may3.csv", "dataset_log_part_1_may4.csv",
    "dataset_log_part_2_may3.csv", "dataset_log_part_2_may4.csv",
    "dataset_log_part_3_may3.csv", "dataset_log_part_3_may4.csv",
    "dataset_log_part_4_may3.csv", "dataset_log_part_4_may4.csv",
    "dataset_log_part_5_may3.csv", "dataset_log_part_5_may4.csv",
    "dataset_log_part_6_may3.csv", "dataset_log_part_6_may4.csv",
    "dataset_log_may_7.csv", "dataset_log_march12.csv",
    "dataset_log_may_18.csv",
]
SPEC_COLUMNS = [
    "power", "stage_2_jitter", "stage_2_eye_max_height", "stage_2_eye_max_width",
]
COL_MAPPING = {
    "eye_maxHeight_norm Vout_2 56G": "stage_2_eye_max_height",
    "eye_maxWidth_norm Vout_2 56G": "stage_2_eye_max_width",
    "eye_p2pJitterAverage_norm stage 2": "stage_2_jitter",
}


def load_specs(data_dir: Path) -> np.ndarray:
    frames = []
    for name in CSV_FILES:
        path = data_dir / name
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            logging.warning("missing: %s", path)
    if not frames:
        raise FileNotFoundError(f"No expected CTLE CSV files found under {data_dir}")
    combined = pd.concat(frames, ignore_index=True).rename(columns=COL_MAPPING)
    missing = [c for c in SPEC_COLUMNS if c not in combined.columns]
    if missing:
        raise ValueError(f"Missing required spec columns: {missing}")
    specs = combined[SPEC_COLUMNS].to_numpy(dtype=np.float64)
    valid = np.isfinite(specs).all(axis=1) & (specs > 0).all(axis=1)
    return specs[valid]


def report(name: str, specs: np.ndarray, lo: np.ndarray, hi: np.ndarray, rail: float) -> None:
    logs = np.log10(np.clip(specs, 1e-12, None))
    normalized = 2.0 * (logs - lo) / np.maximum(hi - lo, 1e-8) - 1.0
    clipped = np.abs(normalized) >= rail
    logging.info("%s: samples=%d", name, len(specs))
    logging.info("  total clipping: %d/%d elements (%.6f%%)",
                 int(clipped.sum()), int(clipped.size), 100.0 * clipped.mean())
    for i, feature in enumerate(SPEC_COLUMNS):
        logging.info("  %-24s %d/%d (%.6f%%), normalized range [%.4f, %.4f]",
                     feature, int(clipped[:, i].sum()), len(specs),
                     100.0 * clipped[:, i].mean(), normalized[:, i].min(), normalized[:, i].max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/home/annaik/Documents/augmented-cvae-ctle"))
    parser.add_argument("--rail", type=float, default=4.0)
    parser.add_argument("--probe-noisy", type=int, default=0,
                        help="Also apply the DAgger input perturbations to this many sampled rows.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    clean_specs = load_specs(args.data_dir)
    logs = np.log10(clean_specs)
    span = np.maximum(logs.max(axis=0) - logs.min(axis=0), 1e-8)
    lo = logs.min(axis=0) - 0.05 * span
    hi = logs.max(axis=0) + 0.05 * span
    logging.info("KNet preprocessing: log10 -> min-max -> clip [-%.1f, %.1f]", args.rail, args.rail)
    logging.info("bounds: lo=%s hi=%s", np.round(lo, 5).tolist(), np.round(hi, 5).tolist())
    report("historical inputs", clean_specs, lo, hi, args.rail)

    if args.probe_noisy > 0:
        rng = np.random.default_rng(args.seed)
        base = clean_specs[rng.integers(0, len(clean_specs), size=args.probe_noisy)].copy()
        noisy = base.copy()
        noisy[:, 0] *= rng.uniform(0.95, 1.05, len(noisy))
        noisy[:, 1] = np.clip(noisy[:, 1] * rng.uniform(0.95, 1.05, len(noisy)), 1.57, 100.0)
        noisy[:, 2] = np.where(noisy[:, 2] > 1, noisy[:, 2] * rng.uniform(0.9, 1.1, len(noisy)), rng.uniform(1e-12, 2, len(noisy)))
        noisy[:, 3] = np.where(noisy[:, 3] > 1, noisy[:, 3] * rng.uniform(0.9, 1.1, len(noisy)), rng.uniform(1e-12, 2, len(noisy)))
        report("DAgger-style noisy probe", noisy, lo, hi, args.rail)


if __name__ == "__main__":
    main()
