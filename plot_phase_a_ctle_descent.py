"""BO-descent plot for CTLE Phase-A: MLP distillation vs KNet distillation.

Data sources:
  MLP : outputs_alliance_can/mlp_bayes_opt/phase_a_mlp_distill_ctle/results.csv
        (trials 0-29, `validation_failure_rate` column, fractions)
  KNet: outputs_alliance_can/kn_bayes_opt/phase_a_mlp_distill_ctle/trial_XXXX.log.txt
        (trials 0-19, `best= X%` = trial's best validation failure rate so far)
        + run_train_knet_single_shot_distillation_2.sh-59014723.out appended as
        trial 20 (completed rerun of the trial_0011 config, best=8.05%).

Both curves are drawn as running minima over trials 1..30 (1-based x).
KNet has 21 points; its running min is flat-extended to x=30.
Curves are padded backward to x=0 with the first point's value.

Usage:
    python plot_phase_a_ctle_descent.py [--output-dir outputs_alliance_can]
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BEST_RE = re.compile(r"best=\s*([\d.]+)\s*%")

MLP_CSV = Path("outputs_alliance_can/mlp_bayes_opt/phase_a_mlp_distill_ctle/results.csv")
KNET_DIR = Path("outputs_alliance_can/kn_bayes_opt/phase_a_mlp_distill_ctle")
KNET_LAST_RUN = KNET_DIR / "run_train_knet_single_shot_distillation_2.sh-59014723.out"

N_TRIALS = 30  # both curves drawn out to trial 30 (1-based x: 1..30)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("outputs_alliance_can"))
    p.add_argument("--output-png", type=str, default="bo_descent_phase_a_ctle.png")
    p.add_argument("--output-csv", type=str, default="bo_descent_phase_a_ctle.csv")
    return p.parse_args()


def _trial_best_from_log(path: Path) -> tuple[float | None, bool]:
    """Return (trial's best val-fail %, completed?) from a Phase-A log.

    completed=True iff the log has a 'final failure' line.
    """
    best: float | None = None
    completed = False
    try:
        with path.open("r", errors="replace") as f:
            for line in f:
                m = BEST_RE.search(line)
                if m:
                    best = float(m.group(1))
                if "final failure" in line:
                    completed = True
    except OSError:
        return None, False
    return best, completed


def load_mlp(csv_path: Path) -> list[tuple[int, float]]:
    """[(trial_0based, val_fail_fraction), ...] sorted by trial."""
    out: list[tuple[int, float]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append((int(row["trial"]), float(row["validation_failure_rate"])))
    out.sort(key=lambda p: p[0])
    return out


def load_knet(knet_dir: Path, last_run: Path) -> tuple[list[tuple[int, float]], list[int]]:
    """[(trial_0based, best_val_fail_pct), ...] for trials 0-19 + last_run as trial 20.

    Returns (points, incomplete_trials).
    """
    points: list[tuple[int, float]] = []
    incomplete: list[int] = []
    for i in range(20):
        p = knet_dir / f"trial_{i:04d}.log.txt"
        best, completed = _trial_best_from_log(p)
        if best is None:
            print(f"  [WARN] no best= in {p.name}; skipping trial {i}")
            continue
        if not completed:
            incomplete.append(i)
        points.append((i, best))
    last_best, _ = _trial_best_from_log(last_run)
    if last_best is None:
        print(f"  [WARN] no best= found in {last_run.name}")
    else:
        points.append((20, last_best))
    points.sort(key=lambda p: p[0])
    return points, incomplete


def running_min_curve(points: list[tuple[int, float]],
                      n_trials: int = N_TRIALS) -> tuple[list[int], list[float]]:
    """Running minimum over 1-based x in 1..n_trials.

    - x = trial_0based + 1
    - prepend (0, first_y) so the curve starts at x=0
    - flat-extend the last running-min value out to n_trials
    """
    if not points:
        return [], []
    xs = [t + 1 for t, _ in points]
    ys = [v for _, v in points]
    run = list(np.minimum.accumulate(ys))
    out_x = [0, xs[0]] if xs[0] > 0 else [xs[0]]
    out_y = [run[0], run[0]] if xs[0] > 0 else [run[0]]
    for x, y in zip(xs[1:], run[1:]):
        if x > n_trials:
            break
        out_x.append(x)
        out_y.append(y)
    if out_x[-1] < n_trials:
        out_x.append(n_trials)
        out_y.append(out_y[-1])
    return out_x, out_y


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mlp = load_mlp(MLP_CSV)
    print(f"MLP trials from results.csv: {len(mlp)} "
          f"(range {mlp[0][0]}..{mlp[-1][0]})")
    knet, incomplete = load_knet(KNET_DIR, KNET_LAST_RUN)
    print(f"KNet trials from logs: {len(knet)} "
          f"(range {knet[0][0]}..{knet[-1][0]})")
    if incomplete:
        print(f"NOTE: KNet trials with truncated logs (best-so-far used): "
              f"{incomplete}")

    # Work in percent for both.
    mlp_pct = [(t, v * 100.0) for t, v in mlp]
    mlp_x, mlp_y = running_min_curve(mlp_pct)
    knet_x, knet_y = running_min_curve(knet)

    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=150)
    ax.plot(mlp_x, mlp_y, color="#1f77b4", marker="o", markevery=3, markersize=5,
            linewidth=1.4,
            label=f"MLP  (best {mlp_y[-1]:.3g}%, n={len(mlp)})")
    ax.plot(knet_x, knet_y, color="#d62728", marker="o", markevery=3, markersize=5,
            linewidth=1.4,
            label=f"KNet (best {knet_y[-1]:.3g}%, n={len(knet)})")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Best validation failure rate so far (%)")
    ax.set_xlim(0, N_TRIALS)
    ax.set_title("CTLE Phase-A distillation: MLP vs KNet")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="upper right", framealpha=0.92)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    png_path = args.output_dir / args.output_png
    data = buf.getvalue()
    fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="_png_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_path, png_path.resolve())
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    print(f"wrote {png_path}")

    csv_path = args.output_dir / args.output_csv
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "trial", "trial_value_pct", "best_so_far_pct"])
        for model, raw in (("mlp", mlp_pct), ("knet", knet)):
            xs = [t + 1 for t, _ in raw]
            ys = [v for _, v in raw]
            run = list(np.minimum.accumulate(ys))
            for x, v, b in zip(xs, ys, run):
                w.writerow([model, x, f"{v:.6f}", f"{b:.6f}"])
    print(f"wrote {csv_path}")

    print("\nKNet per-trial best val_fail (%):")
    for t, v in knet:
        print(f"  trial {t:2d}: {v:6.2f}%")
    print(f"\nMLP best-so-far: {mlp_y[-1]:.3f}%   KNet best-so-far: {knet_y[-1]:.3f}%")


if __name__ == "__main__":
    main()
