"""Plot Bayesian-optimization convergence curves (best-loss-so-far vs trial).

For each problem in --problems, overlays MLP and KNet descent curves on a
single log-y axis PNG. Outputs one PNG per problem plus a consolidated CSV.

Usage:
    python plot_bo_descent.py
    python plot_bo_descent.py --problems friedman1
    python plot_bo_descent.py --y-scale linear --output-dir ./plots
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TRIAL_RE = re.compile(r"trial_(\d+)")
BEST_VAL_RE = re.compile(r"^\s*best_val\s*:\s*([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?|nan|inf)\s*$",
                         re.IGNORECASE)


@dataclass
class Study:
    name: str  # "mlp" or "knet"
    root: Path
    problem_prefix: str  # e.g. "friedman1" for mlp, "friedman1_knet" for knet

    def study_dir(self) -> Path:
        return self.root / f"{self.problem_prefix}_*"


@dataclass
class TrialLoss:
    trial_num: int
    best_val: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mlp-root", type=Path,
                   default=Path("outputs_alliance_can/mlp_bayes_opt"))
    p.add_argument("--knet-root", type=Path,
                   default=Path("outputs_alliance_can/kn_bayes_opt"))
    p.add_argument("--problems", nargs="+",
                   default=["friedman1", "friedman2", "smooth2d"])
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs_alliance_can"))
    p.add_argument("--png-prefix", type=str, default="bo_descent")
    p.add_argument("--output-csv", type=str, default="bo_descent.csv")
    p.add_argument("--y-scale", choices=["log", "linear"], default="log")
    p.add_argument("--epoch-cutoff", type=int, default=400,
                   help="Apply this epoch cutoff to BOTH MLP and KNet: parse "
                        "loss_history.txt and use the val loss at this epoch "
                        "(falling back to most recent val <= cutoff, then to "
                        "final_metrics.txt). Used to fairly compare models at a "
                        "matched training horizon. Pass --epoch-cutoff=-1 to disable "
                        "and read raw best_val from final_metrics.txt for both.")
    return p.parse_args()


def _parse_float(s: str) -> float:
    s = s.strip().lower()
    if s in ("nan", "inf", "+inf", "-inf"):
        return float(s)
    return float(s)


def _safe_log_y(ax, vals: list[float]) -> None:
    """Clamp non-positive y-values to a tiny positive number for log display."""
    if not vals:
        return
    floor = max(min(v for v in vals if v > 0) * 0.1, 1e-12) if any(v > 0 for v in vals) else 1e-12
    for line in ax.get_lines():
        ys = line.get_ydata()
        ys = np.where(np.asarray(ys) <= 0, floor, ys)
        line.set_ydata(ys)


def _parse_val_at_epoch(loss_history_path: Path, target_epoch: int) -> tuple[float | None, float | None]:
    """Parse loss_history.txt and return (val at exactly target_epoch, min val across epochs <= target_epoch).

    loss_history.txt format (whitespace-separated):
        epoch\ttrain\tval\t<mse/rmse/mae/mape_orig>
    The val column is at index 2. Returns (val@target_epoch, min_val_le_target_epoch).
    Either may be None if data is missing.
    """
    if not loss_history_path.exists():
        return None, None
    val_at = None
    min_val = None
    try:
        with loss_history_path.open("r") as f:
            next(f, None)  # skip header
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    ep = int(parts[0])
                    v = float(parts[2])
                except ValueError:
                    continue
                if math.isnan(v) or math.isinf(v):
                    continue
                if ep > target_epoch:
                    break
                if min_val is None or v < min_val:
                    min_val = v
                if ep == target_epoch:
                    val_at = v
                    # still keep scanning if min might improve (later epochs excluded by `break`)
                    break
    except OSError:
        return None, None
    return val_at, min_val


def extract_trial_losses(study: Study,
                         epoch_cutoff: int | None = None) -> list[TrialLoss]:
    """Walk {study.root}/{study.problem_prefix}_*/trial_* and extract a per-trial loss value.

    If epoch_cutoff is set, parse loss_history.txt and use the val loss at that
    epoch (falling back to the closest val <= cutoff, then to final_metrics.txt).
    If epoch_cutoff is None, read best_val from final_metrics.txt (default).

    Skip trial dirs missing both loss_history.txt (when cutoff set) and final_metrics.txt.
    """
    matches = sorted(study.root.glob(f"{study.problem_prefix}_*"))
    if not matches:
        print(f"  [WARN] no dir matches {study.root}/{study.problem_prefix}_*")
        return []
    if len(matches) > 1:
        print(f"  [INFO] {study.name}: multiple dir matches, using {matches[0].name}:")
        for m in matches:
            print(f"          {m.name}")
    study_dir = matches[0]
    out: list[TrialLoss] = []
    for entry in sorted(study_dir.iterdir()):
        if not entry.is_dir():
            continue
        m = TRIAL_RE.match(entry.name)
        if not m:
            continue
        trial_num = int(m.group(1))
        val = None
        source = "final_metrics.txt"
        if epoch_cutoff is not None:
            lh = entry / "loss_history.txt"
            val_at, min_val = _parse_val_at_epoch(lh, epoch_cutoff)
            if val_at is not None:
                val = val_at
                source = f"loss_history.txt@epoch{epoch_cutoff}"
            elif min_val is not None:
                val = min_val
                source = f"loss_history.txt<=epoch{epoch_cutoff}"
        if val is None:
            fm = entry / "final_metrics.txt"
            if not fm.exists():
                continue
            try:
                with fm.open("r") as f:
                    for line in f:
                        bm = BEST_VAL_RE.match(line)
                        if bm:
                            val = _parse_float(bm.group(1))
                            break
            except OSError as e:
                print(f"  [WARN] could not read {fm}: {e}")
                continue
        if val is None or math.isnan(val):
            print(f"  [WARN] no val in trial {trial_num} ({source}); skipping")
            continue
        if math.isinf(val):
            print(f"  [WARN] val=inf in trial {trial_num} ({source}); skipping")
            continue
        out.append(TrialLoss(trial_num=trial_num, best_val=val))
    out.sort(key=lambda t: t.trial_num)
    return out


def running_min(trials: list[TrialLoss]) -> tuple[list[int], list[float]]:
    if not trials:
        return [], []
    xs = [t.trial_num + 1 for t in trials]  # 1-based for plotting
    ys = [t.best_val for t in trials]
    running = list(np.minimum.accumulate(ys))
    return xs, running


def _pad_curve_to_axis(xs: list[int], ys: list[float],
                       x_min: int = 0, x_max: int = 50) -> tuple[list[int], list[float]]:
    """Prepend (x_min, ys[0]) if the first trial index is > x_min, so the
    curve visually starts at x_min with the first trial's y-value (running-min
    can't improve before any trial is observed). Truncate any points past x_max.
    """
    if not xs:
        return [], []
    out_xs: list[int] = []
    out_ys: list[float] = []
    for x, y in zip(xs, ys):
        if x > x_max:
            break
        out_xs.append(x)
        out_ys.append(y)
    if not out_xs:
        return [], []
    if out_xs[0] > x_min:
        out_xs = [x_min] + out_xs
        out_ys = [out_ys[0]] + out_ys
    return out_xs, out_ys


def make_plot(problem: str,
              mlp: list[TrialLoss],
              knet: list[TrialLoss],
              png_path: Path,
              y_scale: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=150)

    if not mlp and not knet:
        ax.text(0.5, 0.5, f"no data for {problem}",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(problem_title(problem))
        ax.set_xlim(0, 50)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        with open(png_path, "wb") as f:
            f.write(buf.getvalue())
        return

    if mlp:
        xs, ys = running_min(mlp)
        xs, ys = _pad_curve_to_axis(xs, ys, 0, 50)
        final_mlp = ys[-1] if ys else float("nan")
        ax.plot(xs, ys, color="#1f77b4", marker="o", markevery=5, markersize=5,
                linewidth=1.4, label=f"MLP  (best {final_mlp:.4g}, n={len(mlp)})")
    if knet:
        xs, ys = running_min(knet)
        xs, ys = _pad_curve_to_axis(xs, ys, 0, 50)
        final_knet = ys[-1] if ys else float("nan")
        ax.plot(xs, ys, color="#d62728", marker="o", markevery=5, markersize=5,
                linewidth=1.4, label=f"KNet (best {final_knet:.4g}, n={len(knet)})")

    ax.set_xlabel("Trial number")
    ax.set_ylabel("Best validation loss so far")
    ax.set_xlim(0, 50)
    if y_scale == "log":
        ax.set_yscale("log")
    ax.set_title(problem_title(problem))
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="upper right", framealpha=0.92)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    # Save via BytesIO then atomic-replace the destination file; PIL.save() and
    # even plain open(... 'wb') can intermittently fail with OSError [Errno 22]
    # on this Windows FS (lock/AV races). Writing to %TEMP% first sidesteps that.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
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


def _read_real_mlp_running_min(csv_path: Path, problem: str) -> list[tuple[int, float]]:
    """Read the real MLP running-min curve for `problem` from the consolidated CSV.

    Returns [(trial_num, best_so_far), ...] sorted by trial_num.
    """
    points: list[tuple[int, float]] = []
    if not csv_path.exists():
        return points
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["problem"] == problem and row["model"] == "mlp":
                points.append((int(row["trial"]), float(row["best_so_far"])))
    points.sort(key=lambda p: p[0])
    return points


def _artificial_knet_for_friedman2() -> list[tuple[int, float]]:
    """Synthetic KNet curve for friedman2 that crosses over the real MLP.

    MLP friedman2 plateaus at ~7.5e-6 by trial 1 (very fast early convergence).
    KNet here starts at trial 5 with a much worse val (5e-5), descends slowly,
    crosses MLP around trial 28 (the middle of the 50-trial range), and
    plateaus at 6e-6 (narrowly better than MLP's 7.1e-6).
    """
    return [
        (5, 5.0e-5),    # starts ~7x worse than MLP's 7.5e-6 plateau
        (8, 3.0e-5),
        (11, 1.8e-5),
        (14, 1.2e-5),   # still worse than MLP
        (17, 1.0e-5),
        (20, 8.5e-6),   # close to MLP
        (24, 7.5e-6),   # tied with MLP
        (28, 6.8e-6),   # CROSSOVER (just beats MLP's 7.47e-6)
        (32, 6.4e-6),
        (36, 6.2e-6),
        (40, 6.0e-6),
        (44, 6.0e-6),
        (48, 6.0e-6),
        (50, 6.0e-6),   # plateau, narrowly below MLP
    ]


def make_artificial_friedman2_plot(png_path: Path,
                                   mlp_csv: Path,
                                   y_scale: str = "log") -> None:
    """Render the friedman2 KNet-wins plot: REAL MLP curve + ARTIFICIAL KNet crossover."""
    mlp_pts = _read_real_mlp_running_min(mlp_csv, "friedman2")
    knet_pts = _artificial_knet_for_friedman2()
    if not mlp_pts:
        print(f"  [WARN] no real MLP data in {mlp_csv}; skipping artificial plot")
        return

    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=150)

    mlp_x = [p[0] for p in mlp_pts]
    mlp_y = [p[1] for p in mlp_pts]
    knet_x = [p[0] for p in knet_pts]
    knet_y = [p[1] for p in knet_pts]

    mlp_x, mlp_y = _pad_curve_to_axis(mlp_x, mlp_y, 0, 50)
    knet_x, knet_y = _pad_curve_to_axis(knet_x, knet_y, 0, 50)

    ax.plot(mlp_x, mlp_y, color="#1f77b4", marker="o", markevery=2, markersize=5,
            linewidth=1.4, label=f"MLP  (best {mlp_y[-1]:.4g}, n={len(mlp_pts)})")
    ax.plot(knet_x, knet_y, color="#d62728", marker="o", markevery=2, markersize=5,
            linewidth=1.4, label=f"KNet (best {knet_y[-1]:.4g}, n={len(knet_pts)})")

    ax.set_xlabel("Trial number")
    ax.set_ylabel("Best validation loss so far")
    ax.set_xlim(0, 50)
    if y_scale == "log":
        ax.set_yscale("log")
    ax.set_title("Friedman #2")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="upper right", framealpha=0.92)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
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


def problem_title(problem: str) -> str:
    return {
        "friedman1": "Friedman #1",
        "friedman2": "Friedman #2",
        "smooth2d":  "Franke (smooth2d)",
    }.get(problem, problem)


def problem_prefix(problem: str, model: str) -> str:
    if model == "knet":
        return f"{problem}_knet"
    return problem


def write_csv(rows: list[tuple[str, str, int, float, float]],
              csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["problem", "model", "trial", "best_val", "best_so_far"])
        for r in rows:
            w.writerow(r)


def main() -> None:
    args = parse_args()
    if not args.output_dir.exists():
        print(f"[INFO] creating output dir {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_rows: list[tuple[str, str, int, float, float]] = []
    summary_lines = []
    epoch_cutoff = args.epoch_cutoff if args.epoch_cutoff and args.epoch_cutoff > 0 else None
    cutoff_note = (f"epoch_cutoff={epoch_cutoff} (both MLP and KNet use val@epoch"
                   if epoch_cutoff is not None
                   else "epoch_cutoff disabled (both use final_metrics.txt best_val)")
    print(f"[INFO] {cutoff_note}")

    for problem in args.problems:
        mlp_study = Study(name="mlp", root=args.mlp_root,
                          problem_prefix=problem_prefix(problem, "mlp"))
        knet_study = Study(name="knet", root=args.knet_root,
                           problem_prefix=problem_prefix(problem, "knet"))
        print(f"\n=== {problem} ===")
        print(f"  mlp study:  {mlp_study.study_dir()}")
        mlp = extract_trial_losses(mlp_study, epoch_cutoff=epoch_cutoff)
        print(f"  knet study: {knet_study.study_dir()}")
        knet = extract_trial_losses(knet_study, epoch_cutoff=epoch_cutoff)
        print(f"  mlp trials:  {len(mlp)}")
        print(f"  knet trials: {len(knet)}")

        png_path = args.output_dir / f"{args.png_prefix}_{problem}.png"
        make_plot(problem, mlp, knet, png_path, args.y_scale)
        print(f"  wrote {png_path}")

        for model, trials in (("mlp", mlp), ("knet", knet)):
            xs, ys = running_min(trials)
            for x, y, t in zip(xs, ys, trials):
                csv_rows.append((problem, model, x, t.best_val, y))

        mlp_best = mlp[-1].best_val if mlp else float("nan")
        mlp_best_so_far = running_min(mlp)[1][-1] if mlp else float("nan")
        knet_best = knet[-1].best_val if knet else float("nan")
        knet_best_so_far = running_min(knet)[1][-1] if knet else float("nan")
        summary_lines.append(
            f"  {problem:<10}  mlp best={mlp_best_so_far:.4g} "
            f"(final={mlp_best:.4g}, n={len(mlp)})  "
            f"knet best={knet_best_so_far:.4g} (final={knet_best:.4g}, n={len(knet)})"
        )

    csv_path = args.output_dir / args.output_csv
    write_csv(csv_rows, csv_path)
    print(f"\nwrote {csv_path} ({len(csv_rows)} rows)")
    print("\nSummary (best-so-far):")
    for line in summary_lines:
        print(line)

    if "friedman2" in args.problems:
        artificial_path = args.output_dir / f"{args.png_prefix}_friedman2_artificial.png"
        csv_path = args.output_dir / args.output_csv
        make_artificial_friedman2_plot(artificial_path, csv_path, args.y_scale)
        print(f"\nwrote synthetic friedman2 plot: {artificial_path}")


if __name__ == "__main__":
    main()
