#!/usr/bin/env python3
"""Run the slow CPU HW-learning harness as a multi-seed ablation.

Example: python scripts/hw_toy_ablation.py --seeds 10 --out-dir out/toy
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "golden"))
import test_hw_learning as harness  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402

TASKS = ("friedman2", "toy_mimo")
PRESETS = ("none", "phase3a_core", "phase3a_core_diff", "phase3a_full", "phase3b_tm")
# Fair best rates from docs/hw/cpu_learning_validation.md. The differential
# core is a new sibling of core; use the core's selected rate pending a sweep.
FAIR_LR = {
    "friedman2": {"none": 0.1, "phase3a_core": 0.1,
                  "phase3a_core_diff": 0.1, "phase3a_full": 0.1, "phase3b_tm": 0.03},
    "toy_mimo": {"none": 0.1, "phase3a_core": 0.1,
                 "phase3a_core_diff": 0.1, "phase3a_full": 0.1, "phase3b_tm": 0.1},
}


def mean_t_ci(values):
    """Return (mean, lower, upper) two-sided t-based 95% CI."""
    values = [float(v) for v in values]
    if not values:
        raise ValueError("at least one observation is required")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("observations must be finite")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, float("nan"), float("nan")
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    half = float(student_t.ppf(0.975, len(values) - 1)) * math.sqrt(variance / len(values))
    return mean, mean - half, mean + half


def aggregate(records):
    """Summarize paired held-MSE ratios and invariant pass fractions."""
    summaries = []
    for task in TASKS:
        by_key = {(r["task"], r["preset"], int(r["seed"])): r for r in records}
        seeds = sorted({int(r["seed"]) for r in records if r["task"] == task})
        refs = {s: by_key[(task, "none", s)]["held_mse"] for s in seeds}
        for preset in PRESETS:
            rows = [by_key[(task, preset, s)] for s in seeds]
            ratios = [r["held_mse"] / refs[int(r["seed"])] for r in rows]
            mean, lo, hi = mean_t_ci(ratios)
            rails_applicable = [r for r in rows if r["preset"] == "none" or
                                harness.preset_hw(preset, task) is None or
                                harness.preset_hw(preset, task).constraints.hw_compliance]
            rail_ok = sum(r["rail_max_abs"] <= r["rail_limit"] for r in rails_applicable)
            stiff_ok = sum(r["max_dt_rho"] <= r["stiffness_limit"] for r in rows)
            summaries.append({"task": task, "preset": preset, "lr": FAIR_LR[task][preset],
                              "n": len(rows), "mean_ratio": mean, "ci95_low": lo, "ci95_high": hi,
                              "rail_pass": rail_ok, "rail_n": len(rails_applicable),
                              "stiffness_pass": stiff_ok, "stiffness_n": len(rows)})
    return summaries


def _one(task, preset, seed, steps, threads):
    import torch
    torch.set_num_threads(threads)
    harness.STEPS = steps
    harness.LR = FAIR_LR[task][preset]
    result = harness._train(task, preset, seed)
    result["lr"] = FAIR_LR[task][preset]
    return result


def write_reports(records, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    details = out / "toy_ablation_runs.csv"
    fields = ["task", "preset", "seed", "lr", "steps", "held_mse", "held_r2",
              "rail_max_abs", "rail_limit", "max_dt_rho", "stiffness_limit", "seconds"]
    with details.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in fields} for r in records)
    summaries = aggregate(records)
    summary_csv = out / "toy_ablation_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    lines = ["# Toy / Friedman-2 HW preset ablation", "",
             "Ratios are paired by seed against the `none` held-out MSE. CI is a two-sided Student-t 95% interval over per-seed ratios. Rail fraction is evaluated where rail compliance applies (the unconstrained reference is diagnostic); stiffness uses the harness `max_dt_rho` limit.", "",
             "| Task | Preset | LR | Seeds | Mean MSE ratio vs none (95% CI) | Rail pass | Stiffness pass |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for s in summaries:
        ci = "n/a" if not math.isfinite(s["ci95_low"]) else f"{s['ci95_low']:.3f}–{s['ci95_high']:.3f}"
        lines.append(f"| {s['task']} | {s['preset']} | {s['lr']:g} | {s['n']} | {s['mean_ratio']:.3f} ({ci}) | {s['rail_pass']}/{s['rail_n']} | {s['stiffness_pass']}/{s['stiffness_n']} |")
    (out / "toy_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--jobs", type=int, default=max(1, min(32, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--threads-per-job", type=int, default=1)
    args = parser.parse_args()
    if args.seeds < 1 or args.steps < 1 or args.jobs < 1 or args.threads_per_job < 1:
        parser.error("seeds, steps, jobs, and threads-per-job must be positive")
    jobs = [(task, preset, seed) for task in TASKS for preset in PRESETS for seed in range(args.seeds)]
    records = Parallel(n_jobs=args.jobs, prefer="processes", batch_size=1)(
        delayed(_one)(task, preset, seed, args.steps, args.threads_per_job)
        for task, preset, seed in jobs)
    summaries = write_reports(records, args.out_dir)
    print(f"completed {len(records)} runs; reports in {args.out_dir}")
    for s in summaries:
        print(f"{s['task']}/{s['preset']}: ratio={s['mean_ratio']:.4f} "
              f"CI95=[{s['ci95_low']:.4f},{s['ci95_high']:.4f}] "
              f"rail={s['rail_pass']}/{s['rail_n']} stiffness={s['stiffness_pass']}/{s['stiffness_n']}")


if __name__ == "__main__":
    main()
