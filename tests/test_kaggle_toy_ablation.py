"""Fast checks for the toy ablation's statistical aggregation."""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hw_toy_ablation import PRESETS, TASKS, aggregate, mean_t_ci


def test_mean_t_ci_centered_and_finite():
    mean, low, high = mean_t_ci([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert low < mean < high
    assert math.isfinite(low) and math.isfinite(high)


def test_mean_t_ci_rejects_empty_and_nonfinite():
    try:
        mean_t_ci([])
    except ValueError:
        pass
    else:
        raise AssertionError("empty input should fail")
    try:
        mean_t_ci([1.0, float("nan")])
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite input should fail")


def test_aggregate_pairs_ratios_and_reports_pass_fractions():
    records = []
    for task in TASKS:
        for preset in PRESETS:
            for seed in (0, 1):
                records.append({
                    "task": task, "preset": preset, "seed": seed,
                    "held_mse": (1.0 + seed) * (2.0 if preset != "none" else 1.0),
                    "rail_max_abs": 1.0, "rail_limit": 2.0,
                    "max_dt_rho": 0.5, "stiffness_limit": 2.0,
                })
    summary = aggregate(records)
    core = next(row for row in summary if row["task"] == "friedman2" and row["preset"] == "phase3a_core")
    assert core["mean_ratio"] == 2.0
    assert core["ci95_low"] == core["ci95_high"] == 2.0
    assert core["rail_n"] == 0  # diagnostic-only: core does not enforce rails
    assert core["stiffness_pass"] == core["stiffness_n"] == 2
