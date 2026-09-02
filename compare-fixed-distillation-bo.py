"""Declare the fixed-distillation comparison result from two BO summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knet-summary", type=Path, required=True)
    parser.add_argument("--mlp-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def extract(summary: dict) -> tuple[float, float]:
    final = summary["final_metrics"]["final"]
    return (
        float(final["zig_failure_comparison"]["test"]["student_failure_rate"]),
        float(final["test"]["logit_mse"]),
    )


def main() -> None:
    args = parse_args()
    knet = json.loads(args.knet_summary.read_text(encoding="utf-8"))
    mlp = json.loads(args.mlp_summary.read_text(encoding="utf-8"))
    if knet["student_kind"] != "knet" or mlp["student_kind"] != "mlp":
        raise ValueError("summaries must be KNet then PlainMLP")
    if knet["fixed_dataset"] != mlp["fixed_dataset"]:
        raise ValueError("the two BO studies did not use the same fixed teacher dataset")
    knet_failure, knet_mse = extract(knet)
    mlp_failure, mlp_mse = extract(mlp)
    result = {
        "knet_failure_rate": knet_failure, "mlp_failure_rate": mlp_failure,
        "knet_test_logit_mse": knet_mse, "mlp_test_logit_mse": mlp_mse,
        "knet_better_failure": knet_failure < mlp_failure,
        "knet_better_imitation": knet_mse < mlp_mse,
        "success": knet_failure < mlp_failure and knet_mse < mlp_mse,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("SUCCESS" if result["success"] else "NOT YET SUCCESS")
    print(json.dumps(result, indent=2))
    if not result["success"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
