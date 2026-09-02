"""CTLE Inverse Design: Plain single-head MLP control (DAgger variant).

Single-file drop-in for ``generative-distillation-improved-dagger-nuance-mlp.py``:
all shared infrastructure (teacher / ZIG / spline flow / RegimeAwareLoss /
DistillationDataset / StudentEvaluator / DAgger schedule) is imported from
:mod:`ctle_dagger_common` so MoE and Plain runs are guaranteed to share the
exact same infra.

This control removes the MoE's gate, regime classifier, and auxiliary regime
cross-entropy loss while preserving the trunk shape (SiLU, no LN, 8-dim Q75 +
raw-log input). The informative first comparison uses ``width=45, layers=3``
(the best current MoE shape); use ``--param-budget`` to auto-derive a
parameter-matched width.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

from ctle_dagger_common import (  # noqa: E402
    PlainMLP, plain_param_count, derive_plain_width,
    PARAM_LOG_BOUNDS,
    setup, run_dagger_training, _logger,
    DEFAULT_TEACHER_DIR, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR,
    Timer,
)


# Sensible defaults match the best MoE shape (W=45, L=3, SiLU, no LN, log features).
DEFAULT_PLAIN_TRUNK_WIDTH = 45
DEFAULT_PLAIN_TRUNK_LAYERS = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="CTLE inverse design: plain single-head MLP DAgger control",
    )
    # Shared DAgger / training flags
    parser.add_argument("--dagger-iterations", type=int, default=None)
    parser.add_argument("--epochs-per-iter", type=int, default=None)
    parser.add_argument("--common-eval-size", type=int, default=None)
    parser.add_argument("--earlystop-eval-every", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output", type=str, default=os.path.join(DEFAULT_OUTPUT_DIR + "_plain"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--teacher-dir", type=str, default=DEFAULT_TEACHER_DIR)
    parser.add_argument("--seed", type=int, default=0)
    # Plain trunk overrides
    parser.add_argument("--plain-trunk-width", type=int, default=DEFAULT_PLAIN_TRUNK_WIDTH,
                        help=f"Trunk hidden width (default {DEFAULT_PLAIN_TRUNK_WIDTH}; "
                             "ignored if --param-budget is set).")
    parser.add_argument("--plain-trunk-layers", type=int, default=DEFAULT_PLAIN_TRUNK_LAYERS)
    parser.add_argument("--plain-activation", choices=["silu", "gelu"], default="silu")
    parser.add_argument("--plain-use-layernorm", action="store_true",
                        help="Insert LayerNorm between each trunk Linear and activation.")
    parser.add_argument("--plain-use-log-features", dest="plain_use_log_features",
                        action="store_true", default=True,
                        help="Append 4 raw log features to the 4 Q75-scaled features (default True).")
    parser.add_argument("--no-plain-log-features", dest="plain_use_log_features",
                        action="store_false",
                        help="Disable the 4-dim raw-log feature concatenation.")
    parser.add_argument("--input-preprocessing", choices=["knet", "q75"], default="knet",
                        help="Input representation: knet=4 log/min-max features clipped to [-4,4]; "
                             "q75=legacy Q75 features plus optional raw-log features.")
    parser.add_argument("--param-budget", type=int, default=None,
                        help="If set, auto-derive plain trunk width via derive_plain_width "
                             "so plain_param_count(W,L) <= param-budget. Overrides --plain-trunk-width.")
    parser.add_argument("--count-params-only", action="store_true",
                        help="Build the student (honoring --param-budget derivation), print "
                             "TRAINABLE_PARAMS=<n> to stdout and exit before loading teacher "
                             "artifacts or training (Bayes-opt preflight hook).")
    return parser.parse_args()


def build_student(args) -> PlainMLP:
    """Construct the plain MLP, applying --param-budget derivation when set."""
    activation = {"silu": torch.nn.SiLU, "gelu": torch.nn.GELU}[args.plain_activation]
    trunk_layers = args.plain_trunk_layers
    if args.param_budget is not None:
        derived_width = derive_plain_width(args.param_budget, trunk_layers)
        if derived_width < 1:
            raise ValueError(
                f"--param-budget {args.param_budget} too small for "
                f"--plain-trunk-layers {trunk_layers}"
            )
        actual = plain_param_count(derived_width, trunk_layers)
        _logger.info(f"[PLAIN] --param-budget {args.param_budget} -> derived width {derived_width} "
                     f"(plain_param_count={actual}, L={trunk_layers})")
        trunk_width = derived_width
    else:
        trunk_width = args.plain_trunk_width
    student = PlainMLP(
        trunk_width=trunk_width,
        trunk_layers=trunk_layers,
        activation=activation,
        use_layernorm=args.plain_use_layernorm,
        use_log_features=(args.plain_use_log_features and args.input_preprocessing == "q75"),
    )
    return student


def _git_commit_short() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8", errors="replace").strip() or None
    except Exception:
        return None


def build_plain_teacher_config(student: PlainMLP, ctx: dict, args: argparse.Namespace) -> dict:
    """Build v2 sidecar config that is sufficient to reconstruct without guessing."""
    # Scaler constants live in ctx after setup()
    scaler_y_p = ctx.get("scaler_y_p")
    try:
        scaler_p_scale = float(scaler_y_p.scale_[0]) if scaler_y_p is not None else None
        scaler_p_mean = float(scaler_y_p.mean_[0]) if scaler_y_p is not None else None
    except Exception:
        scaler_p_scale = scaler_p_mean = None
    input_log_min = ctx.get("input_log_min")
    input_log_max = ctx.get("input_log_max")
    # Normalise to JSON-serialisable lists (or None)
    def _to_list_or_null(arr):
        if arr is None:
            return None
        try:
            return np.asarray(arr, dtype=np.float32).tolist()
        except Exception:
            return None
    try:
        ppc = plain_param_count(int(student.trunk_width), int(student.trunk_layers))
    except Exception:
        ppc = None
    cfg: dict = {
        "schema_version": 2,
        "trunk_width": int(student.trunk_width),
        "trunk_layers": int(student.trunk_layers),
        "activation": str(args.plain_activation),
        "use_layernorm": bool(student.use_layernorm),
        "use_log_features": bool(student.use_log_features),
        "input_preprocessing": str(args.input_preprocessing),
        "param_budget": int(args.param_budget) if args.param_budget is not None else None,
        "plain_param_count": int(ppc) if ppc is not None else None,
        "scaler": {
            "scaler_p_scale": scaler_p_scale,
            "scaler_p_mean": scaler_p_mean,
            "eye_scale_j": float(ctx.get("eye_scale_j")) if ctx.get("eye_scale_j") is not None else None,
            "eye_scale_h": float(ctx.get("eye_scale_h")) if ctx.get("eye_scale_h") is not None else None,
            "eye_scale_w": float(ctx.get("eye_scale_w")) if ctx.get("eye_scale_w") is not None else None,
        },
        "input_log_min": _to_list_or_null(input_log_min),
        "input_log_max": _to_list_or_null(input_log_max),
        "param_log_bounds": {k: [float(v[0]), float(v[1])] for k, v in PARAM_LOG_BOUNDS.items()},
        "checkpoint": "dagger_student_plain.pt",
        "meta": {
            "seed": int(args.seed) if args.seed is not None else None,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": _git_commit_short(),
        },
    }
    return cfg


def main():
    args = parse_args()
    student = build_student(args)
    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in student.parameters())
    _logger.info(
        f"Student model: {n_trainable:,} trainable / {n_total:,} total params  "
        f"(plain W={student.trunk_width} L={student.trunk_layers} "
        f"act={args.plain_activation} LN={student.use_layernorm} "
        f"log_features={student.use_log_features})"
    )
    if args.count_params_only:
        # Machine-readable preflight output (mirrors fixed-mlp-distillation-kirchhoffnet.py).
        print(f"TRAINABLE_PARAMS={n_trainable}")
        print(f"TOTAL_PARAMS={n_total}")
        print(f"TRUNK_WIDTH={student.trunk_width}")
        print(f"TRUNK_LAYERS={student.trunk_layers}")
        print(f"INPUT_PREPROCESSING={args.input_preprocessing}")
        print(f"USE_LOG_FEATURES={student.use_log_features}")
        return
    ctx = setup(args)

    with Timer("plain-mlp DAgger training"):
        result = run_dagger_training(student, ctx, model_name='dagger_student_plain')

    _logger.info("=" * 80)
    _logger.info(f"Final test failure rate: {result['test_failure_rate']*100:.2f}%")
    _logger.info(f"  boundary: {result['boundary_rate']*100:.2f}%")
    _logger.info(f"  interior: {result['interior_rate']*100:.2f}%")
    _logger.info(f"  trainable params: {result['params_count']:,}")
    _logger.info(f"  output: {result['output_dir']}")
    _logger.info(f"  predictions: {result['predictions_csv']}")
    _logger.info("=" * 80)

    teacher_cfg = build_plain_teacher_config(student, ctx, args)
    teacher_cfg_path = os.path.join(result["output_dir"], "teacher_config.json")
    try:
        tmp_path = teacher_cfg_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(teacher_cfg, f, indent=2)
        os.replace(tmp_path, teacher_cfg_path)
        _logger.info(f"Wrote teacher_config.json (v{teacher_cfg.get('schema_version')}) -> {teacher_cfg_path}")
    except Exception as e:
        _logger.warning(f"Failed to write teacher_config.json: {e}")


if __name__ == "__main__":
    main()
