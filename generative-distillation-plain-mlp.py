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
import logging
import os
import sys
import warnings

import torch

warnings.filterwarnings("ignore")

from ctle_dagger_common import (  # noqa: E402
    PlainMLP, plain_param_count, derive_plain_width,
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
    parser.add_argument("--param-budget", type=int, default=None,
                        help="If set, auto-derive plain trunk width via derive_plain_width "
                             "so plain_param_count(W,L) <= param-budget. Overrides --plain-trunk-width.")
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
        use_log_features=args.plain_use_log_features,
    )
    return student


def main():
    args = parse_args()
    ctx = setup(args)
    student = build_student(args)
    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in student.parameters())
    _logger.info(
        f"Student model: {n_trainable:,} trainable / {n_total:,} total params  "
        f"(plain W={student.trunk_width} L={student.trunk_layers} "
        f"act={args.plain_activation} LN={student.use_layernorm} "
        f"log_features={student.use_log_features})"
    )

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


if __name__ == "__main__":
    main()
