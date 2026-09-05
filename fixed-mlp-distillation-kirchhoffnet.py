"""Fixed-dataset PlainMLP -> KirchhoffNet distillation for CTLE.

This is deliberately a *diagnostic* baseline, not a DAgger run.  It makes one
fixed split from the empirical CTLE target specifications, labels every row
once with a frozen PlainMLP teacher, and trains the KirchhoffNet solely with
MSE(student_logits, teacher_logits).  In particular, it has no ZIG model,
validity filtering, k-NN fallback, failure mining, label appending, auxiliary
losses, weighted sampling, or failure-rate-based rollback.

The saved ``fixed_teacher_dataset.npz`` contains the exact inputs, teacher
logits, and split indices used by the run.  Re-running with ``--resume`` uses
that immutable dataset and continues from ``last_checkpoint.pt``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cell_library import make_cell_library
from config import SOLVER, VCA
from ctle_dagger_common import (
    COL_MAPPING,
    DATASET_CSV_FILES,
    HybridHurdleModel,
    PARAM_COLS,
    PARAM_LOG_BOUNDS,
    PlainMLP,
    StudentEvaluator,
    _set_active_context,
    load_plain_mlp_teacher,
)
from kirchhoff_net import format_parameter_breakdown
from topology import build_net_from_config


SPEC_COLS = ["power", "stage_2_jitter", "stage_2_eye_max_height", "stage_2_eye_max_width"]
DEFAULT_TEACHER_CKPT = "outputs/dagger_output_plain_mlp_w48_preprocessing_knet/dagger_student_plain.pt"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    teacher_source = parser.add_mutually_exclusive_group(required=False)
    teacher_source.add_argument("--mlp-teacher-ckpt", default=None,
                                help="Direct path to dagger_student_plain.pt.")
    teacher_source.add_argument("--teacher-dir", default=None,
                                help="Teacher trial directory, or a PlainMLP BO run directory containing bo_summary.json.")
    parser.add_argument("--mlp-teacher-width", type=int, default=48)
    parser.add_argument("--mlp-teacher-layers", type=int, default=3)
    parser.add_argument("--mlp-teacher-activation", choices=["silu", "gelu"], default="silu")
    parser.add_argument("--mlp-teacher-use-layernorm", type=parse_bool, default=False)
    parser.add_argument("--mlp-teacher-input-preprocessing", choices=["knet", "q75"], default="knet")
    parser.add_argument("--dataset-dir", "--data-dir", dest="dataset_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Default 0: no regularization term is added to the distillation objective.")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Disabled by default; set >0 only to guard numerical instability.")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional deterministic cap after shuffling the empirical rows.")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fixed-dataset-path", default=None,
                        help="Shared immutable .npz cache. BO trials must point to one common path.")
    parser.add_argument("--count-params-only", action="store_true")
    parser.add_argument("--prepare-fixed-dataset", action="store_true",
                        help="Create/verify the immutable teacher-labelled cache then exit without student training.")
    parser.add_argument("--bo-mode", action="store_true",
                        help="Use validation metrics only; never evaluate or expose the held-out test split.")
    parser.add_argument("--student-kind", choices=["knet", "mlp"], default="knet")
    parser.add_argument("--student-width", type=int, default=48,
                        help="PlainMLP student width; used only with --student-kind mlp.")
    parser.add_argument("--student-layers", type=int, default=3,
                        help="PlainMLP student trunk layers; used only with --student-kind mlp.")
    parser.add_argument("--student-activation", choices=["silu", "gelu"], default="silu")
    parser.add_argument("--student-use-layernorm", type=parse_bool, default=False)
    parser.add_argument("--zig-artifact-dir", default=None,
                        help="Directory containing the HybridHurdle/ZIG files; defaults to the teacher checkpoint directory.")
    parser.add_argument("--zig-validity-threshold", type=float, default=0.5)
    parser.add_argument("--zig-degrade-threshold", type=float, default=0.20)
    parser.add_argument("--zig-min-degraded-dims", type=int, default=2)
    parser.add_argument("--isolation-tolerance-pp", type=float, default=3.0,
                        help="Maximum absolute teacher/student failure-rate gap, in percentage points, for PASS.")

    # KNet defaults intentionally match dagger-mlp-distillation-kirchhoffnet.py.
    parser.add_argument("--kn-num-stages", type=int, default=4)
    parser.add_argument("--kn-num-hidden", type=int, default=14)
    parser.add_argument("--kn-small-world-k", type=int, default=4)
    parser.add_argument("--kn-small-world-p", type=float, default=0.2)
    parser.add_argument("--kn-small-world-seed", type=int, default=1)
    parser.add_argument("--kn-edge-repeats", type=int, default=2)
    parser.add_argument("--kn-cell-library", default="tanh_free")
    parser.add_argument("--kn-leak-mode", default="non-programmable")
    parser.add_argument("--kn-interstage-activation", default="residual-relu-tanh")
    parser.add_argument("--kn-freeze-read", type=parse_bool, default=True)
    parser.add_argument("--kn-temporal-readout", type=parse_bool, default=True)
    parser.add_argument("--boundary-fan-out", default='{"0": [2, 4], "1": [1, 3], "2": [12, 5], "3": [7, 9]}')
    parser.add_argument("--kn-input-rail", type=float, default=4.0)
    parser.add_argument("--kn-x-max", type=float, default=4.0)
    parser.add_argument("--kn-input-log-pad-frac", type=float, default=0.05)
    parser.add_argument("--kn-solver-scale", type=float, default=1.0,
                        help="Multiplier on per-stage t_span and num_steps (param-count-neutral; "
                             "preserves the SOLVER step-per-t-span resolution). Default 1.0.")
    parser.add_argument("--kn-vca", type=parse_bool, default=True)
    parser.add_argument("--kn-vca-rank", type=int, default=2)
    parser.add_argument("--kn-vca-core", type=parse_bool, default=True)
    parser.add_argument("--kn-vca-gate-shunt", type=parse_bool, default=False)
    parser.add_argument("--kn-vca-separate-core-bus", type=parse_bool, default=True)
    parser.add_argument("--vca-bias", type=parse_bool, default=False)
    return parser.parse_args()


def resolve_teacher_checkpoint(ckpt: str | None, teacher_dir: str | None) -> str:
    """Resolve a direct checkpoint or the selected teacher from a BO run directory."""
    if ckpt:
        path = Path(ckpt).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--mlp-teacher-ckpt does not exist: {path}")
        return str(path)
    if not teacher_dir:
        return DEFAULT_TEACHER_CKPT
    root = Path(teacher_dir).expanduser().resolve()
    direct = root / "dagger_student_plain.pt"
    if direct.is_file():
        return str(direct)
    summary_path = root / "bo_summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            selected = Path(summary["selected_checkpoint"])
            if selected.is_file():
                return str(selected.resolve())
            trial = int(summary["selected_trial"])
            candidate = root / f"trial_{trial:04d}" / "dagger_student_plain.pt"
            if candidate.is_file():
                return str(candidate)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not resolve selected teacher from {summary_path}: {exc}") from exc
    raise FileNotFoundError(
        f"{root} contains neither dagger_student_plain.pt nor a resolvable bo_summary.json"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_ctle_specs(dataset_dir: str) -> pd.DataFrame:
    frames = []
    for name in DATASET_CSV_FILES:
        path = os.path.join(dataset_dir, name)
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
        else:
            logging.warning("Dataset file not found, skipping: %s", path)
    if not frames:
        raise FileNotFoundError(
            f"No expected CTLE CSV files found in {dataset_dir!r}: {', '.join(DATASET_CSV_FILES)}"
        )
    combined = pd.concat(frames, ignore_index=True).rename(columns=COL_MAPPING)
    required = PARAM_COLS + SPEC_COLS
    missing = [name for name in required if name not in combined.columns]
    if missing:
        raise ValueError(f"CTLE files are missing required columns: {missing}")
    df = combined[required].copy()
    positive = (df[SPEC_COLS] > 0).all(axis=1)
    df = df.loc[positive & ~df.isna().any(axis=1)].reset_index(drop=True)
    if len(df) < 3:
        raise ValueError(f"Only {len(df)} valid CTLE rows remain after filtering")
    return df


def input_log_bounds(specs: np.ndarray, pad_frac: float) -> tuple[np.ndarray, np.ndarray]:
    logs = np.log10(np.clip(specs, 1e-12, None))
    lo, hi = logs.min(axis=0), logs.max(axis=0)
    pad = pad_frac * np.maximum(hi - lo, 1e-8)
    return (lo - pad).astype(np.float32), (hi + pad).astype(np.float32)


def parse_boundary_fan_out(spec: str) -> dict[int, list[int]]:
    parsed = json.loads(spec)
    return {int(key): list(value) for key, value in parsed.items()}


class KirchhoffStudent(nn.Module):
    """The same bounded-logit KNet wrapper used by the MLP DAgger script."""

    def __init__(self, args: argparse.Namespace, log_min: np.ndarray, log_max: np.ndarray):
        super().__init__()
        self.input_rail = float(args.kn_input_rail)
        self.param_log_bounds = PARAM_LOG_BOUNDS
        self.register_buffer("input_log_min", torch.as_tensor(log_min, dtype=torch.float32))
        self.register_buffer("input_log_max", torch.as_tensor(log_max, dtype=torch.float32))
        self.register_buffer("log_lo", torch.tensor([v[0] for v in PARAM_LOG_BOUNDS.values()], dtype=torch.float32))
        self.register_buffer("log_hi", torch.tensor([v[1] for v in PARAM_LOG_BOUNDS.values()], dtype=torch.float32))

        solver_scale = float(args.kn_solver_scale)
        stage_t_span = SOLVER["t_span"] / args.kn_num_stages * solver_scale
        stage_steps = max(1, int(round(SOLVER["num_steps"] / args.kn_num_stages * solver_scale)))
        cfg = {
            "stages": [{
                "num_inputs": 4, "num_hidden": args.kn_num_hidden, "num_proj": 0, "num_outputs": 0,
                "hidden_family": "small_world",
                "hidden_kwargs": {"k": args.kn_small_world_k, "p": args.kn_small_world_p,
                                  "seed": args.kn_small_world_seed, "bidirectional": False},
                "input_pattern": "all_to_all", "output_pattern": "all_to_all", "proj_pattern": "all_to_all",
                "edge_repeats": args.kn_edge_repeats, "t_span": stage_t_span, "num_steps": stage_steps,
            } for _ in range(args.kn_num_stages)],
            "out_dim": len(PARAM_COLS), "write_mode": "sparse_proj", "read_mode": "dense",
            "use_robust_input": False,
        }
        self.net = build_net_from_config(
            cfg, cell_lib=make_cell_library(args.kn_cell_library), leak_mode=args.kn_leak_mode,
            freeze_read=args.kn_freeze_read, interstage_activation=args.kn_interstage_activation,
            boundary_fan_out=parse_boundary_fan_out(args.boundary_fan_out),
            enable_temporal_readout=args.kn_temporal_readout, x_max=args.kn_x_max,
            vca_enabled=args.kn_vca, vca_rank=args.kn_vca_rank, vca_core_enabled=args.kn_vca_core,
            vca_gate_shunt=args.kn_vca_gate_shunt,
            vca_separate_core_bus=args.kn_vca_separate_core_bus, vca_bias=args.vca_bias,
        )

    def scale_input(self, specs: torch.Tensor) -> torch.Tensor:
        logs = torch.log10(specs.clamp_min(1e-12))
        scaled = 2.0 * (logs - self.input_log_min.to(specs)) / (
            self.input_log_max.to(specs) - self.input_log_min.to(specs)
        ).clamp_min(1e-8) - 1.0
        return scaled.clamp(-self.input_rail, self.input_rail)

    def forward(self, specs: torch.Tensor) -> torch.Tensor:
        logits, _ = self.net(self.scale_input(specs), store_trajectory=False, solver="heun")
        return logits

    def predict_log_params(self, specs: torch.Tensor) -> torch.Tensor:
        return self.log_lo + (self.log_hi - self.log_lo) * torch.sigmoid(self(specs))


class PlainMLPStudent(PlainMLP):
    """Plain student with the exact KNet log-min/max preprocessing convention."""

    def __init__(self, args: argparse.Namespace, log_min: np.ndarray, log_max: np.ndarray):
        activation = {"silu": nn.SiLU, "gelu": nn.GELU}[args.student_activation]
        super().__init__(trunk_width=args.student_width, trunk_layers=args.student_layers,
                         activation=activation, use_layernorm=args.student_use_layernorm,
                         use_log_features=False)
        self.attach_scaler(1.0, 0.0, 1.0, 1.0, 1.0, input_preprocessing="knet",
                           input_log_min=log_min, input_log_max=log_max)

    def predict_log_params(self, specs: torch.Tensor) -> torch.Tensor:
        return self.log_lo + (self.log_hi - self.log_lo) * torch.sigmoid(self(specs))


def build_student(args: argparse.Namespace, log_min: np.ndarray, log_max: np.ndarray) -> nn.Module:
    if args.student_kind == "mlp":
        return PlainMLPStudent(args, log_min, log_max)
    return KirchhoffStudent(args, log_min, log_max)


def log_model_summary(model: nn.Module, kind: str) -> None:
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    logging.info("TRAINABLE_PARAMS=%d", count)
    if kind == "knet":
        logging.info("%s", format_parameter_breakdown(model.net.parameter_breakdown()))
    else:
        logging.info("PlainMLP student: width=%d layers=%d", model.trunk_width, model.trunk_layers)


def teacher_identity(path: str) -> dict[str, int | str]:
    stat = os.stat(path)
    cfg = Path(path).with_name("teacher_config.json")
    cfg_id = None
    if cfg.is_file():
        cfg_stat = cfg.stat()
        cfg_id = {"size": cfg_stat.st_size, "mtime_ns": cfg_stat.st_mtime_ns}
    return {"path": os.path.abspath(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "teacher_config": cfg_id}


def dataset_fingerprint(args: argparse.Namespace, df: pd.DataFrame) -> str:
    payload = {
        "seed": args.seed, "teacher": teacher_identity(args.mlp_teacher_ckpt), "rows": len(df),
        "max_samples": args.max_samples, "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction, "input_log_pad_frac": args.kn_input_log_pad_frac,
        "teacher_preprocessing": args.mlp_teacher_input_preprocessing,
        "spec_min": np.round(df[SPEC_COLS].min().values, 10).tolist(),
        "spec_max": np.round(df[SPEC_COLS].max().values, 10).tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def make_fixed_dataset(args: argparse.Namespace, df: pd.DataFrame, device: torch.device,
                       log_min: np.ndarray, log_max: np.ndarray, output: str) -> dict[str, np.ndarray]:
    cache_path = args.fixed_dataset_path or os.path.join(output, "fixed_teacher_dataset.npz")
    cache_path = os.path.abspath(cache_path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    fingerprint = dataset_fingerprint(args, df)
    if (args.resume or args.fixed_dataset_path is not None) and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["fingerprint"].item()) != fingerprint:
            raise ValueError("fixed dataset fingerprint differs from this run; use a new shared dataset path")
        return {key: cached[key] for key in cached.files if key != "fingerprint"}

    order = np.random.default_rng(args.seed).permutation(len(df))
    if args.max_samples is not None:
        order = order[:args.max_samples]
    if len(order) < 3:
        raise ValueError("Need at least three fixed samples after --max-samples")
    specs = df.loc[order, SPEC_COLS].to_numpy(dtype=np.float32)

    teacher = load_teacher(args, device, log_min, log_max)
    batches = []
    for start in range(0, len(specs), args.batch_size):
        batches.append(teacher(torch.from_numpy(specs[start:start + args.batch_size]).to(device)).cpu())
    teacher_logits = torch.cat(batches).numpy().astype(np.float32)

    n_train = int(len(specs) * args.train_fraction)
    n_val = int(len(specs) * args.val_fraction)
    if n_train < 1 or n_val < 1 or len(specs) - n_train - n_val < 1:
        raise ValueError("train/val fractions must leave at least one sample in every split")
    fixed = {
        "specs": specs, "teacher_logits": teacher_logits,
        "train_idx": np.arange(0, n_train, dtype=np.int64),
        "val_idx": np.arange(n_train, n_train + n_val, dtype=np.int64),
        "test_idx": np.arange(n_train + n_val, len(specs), dtype=np.int64),
        "input_log_min": log_min, "input_log_max": log_max,
    }
    np.savez_compressed(cache_path, fingerprint=np.array(fingerprint), **fixed)
    logging.info("Wrote fixed teacher-labelled dataset: %s", cache_path)
    return fixed


def loader(fixed: dict[str, np.ndarray], split: str, batch_size: int, shuffle: bool) -> DataLoader:
    idx = fixed[f"{split}_idx"]
    dataset = TensorDataset(torch.from_numpy(fixed["specs"][idx]), torch.from_numpy(fixed["teacher_logits"][idx]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def evaluate(model: KirchhoffStudent, data: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    sq_error = log_abs_error = n = 0.0
    for specs, teacher_logits in data:
        specs, teacher_logits = specs.to(device), teacher_logits.to(device)
        student_logits = model(specs)
        sq_error += torch.sum((student_logits - teacher_logits).square()).item()
        teacher_log_params = model.log_lo + (model.log_hi - model.log_lo) * torch.sigmoid(teacher_logits)
        log_abs_error += torch.sum((model.predict_log_params(specs) - teacher_log_params).abs()).item()
        n += teacher_logits.numel()
    return {"logit_mse": sq_error / n, "log_param_mae": log_abs_error / n}


def load_teacher(args: argparse.Namespace, device: torch.device,
                 log_min: np.ndarray, log_max: np.ndarray) -> nn.Module:
    activation = {"silu": nn.SiLU, "gelu": nn.GELU}[args.mlp_teacher_activation]
    teacher = load_plain_mlp_teacher(
        args.mlp_teacher_ckpt, device, trunk_width=args.mlp_teacher_width,
        trunk_layers=args.mlp_teacher_layers, activation=activation,
        use_layernorm=args.mlp_teacher_use_layernorm,
        input_preprocessing=args.mlp_teacher_input_preprocessing,
    )
    if teacher.input_preprocessing != "knet":
        raise ValueError(f"teacher config requests {teacher.input_preprocessing!r}; fixed benchmark requires knet")
    # Schema-v2 teacher_config.json supplies the exact training scaler/bounds.
    # Only legacy checkpoints require this fallback attachment.
    if teacher.input_log_min is None or teacher.input_log_max is None:
        teacher.attach_scaler(1.0, 0.0, 1.0, 1.0, 1.0, input_preprocessing="knet",
                              input_log_min=log_min, input_log_max=log_max)
        logging.warning("Teacher lacks a schema-v2 scaler config; attached bounds derived from the fixed CTLE data.")
    return teacher


def evaluate_with_zig(args: argparse.Namespace, teacher: nn.Module,
                      student: KirchhoffStudent, specs: np.ndarray,
                      device: torch.device) -> tuple[dict[str, float | bool], dict[str, np.ndarray]]:
    """Evaluate only; no ZIG output participates in labeling or optimization."""
    artifact_dir = args.zig_artifact_dir or os.path.dirname(os.path.abspath(args.mlp_teacher_ckpt))
    required = {
        "model": "hybrid_hurdle_ctle_model.pt",
        "scaler_x": "hybrid_hurdle_scaler_X.pkl",
        "scaler_power": "hybrid_hurdle_scaler_y_power.pkl",
        "config": "hybrid_hurdle_config.pkl",
    }
    paths = {key: os.path.join(artifact_dir, name) for key, name in required.items()}
    absent = [path for path in paths.values() if not os.path.isfile(path)]
    if absent:
        raise FileNotFoundError(
            "Final ZIG evaluation requires artifacts beside --mlp-teacher-ckpt (or --zig-artifact-dir). Missing: "
            + ", ".join(absent)
        )
    scaler_x = joblib.load(paths["scaler_x"])
    scaler_power = joblib.load(paths["scaler_power"])
    zig_config = joblib.load(paths["config"])
    zig = HybridHurdleModel(
        dropout=0.0, per_target=bool(zig_config.get("per_target_hurdle", False))
    ).to(device)
    zig.load_state_dict(torch.load(paths["model"], map_location=device))
    zig.eval()
    for parameter in zig.parameters():
        parameter.requires_grad = False
    _set_active_context({
        "DEVICE": device, "zig_model": zig, "scaler_X": scaler_x, "scaler_y_p": scaler_power,
        "VALIDITY_THRESHOLD": args.zig_validity_threshold,
        "DEGRADE_REL_THRESHOLD": args.zig_degrade_threshold,
        "MIN_DEGRADED_DIMS": args.zig_min_degraded_dims,
        "ERROR_THRESHOLD": 0.10,
    })
    evaluator_kwargs = dict(
        scaler_X=scaler_x, zig_model=zig,
        eye_scale_h=float(zig_config["eye_scale_h"]), eye_scale_w=float(zig_config["eye_scale_w"]),
        eye_scale_j=float(zig_config["eye_scale_j"]), scaler_y_p=scaler_power, device=device,
    )
    teacher_failures, teacher_metrics = StudentEvaluator(teacher, **evaluator_kwargs).identify_failures(specs)
    student_failures, student_metrics = StudentEvaluator(student, **evaluator_kwargs).identify_failures(specs)
    teacher_rate = float(teacher_failures.mean())
    student_rate = float(student_failures.mean())
    gap_pp = abs(student_rate - teacher_rate) * 100.0
    passed = gap_pp <= args.isolation_tolerance_pp
    summary = {
        "teacher_failure_rate": teacher_rate,
        "student_failure_rate": student_rate,
        "absolute_gap_percentage_points": gap_pp,
        "tolerance_percentage_points": args.isolation_tolerance_pp,
        "isolation_pass": passed,
        "teacher_invalid_rate": float(teacher_metrics["invalid_mask"].mean()),
        "student_invalid_rate": float(student_metrics["invalid_mask"].mean()),
    }
    arrays = {
        "specs": specs, "teacher_failure_mask": teacher_failures,
        "student_failure_mask": student_failures,
        "teacher_pred_specs": teacher_metrics["pred_specs"],
        "student_pred_specs": student_metrics["pred_specs"],
    }
    return summary, arrays


def main() -> None:
    args = parse_args()
    args.mlp_teacher_ckpt = resolve_teacher_checkpoint(args.mlp_teacher_ckpt, args.teacher_dir)
    if not 0 < args.train_fraction < 1 or not 0 < args.val_fraction < 1 or args.train_fraction + args.val_fraction >= 1:
        raise ValueError("--train-fraction and --val-fraction must be positive and sum to less than one")
    if args.epochs < 1 or args.batch_size < 1 or args.eval_every < 1:
        raise ValueError("--epochs, --batch-size, and --eval-every must be positive")
    if args.mlp_teacher_input_preprocessing != "knet":
        raise ValueError(
            "This isolation baseline currently supports the requested --mlp-teacher-input-preprocessing knet only. "
            "A q75 teacher requires its original fitted Q75/ZIG scaling artifacts and must not be labelled with placeholders."
        )
    if args.student_kind == "knet" and float(args.kn_solver_scale) <= 0:
        raise ValueError(f"--kn-solver-scale must be positive, got {args.kn_solver_scale}")
    if args.student_kind == "knet" and float(args.kn_x_max) <= 0:
        raise ValueError(f"--kn-x-max must be positive, got {args.kn_x_max}")
    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = resolve_device(args.device)
    set_seed(args.seed)
    logging.info("Fixed MLP -> KNet distillation; device=%s, seed=%d", device, args.seed)

    if args.count_params_only:
        lo = np.array([-4.0, 0.0, -1.0, 0.0], dtype=np.float32)
        hi = np.array([-1.0, 3.0, 3.0, 3.0], dtype=np.float32)
        model = build_student(args, lo, hi)
        log_model_summary(model, args.student_kind)
        return

    df = load_ctle_specs(args.dataset_dir)
    all_specs = df[SPEC_COLS].to_numpy(dtype=np.float32)
    log_min, log_max = input_log_bounds(all_specs, args.kn_input_log_pad_frac)
    fixed = make_fixed_dataset(args, df, device, log_min, log_max, args.output)
    logging.info("Fixed data: train=%d val=%d test=%d; teacher labels generated once",
                 len(fixed["train_idx"]), len(fixed["val_idx"]), len(fixed["test_idx"]))
    if args.prepare_fixed_dataset:
        logging.info("Fixed dataset prepared; exiting before student construction.")
        return

    model = build_student(args, fixed["input_log_min"], fixed["input_log_max"]).to(device)
    log_model_summary(model, args.student_kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    train_data = loader(fixed, "train", args.batch_size, True)
    val_data = loader(fixed, "val", args.batch_size, False)
    test_data = loader(fixed, "test", args.batch_size, False)
    checkpoint_path = os.path.join(args.output, "last_checkpoint.pt")
    best_path = os.path.join(args.output, "best_student.pt")
    start_epoch, best_val = 0, float("inf")
    history: list[dict[str, float]] = []
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, best_val, history = checkpoint["epoch"] + 1, checkpoint["best_val"], checkpoint["history"]
        logging.info("Resuming at epoch %d", start_epoch + 1)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total, count = 0.0, 0
        for specs, teacher_logits in train_data:
            specs, teacher_logits = specs.to(device), teacher_logits.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(specs), teacher_logits)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite distillation loss at epoch {epoch + 1}")
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += loss.item() * teacher_logits.numel()
            count += teacher_logits.numel()
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            val_metrics = evaluate(model, val_data, device)
            row = {"epoch": epoch + 1, "train_logit_mse": total / count, **val_metrics}
            history.append(row)
            logging.info("epoch %4d/%d train_logit_mse=%.7f val_logit_mse=%.7f val_log_param_mae=%.7f",
                         epoch + 1, args.epochs, row["train_logit_mse"], row["logit_mse"], row["log_param_mae"])
            if val_metrics["logit_mse"] < best_val:
                best_val = val_metrics["logit_mse"]
                torch.save(model.state_dict(), best_path)
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "best_val": best_val, "history": history}, checkpoint_path)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    final = {"validation": evaluate(model, val_data, device)}
    test_idx = fixed["test_idx"]
    if not args.bo_mode:
        final["test"] = evaluate(model, test_data, device)
        with torch.no_grad():
            test_specs = torch.from_numpy(fixed["specs"][test_idx]).to(device)
            student_logits = model(test_specs).cpu().numpy()
        np.savez_compressed(
            os.path.join(args.output, "test_teacher_student_predictions.npz"),
            specs=fixed["specs"][test_idx], teacher_logits=fixed["teacher_logits"][test_idx],
            student_logits=student_logits,
        )
    teacher = load_teacher(args, device, fixed["input_log_min"], fixed["input_log_max"])
    zig_val_summary, zig_val_arrays = evaluate_with_zig(
        args, teacher, model, fixed["specs"][fixed["val_idx"]], device
    )
    final["zig_failure_comparison"] = {"validation": zig_val_summary}
    zig_arrays_out = {f"validation_{key}": value for key, value in zig_val_arrays.items()}
    zig_test_summary = None
    if not args.bo_mode:
        zig_test_summary, zig_test_arrays = evaluate_with_zig(
            args, teacher, model, fixed["specs"][test_idx], device
        )
        final["zig_failure_comparison"]["test"] = zig_test_summary
        zig_arrays_out.update({f"test_{key}": value for key, value in zig_test_arrays.items()})
    np.savez_compressed(os.path.join(args.output, "zig_failure_comparison.npz"), **zig_arrays_out)
    torch.save(model.state_dict(), os.path.join(args.output, "student_final.pt"))
    torch.save({"model": model.state_dict(), "input_log_min": fixed["input_log_min"],
                "input_log_max": fixed["input_log_max"], "args": vars(args)},
               os.path.join(args.output, "student_bundle.pt"))
    with open(os.path.join(args.output, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump({"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "objective": "MSE(student_logits, teacher_logits)",
                   "final": final, "history": history, "args": vars(args)}, handle, indent=2)
    logging.info("Best frozen-teacher agreement: validation=%s", final["validation"])
    logging.info("ZIG validation objective: teacher=%.2f%% student=%.2f%% logit_mse=%.8f",
                 100.0 * zig_val_summary["teacher_failure_rate"],
                 100.0 * zig_val_summary["student_failure_rate"], final["validation"]["logit_mse"])
    if args.bo_mode:
        logging.info("BO mode: held-out test evaluation intentionally skipped.")
        return
    assert zig_test_summary is not None
    verdict = "PASS" if zig_test_summary["isolation_pass"] else "FAIL"
    logging.info(
        "ZIG isolation verdict: %s | teacher failure=%.2f%% student failure=%.2f%% gap=%.2f pp (tolerance %.2f pp)",
        verdict, 100.0 * zig_test_summary["teacher_failure_rate"], 100.0 * zig_test_summary["student_failure_rate"],
        zig_test_summary["absolute_gap_percentage_points"], zig_test_summary["tolerance_percentage_points"],
    )
    if not zig_test_summary["isolation_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
