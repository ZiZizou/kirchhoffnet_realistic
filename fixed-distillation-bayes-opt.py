"""Multi-objective BO for the fixed CTLE teacher-distillation benchmark.

Every trial uses the same immutable teacher-labelled split and a fixed number
of epochs.  Optuna sees only validation metrics: (ZIG failure rate,
teacher-logit MSE).  The held-out test split is evaluated once, after BO, for
the lexicographically selected validation configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import optuna
from optuna.samplers import TPESampler


ROOT = Path(__file__).resolve().parent
HARNESS = ROOT / "fixed-mlp-distillation-kirchhoffnet.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-kind", choices=["knet", "mlp"], required=True)
    teacher_source = parser.add_mutually_exclusive_group(required=True)
    teacher_source.add_argument("--mlp-teacher-ckpt")
    teacher_source.add_argument("--teacher-dir",
                                help="PlainMLP BO run directory (bo_summary.json) or selected trial directory.")
    parser.add_argument("--mlp-teacher-width", type=int, default=48)
    parser.add_argument("--mlp-teacher-layers", type=int, default=3)
    parser.add_argument("--mlp-teacher-activation", choices=["silu", "gelu"], default="silu")
    parser.add_argument("--mlp-teacher-use-layernorm", default="False")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--zig-artifact-dir", default=None,
                        help="HybridHurdle/ZIG artifact directory used only for validation and final test evaluation.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed-dataset-path", type=Path, default=None,
                        help="One shared cache path; pass the identical absolute path to both KNet and MLP BO runs.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=110)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--param-budget", type=int, required=True)
    parser.add_argument("--param-tolerance", type=float, default=0.15,
                        help="Each eligible architecture must lie in [budget*(1-tolerance), budget].")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-startup-trials", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def base_command(args: argparse.Namespace, output: Path) -> list[str]:
    teacher_arg = (["--mlp-teacher-ckpt", args.mlp_teacher_ckpt]
                   if args.mlp_teacher_ckpt else ["--teacher-dir", args.teacher_dir])
    command = [
        args.python, str(HARNESS), "--student-kind", args.student_kind,
        *teacher_arg,
        "--mlp-teacher-width", str(args.mlp_teacher_width), "--mlp-teacher-layers", str(args.mlp_teacher_layers),
        "--mlp-teacher-activation", args.mlp_teacher_activation,
        "--mlp-teacher-use-layernorm", str(args.mlp_teacher_use_layernorm),
        "--mlp-teacher-input-preprocessing", "knet",
        "--dataset-dir", args.dataset_dir, "--output", str(output),
        "--fixed-dataset-path", str(args.fixed_dataset_path),
        "--device", args.device, "--seed", str(args.seed), "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size), "--weight-decay", "0", "--bo-mode",
    ]
    if args.zig_artifact_dir:
        command += ["--zig-artifact-dir", args.zig_artifact_dir]
    return command


def sample_config(trial: optuna.Trial, kind: str) -> dict[str, str]:
    cfg: dict[str, str] = {
        "--lr": f"{trial.suggest_float('lr', 2e-4, 5e-3, log=True):.8g}",
        "--batch-size": str(trial.suggest_categorical("batch_size", [128, 256, 512, 1024])),
    }
    if kind == "mlp":
        cfg.update({
            "--student-width": str(trial.suggest_int("student_width", 16, 256)),
            "--student-layers": str(trial.suggest_int("student_layers", 1, 5)),
            "--student-activation": trial.suggest_categorical("student_activation", ["silu", "gelu"]),
            "--student-use-layernorm": str(trial.suggest_categorical("student_layernorm", [False, True])),
        })
    else:
        hidden = trial.suggest_int("kn_num_hidden", 8, 24)
        small_world_k = trial.suggest_categorical("kn_small_world_k", [2, 4, 6, 8])
        if small_world_k >= hidden:
            raise optuna.TrialPruned("small-world degree must be lower than hidden-node count")
        fanout = {str(index): [index, index + 4] for index in range(4)}
        cfg.update({
            "--kn-num-hidden": str(hidden),
            "--kn-num-stages": str(trial.suggest_int("kn_num_stages", 2, 5)),
            "--kn-small-world-k": str(small_world_k),
            "--kn-small-world-p": "0.2", "--kn-vca-rank": str(trial.suggest_int("kn_vca_rank", 1, 4)),
            "--kn-x-max": f"{trial.suggest_float('kn_x_max', 2.0, 6.0):.6g}",
            "--boundary-fan-out": json.dumps(fanout),
        })
    return cfg


def flat_args(cfg: dict[str, str]) -> list[str]:
    return [item for pair in cfg.items() for item in pair]


def param_count(command: list[str]) -> int | None:
    result = subprocess.run(command + ["--count-params-only"], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        return None
    match = re.search(r"TRAINABLE_PARAMS=(\d+)", result.stdout)
    return int(match.group(1)) if match else None


def metrics(path: Path) -> tuple[float, float]:
    with open(path / "metrics.json", encoding="utf-8") as handle:
        result = json.load(handle)
    final = result["final"]
    failure = final["zig_failure_comparison"]["validation"]["student_failure_rate"]
    imitation = final["validation"]["logit_mse"]
    return float(failure), float(imitation)


def main() -> None:
    args = parse_args()
    if args.n_trials < 1 or args.epochs < 1 or not 0 <= args.param_tolerance < 1:
        raise ValueError("invalid BO budget or --param-tolerance")
    args.output.mkdir(parents=True, exist_ok=True)
    fixed_path = (args.fixed_dataset_path or args.output / "fixed_teacher_dataset.npz").resolve()
    args.fixed_dataset_path = fixed_path
    # Serialize cache creation before trials; this guarantees every trial sees byte-identical labels.
    prepare = base_command(args, args.output / "_dataset_prepare") + ["--prepare-fixed-dataset"]
    prepare_result = subprocess.run(prepare, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (args.output / "dataset_prepare.log.txt").write_text(prepare_result.stdout, encoding="utf-8")
    if prepare_result.returncode != 0 or not fixed_path.exists():
        raise RuntimeError("failed to create fixed teacher dataset; see dataset_prepare.log.txt")

    storage = f"sqlite:///{(args.output / 'study.db').resolve()}"
    study_name = f"fixed_ctle_{args.student_kind}"
    if args.resume:
        study = optuna.load_study(study_name=study_name, storage=storage)
    else:
        study = optuna.create_study(study_name=study_name, storage=storage,
                                    directions=["minimize", "minimize"],
                                    sampler=TPESampler(seed=args.seed, n_startup_trials=args.n_startup_trials),
                                    load_if_exists=True)
    low = int(args.param_budget * (1.0 - args.param_tolerance))
    print(f"[fixed-bo] kind={args.student_kind} trials={args.n_trials} epochs={args.epochs} "
          f"param window=[{low}, {args.param_budget}] fixed_dataset={fixed_path}", flush=True)

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        cfg = sample_config(trial, args.student_kind)
        trial_dir = args.output / f"trial_{trial.number:04d}"
        command = base_command(args, trial_dir) + flat_args(cfg)
        actual_params = param_count(command)
        if actual_params is None:
            raise optuna.TrialPruned("parameter preflight failed")
        trial.set_user_attr("actual_params", actual_params)
        trial.set_user_attr("config", cfg)
        if not low <= actual_params <= args.param_budget:
            raise optuna.TrialPruned(f"params={actual_params} outside [{low}, {args.param_budget}]")
        trial_dir.mkdir(parents=True, exist_ok=True)
        run = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (trial_dir / "run.log.txt").write_text(run.stdout, encoding="utf-8")
        if run.returncode != 0:
            raise optuna.TrialPruned(f"training subprocess failed ({run.returncode})")
        failure, imitation = metrics(trial_dir)
        trial.set_user_attr("validation_failure_rate", failure)
        trial.set_user_attr("validation_logit_mse", imitation)
        print(f"[fixed-bo] trial={trial.number:04d} params={actual_params} "
              f"val_failure={failure * 100:.2f}% val_logit_mse={imitation:.8f}", flush=True)
        return failure, imitation

    study.optimize(objective, n_trials=args.n_trials, n_jobs=1)
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError("no eligible completed trials")
    selected = min(complete, key=lambda trial: (float(trial.values[0]), float(trial.values[1])))
    selected_cfg = selected.user_attrs["config"]
    selected_dir = args.output / "selected_final"
    final_command = base_command(args, selected_dir)
    final_command.remove("--bo-mode")
    final_command += flat_args(selected_cfg)
    final_run = subprocess.run(final_command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (selected_dir / "run.log.txt").parent.mkdir(parents=True, exist_ok=True)
    (selected_dir / "run.log.txt").write_text(final_run.stdout, encoding="utf-8")
    # Code 2 is the harness' legitimate "outside teacher-gap tolerance" verdict;
    # metrics are still complete and must be retained for the paper comparison.
    if final_run.returncode not in (0, 2):
        raise RuntimeError(f"selected final evaluation failed ({final_run.returncode})")
    with open(selected_dir / "metrics.json", encoding="utf-8") as handle:
        final_metrics = json.load(handle)
    summary = {
        "student_kind": args.student_kind, "fixed_dataset": str(fixed_path),
        "selection": "lexicographic validation ZIG failure, then validation teacher-logit MSE",
        "selected_trial": selected.number, "selected_params": selected.params,
        "selected_config": selected_cfg, "selected_param_count": selected.user_attrs["actual_params"],
        "selected_validation_values": selected.values, "final_metrics": final_metrics,
        "pareto_trials": [{"trial": trial.number, "values": trial.values,
                           "param_count": trial.user_attrs.get("actual_params")}
                          for trial in study.best_trials],
    }
    with open(args.output / "bo_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(args.output / "results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial", "state", "validation_failure_rate", "validation_logit_mse", "param_count", "params"])
        for trial in study.trials:
            writer.writerow([trial.number, trial.state.name,
                             trial.values[0] if trial.values else "", trial.values[1] if trial.values else "",
                             trial.user_attrs.get("actual_params", ""), json.dumps(trial.params, sort_keys=True)])
    print(f"[fixed-bo] selected trial {selected.number}; artifacts: {args.output}", flush=True)


if __name__ == "__main__":
    main()
