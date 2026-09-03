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
from optuna.study import StudyDirection


ROOT = Path(__file__).resolve().parent
HARNESS = ROOT / "fixed-mlp-distillation-kirchhoffnet.py"

# Finite penalty tuple returned in lieu of optuna.TrialPruned for any infeasible
# configuration. Required because TPESampler._calculate_weights_below_for_multi_objective
# crashes when ``below_trials`` contains pruned trials (values=None): the
# ``lvals *= [-1.0 if d == StudyDirection.MAXIMIZE else 1.0 for d in study.directions]``
# line does ``None * 1.0`` and raises TypeError. See optuna/optuna#5260.
# Failure-rate is in [0, 1] for completed trials; MSE is <<1e3 for valid runs.
# (1.0, 1e6) therefore sorts lexicographically after every real trial and never
# appears Pareto-optimal, but never contains inf/-inf which also crashes the
# multi-objective Parzen estimator (optuna/optuna#3676).
PENALTY_VALUES: tuple[float, float] = (1.0, 1e6)


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
    parser.add_argument("--prepare-device", default="cpu",
                        help="Device for one-time teacher-label cache creation (default CPU, leaving the allocated GPU free for trials).")
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


def sample_config(trial: optuna.Trial, kind: str) -> tuple[dict[str, str], str | None]:
    """Suggest hyperparameters and return ``(cfg, penalty_reason)``.

    ``penalty_reason`` is ``None`` for a valid configuration; otherwise it is a
    short human-readable reason this trial should be penalized. The caller must
    use the return value rather than re-reading ``trial.user_attrs`` to avoid
    storage round-trip dependencies on the live ``Trial`` object.
    """
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
            # Return finite penalty (see PENALTY_VALUES) instead of raising
            # TrialPruned -- the latter produces values=None which crashes the
            # multi-objective TPE sampler (optuna/optuna#5260).
            cfg.update({
                "--kn-num-hidden": str(hidden),
                "--kn-small-world-k": str(small_world_k),
            })
            return cfg, "small-world degree must be lower than hidden-node count"
        fanout = {str(index): [index, index + 4] for index in range(4)}
        cfg.update({
            "--kn-num-hidden": str(hidden),
            "--kn-num-stages": str(trial.suggest_int("kn_num_stages", 2, 5)),
            "--kn-small-world-k": str(small_world_k),
            "--kn-small-world-p": "0.2", "--kn-vca-rank": str(trial.suggest_int("kn_vca_rank", 1, 4)),
            "--kn-x-max": f"{trial.suggest_float('kn_x_max', 2.0, 6.0):.6g}",
            "--boundary-fan-out": json.dumps(fanout),
        })
    return cfg, None


def flat_args(cfg: dict[str, str]) -> list[str]:
    return [item for pair in cfg.items() for item in pair]


def param_count(command: list[str]) -> int | None:
    """Return trainable-param count, or None on any failure (returncode, parse, OSError)."""
    try:
        result = subprocess.run(command + ["--count-params-only"], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[fixed-bo] param_count raised {exc!r}; treating as preflight failure",
              flush=True)
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"TRAINABLE_PARAMS=(\d+)", result.stdout)
    return int(match.group(1)) if match else None


def metrics(path: Path) -> tuple[float, float] | None:
    """Return (failure_rate, logit_mse) or None on any parse/structural error.

    The caller must treat None as a penalty-eligible failure (NOT a TrialPruned)
    because an unhandled exception here propagates out of ``objective``, marks the
    trial as FAIL with ``values=None``, and crashes the next multi-obj TPE trial
    in optuna#5260 territory.

    Rejects NaN/+-inf too: NaN breaks the Parzen estimator's distance/sort
    comparisons, and +-inf triggers optuna#3676. Both must become penalties.
    """
    import math as _math
    try:
        with open(path / "metrics.json", encoding="utf-8") as handle:
            result = json.load(handle)
        final = result["final"]
        failure = float(final["zig_failure_comparison"]["validation"]
                                ["student_failure_rate"])
        imitation = float(final["validation"]["logit_mse"])
        if not (_math.isfinite(failure) and _math.isfinite(imitation)):
            print(f"[fixed-bo] metrics non-finite for {path}: "
                  f"failure={failure} imitation={imitation}", flush=True)
            return None
        return failure, imitation
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"[fixed-bo] metrics parse failed for {path}: {exc!r}", flush=True)
        return None


def main() -> None:
    args = parse_args()
    if args.n_trials < 1 or args.epochs < 1 or not 0 <= args.param_tolerance < 1:
        raise ValueError("invalid BO budget or --param-tolerance")
    args.output.mkdir(parents=True, exist_ok=True)
    fixed_path = (args.fixed_dataset_path or args.output / "fixed_teacher_dataset.npz").resolve()
    args.fixed_dataset_path = fixed_path
    # Serialize cache creation before trials; this guarantees every trial sees byte-identical labels.
    prepare = base_command(args, args.output / "_dataset_prepare") + ["--prepare-fixed-dataset"]
    prepare_device_index = prepare.index("--device") + 1
    prepare[prepare_device_index] = args.prepare_device
    prepare_result = subprocess.run(prepare, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (args.output / "dataset_prepare.log.txt").write_text(prepare_result.stdout, encoding="utf-8")
    if prepare_result.returncode != 0 or not fixed_path.exists():
        raise RuntimeError("failed to create fixed teacher dataset; see dataset_prepare.log.txt")

    storage = f"sqlite:///{(args.output / 'study.db').resolve()}"
    study_name = f"fixed_ctle_{args.student_kind}"
    expected_directions = [StudyDirection.MINIMIZE, StudyDirection.MINIMIZE]
    if args.resume:
        if not (args.output / "study.db").exists():
            raise RuntimeError(
                f"--resume was passed but no study.db exists at {storage}. "
                f"Omit --resume to start a fresh study in {args.output}."
            )
        study = optuna.load_study(study_name=study_name, storage=storage)
        # Guard against silently mismatched study shapes (single-obj DB reused for
        # the multi-obj controller -> ragged t.values -> inhomogeneous shape
        # crash in optuna TPE multi-objective Parzen estimator). Compare against
        # the StudyDirection enum, not the .name string, to avoid casing drift.
        existing = list(study.directions)
        if existing != expected_directions:
            raise RuntimeError(
                f"refusing to resume study '{study_name}' with directions {existing}; "
                f"the fixed-distillation controller requires {expected_directions}. "
                f"Pass --output to a fresh directory or delete {storage}."
            )
    else:
        # Without --resume we refuse to silently reuse an existing sqlite study
        # whose directions may not match. This prevents the inhomogeneous-shape
        # ValueError in optuna.samplers._tpe.sampler._calculate_weights_below_for_multi_objective.
        if (args.output / "study.db").exists():
            raise RuntimeError(
                f"study.db already exists at {storage}. Pass --resume to continue "
                f"the existing study, or choose a fresh --output directory."
            )
        study = optuna.create_study(study_name=study_name, storage=storage,
                                    directions=expected_directions,
                                    sampler=TPESampler(seed=args.seed, n_startup_trials=args.n_startup_trials),
                                    load_if_exists=False)

    # Defense-in-depth: pre-optimize scan rejects pre-existing COMPLETE trials
    # with the wrong values shape. Even with direction-matched studies, a manually
    # added tell()-injected trial could still trigger the TPE ragged-array crash
    # or the lvals*=directions NoneType crash (optuna/optuna#5260).
    n_pruned = n_failed = n_running = 0
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.PRUNED:
            n_pruned += 1
        elif trial.state == optuna.trial.TrialState.FAIL:
            n_failed += 1
        elif trial.state == optuna.trial.TrialState.RUNNING:
            n_running += 1
        elif trial.state == optuna.trial.TrialState.COMPLETE:
            if trial.values is None or len(trial.values) != len(expected_directions):
                raise RuntimeError(
                    f"study '{study_name}' has heterogeneous COMPLETE trial values; "
                    f"trial {trial.number} has "
                    f"{'None' if trial.values is None else len(trial.values)} value(s), "
                    f"expected {len(expected_directions)}. "
                    f"Either migrate via probe_study_db.py or use a fresh --output."
                )
    # HIGH-2: a pre-fix study contains PRUNED/FAIL trials whose values are None.
    # Optuna offers no safe "delete trial" API, so resuming a pre-fix study would
    # still feed those None-valued trials into the multi-obj TPE Parzen estimator
    # and reproduce the optuna#5260 `None * 1.0` TypeError. Fail-fast instead.
    if args.resume and (n_pruned or n_failed):
        raise RuntimeError(
            f"refusing to resume study '{study_name}': {n_pruned} PRUNED and "
            f"{n_failed} FAIL trials present (values=None). Pre-fix runs leave "
            f"these in storage; resuming would re-trigger the optuna#5260 "
            f"`None * float` crash in TPESampler. Pass --output to a fresh "
            f"directory or delete {storage}. Use probe_study_db.py for audit."
        )
    if n_running:
        print(f"[fixed-bo] WARNING: {n_running} RUNNING trial(s) present; "
              "they may be retried by the next trial slot. Reap manually if stale.",
              flush=True)
    low = int(args.param_budget * (1.0 - args.param_tolerance))
    print(f"[fixed-bo] kind={args.student_kind} trials={args.n_trials} epochs={args.epochs} "
          f"param window=[{low}, {args.param_budget}] fixed_dataset={fixed_path}", flush=True)

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        cfg, sample_penalty_reason = sample_config(trial, args.student_kind)
        if sample_penalty_reason is not None:
            trial.set_user_attr("config", cfg)
            trial.set_user_attr("penalized", True)
            trial.set_user_attr("penalty_reason", sample_penalty_reason)
            return PENALTY_VALUES
        trial_dir = args.output / f"trial_{trial.number:04d}"
        command = base_command(args, trial_dir) + flat_args(cfg)
        actual_params = param_count(command)
        if actual_params is None:
            trial.set_user_attr("config", cfg)
            trial.set_user_attr("penalized", True)
            trial.set_user_attr("penalty_reason", "parameter preflight failed")
            return PENALTY_VALUES
        trial.set_user_attr("actual_params", actual_params)
        trial.set_user_attr("config", cfg)
        if not low <= actual_params <= args.param_budget:
            trial.set_user_attr("penalized", True)
            trial.set_user_attr("penalty_reason",
                                f"params={actual_params} outside [{low}, {args.param_budget}]")
            return PENALTY_VALUES
        trial_dir.mkdir(parents=True, exist_ok=True)
        try:
            run = subprocess.run(command, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 timeout=24 * 3600)
            (trial_dir / "run.log.txt").write_text(run.stdout, encoding="utf-8")
            if run.returncode != 0:
                trial.set_user_attr("penalized", True)
                trial.set_user_attr("penalty_reason",
                                    f"training subprocess failed ({run.returncode})")
                return PENALTY_VALUES
        except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as exc:
            print(f"[fixed-bo] training subprocess raised {exc!r}; treating as penalty",
                  flush=True)
            trial.set_user_attr("penalized", True)
            trial.set_user_attr("penalty_reason", f"training subprocess exception ({exc!r})")
            return PENALTY_VALUES
        failure_imitation = metrics(trial_dir)
        if failure_imitation is None:
            trial.set_user_attr("penalized", True)
            trial.set_user_attr("penalty_reason", "metrics parse failed")
            return PENALTY_VALUES
        failure, imitation = failure_imitation
        trial.set_user_attr("validation_failure_rate", failure)
        trial.set_user_attr("validation_logit_mse", imitation)
        print(f"[fixed-bo] trial={trial.number:04d} params={actual_params} "
              f"val_failure={failure * 100:.2f}% val_logit_mse={imitation:.8f}", flush=True)
        return failure, imitation

    study.optimize(objective, n_trials=args.n_trials, n_jobs=1)
    n_expected = len(expected_directions)
    complete = [trial for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE
                and trial.values is not None
                and len(trial.values) == n_expected
                and all(isinstance(v, (int, float)) for v in trial.values)
                and "config" in trial.user_attrs
                and "actual_params" in trial.user_attrs
                and trial.user_attrs.get("penalized") is not True]
    penalized = [trial for trial in study.trials
                 if trial.state == optuna.trial.TrialState.COMPLETE
                 and trial.user_attrs.get("penalized") is True]
    if not complete:
        raise RuntimeError(
            f"no eligible completed trials: {len(study.trials)} total, "
            f"{len(penalized)} penalized (infeasible config). "
            "Reasons: "
            + "; ".join(str(t.user_attrs.get('penalty_reason', '?'))
                        for t in penalized[:5])
        )
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
    try:
        with open(selected_dir / "metrics.json", encoding="utf-8") as handle:
            final_metrics = json.load(handle)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"selected_final metrics.json is malformed ({exc!r}); "
            "bo_summary cannot be written."
        ) from exc
    # Pareto (best_trials) from optuna includes every non-dominated trial; filter
    # out penalized ones because (1.0, 1e6) is never Pareto-optimal against real
    # trials and (2) presenting penalties as Pareto candidates is misleading.
    pareto_trials = [
        {"trial": trial.number, "values": list(trial.values),
         "param_count": trial.user_attrs.get("actual_params")}
        for trial in study.best_trials
        if trial.user_attrs.get("penalized") is not True
    ]
    summary = {
        "student_kind": args.student_kind, "fixed_dataset": str(fixed_path),
        "selection": "lexicographic validation ZIG failure, then validation teacher-logit MSE",
        "selected_trial": selected.number, "selected_params": selected.params,
        "selected_config": selected_cfg, "selected_param_count": selected.user_attrs["actual_params"],
        "selected_validation_values": list(selected.values),
        "final_metrics": final_metrics,
        "n_trials": len(study.trials),
        "n_completed": sum(1 for t in study.trials
                           if t.state == optuna.trial.TrialState.COMPLETE),
        "n_penalized": sum(1 for t in study.trials
                           if t.state == optuna.trial.TrialState.COMPLETE
                           and t.user_attrs.get("penalized") is True),
        "penalty_value": list(PENALTY_VALUES),
        "pareto_trials": pareto_trials,
    }
    with open(args.output / "bo_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(args.output / "results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial", "state", "validation_failure_rate",
                         "validation_logit_mse", "param_count", "penalized",
                         "penalty_reason", "params"])
        for trial in study.trials:
            v0, v1 = "", ""
            if trial.values and len(trial.values) >= 2:
                v0, v1 = trial.values[0], trial.values[1]
            penalized = trial.user_attrs.get("penalized", False)
            writer.writerow([trial.number, trial.state.name, v0, v1,
                             trial.user_attrs.get("actual_params", ""),
                             penalized,
                             trial.user_attrs.get("penalty_reason", ""),
                             json.dumps(trial.params, sort_keys=True)])
    print(f"[fixed-bo] selected trial {selected.number}; artifacts: {args.output}", flush=True)


if __name__ == "__main__":
    main()
