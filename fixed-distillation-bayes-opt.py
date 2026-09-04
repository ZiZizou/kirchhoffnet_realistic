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

import bo_param_sampling as bps


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


def _mlp_param_analytic(width: int, layers: int, layernorm: bool) -> int:
    """Analytic trainable-param count for the fixed-harness PlainMLP.

    Mirrors KirchhoffStudent/PlainMLPStudent construction: first layer 4->W
    (5W), subsequent (L-1) layers W->W (W^2+W), head 7W+7, plus 2WL for
    LayerNorm weight+bias when enabled. Matches harness --count-params-only
    (verified: 48x3 -> 5287, 64x3 -> 9095, etc.).
    """
    trunk = 5 * width + (layers - 1) * (width * width + width)
    head = 7 * width + 7
    ln = 2 * width * layers if layernorm else 0
    return trunk + head + ln


from functools import lru_cache

_feasible_knet_arches_cache: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}


def _get_feasible_knet_arches(low: int, high: int) -> list[tuple[int, int, int, int]]:
    """All (hidden, stages, k, rank) tuples whose param count lies in [low, high].

    Cached per budget window. For the paper budget 7000±15% this is 126 arches.
    """
    key = (low, high)
    if key in _feasible_knet_arches_cache:
        return _feasible_knet_arches_cache[key]
    arches: list[tuple[int, int, int, int]] = []
    for hs in range(8, 25):
        for stages in range(2, 6):
            for k in (2, 4, 6, 8):
                if k >= hs:
                    continue
                for rank in (1, 2, 3, 4):
                    try:
                        p = _knet_param_analytic(hs, stages, k, rank)
                    except Exception:
                        continue
                    if low <= p <= high:
                        arches.append((hs, stages, k, rank))
    _feasible_knet_arches_cache[key] = arches
    return arches


def _knet_param_analytic(num_hidden: int, num_stages: int,
                         small_world_k: int, vca_rank: int) -> int:
    """Fast analytic KNet param count by building the net (no training).

    Uses the same build_net_from_config path as the harness so the count is
    exact. Fanout and x_max do not affect param count except via num_hidden.
    Imported lazily so module import stays lightweight.
    """
    return _knet_param_analytic_cached(num_hidden, num_stages, small_world_k, vca_rank)


@lru_cache(maxsize=None)
def _knet_param_analytic_cached(num_hidden: int, num_stages: int,
                                small_world_k: int, vca_rank: int) -> int:
    # Ensure project root is on sys.path when called from importlib contexts.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from cell_library import make_cell_library
    from topology import build_net_from_config
    from config import SOLVER

    stage_t_span = SOLVER["t_span"] / num_stages
    stage_steps = max(1, int(round(SOLVER["num_steps"] / num_stages)))
    cfg = {
        "stages": [{
            "num_inputs": 4, "num_hidden": num_hidden, "num_proj": 0, "num_outputs": 0,
            "hidden_family": "small_world",
            "hidden_kwargs": {"k": small_world_k, "p": 0.2, "seed": 1, "bidirectional": False},
            "input_pattern": "all_to_all", "output_pattern": "all_to_all", "proj_pattern": "all_to_all",
            "edge_repeats": 2, "t_span": stage_t_span, "num_steps": stage_steps,
        } for _ in range(num_stages)],
        "out_dim": 7, "write_mode": "sparse_proj", "read_mode": "dense",
        "use_robust_input": False,
    }
    fanout = {i: [i, i + 4] for i in range(4)}
    net = build_net_from_config(
        cfg, cell_lib=make_cell_library("tanh_free"), leak_mode="non-programmable",
        freeze_read=True, interstage_activation="residual-relu-tanh",
        boundary_fan_out=fanout,
        enable_temporal_readout=True, x_max=4.0,
        vca_enabled=True, vca_rank=vca_rank, vca_core_enabled=True,
        vca_gate_shunt=False, vca_separate_core_bus=True, vca_bias=False,
    )
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


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


def sample_config(trial: optuna.Trial, kind: str,
                  low: int, high: int,
                  mlp_feasible: list | None = None,
                  knet_feasible: list | None = None) -> tuple[dict[str, str], str | None]:
    """Budget-aware sampler: every suggested architecture lies in [low, high].

    ``low, high`` are derived from --param-budget/--param-tolerance in main().
    For MLP the (layers, width, layernorm) tuple is sampled jointly from a
    precomputed feasible list (``mlp_feasible``) via a single fixed-distribution
    categorical. This replaces the old conditional per-(layers, layernorm)
    width sampling that produced a "CategoricalDistribution does not support
    dynamic value space" crash on trial 1 (RDBStorage rejects the second
    commit's distribution as incompatible with the first trial's). For KNet
    the (hidden, stages, k, rank) tuple is sampled jointly from the
    precomputed feasible list (``knet_feasible``); ``x_max`` and
    ``lr/batch_size`` remain independent (they do not move the param count).
    ``knet_feasible`` falls back to ``_get_feasible_knet_arches(low, high)``
    when not provided (e.g. legacy callers), but the main() path always
    threads the precomputed list to skip the per-trial rebuild.

    Returns ``(cfg, penalty_reason)`` where penalty_reason is None iff the
    configuration is feasible. Callers must branch on the return value, not on
    trial.user_attrs round-trips.
    """
    cfg: dict[str, str] = {
        "--lr": f"{trial.suggest_float('lr', 2e-4, 5e-3, log=True):.8g}",
        "--batch-size": str(trial.suggest_categorical("batch_size", [128, 256, 512, 1024])),
    }
    if kind == "mlp":
        if not mlp_feasible:
            # Should never trigger because ``main`` builds the list and
            # raises a fail-fast ``ValueError`` when empty. Guarded here so
            # trial-time can never silently regress to "no feasible tuple".
            return cfg, f"no MLP architecture in param window [{low}, {high}]"
        arch_idx = int(trial.suggest_categorical(
            "mlp_arch_idx", list(range(len(mlp_feasible)))))
        layers, width, layernorm = mlp_feasible[arch_idx]
        activation = trial.suggest_categorical(
            "student_activation", ["silu", "gelu"])
        cfg.update({
            "--student-width": str(width),
            "--student-layers": str(layers),
            "--student-activation": activation,
            "--student-use-layernorm": str(layernorm),
        })
        return cfg, None
    else:
        # KNet: joint feasible-architecture sampling guarantees every trial
        # lands inside [low, high]. The old per-dim uniform hidden sampling
        # hit the window only ~14% (MLP 1.5%) of the time and caused the
        # 30/30-penalized report. Joint sampling over the precomputed feasible
        # (hidden, stages, k, rank) tuples makes every trial feasible (0%
        # budget-penalized) while still letting TPE explore the full feasible
        # manifold. x_max and lr/batch_size remain independent (they don't
        # affect param count).
        feasible_arches = (knet_feasible
                           if knet_feasible is not None
                           else _get_feasible_knet_arches(low, high))
        if not feasible_arches:
            return cfg, f"no KNet architecture in param window [{low}, {high}]"
        arch_idx = int(trial.suggest_categorical("knet_arch_idx",
                                                 list(range(len(feasible_arches)))))
        hidden, num_stages, small_world_k, vca_rank = feasible_arches[arch_idx]
        x_max = trial.suggest_float('kn_x_max', 2.0, 6.0)
        fanout = {str(index): [index, index + 4] for index in range(4)}
        cfg.update({
            "--kn-num-hidden": str(hidden),
            "--kn-num-stages": str(num_stages),
            "--kn-small-world-k": str(small_world_k),
            "--kn-small-world-p": "0.2", "--kn-vca-rank": str(vca_rank),
            "--kn-x-max": f"{x_max:.6g}",
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

    # Compute the sampling fingerprint BEFORE opening the study so we can
    # reject incompatible resumes (RDBStorage stores distributions per trial;
    # changing budget/tolerance/kind/n_arches between runs causes the second
    # committed trial to crash with "CategoricalDistribution does not support
    # dynamic value space"). The fingerprint lives in ``study.user_attrs``.
    low = int(args.param_budget * (1.0 - args.param_tolerance))
    if args.student_kind == "mlp":
        mlp_feasible_pre = [
            (L, W, ln)
            for L in range(1, 6)
            for ln in (False, True)
            for W in range(16, 257)
            if low <= _mlp_param_analytic(W, L, ln) <= args.param_budget
        ]
        if not mlp_feasible_pre:
            raise ValueError(
                f"param window [{low}, {args.param_budget}] admits no MLP architecture "
                f"(width 16-256, layers 1-5, layernorm True/False). "
                "Widen --param-tolerance or adjust --param-budget."
            )
        feasible_count = len(mlp_feasible_pre)
        arch_param_name = "mlp_arch_idx"
    else:
        if not _get_feasible_knet_arches(low, args.param_budget):
            raise ValueError(
                f"param window [{low}, {args.param_budget}] admits no KNet architecture "
                f"(hidden 8-24, stages 2-5, k 2/4/6/8, rank 1-4). "
                "Widen --param-tolerance or adjust --param-budget."
            )
        feasible_count = len(_get_feasible_knet_arches(low, args.param_budget))
        arch_param_name = "knet_arch_idx"
    sampling_fingerprint = bps.make_sampling_fingerprint({
        "param_budget": args.param_budget,
        "param_tolerance": args.param_tolerance,
        "student_kind": args.student_kind,
        "n_arches": feasible_count,
        "arch_param_name": arch_param_name,
    })

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
        bps.check_sampling_fingerprint(study, sampling_fingerprint)
    else:
        # Without --resume we refuse to silently reuse an existing sqlite study
        # whose directions may not match. This prevents the inhomogeneous-shape
        # ValueError in optuna.samplers._tpe.sampler._calculate_weights_below_for_multi_objective.
        if (args.output / "study.db").exists():
            raise RuntimeError(
                f"study.db already exists at {storage}. Pass --resume to continue "
                "the existing study, or choose a fresh --output directory."
            )
        study = optuna.create_study(study_name=study_name, storage=storage,
                                    directions=expected_directions,
                                    sampler=TPESampler(seed=args.seed, n_startup_trials=args.n_startup_trials),
                                    load_if_exists=False)
        study.set_user_attr("sampling_fingerprint", sampling_fingerprint)

    # Resume fingerprint guard: any change to (budget, tolerance, kind, n_arches)
    # invalidates the SQLite distribution compatibility check on the second
    # committed trial ("CategoricalDistribution does not support dynamic value
    # space"). Refuse incompatible resumes with a clear remediation; the old
    # trials' distributions cannot be migrated in-place. Skipped on fresh
    # creation (fingerprint already set above).

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
    print(f"[fixed-bo] kind={args.student_kind} trials={args.n_trials} epochs={args.epochs} "
          f"param window=[{low}, {args.param_budget}] fixed_dataset={fixed_path}", flush=True)

    # The sampling_fingerprint and joint feasible list are computed BEFORE
    # study create/load (above) so the resume fingerprint guard runs before
    # the second commit would crash RDBStorage. The actual mlp_feasible used
    # by the objective is the same tuple built for the fingerprint; just
    # alias it here. Same for the KNet list: precompute once, thread through
    # the closure so the objective avoids a per-trial rebuild of
    # ``_get_feasible_knet_arches``.
    mlp_feasible = mlp_feasible_pre if args.student_kind == "mlp" else None
    knet_feasible = (None if args.student_kind == "mlp"
                     else _get_feasible_knet_arches(low, args.param_budget))
    if args.student_kind == "mlp":
        print(f"[fixed-bo] MLP feasible tuples: {len(mlp_feasible)} "
              f"(window=[{low}, {args.param_budget}])", flush=True)
    else:
        print(f"[fixed-bo] KNet feasible tuples: {len(knet_feasible)} "
              f"(window=[{low}, {args.param_budget}])", flush=True)

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        cfg, sample_penalty_reason = sample_config(
            trial, args.student_kind, low, args.param_budget,
            mlp_feasible=mlp_feasible, knet_feasible=knet_feasible)
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
