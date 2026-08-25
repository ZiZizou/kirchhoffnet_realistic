"""Optuna-driven Bayesian hyperparameter optimization for KirchhoffNet (KNet).

Tunes the topology + solver hyperparameters of ``train_script.py`` over
multiple optuna trials, while keeping a **fixed epoch budget** per trial
(default 800, override via ``--epochs``). Early stopping is disabled
(``--no-early-stop``) so the full fixed budget always runs.

Permanently-on flags (every trial):
    --freeze-read --temporal-readout --no-early-stop
    --leak non-programmable --vca --vca-core --vca-separate-core-bus
    --cell-library tanh_free --hidden-family small_world
    --boundary-fan-out <json>

Search dimensions (15 dims per trial):
    Topology:    num_hidden, small_world_k, num_stages, fanout_count
    Solver:      t_span (num_steps is derived at a fixed resolution)
    Optimizer:   lr, weight_decay, batch_size
    Physics:     x_max
    Cell bounds: gm_max, isat_max        (gm_min / isat_min fixed at config default)
    Regularizer: device_l2_lambda (sparsity=0, entropy=1e-6 fixed)
    Freeze:      freeze_boundary, freeze_temporal_read

Validity:
    num_hidden >= in_dim * fanout_count (enough distinct fanout targets)
    small_world_k even, 2 <= k < num_hidden, capped at 14
    TrialPruned on any invalid combo

Seed trial (trial 0):
    For datasets listed in ``START_POINTS`` (friedman1, friedman2, smooth2d)
    the user's validated 800-epoch config is enqueued as trial 0 via
    ``study.enqueue_trial``. Trial 0 therefore runs the exact boundary
    fan-out map, fixed flags, and config defaults that were used to
    produce the reference numbers; subsequent trials explore the 18-dim
    space around it.

    Fixed flags for friedman1 / friedman2 / smooth2d (all trials on those
    datasets):
        --solver heun
        --interstage-activation residual-relu-tanh
        --mapper-lr-scale 1.0  --struct-lr-scale 4  --dyn-lr-scale 1.0
        --grad-log --grad-log-every 10  --validate-every 10
        --seed 100

Each trial is a subprocess of ``train_script.py``, so the GPU/CPU isolation
of the original training script is preserved. Optuna ``n_jobs`` concurrently
spawns up to ``n_workers`` trial subprocesses; trials are pinned round-robin
to individual GPUs via ``CUDA_VISIBLE_DEVICES`` when CUDA is available.

Dataset-in / -out dimensions and per-dataset ``num_hidden`` range are
hardcoded in ``DATASETS``. Default ranges favor the existing small_world
configurations used in recent friedman2 grid runs.

Outputs (in ``--output/<dataset>_knet_e<E>/``):
    <study_name>.db   optuna sqlite (resumed automatically if it exists;
                       --resume is retained as a backwards-compatible flag)
    best_hyperparams.txt best config + metrics + actual param count
    results.csv        every trial: HPs + metrics + param_count
    objective_history.png  trial values + best-so-far curve
    trial_<NNNN>/log.txt   per-trial subprocess stdout/stderr
    trial_<NNNN>/final_metrics.txt (inherited from train_script.py)

CLI:
    python kn_bayes_opt.py --dataset friedman2 --epochs 800 --n-trials 30
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import optuna
import torch
import logging

_logger = logging.getLogger("kn_bayes_opt")
from optuna.samplers import TPESampler


DATASETS: dict[str, dict[str, Any]] = {
    "housing": {
        "in_dim": 8,
        "out_dim": 1,
        "num_hidden_range": (16, 32),
        "max_fanout_count": 2,
        "default_budget": 8000,
    },
    "smooth2d": {
        "in_dim": 2,
        "out_dim": 1,
        "num_hidden_range": (8, 24),
        "max_fanout_count": 2,
        "default_budget": 7000,
    },
    "friedman1": {
        "in_dim": 10,
        "out_dim": 1,
        "num_hidden_range": (20, 32),
        "max_fanout_count": 2,
        "default_budget": 7000,
    },
    "friedman2": {
        "in_dim": 4,
        "out_dim": 1,
        "num_hidden_range": (8, 32),
        "max_fanout_count": 2,
        "default_budget": 7000,
    },
    "friedman3": {
        "in_dim": 4,
        "out_dim": 1,
        "num_hidden_range": (8, 24),
        "max_fanout_count": 2,
        "default_budget": 7000,
    },
    "ctle": {
        "in_dim": 4,
        "out_dim": 7,
        "num_hidden_range": (10, 20),
        "max_fanout_count": 2,
        "default_budget": 6000,
    },
}

BATCH_SIZE_CHOICES = [512, 1024, 2048, 4096]
FANOUT_COUNT_CHOICES = [1, 2]
SMALL_WORLD_K_MAX = 14
SMALL_WORLD_K_CHOICES = (2, 4, 6, 8)
SMALL_WORLD_P_FIXED = 0.2
STEPS_PER_T_SPAN = 10.0
SPARSITY_LAMBDA_FIXED = 0.0
ENTROPY_LAMBDA_FIXED = 1e-6

START_POINTS: dict[str, dict[str, Any]] = {
    "friedman1": {
        "boundary_fan_out": {
            "0": [2, 12], "1": [7, 17], "2": [22, 5], "3": [10, 15],
            "4": [4, 14], "5": [8, 19], "6": [6, 16], "7": [9, 23],
            "8": [1, 13], "9": [11, 24],
        },
        "num_hidden": 25, "small_world_k": 4, "small_world_p": 0.2,
        "num_stages": 5, "fanout_count": 2,
        "t_span": 7.0, "num_steps": 70,
        "lr": 1.2e-3, "weight_decay": 1e-4, "batch_size": 4096,
        "x_max": 3.0, "gm_max": 10.0, "isat_max": 10.0,
        "sparsity_lambda": 0.0, "entropy_lambda": 1e-6,
        "device_l2_lambda": 0.0, "freeze_boundary": 0,
        "freeze_temporal_read": 0,
    },
    "friedman2": {
        "boundary_fan_out": {
            "0": [2, 12], "1": [7, 17], "2": [22, 5], "3": [10, 15],
        },
        "num_hidden": 25, "small_world_k": 4, "small_world_p": 0.2,
        "num_stages": 5, "fanout_count": 2,
        "t_span": 7.0, "num_steps": 70,
        "lr": 1.2e-3, "weight_decay": 1e-4, "batch_size": 4096,
        "x_max": 3.0, "gm_max": 10.0, "isat_max": 10.0,
        "sparsity_lambda": 0.0, "entropy_lambda": 1e-6,
        "device_l2_lambda": 0.0, "freeze_boundary": 0,
        "freeze_temporal_read": 0,
    },
    "smooth2d": {
        "boundary_fan_out": {
            "0": [0, 1], "1": [3, 4],
        },
        "num_hidden": 14, "small_world_k": 4, "small_world_p": 0.2,
        "num_stages": 10, "fanout_count": 2,
        "t_span": 7.0, "num_steps": 70,
        "lr": 1.2e-3, "weight_decay": 1e-4, "batch_size": 4096,
        "x_max": 3.0, "gm_max": 10.0, "isat_max": 10.0,
        "sparsity_lambda": 0.0, "entropy_lambda": 1e-6,
        "device_l2_lambda": 0.0, "freeze_boundary": 0,
        "freeze_temporal_read": 0,
    },
    "ctle": {
        # Seed is the current default dagger config (4 stages, 14 hidden, rank 2, t-span ~4.0 via SOLVER)
        # plus the Friedman2 winner t-span 7.12 as prior for BO to explore.
        "boundary_fan_out": {
            "0": [2, 4], "1": [1, 3], "2": [12, 5], "3": [7, 9],
        },
        "num_hidden": 14, "small_world_k": 4, "small_world_p": 0.2,
        "num_stages": 4, "fanout_count": 2,
        "t_span": 4.0, "num_steps": 40,
        "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 256,
        "x_max": 4.0, "gm_max": 10.0, "isat_max": 10.0,
        "sparsity_lambda": 0.0, "entropy_lambda": 1e-6,
        "device_l2_lambda": 0.0, "freeze_boundary": 0,
        "freeze_temporal_read": 0,
        "vca_rank": 2,
    },
}

EXTRA_FLAGS_FOR_PROBLEM: dict[str, list[str]] = {
    "friedman1": [
        "--solver", "heun",
        "--interstage-activation", "residual-relu-tanh",
        "--mapper-lr-scale", "1.0",
        "--struct-lr-scale", "4",
        "--dyn-lr-scale", "1.0",
        "--grad-log",
        "--grad-log-every", "10",
        "--validate-every", "10",
    ],
    "friedman2": [
        "--solver", "heun",
        "--interstage-activation", "residual-relu-tanh",
        "--mapper-lr-scale", "1.0",
        "--struct-lr-scale", "4",
        "--dyn-lr-scale", "1.0",
        "--grad-log",
        "--grad-log-every", "10",
        "--validate-every", "10",
    ],
    "smooth2d": [
        "--solver", "heun",
        "--interstage-activation", "residual-relu-tanh",
        "--mapper-lr-scale", "1.0",
        "--struct-lr-scale", "4",
        "--dyn-lr-scale", "1.0",
        "--grad-log",
        "--grad-log-every", "10",
        "--validate-every", "10",
    ],
}

OBJECTIVE_KEYS = {
    "best_val",
    "best_rmse_orig",
    "best_mse_orig",
    "best_mae_orig",
    "best_mape_orig",
}


def _penalized_objective(metric: float, actual_params: int,
                         reference_params: int, strength: float) -> float:
    """Scale the metric upward in proportion to the model size."""
    if actual_params < 0:
        return float("inf")
    normalized_params = actual_params / max(1, reference_params)
    return metric * (1.0 + strength * normalized_params)


def _over_budget_objective(actual_params: int, budget: int,
                           base: float) -> float:
    """Return a finite, graded objective for an over-budget architecture."""
    ratio = actual_params / max(1, budget)
    return base * ratio * ratio


def build_boundary_fan_out(in_dim: int, fanout_count: int, num_hidden: int) -> dict:
    """Fixed-spread boundary fanout map.

    Args:
        in_dim: number of input features.
        fanout_count: connections per input (1 or 2).
        num_hidden: hidden node count; targets must be < num_hidden.

    Returns:
        dict mapping each input index to a list of target node indices.

    With fanout_count=2 the pattern is ``input i -> [i, i + in_dim]`` (even
    spread across two columns of the hidden grid). With fanout_count=1 the
    pattern is ``input i -> [i]``. Targets are always unique across inputs
    when ``num_hidden >= in_dim * fanout_count``.
    """
    if fanout_count < 1 or fanout_count > 2:
        raise ValueError(f"fanout_count must be 1 or 2, got {fanout_count}")
    if num_hidden < in_dim * fanout_count:
        raise ValueError(
            f"num_hidden={num_hidden} too small for "
            f"in_dim={in_dim} * fanout_count={fanout_count}"
        )
    fanout: dict[int, list[int]] = {}
    for i in range(in_dim):
        if fanout_count == 2:
            fanout[i] = [i, i + in_dim]
        else:
            fanout[i] = [i]
    return fanout


def valid_small_world_k_choices(num_hidden: int) -> list[int]:
    """Meaningful even k values that are valid for ``num_hidden``."""
    return [k for k in SMALL_WORLD_K_CHOICES if k < num_hidden]


def _resolve_trial_dir(run_dir: Path, trial_number: int) -> Path | None:
    """Locate the actual trial output directory.

    ``train_script.py`` calls ``_ensure_dir`` which appends a timestamp
    suffix when the requested dir already exists. Locate the latest
    directory matching ``trial_NNNN*`` (preferring exact match if it
    exists).

    Returns:
        The resolved directory Path, or ``None`` if no matching dir found.
    """
    exact = run_dir / f"trial_{trial_number:04d}"
    if exact.is_dir() and (exact / "final_metrics.txt").exists():
        return exact
    candidates = sorted(
        run_dir.glob(f"trial_{trial_number:04d}*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _parse_final_metrics(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, val = line.partition(":")
            try:
                out[key.strip()] = float(val.strip())
            except ValueError:
                continue
    return out


def _parse_trainable_param_count(text: str) -> int | None:
    """Extract train_script.py's pre-training trainable-parameter count."""
    match = re.search(r"trainable params:\s*([0-9][0-9,]*)", text)
    if match is None:
        return None
    return int(match.group(1).replace(",", ""))


def _build_command(
    *,
    python: str,
    script: str,
    problem: str,
    seed: int,
    epochs: int,
    num_hidden: int,
    small_world_k: int,
    small_world_p: float,
    num_stages: int,
    t_span: float,
    num_steps: int,
    vca_rank: int,
    fanout_count: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    x_max: float,
    gm_max: float,
    isat_max: float,
    sparsity_lambda: float,
    entropy_lambda: float,
    device_l2_lambda: float,
    freeze_boundary: int,
    freeze_temporal_read: int,
    output: Path,
    device: str,
    boundary_fan_out: dict | None = None,
) -> list[str]:
    if boundary_fan_out is None:
        boundary_fan_out = build_boundary_fan_out(
            in_dim=DATASETS[problem]["in_dim"],
            fanout_count=fanout_count,
            num_hidden=num_hidden,
        )
    cmd = [
        python,
        script,
        "--problem", problem,
        "--epochs", str(epochs),
        "--num-hidden", str(num_hidden),
        "--small-world-k", str(small_world_k),
        "--small-world-p", f"{small_world_p:.6f}",
        "--num-stages", str(num_stages),
        "--t-span", f"{t_span:.6f}",
        "--num-steps", str(num_steps),
        "--vca-rank", str(vca_rank),
        "--boundary-fan-out", json.dumps(boundary_fan_out),
        "--cell-library", "tanh_free",
        "--leak", "non-programmable",
        "--temporal-readout",
        "--freeze-read",
        "--vca",
        "--vca-core",
        "--vca-separate-core-bus",
        "--no-early-stop",
        "--hidden-family", "small_world",
        "--lr", f"{lr:.6e}",
        "--weight-decay", f"{weight_decay:.6e}",
        "--batch-size", str(batch_size),
        "--x-max", f"{x_max:.6e}",
        "--gm-max", f"{gm_max:.6e}",
        "--isat-max", f"{isat_max:.6e}",
        "--sparsity-lambda", f"{sparsity_lambda:.6e}",
        "--entropy-lambda", f"{entropy_lambda:.6e}",
        "--device-l2-lambda", f"{device_l2_lambda:.6e}",
        "--device", device,
        "--output", str(output),
    ]
    if freeze_boundary:
        cmd += ["--freeze-boundary"]
    if freeze_temporal_read:
        cmd += ["--freeze-temporal-read"]
    if problem in EXTRA_FLAGS_FOR_PROBLEM:
        cmd += EXTRA_FLAGS_FOR_PROBLEM[problem]
    cmd += ["--seed", str(seed)]
    if problem.startswith("friedman"):
        cmd += ["--target-noise-std", "1.0"]
    return cmd


def _build_dagger_command(
    *,
    python: str,
    script: str,
    dagger_iterations: int,
    epochs_per_iter: int,
    common_eval_size: int,
    kn_num_stages: int,
    kn_num_hidden: int,
    kn_small_world_k: int,
    kn_small_world_p: float,
    kn_vca_rank: int,
    kn_x_max: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    output: Path,
    device: str,
    boundary_fan_out: dict | None = None,
    t_span: float | None = None,
    seed: int = 0,
) -> list[str]:
    if boundary_fan_out is None:
        # CTLE in_dim=4
        boundary_fan_out = build_boundary_fan_out(
            in_dim=4, fanout_count=2, num_hidden=kn_num_hidden
        )
    cmd = [
        python,
        script,
        "--dagger-iterations", str(dagger_iterations),
        "--epochs-per-iter", str(epochs_per_iter),
        "--common-eval-size", str(common_eval_size),
        "--kn-num-stages", str(kn_num_stages),
        "--kn-num-hidden", str(kn_num_hidden),
        "--kn-small-world-k", str(kn_small_world_k),
        "--kn-small-world-p", f"{kn_small_world_p:.6f}",
        "--kn-vca-rank", str(kn_vca_rank),
        "--kn-x-max", f"{kn_x_max:.6f}",
        "--boundary-fan-out", json.dumps(boundary_fan_out),
        "--lr", f"{lr:.6e}",
        "--weight-decay", f"{weight_decay:.6e}",
        "--batch-size", str(batch_size),
        "--output", str(output),
        "--device", device,
        "--seed", str(seed),
    ]
    if t_span is not None:
        cmd += ["--t-span", f"{t_span:.6f}"]
    return cmd


def _parse_dagger_test_failure(log_text: str) -> float | None:
    """Extract Test failure rate: X% from dagger log. Returns fraction 0-1."""
    # Prefer last occurrence (final test)
    matches = re.findall(r"Test failure rate:\s*([\d\.]+)%", log_text)
    if not matches:
        return None
    return float(matches[-1]) / 100.0


def _plot_history(study: optuna.Study, path: Path, *, title: str,
                  objective: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[kn_bayes_opt] matplotlib not installed; skipping plot")
        return
    vals: list[float] = []
    for t in study.trials:
        if t.value is None or t.value == float("inf"):
            continue
        vals.append(float(t.value))
    if not vals:
        return
    best_so_far: list[float] = []
    cur = float("inf")
    for v in vals:
        cur = min(cur, v)
        best_so_far.append(cur)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(vals, "o", color="C0", alpha=0.5, label="trial")
    ax.plot(best_so_far, "-", color="C3", label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel(objective)
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _recover_unfinished_trials(
    study: optuna.Study, dataset: str,
) -> tuple[list[dict[str, Any]], set[int]]:
    """Recover trials abandoned when the previous allocation expired.

    Optuna stores a trial as RUNNING while its objective is executing.  If
    Slurm kills the allocation, that state can remain in SQLite forever and
    ``study.optimize`` will not execute the trial again.  Convert those
    abandoned trials to FAIL and enqueue their exact parameters so the next
    allocation retries them.  The returned set identifies retries of the
    special START_POINTS trial, whose boundary map is not an Optuna param.

    This is intentionally a startup operation: a RUNNING trial is assumed to
    belong to an older allocation.  Do not start two jobs against the same
    study at the same time.
    """
    running = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.RUNNING
    ]
    if not running:
        return [], set()

    retry_params: list[dict[str, Any]] = []
    recovered_seed_indices: set[int] = set()
    for trial in running:
        params = dict(trial.params)
        if trial.user_attrs.get("seed_trial") is True:
            # boundary_fan_out is supplied by START_POINTS rather than
            # suggested by Optuna, so recover the complete seed configuration.
            retry_params.append(dict(START_POINTS[dataset]))
            recovered_seed_indices.add(len(retry_params) - 1)
        else:
            retry_params.append(params)

        try:
            # Optuna has no public transition for an already-running trial;
            # this is the storage-level operation used to finalize it.
            study._storage.set_trial_state_values(  # type: ignore[attr-defined]
                trial._trial_id, optuna.trial.TrialState.FAIL, None
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not mark abandoned trial {trial.number} as FAIL"
            ) from exc

    return retry_params, recovered_seed_indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--epochs", type=int, default=800,
                        help="Fixed epoch budget per trial (default: 800).")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=None,
                        help="Optional wall-clock timeout in seconds.")
    parser.add_argument("--seed", type=int, default=100,
                        help="Seed used for both the Optuna TPE sampler and "
                             "every train_script.py subprocess (last-wins). "
                             "Overrides per-problem defaults. "
                             "Default: 100 (matches validated START_POINTS "
                             "runs).")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Concurrent trial subprocesses. Default: number "
                             "of visible CUDA GPUs when available, else "
                             "os.cpu_count(). Trials are pinned round-robin "
                             "to GPUs via CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Device 'auto' | 'cpu' | 'cuda' (default: auto).")
    parser.add_argument("--objective", default="best_val",
                        choices=sorted(OBJECTIVE_KEYS),
                        help="Metric to minimize (default: best_val).")
    parser.add_argument("--param-penalty", type=float, default=0.25,
                        help="Dimensionless multiplier for the parameter-count "
                             "penalty (default: 0.25; 0 disables it).")
    parser.add_argument("--param-budget", type=int, default=None,
                        help="Hard maximum on trainable parameters. "
                              "Defaults to per-dataset budget (housing 8000, "
                              "friedman1/2/3 7000, smooth2d 7000, ctle 6000); "
                              "trials exceeding budget*(1+tolerance) are "
                              "rejected via an upfront --count-params-only "
                              "preflight without training.")
    parser.add_argument("--param-tolerance", type=float, default=0.10,
                        help="Allowed fractional overage above param-budget "
                             "before rejection (default: 0.10 = 10%%).")
    parser.add_argument("--invalid-param-objective", type=float, default=1e6,
                        help="Base objective for over-budget models. Their "
                             "finite penalty is this value times the squared "
                             "parameter-count/budget ratio (default: 1e6).")
    parser.add_argument("--param-reference", type=int, default=None,
                        help="Parameter count corresponding to normalized size "
                             "1.0 for the penalty (default: param-budget or 10000).")
    parser.add_argument("--num-hidden-min", type=int, default=None,
                        help="Override min num_hidden (default: "
                             "max(default_min_from_DATASETS, "
                             "in_dim * fanout_count_min)).")
    parser.add_argument("--num-hidden-max", type=int, default=None,
                        help="Override max num_hidden (default: per-dataset).")
    parser.add_argument("--t-span-min", type=float, default=0.5)
    parser.add_argument("--t-span-max", type=float, default=10.0)
    parser.add_argument("--num-steps-min", type=int, default=10)
    parser.add_argument("--num-steps-max", type=int, default=150)
    parser.add_argument("--vca-rank-min", type=int, default=1,
                        help="Minimum VCA projection rank (default: 1).")
    parser.add_argument("--vca-rank-max", type=int, default=8,
                        help="Maximum VCA projection rank (default: 8).")
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--lr-max", type=float, default=1e-2)
    parser.add_argument("--wd-min", type=float, default=1e-6)
    parser.add_argument("--wd-max", type=float, default=1e-2)
    parser.add_argument("--num-stages-max", type=int, default=10,
                        help="Upper bound on num_stages (default: 10; smooth2d "
                             "start = 10).")
    parser.add_argument("--x-max-min", type=float, default=0.5,
                        help="Lower bound for x_max (ODE rail) search "
                             "(default: 0.5; config default is 3.0).")
    parser.add_argument("--x-max-max", type=float, default=8.0,
                        help="Upper bound for x_max (ODE rail) search "
                             "(default: 8.0; config default is 3.0).")
    parser.add_argument("--gm-max-min", type=float, default=1.0,
                        help="Lower bound for log-uniform gm_max search "
                             "(default: 1.0; config default is 10.0). gm_min "
                             "is fixed at config default (0.01).")
    parser.add_argument("--gm-max-max", type=float, default=50.0,
                        help="Upper bound for log-uniform gm_max search "
                             "(default: 50.0).")
    parser.add_argument("--isat-max-min", type=float, default=1.0,
                        help="Lower bound for log-uniform isat_max search "
                             "(default: 1.0; config default is 10.0). "
                             "isat_min is fixed at config default (0.01).")
    parser.add_argument("--isat-max-max", type=float, default=50.0,
                        help="Upper bound for log-uniform isat_max search "
                             "(default: 50.0).")
    parser.add_argument("--sparsity-lambda-min", type=float, default=1e-7,
                        help="Lower bound for log-uniform sparsity lambda "
                             "search (default: 1e-7).")
    parser.add_argument("--sparsity-lambda-max", type=float, default=1e-3,
                        help="Upper bound for log-uniform sparsity lambda "
                             "search (default: 1e-3).")
    parser.add_argument("--entropy-lambda-min", type=float, default=1e-6,
                        help="Lower bound for log-uniform entropy lambda "
                             "search (default: 1e-6).")
    parser.add_argument("--entropy-lambda-max", type=float, default=1e-2,
                        help="Upper bound for log-uniform entropy lambda "
                             "search (default: 1e-2).")
    parser.add_argument("--device-l2-lambda-min", type=float, default=0.0,
                        help="Deprecated; ignored. The device_l2_lambda search "
                             "range is always linear [0.0, --device-l2-lambda-max] "
                             "so that 0.0 (penalty off) is representable. "
                             "Default: 0.0.")
    parser.add_argument("--device-l2-lambda-max", type=float, default=1e-3,
                        help="Upper bound for linear device_l2_lambda "
                             "search (default: 1e-3).")
    parser.add_argument("--output", type=Path,
                        default=Path("./outputs/kn_bayes_opt"))
    parser.add_argument("--study-name", default=None,
                        help="Optuna study name. Default: "
                             "<dataset>_knet_e<E>.")
    parser.add_argument("--resume", action="store_true",
                        help="Accepted for backwards compatibility. Existing "
                             "studies are resumed automatically.")
    parser.add_argument("--no-seed-trial", action="store_true",
                        help="Skip enqueueing the START_POINTS seed trial "
                             "(trial 0 explores the search space from a "
                             "random TPE sample instead).")
    args = parser.parse_args()

    if args.param_budget is not None and args.param_budget < 1:
        raise ValueError("--param-budget must be >= 1")
    if args.param_reference is not None and args.param_reference < 1:
        raise ValueError("--param-reference must be >= 1")
    if args.param_tolerance < 0.0:
        raise ValueError("--param-tolerance must be >= 0")
    if args.vca_rank_min < 1 or args.vca_rank_min > args.vca_rank_max:
        raise ValueError("Invalid VCA rank range")
    if args.invalid_param_objective <= 0.0:
        raise ValueError("--invalid-param-objective must be > 0")

    cfg = DATASETS[args.dataset]
    in_dim: int = cfg["in_dim"]
    out_dim: int = cfg["out_dim"]
    default_min, default_max = cfg["num_hidden_range"]
    num_hidden_min = (
        args.num_hidden_min if args.num_hidden_min is not None
        else max(default_min, in_dim * min(FANOUT_COUNT_CHOICES))
    )
    num_hidden_max = (
        args.num_hidden_max if args.num_hidden_max is not None
        else default_max
    )
    if num_hidden_min > num_hidden_max:
        raise ValueError(
            f"--num-hidden-min ({num_hidden_min}) > "
            f"--num-hidden-max ({num_hidden_max})"
        )

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / f"{args.dataset}_knet_e{args.epochs}"
    run_dir.mkdir(parents=True, exist_ok=True)

    study_name = args.study_name or f"{args.dataset}_knet_e{args.epochs}"
    storage = f"sqlite:///{run_dir / (study_name + '.db')}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            print("[kn_bayes_opt] WARNING: --device cuda requested but no CUDA "
                  "GPU detected; falling back to CPU")
            device = "cpu"

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if args.n_workers is None:
        if device == "cuda" and n_gpus >= 1:
            n_workers = n_gpus
        else:
            n_workers = max(1, os.cpu_count() or 1)
    else:
        n_workers = max(1, args.n_workers)

    db_path = run_dir / (study_name + ".db")
    sampler = TPESampler(seed=args.seed)
    study_was_resumed = db_path.exists()
    if study_was_resumed:
        study = optuna.load_study(study_name=study_name, storage=storage,
                                  sampler=sampler)
        retry_params, recovered_seed_indices = _recover_unfinished_trials(
            study, args.dataset
        )
        if retry_params:
            print(f"[kn_bayes_opt] recovered {len(retry_params)} unfinished "
                  "trial(s); they will be retried before new trials")
        else:
            recovered_seed_indices = set()
    else:
        study = optuna.create_study(study_name=study_name, storage=storage,
                                    sampler=sampler, direction="minimize")
        retry_params = []
        recovered_seed_indices = set()

    # Enqueue recovered parameter sets after the old RUNNING rows have been
    # finalized. Optuna will assign each retry a new trial number.
    for retry_index, params in enumerate(retry_params):
        study.enqueue_trial(
            params,
            user_attrs={
                "recovered_trial": True,
                "recovered_seed_trial": retry_index in recovered_seed_indices,
            },
        )

    if (args.dataset in START_POINTS
            and not args.no_seed_trial):
        seed_already_complete = any(
            t.user_attrs.get("seed_trial") is True
            and t.state == optuna.trial.TrialState.COMPLETE
            for t in study.trials
        )
        seed_already_queued = any(
            t.state == optuna.trial.TrialState.WAITING
            and (
                t.user_attrs.get("seed_trial") is True
                or t.user_attrs.get("recovered_seed_trial") is True
            )
            for t in study.trials
        )
        if not seed_already_complete and not seed_already_queued:
            study.enqueue_trial(
                START_POINTS[args.dataset],
                user_attrs={"seed_trial": True},
            )

    repo_dir = Path(__file__).resolve().parent
    script_path = repo_dir / "train_script.py"
    python_exe = sys.executable
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    print(f"[kn_bayes_opt] dataset={args.dataset} in_dim={in_dim} out_dim={out_dim} "
          f"num_hidden_range=({num_hidden_min},{num_hidden_max}) "
          f"epochs={args.epochs} n_trials={args.n_trials} n_workers={n_workers} "
          f"objective={args.objective} device={device}")
    print(f"[kn_bayes_opt] cuda_available={torch.cuda.is_available()} "
          f"n_gpus={n_gpus} study_name={study_name} storage={storage}")
    print(f"[kn_bayes_opt] run_dir={run_dir}")
    print(f"[kn_bayes_opt] script={script_path}")
    print(f"[kn_bayes_opt] study_resumed={study_was_resumed}")
    # Resolve effective param budget: CLI override > per-dataset default.
    # This makes the upfront --count-params-only gate active for every
    # dataset including housing (previously housing had no default budget).
    param_budget: int | None = (
        args.param_budget if args.param_budget is not None
        else cfg.get("default_budget")
    )
    # param_reference: explicit CLI > resolved budget > 10000 fallback.
    if args.param_reference is not None:
        param_reference = args.param_reference
    elif param_budget is not None:
        param_reference = param_budget
    else:
        param_reference = 10000
    param_limit: float | None = (
        param_budget * (1.0 + args.param_tolerance)
        if param_budget is not None else None
    )
    print(f"[kn_bayes_opt] param_budget={param_budget} "
          f"param_limit={param_limit} param_reference={param_reference} "
          f"param_tolerance={args.param_tolerance}")

    def objective(trial: optuna.Trial) -> float:
        is_seed_trial = (
            trial.user_attrs.get("seed_trial") is True
            or trial.user_attrs.get("recovered_seed_trial") is True
        )

        # ── CTLE fast DAgger proxy (4×100, Test 1000 objective) ──────────
        if args.dataset == "ctle":
            # Use defaults as seed-trial, otherwise sample 4×100 proxy space.
            if is_seed_trial:
                sp = START_POINTS["ctle"]
                fanout_count = sp["fanout_count"]
                num_hidden = sp["num_hidden"]
                small_world_k = sp["small_world_k"]
                small_world_p = sp["small_world_p"]
                num_stages = sp["num_stages"]
                t_span = sp["t_span"]
                vca_rank = sp.get("vca_rank", 2)
                lr = sp["lr"]
                weight_decay = sp["weight_decay"]
                batch_size = sp["batch_size"]
                x_max = sp["x_max"]
                seed_boundary_map = sp["boundary_fan_out"]
            else:
                fanout_count = trial.suggest_categorical("fanout_count", FANOUT_COUNT_CHOICES)
                min_hidden_for_fanout = in_dim * fanout_count
                nh_low = max(num_hidden_min, min_hidden_for_fanout)
                if nh_low > num_hidden_max:
                    raise optuna.TrialPruned(f"fanout_count={fanout_count} requires num_hidden >= {min_hidden_for_fanout}, but max={num_hidden_max}")
                num_hidden = trial.suggest_int("num_hidden", nh_low, num_hidden_max)
                small_world_k = trial.suggest_categorical("small_world_k", SMALL_WORLD_K_CHOICES)
                if small_world_k >= num_hidden:
                    raise optuna.TrialPruned(f"small_world_k={small_world_k} must be < num_hidden={num_hidden}")
                small_world_p = SMALL_WORLD_P_FIXED
                num_stages = trial.suggest_int("num_stages", 1, args.num_stages_max)
                t_span = trial.suggest_float("t_span", args.t_span_min, args.t_span_max)
                vca_rank = trial.suggest_int("vca_rank", args.vca_rank_min, args.vca_rank_max)
                lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
                weight_decay = trial.suggest_float("weight_decay", args.wd_min, args.wd_max, log=True)
                batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES)
                x_max = trial.suggest_float("x_max", args.x_max_min, args.x_max_max)
                seed_boundary_map = None

            dagger_iterations = 4
            epochs_per_iter = 100
            common_eval_size = 1000
            # CTLE uses fixed 4-dim spec, so build_boundary_fan_out needs in_dim=4
            trial_dir = run_dir / f"trial_{trial.number:04d}"
            log_path = run_dir / f"trial_{trial.number:04d}.log.txt"
            dagger_script = str((Path(__file__).parent / "dagger-nuance-distillation-kirchhoffnet.py").resolve())
            cmd = _build_dagger_command(
                python=python_exe, script=dagger_script,
                dagger_iterations=dagger_iterations, epochs_per_iter=epochs_per_iter,
                common_eval_size=common_eval_size,
                kn_num_stages=num_stages, kn_num_hidden=num_hidden,
                kn_small_world_k=small_world_k, kn_small_world_p=small_world_p,
                kn_vca_rank=vca_rank, kn_x_max=x_max,
                lr=lr, weight_decay=weight_decay, batch_size=batch_size,
                output=trial_dir, device=device,
                boundary_fan_out=seed_boundary_map, t_span=t_span, seed=args.seed,
            )
            # seed-trial attrs
            if is_seed_trial:
                trial.set_user_attr("seed_trial", True)
                trial.set_user_attr("start_point", json.dumps(START_POINTS["ctle"]))
            else:
                trial.set_user_attr("seed_trial", False)
            # GPU pinning
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            sub_env = os.environ.copy()
            sub_env["PYTHONIOENCODING"] = "utf-8"
            if device == "cuda" and n_gpus >= 1:
                gpu_idx = trial.number % n_gpus
                sub_env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            # Run dagger (4×100)
            trial_dir.mkdir(parents=True, exist_ok=True)
            print(f"[ctle] trial {trial.number}: {' '.join(cmd)}", flush=True)
            with open(log_path, "w", encoding="utf-8") as logf:
                logf.write(f"$ {' '.join(cmd)}\n")
                logf.flush()
                proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True, cwd=str(Path(__file__).parent), env=sub_env)
            if proc.returncode != 0:
                print(f"[ctle] trial {trial.number} subprocess failed (code {proc.returncode})", flush=True)
                return float("inf")
            log_text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
            test_rate = _parse_dagger_test_failure(log_text)
            if test_rate is None:
                print(f"[ctle] trial {trial.number} could not parse Test failure rate", flush=True)
                return float("inf")
            # Parse param count for penalty (optional)
            param_count = _parse_trainable_param_count(log_text)
            trial.set_user_attr("test_failure_rate", test_rate)
            trial.set_user_attr("param_count", param_count if param_count is not None else 0)
            # Penalized objective (same as kn_bayes for friedman)
            # Upfront gate for CTLE: use --count-params-only would require
            # dagger preflight; for now we gate after the run (CTLE runs are
            # 4x100 fast proxy). Also enforce hard gate before penalty.
            base_metric = test_rate  # minimize Test 1000
            if param_count is not None and param_budget is not None:
                # over-budget -> finite graded penalty, no extra training waste
                # for subsequent trials (TPE learns to avoid large configs)
                if param_limit is not None and param_count > param_limit:
                    ratio = param_count / max(1, param_limit)
                    return float(args.invalid_param_objective) * (ratio ** 2)
            penalized = _penalized_objective(base_metric, param_count or 0, param_reference, args.param_penalty)
            trial.set_user_attr("raw_test_rate", base_metric)
            trial.set_user_attr("penalized", penalized)
            print(f"[ctle] trial {trial.number} Test {test_rate*100:.2f}% penalized {penalized*100:.2f}%", flush=True)
            return penalized

        fanout_count = trial.suggest_categorical("fanout_count",
                                                  FANOUT_COUNT_CHOICES)
        min_hidden_for_fanout = in_dim * fanout_count
        nh_low = max(num_hidden_min, min_hidden_for_fanout)
        if nh_low > num_hidden_max:
            raise optuna.TrialPruned(
                f"fanout_count={fanout_count} requires "
                f"num_hidden >= {min_hidden_for_fanout}, "
                f"but max={num_hidden_max}"
            )
        num_hidden = trial.suggest_int("num_hidden", nh_low, num_hidden_max)
        # Keep this distribution fixed across all trials. Optuna/RDBStorage
        # rejects a parameter whose categorical choices change between trials
        # (for example, num_hidden=8 makes k=8 invalid while num_hidden=25
        # allows it). Invalid combinations are pruned below instead.
        small_world_k = trial.suggest_categorical(
            "small_world_k", SMALL_WORLD_K_CHOICES)
        if small_world_k >= num_hidden:
            raise optuna.TrialPruned(
                f"small_world_k={small_world_k} must be < num_hidden={num_hidden}"
            )
        small_world_p = SMALL_WORLD_P_FIXED
        num_stages = trial.suggest_int(
            "num_stages", 1, args.num_stages_max)
        t_span = trial.suggest_float(
            "t_span", args.t_span_min, args.t_span_max)
        num_steps = max(1, round(STEPS_PER_T_SPAN * t_span))
        vca_rank = trial.suggest_int(
            "vca_rank", args.vca_rank_min, args.vca_rank_max)
        lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        weight_decay = trial.suggest_float("weight_decay",
                                           args.wd_min, args.wd_max,
                                           log=True)
        batch_size = trial.suggest_categorical("batch_size",
                                                BATCH_SIZE_CHOICES)

        x_max = trial.suggest_float(
            "x_max", args.x_max_min, args.x_max_max)
        gm_max = trial.suggest_float(
            "gm_max", args.gm_max_min, args.gm_max_max, log=True)
        isat_max = trial.suggest_float(
            "isat_max", args.isat_max_min, args.isat_max_max, log=True)
        sparsity_lambda = SPARSITY_LAMBDA_FIXED
        entropy_lambda = ENTROPY_LAMBDA_FIXED
        device_l2_lambda = trial.suggest_float(
            "device_l2_lambda", 0.0, args.device_l2_lambda_max)
        freeze_boundary = trial.suggest_categorical(
            "freeze_boundary", [0, 1])
        freeze_temporal_read = trial.suggest_categorical(
            "freeze_temporal_read", [0, 1])

        seed_boundary_map: dict | None = None
        if is_seed_trial:
            seed_boundary_map = START_POINTS[args.dataset]["boundary_fan_out"]

        trial_dir = run_dir / f"trial_{trial.number:04d}"
        log_path = run_dir / f"trial_{trial.number:04d}.log.txt"
        resolved_trial_dir: Path | None = None

        cmd = _build_command(
            python=python_exe, script=str(script_path),
            problem=args.dataset, seed=args.seed,
            epochs=args.epochs,
            num_hidden=num_hidden, small_world_k=small_world_k,
            small_world_p=small_world_p, num_stages=num_stages,
            t_span=t_span, num_steps=num_steps, vca_rank=vca_rank,
            fanout_count=fanout_count, lr=lr, weight_decay=weight_decay,
            batch_size=batch_size, x_max=x_max,
            gm_max=gm_max, isat_max=isat_max,
            sparsity_lambda=sparsity_lambda,
            entropy_lambda=entropy_lambda,
            device_l2_lambda=device_l2_lambda,
            freeze_boundary=freeze_boundary,
            freeze_temporal_read=freeze_temporal_read,
            output=trial_dir, device=device,
            boundary_fan_out=seed_boundary_map,
        )
        if is_seed_trial:
            trial.set_user_attr("seed_trial", True)
            trial.set_user_attr("start_point",
                                json.dumps(START_POINTS[args.dataset]))
        else:
            trial.set_user_attr("seed_trial", False)

        sub_env = os.environ.copy()
        sub_env.setdefault("PYTHONIOENCODING", "utf-8")
        sub_env.setdefault("PYTHONUTF8", "1")
        gpu_idx = -1
        if device == "cuda" and n_gpus >= 1:
            gpu_idx = trial.number % n_gpus
            sub_env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        preflight_cmd = cmd.copy()
        preflight_output = trial_dir / "_preflight"
        output_arg_index = preflight_cmd.index("--output") + 1
        preflight_cmd[output_arg_index] = str(preflight_output)
        # Keep the same auto-detected device for the preflight.  Appending a
        # second ``--device cpu`` here used to override the selected CUDA
        # device on Alliance, making the run appear CPU-only.
        preflight_cmd += ["--count-params-only"]
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write(f"$ {' '.join(preflight_cmd)}\n")
            logf.flush()
            preflight = subprocess.run(
                preflight_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=str(repo_dir), creationflags=creationflags,
                env=sub_env)
            logf.write(preflight.stdout)
            logf.flush()
        preflight_params = _parse_trainable_param_count(preflight.stdout)
        if preflight.returncode != 0 or preflight_params is None:
            trial.set_user_attr("subprocess_failed", True)
            trial.set_user_attr("preflight_failed", True)
            trial.set_user_attr("preflight_returncode", preflight.returncode)
            return float("inf")
        trial.set_user_attr("preflight_params", preflight_params)
        # Upfront gate: reject over-budget configs before training.
        # Uses --count-params-only preflight so housing and all other
        # datasets are checked identically; no training time is wasted.
        if (param_limit is not None and preflight_params > param_limit):
            invalid_objective = _over_budget_objective(
                preflight_params, int(param_limit),
                args.invalid_param_objective)
            trial.set_user_attr("param_budget_exceeded", True)
            trial.set_user_attr("param_budget", param_budget)
            trial.set_user_attr("param_limit", param_limit)
            trial.set_user_attr("actual_params", preflight_params)
            trial.set_user_attr("normalized_param_count",
                                preflight_params / max(1, param_budget))
            trial.set_user_attr("invalid_param_objective", invalid_objective)
            print(f"[kn_bayes_opt] trial {trial.number:04d} rejected before "
                  f"training: params={preflight_params} > "
                  f"param_limit={param_limit:.0f}; "
                  f"objective={invalid_objective:.6g}")
            return invalid_objective

        t0 = time.time()
        with open(log_path, "a", encoding="utf-8") as logf:
            logf.write("[preflight completed; launching training]\n")
            logf.write(f"$ {' '.join(cmd)}\n")
            logf.flush()
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                  text=True, cwd=str(repo_dir),
                                  creationflags=creationflags,
                                  env=sub_env)
        elapsed = time.time() - t0

        resolved_trial_dir = _resolve_trial_dir(run_dir, trial.number)
        metrics_path = (
            resolved_trial_dir / "final_metrics.txt"
            if resolved_trial_dir is not None
            else trial_dir / "final_metrics.txt"
        )
        if proc.returncode != 0 or not metrics_path.exists():
            trial.set_user_attr("subprocess_failed", True)
            trial.set_user_attr("subprocess_returncode", proc.returncode)
            trial.set_user_attr("subprocess_seconds", elapsed)
            if resolved_trial_dir is not None:
                trial.set_user_attr("resolved_trial_dir",
                                    str(resolved_trial_dir))
            return float("inf")

        metrics = _parse_final_metrics(metrics_path)
        if args.objective not in metrics:
            trial.set_user_attr("subprocess_failed", True)
            trial.set_user_attr("subprocess_seconds", elapsed)
            return float("inf")

        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        actual_params = int(metrics.get("param_count", -1))
        if (param_limit is not None and actual_params > param_limit):
            invalid_objective = _over_budget_objective(
                actual_params, int(param_limit),
                args.invalid_param_objective)
            trial.set_user_attr("param_budget_exceeded", True)
            trial.set_user_attr("param_budget", param_budget)
            trial.set_user_attr("param_limit", param_limit)
            trial.set_user_attr("actual_params", actual_params)
            trial.set_user_attr("normalized_param_count",
                                actual_params / max(1, param_budget))
            trial.set_user_attr("invalid_param_objective", invalid_objective)
            print(f"[kn_bayes_opt] trial {trial.number:04d} rejected after "
                  f"training: params={actual_params} > "
                  f"param_limit={param_limit:.0f}; "
                  f"objective={invalid_objective:.6g}")
            return invalid_objective
        raw_objective = float(metrics[args.objective])
        normalized_params = actual_params / max(1, param_reference)
        objective_value = _penalized_objective(
            raw_objective, actual_params, param_reference,
            args.param_penalty)
        trial.set_user_attr("actual_params", actual_params)
        trial.set_user_attr("raw_objective", raw_objective)
        trial.set_user_attr("normalized_param_count", normalized_params)
        trial.set_user_attr("param_penalty", objective_value - raw_objective)
        if resolved_trial_dir is not None:
            trial.set_user_attr("resolved_trial_dir",
                                str(resolved_trial_dir))
        trial.set_user_attr("subprocess_seconds", elapsed)
        trial.set_user_attr("gpu", gpu_idx)
        if seed_boundary_map is not None:
            trial.set_user_attr("boundary_fan_out",
                                json.dumps(seed_boundary_map))
        else:
            trial.set_user_attr("boundary_fan_out",
                                json.dumps(build_boundary_fan_out(
                                    in_dim=in_dim,
                                    fanout_count=fanout_count,
                                    num_hidden=num_hidden)))
        seed_tag = " [SEED]" if is_seed_trial else ""
        print(f"[kn_bayes_opt] trial {trial.number:04d}{seed_tag} "
              f"nh={num_hidden} k={small_world_k} p={small_world_p:.2f} "
              f"st={num_stages} ts={t_span:.2f} ns={num_steps} "
              f"vca_rank={vca_rank} "
              f"fc={fanout_count} lr={lr:.2e} wd={weight_decay:.2e} "
              f"bs={batch_size} xmx={x_max:.2f} gm={gm_max:.2f} "
              f"isat={isat_max:.2f} d2l={device_l2_lambda:.2e} "
              f"fb={freeze_boundary} ftr={freeze_temporal_read} "
              f"params={actual_params} "
              f"gpu={gpu_idx} -> {args.objective}={raw_objective:.6f} "
              f"penalized={objective_value:.6f} "
              f"({elapsed:.1f}s)")
        return objective_value

    study.optimize(objective, n_trials=args.n_trials, n_jobs=n_workers,
                   timeout=args.timeout, show_progress_bar=False)

    completed = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"[kn_bayes_opt] done. {len(completed)}/{len(study.trials)} trials "
          "completed.")
    if completed:
        print(f"[kn_bayes_opt] best_value={study.best_value:.6f}")
        print(f"[kn_bayes_opt] best_params={study.best_params}")

    with open(run_dir / "best_hyperparams.txt", "w") as f:
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"in_dim: {in_dim}\n")
        f.write(f"out_dim: {out_dim}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"objective: {args.objective}\n")
        f.write(f"param_penalty: {args.param_penalty}\n")
        f.write(f"param_budget: {param_budget}\n")
        f.write(f"param_tolerance: {args.param_tolerance}\n")
        f.write(f"param_limit: {param_limit}\n")
        f.write(f"invalid_param_objective: {args.invalid_param_objective}\n")
        f.write(f"param_reference: {param_reference}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"n_trials: {args.n_trials}\n")
        f.write(f"n_workers: {n_workers}\n")
        f.write(f"device: {device}\n")
        f.write(f"n_gpus: {n_gpus}\n")
        f.write(f"search_dims: 15 (fanout_count, num_hidden, small_world_k, "
                "num_stages, t_span, lr, "
                "vca_rank, "
                "weight_decay, batch_size, x_max, gm_max, isat_max, "
                "device_l2_lambda, "
                "freeze_boundary, freeze_temporal_read)\n")
        f.write(f"has_start_point: {args.dataset in START_POINTS}\n")
        if args.dataset in START_POINTS:
            f.write(f"start_point: {json.dumps(START_POINTS[args.dataset])}\n")
            f.write(f"extra_flags_for_problem: "
                    f"{json.dumps(EXTRA_FLAGS_FOR_PROBLEM.get(args.dataset, []))}\n")
        if completed:
            bt = study.best_trial
            f.write(f"best_trial_number: {bt.number}\n")
            f.write(f"best_trial_is_seed: {bt.user_attrs.get('seed_trial', False)}\n")
            f.write(f"best_value: {study.best_value:.6f}\n")
            f.write(f"actual_params: {bt.user_attrs.get('actual_params')}\n")
            f.write(f"raw_objective: {bt.user_attrs.get('raw_objective')}\n")
            f.write(f"normalized_param_count: {bt.user_attrs.get('normalized_param_count')}\n")
            f.write(f"subprocess_seconds: "
                    f"{bt.user_attrs.get('subprocess_seconds')}\n")
            f.write(f"gpu: {bt.user_attrs.get('gpu', -1)}\n")
            f.write(f"boundary_fan_out: {bt.user_attrs.get('boundary_fan_out')}\n")
            f.write("params:\n")
            for k, v in bt.params.items():
                f.write(f"  {k}: {v}\n")
            f.write("metrics:\n")
            for k in OBJECTIVE_KEYS | {"param_count", "epochs_run",
                                       "elapsed_seconds"}:
                if k in bt.user_attrs:
                    f.write(f"  {k}: {bt.user_attrs[k]}\n")

    csv_path = run_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trial", "state", "seed_trial", "num_hidden", "small_world_k",
            "small_world_p", "num_stages", "t_span", "num_steps",
            "vca_rank", "fanout_count", "boundary_fan_out", "lr", "weight_decay",
            "batch_size", "x_max", "gm_max", "isat_max", "sparsity_lambda",
            "entropy_lambda", "device_l2_lambda", "freeze_boundary",
            "freeze_temporal_read", "device", "gpu", "objective",
            "objective_value", "actual_params", "best_val", "best_epoch",
            "best_rmse_orig", "best_mae_orig", "best_mape_orig",
            "epochs_run", "elapsed_seconds", "raw_objective",
            "normalized_param_count", "param_penalty",
            "invalid_param_objective",
        ])
        for t in study.trials:
            w.writerow([
                t.number, t.state.name,
                t.user_attrs.get("seed_trial", False),
                t.params.get("num_hidden"),
                t.params.get("small_world_k"),
                SMALL_WORLD_P_FIXED,
                t.params.get("num_stages"),
                t.params.get("t_span"),
                (max(1, round(STEPS_PER_T_SPAN * t.params.get("t_span")))
                 if t.params.get("t_span") is not None else None),
                t.params.get("vca_rank"),
                t.params.get("fanout_count"),
                t.user_attrs.get("boundary_fan_out"),
                t.params.get("lr"),
                t.params.get("weight_decay"),
                t.params.get("batch_size"),
                t.params.get("x_max"),
                t.params.get("gm_max"),
                t.params.get("isat_max"),
                SPARSITY_LAMBDA_FIXED,
                ENTROPY_LAMBDA_FIXED,
                t.params.get("device_l2_lambda"),
                t.params.get("freeze_boundary"),
                t.params.get("freeze_temporal_read"),
                device, t.user_attrs.get("gpu", -1),
                args.objective, t.value if t.value is not None else float("inf"),
                t.user_attrs.get("actual_params"),
                t.user_attrs.get("best_val"),
                t.user_attrs.get("best_epoch"),
                t.user_attrs.get("best_rmse_orig"),
                t.user_attrs.get("best_mae_orig"),
                t.user_attrs.get("best_mape_orig"),
                t.user_attrs.get("epochs_run"),
                t.user_attrs.get("elapsed_seconds"),
                t.user_attrs.get("raw_objective"),
                t.user_attrs.get("normalized_param_count"),
                t.user_attrs.get("param_penalty"),
                t.user_attrs.get("invalid_param_objective"),
            ])

    _plot_history(
        study, run_dir / "objective_history.png",
        title=(f"Optuna @ {args.dataset} (epochs={args.epochs})"),
        objective=args.objective,
    )

    print(f"[kn_bayes_opt] artifacts in {run_dir}")


if __name__ == "__main__":
    main()
