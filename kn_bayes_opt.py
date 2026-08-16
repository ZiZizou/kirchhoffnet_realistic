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

Search dimensions (per trial):
    num_hidden (small_world node count; min bounded by in_dim * fanout_count)
    small_world_k (even categorical in [2, min(14, num_hidden-1)])
    small_world_p (continuous [0, 1])
    num_stages (int [1, 3])
    t_span (total ODE horizon [0.5, 10.0])
    num_steps (total Heun steps [10, 150])
    fanout_count (categorical {1, 2}; fixed-spread target map)
    lr, weight_decay, batch_size (optimizer knob cluster)

Validity:
    num_hidden >= in_dim * fanout_count (enough distinct fanout targets)
    small_world_k even, 2 <= k < num_hidden, capped at 14
    TrialPruned on any invalid combo

Each trial is a subprocess of ``train_script.py``, so the GPU/CPU isolation
of the original training script is preserved. Optuna ``n_jobs`` concurrently
spawns up to ``n_workers`` trial subprocesses; trials are pinned round-robin
to individual GPUs via ``CUDA_VISIBLE_DEVICES`` when CUDA is available.

Dataset-in / -out dimensions and per-dataset ``num_hidden`` range are
hardcoded in ``DATASETS``. Default ranges favor the existing small_world
configurations used in recent friedman2 grid runs.

Outputs (in ``--output/<dataset>_knet_e<E>/``):
    <study_name>.db   optuna sqlite (resumable via --resume)
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
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import optuna
import torch
from optuna.samplers import TPESampler


DATASETS: dict[str, dict[str, Any]] = {
    "housing": {
        "in_dim": 8,
        "out_dim": 1,
        "num_hidden_range": (16, 32),
        "max_fanout_count": 2,
    },
    "smooth2d": {
        "in_dim": 2,
        "out_dim": 1,
        "num_hidden_range": (8, 24),
        "max_fanout_count": 2,
    },
    "friedman1": {
        "in_dim": 10,
        "out_dim": 1,
        "num_hidden_range": (20, 32),
        "max_fanout_count": 2,
    },
    "friedman2": {
        "in_dim": 4,
        "out_dim": 1,
        "num_hidden_range": (8, 24),
        "max_fanout_count": 2,
    },
    "friedman3": {
        "in_dim": 4,
        "out_dim": 1,
        "num_hidden_range": (8, 24),
        "max_fanout_count": 2,
    },
}

BATCH_SIZE_CHOICES = [512, 1024, 2048, 4096]
FANOUT_COUNT_CHOICES = [1, 2]
NUM_STAGES_CHOICES = [1, 2, 3]
SMALL_WORLD_K_MAX = 14

OBJECTIVE_KEYS = {
    "best_val",
    "best_rmse_orig",
    "best_mse_orig",
    "best_mae_orig",
    "best_mape_orig",
}


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
    """Even k values in [2, min(SMALL_WORLD_K_MAX, num_hidden-1)]."""
    upper = min(SMALL_WORLD_K_MAX, num_hidden - 1)
    return [k for k in range(2, upper + 1) if k % 2 == 0]


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
    if exact.is_dir():
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
    fanout_count: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    c_eff: float,
    x_max: float,
    gm_min: float,
    gm_max: float,
    isat_min: float,
    isat_max: float,
    sparsity_lambda: float,
    entropy_lambda: float,
    output: Path,
    device: str,
) -> list[str]:
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
        "--c-eff", f"{c_eff:.6e}",
        "--x-max", f"{x_max:.6e}",
        "--gm-min", f"{gm_min:.6e}",
        "--gm-max", f"{gm_max:.6e}",
        "--isat-min", f"{isat_min:.6e}",
        "--isat-max", f"{isat_max:.6e}",
        "--sparsity-lambda", f"{sparsity_lambda:.6e}",
        "--entropy-lambda", f"{entropy_lambda:.6e}",
        "--seed", str(seed),
        "--device", device,
        "--output", str(output),
    ]
    if problem.startswith("friedman"):
        cmd += ["--target-noise-std", "1.0"]
    return cmd


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
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed shared across all trials (default: 0). "
                             "Also seeds the small_world rewiring RNG.")
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
    parser.add_argument("--num-hidden-min", type=int, default=None,
                        help="Override min num_hidden (default: max(2, "
                             "in_dim * fanout_count_min)).")
    parser.add_argument("--num-hidden-max", type=int, default=None,
                        help="Override max num_hidden (default: per-dataset).")
    parser.add_argument("--t-span-min", type=float, default=0.5)
    parser.add_argument("--t-span-max", type=float, default=10.0)
    parser.add_argument("--num-steps-min", type=int, default=10)
    parser.add_argument("--num-steps-max", type=int, default=150)
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--lr-max", type=float, default=1e-2)
    parser.add_argument("--wd-min", type=float, default=1e-6)
    parser.add_argument("--wd-max", type=float, default=1e-2)
    parser.add_argument("--num-stages-max", type=int, default=3,
                        help="Upper bound on num_stages (default: 3).")
    parser.add_argument("--c-eff-min", type=float, default=0.1,
                        help="Lower bound for log-uniform C_eff search "
                             "(default: 0.1).")
    parser.add_argument("--c-eff-max", type=float, default=10.0,
                        help="Upper bound for log-uniform C_eff search "
                             "(default: 10.0).")
    parser.add_argument("--x-max-min", type=float, default=0.5,
                        help="Lower bound for x_max (ODE rail) search "
                             "(default: 0.5; config default is 3.0).")
    parser.add_argument("--x-max-max", type=float, default=8.0,
                        help="Upper bound for x_max (ODE rail) search "
                             "(default: 8.0; config default is 3.0).")
    parser.add_argument("--gm-min-min", type=float, default=0.001,
                        help="Lower bound for log-uniform gm_min search "
                             "(default: 0.001; config default is 0.01).")
    parser.add_argument("--gm-min-max", type=float, default=0.1,
                        help="Upper bound for log-uniform gm_min search "
                             "(default: 0.1).")
    parser.add_argument("--gm-max-min", type=float, default=1.0,
                        help="Lower bound for log-uniform gm_max search "
                             "(default: 1.0; config default is 10.0).")
    parser.add_argument("--gm-max-max", type=float, default=50.0,
                        help="Upper bound for log-uniform gm_max search "
                             "(default: 50.0).")
    parser.add_argument("--isat-min-min", type=float, default=0.001,
                        help="Lower bound for log-uniform isat_min search "
                             "(default: 0.001; config default is 0.01).")
    parser.add_argument("--isat-min-max", type=float, default=0.1,
                        help="Upper bound for log-uniform isat_min search "
                             "(default: 0.1).")
    parser.add_argument("--isat-max-min", type=float, default=1.0,
                        help="Lower bound for log-uniform isat_max search "
                             "(default: 1.0; config default is 10.0).")
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
    parser.add_argument("--output", type=Path,
                        default=Path("./outputs/kn_bayes_opt"))
    parser.add_argument("--study-name", default=None,
                        help="Optuna study name. Default: "
                             "<dataset>_knet_e<E>.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing study from <run_dir>/"
                             "<study_name>.db")
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    in_dim: int = cfg["in_dim"]
    out_dim: int = cfg["out_dim"]
    default_min, default_max = cfg["num_hidden_range"]
    num_hidden_min = (
        args.num_hidden_min if args.num_hidden_min is not None
        else max(2, in_dim * min(FANOUT_COUNT_CHOICES))
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

    sampler = TPESampler(seed=args.seed)
    if args.resume and (run_dir / (study_name + ".db")).exists():
        study = optuna.load_study(study_name=study_name, storage=storage,
                                  sampler=sampler)
    else:
        study = optuna.create_study(study_name=study_name, storage=storage,
                                    sampler=sampler, direction="minimize")

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

    def objective(trial: optuna.Trial) -> float:
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
        small_world_k = trial.suggest_categorical(
            "small_world_k", valid_small_world_k_choices(SMALL_WORLD_K_MAX + 1))
        if small_world_k >= num_hidden:
            raise optuna.TrialPruned(
                f"small_world_k={small_world_k} must be < num_hidden={num_hidden}"
            )
        small_world_p = trial.suggest_float("small_world_p", 0.0, 1.0)
        num_stages = trial.suggest_int(
            "num_stages", 1, args.num_stages_max)
        t_span = trial.suggest_float(
            "t_span", args.t_span_min, args.t_span_max)
        num_steps = trial.suggest_int(
            "num_steps", args.num_steps_min, args.num_steps_max)
        lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        weight_decay = trial.suggest_float("weight_decay",
                                           args.wd_min, args.wd_max,
                                           log=True)
        batch_size = trial.suggest_categorical("batch_size",
                                                BATCH_SIZE_CHOICES)

        c_eff = trial.suggest_float(
            "c_eff", args.c_eff_min, args.c_eff_max, log=True)
        x_max = trial.suggest_float(
            "x_max", args.x_max_min, args.x_max_max)
        gm_min = trial.suggest_float(
            "gm_min", args.gm_min_min, args.gm_min_max, log=True)
        gm_max = trial.suggest_float(
            "gm_max", args.gm_max_min, args.gm_max_max, log=True)
        isat_min = trial.suggest_float(
            "isat_min", args.isat_min_min, args.isat_min_max, log=True)
        isat_max = trial.suggest_float(
            "isat_max", args.isat_max_min, args.isat_max_max, log=True)
        sparsity_lambda = trial.suggest_float(
            "sparsity_lambda", args.sparsity_lambda_min,
            args.sparsity_lambda_max, log=True)
        entropy_lambda = trial.suggest_float(
            "entropy_lambda", args.entropy_lambda_min,
            args.entropy_lambda_max, log=True)
        if gm_max <= gm_min:
            raise optuna.TrialPruned(
                f"gm_max={gm_max} must be > gm_min={gm_min}")
        if isat_max <= isat_min:
            raise optuna.TrialPruned(
                f"isat_max={isat_max} must be > isat_min={isat_min}")

        trial_dir = run_dir / f"trial_{trial.number:04d}"
        log_path = run_dir / f"trial_{trial.number:04d}.log.txt"
        resolved_trial_dir: Path | None = None

        cmd = _build_command(
            python=python_exe, script=str(script_path),
            problem=args.dataset, seed=args.seed,
            epochs=args.epochs,
            num_hidden=num_hidden, small_world_k=small_world_k,
            small_world_p=small_world_p, num_stages=num_stages,
            t_span=t_span, num_steps=num_steps,
            fanout_count=fanout_count, lr=lr, weight_decay=weight_decay,
            batch_size=batch_size, c_eff=c_eff, x_max=x_max,
            gm_min=gm_min, gm_max=gm_max, isat_min=isat_min,
            isat_max=isat_max, sparsity_lambda=sparsity_lambda,
            entropy_lambda=entropy_lambda,
            output=trial_dir, device=device,
        )

        t0 = time.time()
        sub_env = os.environ.copy()
        sub_env.setdefault("PYTHONIOENCODING", "utf-8")
        sub_env.setdefault("PYTHONUTF8", "1")
        gpu_idx = -1
        if device == "cuda" and n_gpus >= 1:
            gpu_idx = trial.number % n_gpus
            sub_env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        with open(log_path, "w", encoding="utf-8") as logf:
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
        trial.set_user_attr("actual_params", int(metrics.get("param_count", -1)))
        if resolved_trial_dir is not None:
            trial.set_user_attr("resolved_trial_dir",
                                str(resolved_trial_dir))
        trial.set_user_attr("subprocess_seconds", elapsed)
        trial.set_user_attr("gpu", gpu_idx)
        trial.set_user_attr("boundary_fan_out",
                            json.dumps(build_boundary_fan_out(
                                in_dim=in_dim, fanout_count=fanout_count,
                                num_hidden=num_hidden)))
        print(f"[kn_bayes_opt] trial {trial.number:04d} "
              f"nh={num_hidden} k={small_world_k} p={small_world_p:.2f} "
              f"st={num_stages} ts={t_span:.2f} ns={num_steps} "
              f"fc={fanout_count} lr={lr:.2e} wd={weight_decay:.2e} "
              f"bs={batch_size} params={int(metrics.get('param_count',-1))} "
              f"gpu={gpu_idx} -> {args.objective}={metrics[args.objective]:.6f} "
              f"({elapsed:.1f}s)")
        return float(metrics[args.objective])

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
        f.write(f"seed: {args.seed}\n")
        f.write(f"n_trials: {args.n_trials}\n")
        f.write(f"n_workers: {n_workers}\n")
        f.write(f"device: {device}\n")
        f.write(f"n_gpus: {n_gpus}\n")
        if completed:
            bt = study.best_trial
            f.write(f"best_trial_number: {bt.number}\n")
            f.write(f"best_value: {study.best_value:.6f}\n")
            f.write(f"actual_params: {bt.user_attrs.get('actual_params')}\n")
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
            "trial", "state", "num_hidden", "small_world_k", "small_world_p",
            "num_stages", "t_span", "num_steps", "fanout_count",
            "boundary_fan_out", "lr", "weight_decay", "batch_size",
            "device", "gpu", "objective", "objective_value",
            "actual_params", "best_val", "best_epoch", "best_rmse_orig",
            "best_mae_orig", "best_mape_orig", "epochs_run",
            "elapsed_seconds",
        ])
        for t in study.trials:
            w.writerow([
                t.number, t.state.name,
                t.params.get("num_hidden"),
                t.params.get("small_world_k"),
                t.params.get("small_world_p"),
                t.params.get("num_stages"),
                t.params.get("t_span"),
                t.params.get("num_steps"),
                t.params.get("fanout_count"),
                t.user_attrs.get("boundary_fan_out"),
                t.params.get("lr"),
                t.params.get("weight_decay"),
                t.params.get("batch_size"),
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
            ])

    _plot_history(
        study, run_dir / "objective_history.png",
        title=(f"Optuna @ {args.dataset} (epochs={args.epochs})"),
        objective=args.objective,
    )

    print(f"[kn_bayes_opt] artifacts in {run_dir}")


if __name__ == "__main__":
    main()
