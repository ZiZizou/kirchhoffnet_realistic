"""Optuna-driven Bayesian hyperparameter optimization for MLP baselines.

Tunes the training hyperparameters of ``mlp_benchmark_{dataset}.py`` over
multiple optuna trials, while keeping the **parameter budget** and **epoch
budget** fixed per trial:

- Fixed parameter budget: ``hidden_dim`` is derived from
  ``(num_layers, budget)`` by solving the MLP param-count formula for the
  nearest integer. The optimizer explores the depth/width degeneracy at
  constant param count.
- Fixed epoch budget: ``--epochs`` is passed explicitly and ``--patience``
  defaults to ``epochs`` so early stopping never curtails the run.

Each trial is executed as a subprocess of the corresponding
``mlp_benchmark_{dataset}.py``, so the GPU/CPU isolation of the original
benchmark is preserved and the only artefact contract is the
``final_metrics.txt`` file. Parallelism is provided by optuna's
``n_jobs`` (ThreadPoolExecutor) which concurrently spawns up to ``n_workers``
trial subprocesses - this is the recommended mode on Kaggle (CPU cores or
single GPU). When ``torch.cuda.is_available()`` the default is ``n_workers=1``
because each subprocess grabs the GPU.

Dataset-in / -out dimensions are hardcoded per dataset (housing=8,
friedman1=10, friedman2/3=4, smooth2d=2, all out_dim=1). Default param
budgets match the problem-statement scales (housing 2000, friedman1 6600,
friedman2/3 7000, smooth2d 1842) and are overridable via ``--param-budget``.

Outputs (in ``--output/<dataset>_budget<P>_e<E>/``):
  - ``<study_name>.db``              optuna sqlite (resumable via ``--resume``)
  - ``best_hyperparams.txt``         best config + metrics + actual param count
  - ``results.csv``                  every trial: HPs + metrics + param_count
  - ``objective_history.png``        trial values + best-so-far curve
  - ``trial_<NNNN>/log.txt``         per-trial subprocess stdout/stderr
  - ``trial_<NNNN>/final_metrics.txt`` (inherited from mlp_benchmark_*.py)

CLI:
    python mlp_bayes_opt.py --dataset housing --param-budget 2000 \\
        --epochs 800 --n-trials 30 --n-workers 2 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import math
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
        "script": "mlp_benchmark_housing.py",
        "in_dim": 8,
        "out_dim": 1,
        "default_budget": 2000,
    },
    "friedman1": {
        "script": "mlp_benchmark_friedman1.py",
        "in_dim": 10,
        "out_dim": 1,
        "default_budget": 6600,
    },
    "friedman2": {
        "script": "mlp_benchmark_friedman2.py",
        "in_dim": 4,
        "out_dim": 1,
        "default_budget": 7000,
    },
    "friedman3": {
        "script": "mlp_benchmark_friedman3.py",
        "in_dim": 4,
        "out_dim": 1,
        "default_budget": 7000,
    },
    "smooth2d": {
        "script": "mlp_benchmark.py",
        "in_dim": 2,
        "out_dim": 1,
        "default_budget": 1842,
    },
}

BATCH_CHOICES = [512, 1024, 2048, 4096]
ACTIVATION_CHOICES = ["relu", "tanh"]
OBJECTIVE_KEYS = {
    "best_val",
    "best_rmse_orig",
    "best_mse_orig",
    "best_mae_orig",
    "best_mape_orig",
    "final_val",
}


def _param_count(h: int, num_layers: int, in_dim: int, out_dim: int) -> int:
    """Closed-form MLP parameter count.

    params = (L-2)h^2 + (in + L - 1 + out)h + out
    where L = num_layers, h = hidden_dim, in = in_dim, out = out_dim.
    """
    return (num_layers - 2) * h * h + (in_dim + num_layers - 1 + out_dim) * h + out_dim


def derive_hidden_dim(*, num_layers: int, budget: int,
                      in_dim: int, out_dim: int) -> int:
    """Return the largest integer h >= 1 with params(h) <= budget.

    Solves the MLP param-count formula for the exact positive root and
    rounds to nearest integer. If that rounded h exceeds the budget we
    step down until it fits. Returns -1 if no valid h exists (e.g. budget
    too small for the requested num_layers).
    """
    if num_layers < 2:
        raise ValueError(f"num_layers must be >= 2, got {num_layers}")
    a = num_layers - 2
    b = in_dim + num_layers - 1 + out_dim
    c = out_dim - budget
    if a == 0:
        h_exact = (budget - out_dim) / b
        h = max(1, round(h_exact))
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            return -1
        h_exact = (-b + math.sqrt(disc)) / (2 * a)
        if h_exact <= 0:
            return -1
        h = max(1, round(h_exact))
    while h > 0 and _param_count(h, num_layers, in_dim, out_dim) > budget:
        h -= 1
    if h < 1:
        return -1
    return h


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
    hidden_dim: int,
    num_layers: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    activation: str,
    loss: str,
    seed: int,
    output: Path,
    patience: int,
    validate_every: int,
    dataset: str,
    device: str,
) -> list[str]:
    cmd = [
        python,
        script,
        "--hidden-dim", str(hidden_dim),
        "--num-layers", str(num_layers),
        "--epochs", str(epochs),
        "--lr", f"{lr:.6e}",
        "--weight-decay", f"{weight_decay:.6e}",
        "--batch-size", str(batch_size),
        "--activation", activation,
        "--loss", loss,
        "--seed", str(seed),
        "--patience", str(patience),
        "--validate-every", str(validate_every),
        "--device", device,
        "--output", str(output),
    ]
    if dataset.startswith("friedman"):
        cmd += ["--target-noise-std", "1.0"]
    return cmd


def _plot_history(study: optuna.Study, path: Path, *, title: str,
                  objective: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[mlp_bayes_opt] matplotlib not installed; skipping plot")
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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--param-budget", type=int, default=None,
                        help="Target parameter count (default: per-dataset).")
    parser.add_argument("--epochs", type=int, default=800,
                        help="Fixed epoch budget (default: 800).")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=None,
                        help="Optional wall-clock timeout in seconds.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed shared across all trials (default: 0).")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Concurrent trial subprocesses. Default: 1 if "
                             "CUDA is available, else os.cpu_count().")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Device 'auto' | 'cpu' | 'cuda' (default: auto-detect). "
                             "'auto' uses CUDA when available and is forwarded to each "
                             "trial subprocess. An explicit 'cuda' with no GPU falls "
                             "back to CPU.")
    parser.add_argument("--n-layers-max", type=int, default=6,
                        help="Upper bound on num_layers (default: 6).")
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--lr-max", type=float, default=1e-2)
    parser.add_argument("--wd-min", type=float, default=1e-6)
    parser.add_argument("--wd-max", type=float, default=1e-2)
    parser.add_argument("--objective", default="best_val",
                        choices=sorted(OBJECTIVE_KEYS),
                        help="Metric to minimize (default: best_val).")
    parser.add_argument("--loss", choices=["huber", "mse"], default="huber",
                        help="Training loss (default: huber, matches KNet).")
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=Path("./outputs/mlp_bayes_opt"))
    parser.add_argument("--study-name", default=None,
                        help="Optuna study name. Default: "
                             "<dataset>_budget<P>_e<E>.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing study from <run_dir>/"
                             "<study_name>.db")
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    in_dim: int = cfg["in_dim"]
    out_dim: int = cfg["out_dim"]
    script: str = cfg["script"]
    param_budget: int = (args.param_budget if args.param_budget is not None
                         else cfg["default_budget"])

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / f"{args.dataset}_budget{param_budget}_e{args.epochs}"
    run_dir.mkdir(parents=True, exist_ok=True)

    study_name = (args.study_name
                  or f"{args.dataset}_budget{param_budget}_e{args.epochs}")
    storage = f"sqlite:///{run_dir / (study_name + '.db')}"

    if args.n_workers is None:
        n_workers = 1 if torch.cuda.is_available() else max(1, os.cpu_count() or 1)
    else:
        n_workers = max(1, args.n_workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            print("[mlp_bayes_opt] WARNING: --device cuda requested but no CUDA GPU "
                  "detected; falling back to CPU")
            device = "cpu"

    patience = args.epochs  # full fixed-budget run, no early stopping

    sampler = TPESampler(seed=args.seed)
    if args.resume and (run_dir / (study_name + ".db")).exists():
        study = optuna.load_study(study_name=study_name, storage=storage,
                                  sampler=sampler)
    else:
        study = optuna.create_study(study_name=study_name, storage=storage,
                                    sampler=sampler, direction="minimize")

    repo_dir = Path(__file__).resolve().parent
    script_path = repo_dir / script
    python_exe = sys.executable
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    print(f"[mlp_bayes_opt] dataset={args.dataset} in_dim={in_dim} out_dim={out_dim} "
          f"budget={param_budget} epochs={args.epochs} n_trials={args.n_trials} "
          f"n_workers={n_workers} objective={args.objective} loss={args.loss} "
          f"device={device}")
    print(f"[mlp_bayes_opt] cuda_available={torch.cuda.is_available()} "
          f"study_name={study_name} storage={storage}")
    print(f"[mlp_bayes_opt] run_dir={run_dir}")
    print(f"[mlp_bayes_opt] script={script_path}")

    def objective(trial: optuna.Trial) -> float:
        num_layers = trial.suggest_int("num_layers", 2, args.n_layers_max)
        hidden_dim = derive_hidden_dim(num_layers=num_layers,
                                       budget=param_budget,
                                       in_dim=in_dim, out_dim=out_dim)
        if hidden_dim < 1:
            raise optuna.TrialPruned(
                f"budget={param_budget} too small for num_layers={num_layers}")
        lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        weight_decay = trial.suggest_float("weight_decay",
                                           args.wd_min, args.wd_max, log=True)
        batch_size = trial.suggest_categorical("batch_size", BATCH_CHOICES)
        activation = trial.suggest_categorical("activation", ACTIVATION_CHOICES)

        trial_dir = run_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        log_path = trial_dir / "log.txt"

        cmd = _build_command(
            python=python_exe, script=str(script_path),
            hidden_dim=hidden_dim, num_layers=num_layers,
            epochs=args.epochs, lr=lr, weight_decay=weight_decay,
            batch_size=batch_size, activation=activation, loss=args.loss,
            seed=args.seed, output=trial_dir, patience=patience,
            validate_every=args.validate_every, dataset=args.dataset,
            device=device,
        )
        t0 = time.time()
        sub_env = os.environ.copy()
        sub_env.setdefault("PYTHONIOENCODING", "utf-8")
        sub_env.setdefault("PYTHONUTF8", "1")
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write(f"$ {' '.join(cmd)}\n")
            logf.flush()
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                  text=True, cwd=str(repo_dir),
                                  creationflags=creationflags,
                                  env=sub_env)
        elapsed = time.time() - t0

        metrics_path = trial_dir / "final_metrics.txt"
        if proc.returncode != 0 or not metrics_path.exists():
            trial.set_user_attr("subprocess_failed", True)
            trial.set_user_attr("subprocess_returncode", proc.returncode)
            trial.set_user_attr("subprocess_seconds", elapsed)
            return float("inf")

        metrics = _parse_final_metrics(metrics_path)
        if args.objective not in metrics:
            trial.set_user_attr("subprocess_failed", True)
            trial.set_user_attr("subprocess_seconds", elapsed)
            return float("inf")

        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("hidden_dim", hidden_dim)
        trial.set_user_attr("actual_params", int(metrics.get("param_count", -1)))
        trial.set_user_attr("subprocess_seconds", elapsed)
        print(f"[mlp_bayes_opt] trial {trial.number:04d} "
              f"L={num_layers} h={hidden_dim} params={int(metrics.get('param_count',-1))} "
              f"lr={lr:.2e} wd={weight_decay:.2e} bs={batch_size} act={activation} "
              f"-> {args.objective}={metrics[args.objective]:.6f} "
              f"({elapsed:.1f}s)")
        return float(metrics[args.objective])

    study.optimize(objective, n_trials=args.n_trials, n_jobs=n_workers,
                   timeout=args.timeout, show_progress_bar=False)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"[mlp_bayes_opt] done. {len(completed)}/{len(study.trials)} trials completed.")
    if completed:
        print(f"[mlp_bayes_opt] best_value={study.best_value:.6f}")
        print(f"[mlp_bayes_opt] best_params={study.best_params}")

    with open(run_dir / "best_hyperparams.txt", "w") as f:
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"in_dim: {in_dim}\n")
        f.write(f"out_dim: {out_dim}\n")
        f.write(f"param_budget: {param_budget}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"loss: {args.loss}\n")
        f.write(f"objective: {args.objective}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"n_trials: {args.n_trials}\n")
        f.write(f"n_workers: {n_workers}\n")
        f.write(f"device: {device}\n")
        if completed:
            bt = study.best_trial
            f.write(f"best_trial_number: {bt.number}\n")
            f.write(f"best_value: {study.best_value:.6f}\n")
            f.write(f"hidden_dim: {bt.user_attrs.get('hidden_dim')}\n")
            f.write(f"actual_params: {bt.user_attrs.get('actual_params')}\n")
            f.write(f"subprocess_seconds: {bt.user_attrs.get('subprocess_seconds')}\n")
            f.write("params:\n")
            for k, v in bt.params.items():
                f.write(f"  {k}: {v}\n")
            f.write("metrics:\n")
            for k in OBJECTIVE_KEYS | {"param_count", "epochs_run", "elapsed_seconds"}:
                if k in bt.user_attrs:
                    f.write(f"  {k}: {bt.user_attrs[k]}\n")

    csv_path = run_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trial", "state", "hidden_dim", "actual_params", "num_layers",
            "lr", "weight_decay", "batch_size", "activation", "loss",
            "device", "objective", "objective_value",
            "best_val", "best_epoch", "best_rmse_orig", "best_mae_orig",
            "best_mape_orig", "final_val", "elapsed_seconds",
        ])
        for t in study.trials:
            actual_params = t.user_attrs.get("actual_params", -1)
            hidden_dim = t.user_attrs.get("hidden_dim", -1)
            w.writerow([
                t.number, t.state.name,
                hidden_dim, actual_params,
                t.params.get("num_layers"),
                t.params.get("lr"),
                t.params.get("weight_decay"),
                t.params.get("batch_size"),
                t.params.get("activation"),
                args.loss,
                device,
                args.objective, t.value if t.value is not None else float("inf"),
                t.user_attrs.get("best_val"),
                t.user_attrs.get("best_epoch"),
                t.user_attrs.get("best_rmse_orig"),
                t.user_attrs.get("best_mae_orig"),
                t.user_attrs.get("best_mape_orig"),
                t.user_attrs.get("final_val"),
                t.user_attrs.get("subprocess_seconds"),
            ])

    _plot_history(
        study, run_dir / "objective_history.png",
        title=(f"Optuna @ {args.dataset} "
               f"(budget={param_budget}, epochs={args.epochs})"),
        objective=args.objective,
    )

    print(f"[mlp_bayes_opt] artifacts in {run_dir}")


if __name__ == "__main__":
    main()
