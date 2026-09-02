"""Optuna-driven Bayesian optimization for the plain single-head MLP DAgger control.

Every trial is a subprocess of ``generative-distillation-plain-mlp.py`` and is
**locked to ``--input-preprocessing knet``**: the controller never suggests a
preprocessing choice, it is hardcoded in :func:`base_command`. All sampled
architectures therefore run the 4-feature ``PlainMLP.scale_input()`` knet
branch (log / min-max, clipped to [-4, 4]).

Width mode: **derived at budget**. The controller samples
``plain_trunk_layers`` and derives ``plain_trunk_width`` so the network sits
just under ``--param-budget``, exploring the depth/width degeneracy at (near)
constant parameter count. Param counting uses a knet-corrected closed form
(``_knet_param_count``) because the legacy :func:`plain_param_count` assumes an
8->W first layer (Q75) and overcounts knet nets (4->W) by 4W. Budget gating
uses the subprocess preflight (--count-params-only), so the gate is against
PyTorch's own count, not the closed form.

Fidelity: BO trials default to a fast DAgger proxy (4 iterations x 100 epochs,
common-eval 1000); ``--fidelity full`` restores the production 10 x 200
schedule with common-eval 2000.

Objective: single minimize of the trial's final test failure rate (fraction),
optionally scaled by a parameter penalty (``--param-penalty``).

CLI example:
    python generative-distillation-plain-mlp-bayes-opt.py --param-budget 5559 \\
        --n-trials 30 --seed 100
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
from optuna.samplers import TPESampler


SCRIPT = Path(__file__).resolve().parent / "generative-distillation-plain-mlp.py"

# Hard requirement for this controller variant: no search over preprocessing.
INPUT_PREPROCESSING = "knet"

DEFAULT_PARAM_BUDGET = 5559  # reference run: derives W=49 L=3 knet (legacy 8->W count gives 48)

SEED_TRIAL = {
    "plain_trunk_layers": 3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 256,
}

FIDELITY_PRESETS = {
    "proxy": {"dagger_iterations": 4, "epochs_per_iter": 100,
              "common_eval_size": 1000, "earlystop_eval_every": 5},
    "full": {"dagger_iterations": 10, "epochs_per_iter": 200,
             "common_eval_size": 2000, "earlystop_eval_every": 1},
}

BATCH_CHOICES = [256, 512, 1024]


def _recover_unfinished_trials(study: optuna.Study) -> list[dict[str, Any]]:
    """Mark abandoned RUNNING trials as FAIL and re-queue their params.

    A trial stays RUNNING in SQLite if the host was killed mid-run;
    study.optimize would otherwise never retry it.
    """
    running = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
    retry_params: list[dict[str, Any]] = []
    for trial in running:
        params = dict(trial.params)
        retry_params.append(params)
        try:
            study._storage.set_trial_state_values(  # type: ignore[attr-defined]
                trial._trial_id, optuna.trial.TrialState.FAIL, None
            )
        except Exception as exc:
            raise RuntimeError(f"Could not mark abandoned trial {trial.number} as FAIL") from exc
    return retry_params


def _knet_param_count(trunk_width: int, trunk_layers: int) -> int:
    """Exact trainable count of a knet PlainMLP (4 -> W first layer).

    trunk = 5W + (L-1)(W^2 + W);  head = 7W + 7.
    The legacy :func:`plain_param_count` assumes an 8 -> W first layer and
    returns 4W more; it is kept for comparability in the CSV only.

    Note: this closed form does not include ``LayerNorm`` parameters
    (``2*W`` per trunk layer when ``plain_use_layernorm=True``). The BO budget
    gate is therefore optimistic; the subprocess ``--count-params-only``
    preflight supplies the ground-truth torch count including LayerNorm.
    """
    if trunk_width < 1 or trunk_layers < 1:
        raise ValueError("trunk dimensions must be positive")
    trunk = 5 * trunk_width + (trunk_layers - 1) * (trunk_width ** 2 + trunk_width)
    head = 7 * trunk_width + 7
    return trunk + head


def _legacy_param_count(trunk_width: int, trunk_layers: int) -> int:
    """Legacy plain_param_count (8 -> W first layer): 9W + (L-1)(W^2+W) + 7W + 7."""
    trunk = 9 * trunk_width + (trunk_layers - 1) * (trunk_width ** 2 + trunk_width)
    head = 7 * trunk_width + 7
    return trunk + head


def derive_knet_width(budget: int, trunk_layers: int) -> int:
    """Largest W >= 1 with :func:`_knet_param_count`(W, L) <= budget.

    Returns -1 when no width fits (budget too small for the depth).
    Mirrors :func:`derive_plain_width` semantics for the knet-corrected count.
    """
    if trunk_layers < 1:
        raise ValueError("trunk_layers must be >= 1")
    if _knet_param_count(1, trunk_layers) > budget:
        return -1
    w = 1
    while _knet_param_count(w + 1, trunk_layers) <= budget:
        w += 1
    return w


def max_knet_width(*, trunk_layers: int, param_limit: int,
                   lower: int, upper: int) -> int:
    """Largest permitted knet width in [lower, upper] under param_limit.

    Currently unused (width is derived via :func:`derive_knet_width`);
    retained for a future sampled-width mode and for symmetry with
    :func:`mlp_bayes_opt.max_moe_width`.
    """
    valid = [w for w in range(lower, upper + 1)
             if _knet_param_count(w, trunk_layers) <= param_limit]
    return max(valid, default=-1)


def _penalized_objective(metric: float, actual_params: int,
                         reference_params: int, strength: float) -> float:
    """Scale the metric upward in proportion to model size (0 strength disables)."""
    if actual_params < 0:
        return float("inf")
    return metric * (1.0 + strength * actual_params / max(1, reference_params))


def _over_budget_objective(actual_params: int, budget: int, base: float) -> float:
    """Finite graded objective for architectures above the soft cap."""
    ratio = actual_params / max(1, budget)
    return base * ratio * ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("./outputs/generative_plain_bayes_opt"))
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=None,
                        help="Optional wall-clock timeout in seconds.")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--param-budget", type=int, default=DEFAULT_PARAM_BUDGET,
                        help=f"Parameter budget; width is derived to fit under it "
                             f"(default {DEFAULT_PARAM_BUDGET}, the legacy reference run).")
    parser.add_argument("--param-tolerance", type=float, default=0.10,
                        help="Soft cap fraction above --param-budget before a trial "
                             "is rejected (default 0.10).")
    parser.add_argument("--param-penalty", type=float, default=0.25,
                        help="Dimensionless multiplier for the parameter-count penalty "
                             "(default 0.25; 0 disables). Objective = test_failure_rate "
                             "* (1 + penalty * params / param_budget).")
    parser.add_argument("--invalid-param-objective", type=float, default=1e6,
                        help="Base finite objective for architectures above the soft cap.")
    parser.add_argument("--min-width", type=int, default=32)
    parser.add_argument("--max-width", type=int, default=96,
                        help="Hard upper bound on the derived trunk width.")
    parser.add_argument("--layers-min", type=int, default=2)
    parser.add_argument("--layers-max", type=int, default=3)
    parser.add_argument("--activation-choices", nargs="+", default=["silu", "gelu"],
                        choices=["silu", "gelu"],
                        help="Trunk activations to sample.")
    parser.add_argument("--layernorm-choices", nargs="+", default=["False", "True"],
                        choices=["False", "True"],
                        help="LayerNorm options to sample (default: both).")
    parser.add_argument("--batch-choices", nargs="+", type=int, default=BATCH_CHOICES)
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--lr-max", type=float, default=1e-2)
    parser.add_argument("--wd-min", type=float, default=1e-6)
    parser.add_argument("--wd-max", type=float, default=1e-2)
    parser.add_argument("--fidelity", choices=sorted(FIDELITY_PRESETS), default="proxy",
                        help="Trial schedule: proxy=4x100/eval1000 (fast BO), "
                             "full=10x200/eval2000 (production).")
    parser.add_argument("--dagger-iterations", type=int, default=None,
                        help="Override the fidelity preset.")
    parser.add_argument("--epochs-per-iter", type=int, default=None)
    parser.add_argument("--common-eval-size", type=int, default=None)
    parser.add_argument("--earlystop-eval-every", type=int, default=None)
    parser.add_argument("--teacher-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Concurrent trial subprocesses. >1 only on multi-GPU hosts: "
                             "trials are pinned round-robin via CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--study-name", default=None,
                        help="Optuna study name (default: plain_knet_budget<P>_f<FIDELITY>).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the study from <run_dir>/<study_name>.db.")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip the --count-params-only subprocess preflight and trust "
                             "the closed-form count (saves ~2s/trial).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the sampled trial commands instead of running them.")
    return parser.parse_args()


def base_command(args: argparse.Namespace, trial_dir: Path, device: str, seed: int) -> list[str]:
    """Static part of the trial command; knet is hardcoded here by design.

    Contract: BO derives ``plain_trunk_width`` via the knet-corrected
    :func:`_knet_param_count` and forwards it explicitly. ``--param-budget``
    is intentionally NOT forwarded to the child — the harness would re-derive
    via the legacy 8->W :func:`plain_param_count` (48 vs 49 at 5559) and
    diverge from the BO's knet count. Gating uses the subprocess
    ``TRAINABLE_PARAMS`` preflight, i.e. ground truth.
    """
    cmd = [
        sys.executable, str(SCRIPT),
        "--dagger-iterations", str(args.dagger_iterations),
        "--epochs-per-iter", str(args.epochs_per_iter),
        "--common-eval-size", str(args.common_eval_size),
        "--earlystop-eval-every", str(args.earlystop_eval_every),
        "--lr", f"{args.lr:.8g}",
        "--weight-decay", f"{args.weight_decay:.8g}",
        "--batch-size", str(args.batch_size),
        "--plain-trunk-width", str(args.plain_trunk_width),
        "--plain-trunk-layers", str(args.plain_trunk_layers),
        "--plain-activation", args.plain_activation,
        "--input-preprocessing", INPUT_PREPROCESSING,
        "--seed", str(seed),
        "--device", device,
        "--output", str(trial_dir),
    ]
    if args.plain_use_layernorm:
        cmd.append("--plain-use-layernorm")
    if args.teacher_dir:
        cmd += ["--teacher-dir", args.teacher_dir]
    if args.data_dir:
        cmd += ["--data-dir", args.data_dir]
    return cmd


def preflight_param_count(cmd: list[str], repo_dir: Path) -> int | None:
    """Run --count-params-only and return the trainable count (None on failure)."""
    result = subprocess.run(cmd + ["--count-params-only"], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            cwd=str(repo_dir))
    if result.returncode != 0:
        return None
    match = re.search(r"TRAINABLE_PARAMS=(\d+)", result.stdout)
    return int(match.group(1)) if match else None


def _plot_history(study: optuna.Study, path: Path, *, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plain-bo] matplotlib not installed; skipping plot")
        return
    vals = [float(t.value) for t in study.trials
            if t.value is not None and t.value != float("inf")]
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
    ax.set_ylabel("penalized test failure rate")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.n_trials < 1:
        raise ValueError("--n-trials must be >= 1")
    if args.param_budget < 1:
        raise ValueError("--param-budget must be >= 1")
    if not 0.0 <= args.param_tolerance < 1.0:
        raise ValueError("--param-tolerance must be in [0, 1)")
    if args.param_penalty < 0.0:
        raise ValueError("--param-penalty must be >= 0")
    if args.invalid_param_objective <= 0.0:
        raise ValueError("--invalid-param-objective must be > 0")
    if not 1 <= args.layers_min <= args.layers_max:
        raise ValueError("require 1 <= --layers-min <= --layers-max")
    # Allow L=1 (single Linear 4->W -> head); PlainMLP supports it and the
    # knet/layers degeneracy at budget includes it.  Reject only L<1.
    if args.layers_min < 1 or args.layers_max < 1:
        raise ValueError("--layers-min/max must be >= 1")
    if args.min_width < 8:
        raise ValueError("--min-width must be >= 8")

    # Fidelity preset with per-knob overrides
    preset = dict(FIDELITY_PRESETS[args.fidelity])
    if args.dagger_iterations is not None:
        preset["dagger_iterations"] = args.dagger_iterations
    if args.epochs_per_iter is not None:
        preset["epochs_per_iter"] = args.epochs_per_iter
    if args.common_eval_size is not None:
        preset["common_eval_size"] = args.common_eval_size
    if args.earlystop_eval_every is not None:
        preset["earlystop_eval_every"] = args.earlystop_eval_every
    for key in ("dagger_iterations", "epochs_per_iter", "common_eval_size", "earlystop_eval_every"):
        if preset[key] < 1:
            raise ValueError(f"{key} must be >= 1")
    args.dagger_iterations = preset["dagger_iterations"]
    args.epochs_per_iter = preset["epochs_per_iter"]
    args.common_eval_size = preset["common_eval_size"]
    args.earlystop_eval_every = preset["earlystop_eval_every"]

    run_dir = (args.output / f"plain_knet_budget{args.param_budget}_f{args.fidelity}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    study_name = args.study_name or f"plain_knet_budget{args.param_budget}_f{args.fidelity}"
    storage = f"sqlite:///{run_dir / (study_name + '.db')}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            print("[plain-bo] WARNING: --device cuda requested but no CUDA GPU detected; "
                  "falling back to CPU")
            device = "cpu"

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_workers = max(1, args.n_workers)
    if n_workers > 1 and n_gpus < 1:
        raise ValueError("--n-workers > 1 requires CUDA GPUs (trials are GPU-pinned; "
                         "DAgger on CPU is too slow to parallelize usefully)")
    if n_workers > n_gpus and device == "cuda":
        print(f"[plain-bo] WARNING: n_workers={n_workers} > n_gpus={n_gpus}; "
              "some GPUs will host multiple concurrent trials")

    repo_dir = Path(__file__).resolve().parent
    soft_limit = int(args.param_budget * (1.0 + args.param_tolerance))
    seed_int = args.seed
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    sampler = TPESampler(seed=seed_int, multivariate=True, group=True)
    db_path = run_dir / (study_name + ".db")
    if args.resume and db_path.exists():
        study = optuna.load_study(study_name=study_name, storage=storage, sampler=sampler)
        retry_params = _recover_unfinished_trials(study)
        if retry_params:
            print(f"[plain-bo] recovered {len(retry_params)} unfinished trial(s); they will be retried")
            for params in retry_params:
                study.enqueue_trial(params)
    else:
        study = optuna.create_study(study_name=study_name, storage=storage,
                                    sampler=sampler, direction="minimize")

    # Deterministic seed trial: current best plain shape (W=49 L=3 SiLU no-LN at
    # budget 5559, knet; legacy 8->W count gives 48) as trial 0, mirroring the
    # queued baselines of the other BO controllers.
    seed_complete = any(
        t.user_attrs.get("seed_trial") is True
        and t.state == optuna.trial.TrialState.COMPLETE
        for t in study.trials
    )
    seed_waiting = any(
        t.state == optuna.trial.TrialState.WAITING
        and t.user_attrs.get("seed_trial") is True
        for t in study.trials
    )
    if not seed_complete and not seed_waiting:
        study.enqueue_trial(dict(SEED_TRIAL), user_attrs={"seed_trial": True})

    print(f"[plain-bo] input_preprocessing={INPUT_PREPROCESSING} (locked, not searched)")
    print(f"[plain-bo] budget={args.param_budget} soft_limit={soft_limit} "
          f"trials={args.n_trials} workers={n_workers} gpus={n_gpus} device={device}")
    print(f"[plain-bo] fidelity={args.fidelity} -> "
          f"{args.dagger_iterations}x{args.epochs_per_iter} iters, "
          f"eval={args.common_eval_size} every {args.earlystop_eval_every} epochs")
    print(f"[plain-bo] run_dir={run_dir}")
    print(f"[plain-bo] script={SCRIPT}")
    print(f"[plain-bo] study={study_name} storage={storage}")

    if args.dry_run:
        print("[plain-bo] DRY RUN: sampling commands only; nothing is executed")

    def objective(trial: optuna.Trial) -> float:
        is_seed = trial.user_attrs.get("seed_trial") is True
        if is_seed:
            trunk_layers = int(SEED_TRIAL["plain_trunk_layers"])
            lr = float(SEED_TRIAL["lr"])
            weight_decay = float(SEED_TRIAL["weight_decay"])
            batch_size = int(SEED_TRIAL["batch_size"])
            activation = "silu"
            use_layernorm = False
        else:
            trunk_layers = trial.suggest_int("plain_trunk_layers",
                                             args.layers_min, args.layers_max)
            lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
            weight_decay = trial.suggest_float("weight_decay", args.wd_min, args.wd_max, log=True)
            batch_size = trial.suggest_categorical("batch_size", args.batch_choices)
            activation = trial.suggest_categorical("plain_activation", args.activation_choices)
            use_layernorm = trial.suggest_categorical(
                "plain_layernorm", [s == "True" for s in args.layernorm_choices])

        trunk_width = derive_knet_width(args.param_budget, trunk_layers)
        if trunk_width == -1:
            raise optuna.TrialPruned(
                f"budget={args.param_budget} too small for L={trunk_layers} "
                f"(no W>=1 fits)")
        if trunk_width < args.min_width:
            raise optuna.TrialPruned(
                f"budget={args.param_budget} derives width {trunk_width} < min {args.min_width} "
                f"for L={trunk_layers}")
        trunk_width = min(trunk_width, args.max_width)
        if _knet_param_count(trunk_width, trunk_layers) > soft_limit:
            raise optuna.TrialPruned(
                f"closed-form params {_knet_param_count(trunk_width, trunk_layers)} "
                f"> soft cap {soft_limit}")

        trial_dir = run_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        log_path = trial_dir / "log.txt"

        targs = argparse.Namespace(**vars(args))
        targs.plain_trunk_width = trunk_width
        targs.plain_trunk_layers = trunk_layers
        targs.lr = lr
        targs.weight_decay = weight_decay
        targs.batch_size = batch_size
        targs.plain_activation = activation
        targs.plain_use_layernorm = use_layernorm
        cmd = base_command(targs, trial_dir, device, seed_int)

        if args.dry_run:
            print(f"[plain-bo] trial {trial.number:04d} (dry): {' '.join(cmd)}")
            raise optuna.TrialPruned("dry run")

        if args.skip_preflight:
            expected_params = _knet_param_count(trunk_width, trunk_layers)
        else:
            expected_params = preflight_param_count(cmd, repo_dir)
            if expected_params is None:
                raise optuna.TrialPruned("parameter preflight failed")
        trial.set_user_attr("param_count", expected_params)
        trial.set_user_attr("knet_param_count", expected_params)
        trial.set_user_attr("legacy_param_count",
                            _legacy_param_count(trunk_width, trunk_layers))
        if expected_params > soft_limit:
            return _over_budget_objective(expected_params, soft_limit,
                                          args.invalid_param_objective)

        sub_env = os.environ.copy()
        sub_env.setdefault("PYTHONIOENCODING", "utf-8")
        sub_env.setdefault("PYTHONUTF8", "1")
        gpu_idx = -1
        if device == "cuda" and n_gpus >= 1:
            gpu_idx = trial.number % n_gpus
            sub_env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write(f"$ {' '.join(cmd)}\n")
            logf.flush()
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True,
                                  cwd=str(repo_dir), creationflags=creationflags, env=sub_env)
        elapsed = time.time() - t0

        if proc.returncode != 0:
            trial.set_user_attr("subprocess_failed", True)
            trial.set_user_attr("subprocess_returncode", proc.returncode)
            trial.set_user_attr("subprocess_seconds", elapsed)
            return float("inf")

        text = log_path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"Test failure rate:\s*([\d\.]+)%", text)
        if not m:
            trial.set_user_attr("subprocess_failed", True)
            trial.set_user_attr("missing_test_rate", True)
            return float("inf")
        test_rate = float(m.group(1)) / 100.0

        # Subprocess-logged param count (source of truth for the penalty)
        pm = re.search(r"trainable params:\s*([0-9,]+)", text)
        if pm is None:
            pm = re.search(r"Student model:\s*([0-9,]+) trainable", text)
        param_count = int(pm.group(1).replace(",", "")) if pm else expected_params

        vm = re.findall(r"Validation failure rate:\s*([\d\.]+)%", text)
        val_rate = float(vm[-1]) / 100.0 if vm else None

        trial.set_user_attr("test_failure_rate", test_rate)
        trial.set_user_attr("validation_failure_rate", val_rate)
        trial.set_user_attr("param_count", param_count)
        trial.set_user_attr("plain_trunk_width", trunk_width)
        trial.set_user_attr("plain_trunk_layers", trunk_layers)
        trial.set_user_attr("plain_activation", activation)
        trial.set_user_attr("plain_layernorm", use_layernorm)
        trial.set_user_attr("subprocess_seconds", elapsed)
        trial.set_user_attr("gpu", gpu_idx)
        if param_count > soft_limit:
            penalized = _over_budget_objective(param_count, soft_limit,
                                               args.invalid_param_objective)
        else:
            penalized = _penalized_objective(test_rate, param_count,
                                             args.param_budget, args.param_penalty)
        trial.set_user_attr("raw_test_rate", test_rate)
        trial.set_user_attr("penalized", penalized)
        print(f"[plain-bo] trial {trial.number:04d} W={trunk_width} L={trunk_layers} "
              f"act={activation} LN={int(use_layernorm)} bs={batch_size} "
              f"lr={lr:.2e} wd={weight_decay:.2e} params={param_count} gpu={gpu_idx} "
              f"-> test_rate={test_rate*100:.2f}% penalized={penalized:.6f} ({elapsed:.1f}s)",
              flush=True)
        return penalized

    study.optimize(objective, n_trials=args.n_trials, n_jobs=n_workers,
                   timeout=args.timeout, show_progress_bar=False)

    if args.dry_run:
        print(f"[plain-bo] dry run complete; {len(study.trials)} command(s) printed; "
              f"no study state was evaluated")
        return

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"[plain-bo] done. {len(completed)}/{len(study.trials)} completed, {len(pruned)} pruned.")
    if not completed:
        print("[plain-bo] no completed trials; nothing to summarize")
        return

    # Lexicographic selection: penalized objective first, raw test failure rate second.
    def _select_key(t: optuna.trial.FrozenTrial) -> tuple[float, float]:
        return (float(t.value), float(t.user_attrs.get("raw_test_rate", float("inf"))))

    best = min(completed, key=_select_key)
    print(f"[plain-bo] selected trial {best.number}: value={best.value:.6f} "
          f"raw_test_rate={best.user_attrs.get('raw_test_rate')*100:.2f}% "
          f"params={best.user_attrs.get('param_count')}")

    with open(run_dir / "best_hyperparams.txt", "w", encoding="utf-8") as f:
        f.write(f"input_preprocessing: {INPUT_PREPROCESSING}\n")
        f.write(f"param_budget: {args.param_budget}\n")
        f.write(f"param_tolerance: {args.param_tolerance}\n")
        f.write(f"param_penalty: {args.param_penalty}\n")
        f.write(f"fidelity: {args.fidelity}\n")
        f.write(f"dagger_iterations: {args.dagger_iterations}\n")
        f.write(f"epochs_per_iter: {args.epochs_per_iter}\n")
        f.write(f"common_eval_size: {args.common_eval_size}\n")
        f.write(f"earlystop_eval_every: {args.earlystop_eval_every}\n")
        f.write(f"seed: {seed_int}\n")
        f.write(f"n_trials: {args.n_trials}\n")
        f.write(f"n_workers: {n_workers}\n")
        f.write(f"device: {device}\n")
        f.write(f"n_gpus: {n_gpus}\n")
        f.write(f"best_trial_number: {best.number}\n")
        f.write(f"best_value: {best.value:.6f}\n")
        f.write(f"plain_trunk_width: {best.user_attrs.get('plain_trunk_width')}\n")
        f.write(f"plain_trunk_layers: {best.user_attrs.get('plain_trunk_layers')}\n")
        f.write(f"plain_activation: {best.user_attrs.get('plain_activation')}\n")
        f.write(f"plain_use_layernorm: {best.user_attrs.get('plain_layernorm')}\n")
        f.write(f"param_count: {best.user_attrs.get('param_count')}\n")
        f.write(f"legacy_param_count: {best.user_attrs.get('legacy_param_count')}\n")
        f.write(f"test_failure_rate: {best.user_attrs.get('test_failure_rate')}\n")
        f.write(f"validation_failure_rate: {best.user_attrs.get('validation_failure_rate')}\n")
        f.write("params:\n")
        for k, v in best.params.items():
            f.write(f"  {k}: {v}\n")

    csv_path = run_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial", "state", "plain_trunk_width", "plain_trunk_layers",
                    "plain_activation", "plain_layernorm", "lr", "weight_decay",
                    "batch_size", "param_count", "knet_param_count", "legacy_param_count",
                    "test_failure_rate", "validation_failure_rate", "raw_test_rate",
                    "penalized", "objective_value", "gpu", "subprocess_seconds"])
        for t in study.trials:
            ua = t.user_attrs
            w.writerow([
                t.number, t.state.name,
                ua.get("plain_trunk_width", ""),
                ua.get("plain_trunk_layers", ""),
                ua.get("plain_activation", ""),
                ua.get("plain_layernorm", ""),
                t.params.get("lr", ""),
                t.params.get("weight_decay", ""),
                t.params.get("batch_size", ""),
                ua.get("param_count", ""),
                ua.get("knet_param_count", ""),
                ua.get("legacy_param_count", ""),
                ua.get("test_failure_rate", ""),
                ua.get("validation_failure_rate", ""),
                ua.get("raw_test_rate", ""),
                ua.get("penalized", ""),
                t.value if t.value is not None else "",
                ua.get("gpu", -1),
                ua.get("subprocess_seconds", ""),
            ])

    _plot_history(study, run_dir / "objective_history.png",
                  title=(f"PlainMLP BO @ knet budget={args.param_budget} "
                         f"fidelity={args.fidelity}"))

    summary = {
        "selection": "lexicographic: penalized objective, then raw test failure rate",
        "input_preprocessing": INPUT_PREPROCESSING,
        "param_budget": args.param_budget,
        "param_tolerance": args.param_tolerance,
        "param_penalty": args.param_penalty,
        "fidelity": args.fidelity,
        "dagger_iterations": args.dagger_iterations,
        "epochs_per_iter": args.epochs_per_iter,
        "common_eval_size": args.common_eval_size,
        "earlystop_eval_every": args.earlystop_eval_every,
        "seed": seed_int,
        "n_trials_requested": args.n_trials,
        "n_trials_completed": len(completed),
        "n_trials_pruned": len(pruned),
        "selected_trial": best.number,
        "selected_params": dict(best.params),
        "selected_arch": {
            "plain_trunk_width": best.user_attrs.get("plain_trunk_width"),
            "plain_trunk_layers": best.user_attrs.get("plain_trunk_layers"),
            "plain_activation": best.user_attrs.get("plain_activation"),
            "plain_use_layernorm": best.user_attrs.get("plain_layernorm"),
        },
        "selected_param_count": best.user_attrs.get("param_count"),
        "selected_legacy_param_count": best.user_attrs.get("legacy_param_count"),
        "selected_values": {
            "penalized": best.value,
            "test_failure_rate": best.user_attrs.get("test_failure_rate"),
            "validation_failure_rate": best.user_attrs.get("validation_failure_rate"),
        },
        "selected_checkpoint": str((run_dir / f"trial_{best.number:04d}"
                                    / "dagger_student_plain.pt").resolve()),
        "selected_teacher_config": str((run_dir / f"trial_{best.number:04d}"
                                        / "teacher_config.json").resolve()),
    }
    summary_path = run_dir / "bo_summary.json"
    summary_tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    try:
        with open(summary_tmp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        os.replace(summary_tmp, summary_path)
    except Exception as e:
        try:
            if summary_tmp.exists():
                summary_tmp.unlink()
        except Exception:
            pass
        print(f"[plain-bo] WARNING: failed to write bo_summary.json atomically: {e}")

    print(f"[plain-bo] artifacts in {run_dir}")


if __name__ == "__main__":
    main()
