"""Tree-based & gradient-boosting benchmark for housing + Friedman 1/2/3.

Mirrors the existing ``mlp_benchmark_{housing,friedman{1,2,3}}.py`` and the
(now-deprecated ``xgb_benchmark_housing.py``) so that tree/boosting results are
directly comparable to the MLP and KirchhoffNet baselines on the same dataset,
splits, preprocessing, and metrics.

Methods (select with --method):
    xgboost               - XGBoost XGBRegressor (reg:squarederror, hist)
    lightgbm              - LightGBM LGBMRegressor (regression L2)
    catboost              - CatBoost CatBoostRegressor (RMSE)
    gradient_boosting     - sklearn GradientBoostingRegressor (squared_error)
    hist_gradient_boosting- sklearn HistGradientBoostingRegressor (squared_error)
    random_forest         - sklearn RandomForestRegressor
    extra_trees           - sklearn ExtraTreesRegressor

Datasets (select with --dataset):
    housing   - California Housing (8 features, regression)
    friedman1  - Friedman #1 synthetic (10 features, 5 relevant + 5 noise)
    friedman2  - Friedman #2 synthetic (4 features, ranges per-dim)
    friedman3  - Friedman #3 synthetic (4 features, ranges per-dim)

Data pipeline:
  - housing:    reuses ``train_script._load_california_housing_data`` and
                ``_make_data_split`` (80/20, seed=42) so the split is
                byte-identical to ``mlp_benchmark_housing.py``. Tree models are
                trained on RAW (un-normalized) features AND RAW targets
                (USD x 100k) per the finding in ``xgb_benchmark_housing.py`` /
                ARCHITECTURE.md line ~764: tree models are invariant to
                monotone feature transforms, and the MLP's min-max+standardized
                scaling actively hurts tree performance.
  - friedmanN:  reuses ``train_script.make_data_friedmanN`` loaders (20k LHS
                train + 4k uniform val, seed=42, sigma=1.0) so the splits are
                byte-identical to ``mlp_benchmark_friedmanN.py``. Tree models
                are trained on the normalized [0,1]^d inputs (which are
                monotone per-feature transforms of the raw inputs - tree
                splits are invariant) AND raw (denormalized) targets.

Capacity control (set params via --max-features / --min-samples-leaf /
--max-leaf-nodes / --n-estimators / --max-depth):
  - Approximate "params" = total internal (split) nodes across all trees (the
    same proxy used by the XGBoost benchmark). Reported as
    ``approx_node_count`` in ``final_metrics.txt``.
  - For sklearn ensembles the per-tree max parameters drive total capacity:
      max_leaf_nodes=2^max_depth would approximate a depth-bounded tree
      internal-node count; we expose both ``--max-depth`` and ``--max-leaf-nodes``
      so the user can dial capacity at a familiar level.
  - For XGBoost / LightGBM / CatBoost the standard ``max_depth`` knob is used.
  - For RandomForest / ExtraTrees use ``--n-estimators`` + ``--max-features``
    + ``--min-samples-leaf`` for capacity (no ``max_depth`` clamp).

Metrics:
  - Huber loss (delta=1.0) on STANDARDIZED targets, reported per round on
    train+val: directly comparable to MLP/KirchhoffNet reported val Huber.
  - MSE / RMSE / MAE / MAPE on RAW (original) targets in ``final_metrics.txt``
    and tracked per round alongside Huber.
  - Best round by val Huber is selected; final metrics use that round.

Outputs to --output (default ``./output/{method}_{dataset}``):
    config_snapshot.txt
    loss_history.txt       (round\ttrain_huber\tval_huber\tmae_orig\trmse_orig)
    loss_curve.png         (train+val Huber vs round, log y)
    output_fit.png         (scatter of pred vs target on val)
    final_metrics.txt
    model.{ext}            (json|txt|cbm|pkl depending on method)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

_THIS_FILE = Path(__file__).resolve()
sys.path.insert(0, str(_THIS_FILE.parent))

from config import OPTIM
from train_script import (
    _load_california_housing_data,
    _make_data_split,
    make_data_friedman1,
    make_data_friedman2,
    make_data_friedman3,
)


__all__ = [
    "COUNT_TREE_PARAMS",
    "TreeFitResult",
    "count_tree_params",
]

# Directory under output where fitted models will be pickled. When the chosen
# method supports a native binary dump (xgboost / lightgbm / catboost) the
# model is also written in that format alongside model.pkl for inspection.
_MODEL_EXT = {
    "xgboost": "json",
    "lightgbm": "txt",
    "catboost": "cbm",
    "gradient_boosting": "pkl",
    "hist_gradient_boosting": "pkl",
    "random_forest": "pkl",
    "extra_trees": "pkl",
}

COUNT_TREE_PARAMS = True


def _resolve_device(choice: str) -> str:
    """Resolve a ``--device`` choice ('auto' | 'cpu' | 'cuda') to a concrete
    device string. ``auto`` returns 'cuda' when a CUDA GPU is available,
    otherwise 'cpu'. An explicit 'cuda' on a GPU-less machine is honored here;
    the caller decides whether to warn + fall back to CPU."""
    if choice == "cuda":
        return "cuda"
    if choice == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Metrics / plotting helpers
# ============================================================================


def _huber_np(y_pred: np.ndarray, y_target: np.ndarray, delta: float = 1.0) -> float:
    """Huber loss (numpy, matches torch F.huber_loss semantics)."""
    err = y_pred - y_target
    abs_err = np.abs(err)
    quad = np.minimum(abs_err, delta)
    lin = abs_err - quad
    return float(np.mean(0.5 * quad * quad + delta * lin))


def _mape_np(y_pred: np.ndarray, y_target: np.ndarray) -> float:
    """Mean absolute percent error (%). Clips |y_target| denom at 1e-8."""
    return float(np.mean(np.abs((y_pred - y_target) / np.maximum(np.abs(y_target), 1e-8))) * 100.0)


def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        warnings.warn("matplotlib not installed; skipping plots", stacklevel=2)
        return None


def plot_loss_curve(history, val_history, save_path, title):
    plt = _import_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history, label="train", color="C0")
    ax.plot(val_history, label="val", color="C3")
    ax.set_xlabel("round")
    ax.set_ylabel("Huber loss (standardized)")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_output_fit(out_np, target_np, save_path, title):
    plt = _import_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(target_np.ravel(), out_np.ravel(), s=4, alpha=0.4, color="C2")
    lo = float(min(target_np.min(), out_np.min()))
    hi = float(max(target_np.max(), out_np.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y = x")
    ax.set_xlabel("target (raw units)")
    ax.set_ylabel("prediction (raw units)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Tree capacity counting
# ============================================================================


def _sklearn_internal_node_count(tree_) -> int:
    """Internal-node count for a single sklearn DecisionTreeRegressor (Tree)."""
    return int(getattr(tree_, "node_count", 0) - getattr(tree_, "n_leaves", 0))


def _sklearn_tree_predictor_internal_nodes(pred) -> int:
    """For HistGradientBoosting's private TreePredictor objects."""
    try:
        n_leaves = int(pred.get_n_leaf_nodes())
        n_nodes = int(len(pred.nodes))
        return max(0, n_nodes - n_leaves)
    except Exception:
        return 0


def _xgboost_internal_nodes(booster) -> int:
    """From ``xgb_benchmark_housing.count_boosted_params``."""
    try:
        dump = booster.get_dump()
        total_internal = 0
        for tree_str in dump:
            for line in tree_str.splitlines():
                if "yes=" in line and "leaf=" not in line:
                    total_internal += 1
        return total_internal
    except Exception:
        return 0


def _lightgbm_internal_nodes(booster) -> int:
    """LightGBM ``dump_model()['tree_info']`` lists each tree with its
    ``num_leaves``. For leaf-wise growth, internal nodes per tree ≈ num_leaves
    - 1, since leaf-wise trees are level-balanced minus the last layer.
    """
    try:
        model_dump = booster.dump_model()
        tree_info = model_dump.get("tree_info", []) or []
        if not tree_info:
            return 0
        total_internal = 0
        for t in tree_info:
            n_leaves = int(t.get("num_leaves", 0) or 0)
            if n_leaves > 0:
                total_internal += n_leaves - 1
        return max(0, total_internal)
    except Exception:
        return 0


def _catboost_internal_nodes(model) -> int:
    """CatBoost exposes ``get_tree_leaf_counts()`` -> per-tree leaf counts.
    For an oblivious tree with L leaves, there are (L - 1) internal split
    nodes; CatBoost trees are oblivious so this is exact.
    """
    try:
        n_trees = int(getattr(model, "tree_count_", 0))
        if n_trees <= 0:
            return 0
        leaves_per_tree = list(model.get_tree_leaf_counts())
        if not leaves_per_tree:
            return 0
        if len(leaves_per_tree) != n_trees:
            n_trees = len(leaves_per_tree)
        total_leaves = sum(int(x) for x in leaves_per_tree)
        return max(0, total_leaves - n_trees)
    except Exception:
        return 0


def _sklearn_ensemble_internal_nodes(estimator) -> int:
    """For sklearn ensembles. Handles RandomForest/ExtraTrees (estimators_
    is a 2-D ndarray of trees), GradientBoosting (estimators_ is a 1-D list
    of trees), and HistGradientBoosting (``_predictors`` is a list of iter ->
    [TreePredictor]) via a separate helper.
    """
    # HistGradientBoosting special path
    predictors = getattr(estimator, "_predictors", None)
    if predictors:
        total = 0
        try:
            for iter_preds in predictors:
                for pred in iter_preds:
                    total += _sklearn_tree_predictor_internal_nodes(pred)
        except Exception:
            pass
        if total > 0:
            return total

    # Standard ensembles: try estimators_ (1-D or 2-D iterable of trees).
    try:
        estimators_attr = getattr(estimator, "estimators_", None)
        if estimators_attr is None:
            return 0
        try:
            trees = list(estimators_attr)
        except TypeError:
            trees = [estimators_attr]
    except Exception:
        return 0

    flat: list = []
    for item in trees:
        try:
            row = list(item)
        except TypeError:
            row = [item]
        # For forest 2-D layout: each row already contains individual tree
        # objects. For boosting 1-D layout: each entry is a single tree.
        flat.extend(row)

    total = 0
    for t in flat:
        tree_obj = getattr(t, "tree_", None)
        if tree_obj is not None:
            total += _sklearn_internal_node_count(tree_obj)
        else:
            # Could be a HistGB TreePredictor or similar
            total += _sklearn_tree_predictor_internal_nodes(t)
    return total


def count_tree_params(model: object, method: str) -> int:
    """Best-effort total internal-split-node count across all trees."""
    try:
        if method == "xgboost":
            return _xgboost_internal_nodes(model)
        if method == "lightgbm":
            return _lightgbm_internal_nodes(model)
        if method == "catboost":
            return _catboost_internal_nodes(model)
        if method in ("gradient_boosting", "hist_gradient_boosting"):
            return _sklearn_ensemble_internal_nodes(model)
        if method in ("random_forest", "extra_trees"):
            return _sklearn_ensemble_internal_nodes(model)
    except Exception:
        pass
    return 0


# ============================================================================
# Data loaders (one per dataset)
# ============================================================================


def _stack_loader(loader) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate a torch DataLoader into numpy arrays (X, y)."""
    xs, ys = [], []
    for u, y in loader:
        xs.append(u)
        ys.append(y)
    return torch.cat(xs, dim=0).cpu().numpy().astype(np.float32), torch.cat(
        ys, dim=0
    ).cpu().numpy().astype(np.float32)


def _to_array(loader) -> tuple[np.ndarray, np.ndarray]:
    return _stack_loader(loader)


def _load_housing():
    """Returns (X_train, y_train_orig, X_val, y_val_orig, y_mean, y_std, meta).

    Targets are returned in RAW USD x 100k units (not z-scored). Features are
    kept RAW (un-normalized) because tree models are invariant to monotone
    feature transforms and the original xgb baseline showed that min-max
    features + standardized targets degraded tree val RMSE.
    """
    X, y_norm, y_mean, y_std = _load_california_housing_data()
    y_mean_f = float(y_mean)
    y_std_f = float(y_std)
    y_orig = y_norm * y_std_f + y_mean_f

    # Build a deterministic train/val split using the same seed and 80/20 ratio.
    n = X.shape[0]
    rng = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=rng)
    n_train = int(0.8 * n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    X_train = X[train_idx].numpy().astype(np.float32)
    X_val = X[val_idx].numpy().astype(np.float32)
    y_train = y_orig[train_idx].numpy().astype(np.float32).ravel()
    y_val = y_orig[val_idx].numpy().astype(np.float32).ravel()

    meta = dict(
        name="California Housing",
        n_train=int(X_train.shape[0]),
        n_val=int(X_val.shape[0]),
        n_features=int(X_train.shape[1]),
        target_units="USD x 100k (raw)",
        split="deterministic 80/20, seed=42, via _make_data_split",
        target_normalization_for_huber="z-score (mean=" + f"{y_mean_f:.6f}, std={y_std_f:.6f})",
    )
    return X_train, y_train, X_val, y_val, y_mean_f, y_std_f, meta


def _load_friedman(make_data_fn, name: str, target_noise_std: float):
    """Returns (X_train_norm, y_train_orig, X_val_norm, y_val_orig, ...).

    Inputs are min-max-normalized to [0,1] (identical to mlp/knet loaders).
    Targets are denormalized back to RAW target units for tree training, since
    tree models on raw vs standardized targets give materially different RMSE
    when the noise floor is comparable to the y_mean scale (sigma=1.0).
    """
    train_loader, val_loader, _, inverse_stats = make_data_fn(
        batch_size=int(OPTIM["batch_size"]),
        noise_std=target_noise_std,
        normalize_inputs=True,
    )
    X_train_norm, y_train_norm = _to_array(train_loader)
    X_val_norm, y_val_norm = _to_array(val_loader)

    y_mean_f = float(inverse_stats["y_mean"])
    y_std_f = float(inverse_stats["y_std"])
    y_train_orig = (y_train_norm * y_std_f) + y_mean_f
    y_val_orig = (y_val_norm * y_std_f) + y_mean_f
    y_train_orig = y_train_orig.astype(np.float32).ravel()
    y_val_orig = y_val_orig.astype(np.float32).ravel()

    meta = dict(
        name=name,
        n_train=int(X_train_norm.shape[0]),
        n_val=int(X_val_norm.shape[0]),
        n_features=int(X_train_norm.shape[1]),
        target_units="raw Friedman target units",
        split="20k LHS train / 4k uniform val (seed=42)",
        target_normalization_for_huber="z-score (mean="
        + f"{y_mean_f:.6f}, std={y_std_f:.6f})",
        target_noise_std=target_noise_std,
        input_normalization="min-max [0,1] per feature (train stats)",
    )
    return (
        X_train_norm,
        y_train_orig,
        X_val_norm,
        y_val_orig,
        y_mean_f,
        y_std_f,
        meta,
    )


DATASET_LOADERS = {
    "housing": _load_housing,
    "friedman1": lambda: _load_friedman(make_data_friedman1, "Friedman #1", 1.0),
    "friedman2": lambda: _load_friedman(make_data_friedman2, "Friedman #2", 1.0),
    "friedman3": lambda: _load_friedman(make_data_friedman3, "Friedman #3", 1.0),
}


# ============================================================================
# Per-round evaluation helper (computes 5 metrics from current predictions)
# ============================================================================


def _eval_round(
    pred_train_orig: np.ndarray,
    y_train_orig: np.ndarray,
    pred_val_orig: np.ndarray,
    y_val_orig: np.ndarray,
    y_mean_f: float,
    y_std_f: float,
    huber_delta: float,
) -> dict:
    """One round's training+validation metrics."""
    pred_train_norm = (pred_train_orig - y_mean_f) / y_std_f
    pred_val_norm = (pred_val_orig - y_mean_f) / y_std_f
    y_train_norm = (y_train_orig - y_mean_f) / y_std_f
    y_val_norm = (y_val_orig - y_mean_f) / y_std_f

    return dict(
        train_huber=_huber_np(pred_train_norm, y_train_norm, delta=huber_delta),
        val_huber=_huber_np(pred_val_norm, y_val_norm, delta=huber_delta),
        train_mae=float(np.mean(np.abs(pred_train_orig - y_train_orig))),
        val_mae=float(np.mean(np.abs(pred_val_orig - y_val_orig))),
        train_rmse=float(np.sqrt(np.mean((pred_train_orig - y_train_orig) ** 2))),
        val_rmse=float(np.sqrt(np.mean((pred_val_orig - y_val_orig) ** 2))),
        val_mape=_mape_np(pred_val_orig, y_val_orig),
    )


# ============================================================================
# Per-method training adapters
# ============================================================================


class TreeFitResult:
    """Mutable result filled in by each method's training adapter."""

    def __init__(self):
        self.model = None
        self.model_path: str | None = None
        self.history: list[dict] = []
        self.best_round: int = -1
        self.best_val: float = math.inf
        self.rounds_run: int = 0
        self.elapsed_s: float = 0.0
        self.extra: dict = {}


# ----- XGBoost ------------------------------------------------------------------


def _train_xgboost(
    X_train: np.ndarray,
    y_train_orig: np.ndarray,
    X_val: np.ndarray,
    y_val_orig: np.ndarray,
    y_mean_f: float,
    y_std_f: float,
    huber_delta: float,
    args,
    out_dir: Path,
) -> TreeFitResult:
    import xgboost as xgb

    esr = args.early_stopping_rounds if args.early_stopping_rounds > 0 else None

    params = dict(
        max_depth=args.max_depth,
        eta=args.lr,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        objective="reg:squarederror",
        tree_method="hist",
        seed=args.seed,
    )
    if args.gpu_available:
        params["device"] = "cuda"

    dtrain = xgb.DMatrix(X_train, label=y_train_orig)
    dval = xgb.DMatrix(X_val, label=y_val_orig)

    res = TreeFitResult()
    res.extra["params"] = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in params.items()}

    evals_result: dict = {}

    start = time.time()
    if esr is not None:
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=args.n_estimators,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=esr,
            maximize=False,
            verbose_eval=False,
            evals_result=evals_result,
        )
        best_iter = booster.best_iteration if hasattr(booster, "best_iteration") and booster.best_iteration is not None else len(evals_result.get("train", {}).get("rmse", [])) - 1
    else:
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=args.n_estimators,
            evals=[(dtrain, "train"), (dval, "val")],
            verbose_eval=False,
            evals_result=evals_result,
        )
        best_iter = len(evals_result.get("train", {}).get("rmse", [])) - 1

    rmse_history = evals_result.get("train", {}).get("rmse", [])
    rounds_run = len(rmse_history)

    # XGBoost reports native rmse (on raw targets) per round. Compute Huber in
    # standardized space and MAE/RMSE in raw space for the loss_history file.
    best_val = math.inf
    best_round = -1
    rounds_to_use = min(rounds_run, args.n_estimators)
    print_every = max(1, args.validate_every)
    for i in range(rounds_to_use):
        pred_val = booster.predict(dval, iteration_range=(0, i + 1))
        pred_train = booster.predict(dtrain, iteration_range=(0, i + 1))

        m = _eval_round(
            pred_train, y_train_orig,
            pred_val, y_val_orig,
            y_mean_f, y_std_f, huber_delta,
        )
        res.history.append({"round": i, **m})
        if m["val_huber"] < best_val - args.min_delta:
            best_val = float(m["val_huber"])
            best_round = i
        if args.verbose_history and (i % print_every == 0 or i == rounds_to_use - 1):
            print(
                f"  round {i:4d}  train={m['train_huber']:.6f}  val={m['val_huber']:.6f}  "
                f"val_mae={m['val_mae']:.4f}  val_rmse={m['val_rmse']:.4f}"
            )

    res.elapsed_s = time.time() - start
    res.rounds_run = rounds_to_use
    res.best_round = best_round if best_round >= 0 else (best_iter if best_iter >= 0 else rounds_to_use - 1)
    res.best_val = best_val if best_val != math.inf else (res.history[-1]["val_huber"] if res.history else float("nan"))
    res.best_iteration = int(best_iter) if best_iter is not None and best_iter >= 0 else res.best_round
    res.model = booster
    res.model_path = str(out_dir / f"model.{_MODEL_EXT['xgboost']}")
    booster.save_model(res.model_path)

    # Save predictions at the best round for downstream plots.
    res.final_pred_val_orig = booster.predict(dval, iteration_range=(0, res.best_round + 1))

    return res


# ----- LightGBM -----------------------------------------------------------------


def _train_lightgbm(
    X_train: np.ndarray,
    y_train_orig: np.ndarray,
    X_val: np.ndarray,
    y_val_orig: np.ndarray,
    y_mean_f: float,
    y_std_f: float,
    huber_delta: float,
    args,
    out_dir: Path,
) -> TreeFitResult:
    import lightgbm as lgb

    base_kwargs = dict(
        objective="regression",
        metric="rmse",
        learning_rate=args.lr,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth if args.max_depth > 0 else -1,
        num_leaves=args.max_leaf_nodes if args.max_leaf_nodes > 0 else (2 ** max(1, args.max_depth)),
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        min_child_weight=args.min_child_weight,
        random_state=args.seed,
        verbosity=-1,
    )
    if args.max_features is not None:
        if args.max_features <= 0.0 or args.max_features > 1.0:
            raise ValueError("--max-features must be in (0, 1] for LightGBM")
        base_kwargs["colsample_bytree"] = args.max_features
    res = TreeFitResult()
    res.extra["params"] = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in base_kwargs.items()}

    esr = args.early_stopping_rounds if args.early_stopping_rounds > 0 else 0

    dtrain = lgb.Dataset(X_train, label=y_train_orig)
    dval = lgb.Dataset(X_val, label=y_val_orig, reference=dtrain)

    callbacks = []
    if esr > 0:
        callbacks.append(lgb.early_stopping(stopping_rounds=esr, verbose=False))
    eval_log: dict = {}
    callbacks.append(lgb.record_evaluation(eval_log))

    params = {k: v for k, v in base_kwargs.items() if k != "n_estimators"}
    if args.gpu_available:
        params["device"] = "gpu"

    start = time.time()
    try:
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=args.n_estimators,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )
    except Exception:
        if params.get("device") == "gpu":
            print("[tree_benchmark] WARNING: LightGBM GPU build unavailable; falling back to CPU")
            params["device"] = "cpu"
            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=args.n_estimators,
                valid_sets=[dtrain, dval],
                valid_names=["train", "val"],
                callbacks=callbacks,
            )
        else:
            raise
    res.elapsed_s = time.time() - start

    rounds_run = booster.current_iteration()
    res.rounds_run = rounds_to_use = rounds_run

    # eval_log now has {split_name: {metric: [values]}} from record_evaluation
    rmse_history_val = eval_log.get("val", {}).get("rmse", [])

    best_val = math.inf
    best_round = -1
    best_iter = int(getattr(booster, "best_iteration", -1) or -1)
    print_every = max(1, args.validate_every)
    for i in range(rounds_to_use):
        pred_val = booster.predict(X_val, num_iteration=i + 1)
        pred_train = booster.predict(X_train, num_iteration=i + 1)
        m = _eval_round(
            pred_train, y_train_orig,
            pred_val, y_val_orig,
            y_mean_f, y_std_f, huber_delta,
        )
        res.history.append({"round": i, **m})
        if m["val_huber"] < best_val - args.min_delta:
            best_val = float(m["val_huber"])
            best_round = i
        if args.verbose_history and (i % print_every == 0 or i == rounds_to_use - 1):
            print(
                f"  round {i:4d}  train={m['train_huber']:.6f}  val={m['val_huber']:.6f}  "
                f"val_mae={m['val_mae']:.4f}  val_rmse={m['val_rmse']:.4f}"
            )

    res.best_round = best_round if best_round >= 0 else (best_iter if best_iter >= 0 else rounds_to_use - 1)
    res.best_val = best_val if best_val != math.inf else (res.history[-1]["val_huber"] if res.history else float("nan"))
    res.best_iteration = best_iter if best_iter >= 0 else res.best_round
    res.model = booster
    res.model_path = str(out_dir / f"model.{_MODEL_EXT['lightgbm']}")
    booster.save_model(res.model_path)

    res.final_pred_val_orig = booster.predict(X_val, num_iteration=res.best_round + 1)
    return res


# ----- CatBoost -----------------------------------------------------------------


def _train_catboost(
    X_train: np.ndarray,
    y_train_orig: np.ndarray,
    X_val: np.ndarray,
    y_val_orig: np.ndarray,
    y_mean_f: float,
    y_std_f: float,
    huber_delta: float,
    args,
    out_dir: Path,
) -> TreeFitResult:
    import catboost as cb

    train_kwargs = dict(
        loss_function="RMSE",
        learning_rate=args.lr,
        depth=args.max_depth if args.max_depth > 0 else 6,
        iterations=args.n_estimators,
        l2_leaf_reg=args.reg_lambda,
        random_seed=args.seed,
        verbose=False,
    )
    if args.early_stopping_rounds and args.early_stopping_rounds > 0:
        train_kwargs.update(
            early_stopping_rounds=args.early_stopping_rounds,
            eval_metric="RMSE",
            use_best_model=True,
        )
    if args.gpu_available:
        train_kwargs.update(task_type="GPU", devices="0")

    res = TreeFitResult()
    res.extra["params"] = {k: v for k, v in train_kwargs.items() if k != "verbose"}

    start = time.time()
    model = cb.CatBoostRegressor(**train_kwargs)
    try:
        if args.early_stopping_rounds and args.early_stopping_rounds > 0:
            model.fit(
                X_train,
                y_train_orig,
                eval_set=(X_val, y_val_orig),
            )
        else:
            model.fit(X_train, y_train_orig)
    except Exception:
        if train_kwargs.get("task_type") == "GPU":
            print("[tree_benchmark] WARNING: CatBoost GPU unavailable; falling back to CPU")
            train_kwargs["task_type"] = "CPU"
            train_kwargs.pop("devices", None)
            res.extra["params"] = {k: v for k, v in train_kwargs.items() if k != "verbose"}
            model = cb.CatBoostRegressor(**train_kwargs)
            if args.early_stopping_rounds and args.early_stopping_rounds > 0:
                model.fit(
                    X_train,
                    y_train_orig,
                    eval_set=(X_val, y_val_orig),
                )
            else:
                model.fit(X_train, y_train_orig)
        else:
            raise
    res.elapsed_s = time.time() - start

    # CatBoost exposes staged_predict via get_metric_on_step? No. Use a loop
    # calling model.predict with a tree_count parameter to get staged predictions
    # up to the best round.
    res.rounds_run = int(model.tree_count_)
    rounds_to_use = res.rounds_run

    best_val = math.inf
    best_round = -1
    print_every = max(1, args.validate_every)
    for i in range(rounds_to_use):
        pred_val = model.predict(X_val, ntree_end=i + 1, ntree_start=0)
        pred_train = model.predict(X_train, ntree_end=i + 1, ntree_start=0)
        m = _eval_round(
            pred_train, y_train_orig,
            pred_val, y_val_orig,
            y_mean_f, y_std_f, huber_delta,
        )
        res.history.append({"round": i, **m})
        if m["val_huber"] < best_val - args.min_delta:
            best_val = float(m["val_huber"])
            best_round = i
        if args.verbose_history and (i % print_every == 0 or i == rounds_to_use - 1):
            print(
                f"  round {i:4d}  train={m['train_huber']:.6f}  val={m['val_huber']:.6f}  "
                f"val_mae={m['val_mae']:.4f}  val_rmse={m['val_rmse']:.4f}"
            )

    res.best_round = best_round if best_round >= 0 else rounds_to_use - 1
    res.best_val = best_val if best_val != math.inf else (res.history[-1]["val_huber"] if res.history else float("nan"))
    res.model = model
    res.model_path = str(out_dir / f"model.{_MODEL_EXT['catboost']}")
    model.save_model(res.model_path)

    res.final_pred_val_orig = model.predict(X_val, ntree_end=res.best_round + 1, ntree_start=0)
    return res


# ----- sklearn GradientBoostingRegressor ----------------------------------------


def _train_sklearn_gradient_boosting(
    X_train: np.ndarray,
    y_train_orig: np.ndarray,
    X_val: np.ndarray,
    y_val_orig: np.ndarray,
    y_mean_f: float,
    y_std_f: float,
    huber_delta: float,
    args,
    out_dir: Path,
) -> TreeFitResult:
    from sklearn.ensemble import GradientBoostingRegressor

    kwargs = dict(
        n_estimators=args.n_estimators,
        learning_rate=args.lr,
        max_depth=args.max_depth if args.max_depth > 0 else 3,
        subsample=args.subsample,
        random_state=args.seed,
    )
    if args.max_features is not None and args.max_features > 0.0 and args.max_features <= 1.0:
        kwargs["max_features"] = args.max_features
    if args.min_samples_leaf is not None and args.min_samples_leaf > 0:
        kwargs["min_samples_leaf"] = int(args.min_samples_leaf)
    if args.max_leaf_nodes is not None and args.max_leaf_nodes > 0:
        kwargs["max_leaf_nodes"] = int(args.max_leaf_nodes)

    res = TreeFitResult()
    res.extra["params"] = kwargs

    # Skip X/y reshaping - sklearn accepts float64; cast for speed.
    X_train64 = X_train.astype(np.float64, copy=False)
    X_val64 = X_val.astype(np.float64, copy=False)
    y_train64 = y_train_orig.astype(np.float64, copy=False)

    if args.early_stopping_rounds and args.early_stopping_rounds > 0:
        kwargs.update(
            n_iter_no_change=int(args.early_stopping_rounds),
            validation_fraction=0.1,
            tol=1e-6,
        )

    start = time.time()
    model = GradientBoostingRegressor(**kwargs)
    model.fit(X_train64, y_train64)
    res.elapsed_s = time.time() - start

    rounds_run = len(list(model.staged_predict(X_val64)))
    res.rounds_run = rounds_run

    best_val = math.inf
    best_round = -1
    staged_val = list(model.staged_predict(X_val64))
    staged_train = list(model.staged_predict(X_train64))
    print_every = max(1, args.validate_every)
    for i, (pred_val, pred_train) in enumerate(zip(staged_val, staged_train)):
        pred_val_arr = np.asarray(pred_val).ravel().astype(np.float32)
        pred_train_arr = np.asarray(pred_train).ravel().astype(np.float32)
        m = _eval_round(
            pred_train_arr, y_train_orig,
            pred_val_arr, y_val_orig,
            y_mean_f, y_std_f, huber_delta,
        )
        res.history.append({"round": i, **m})
        if m["val_huber"] < best_val - args.min_delta:
            best_val = float(m["val_huber"])
            best_round = i
        if args.verbose_history and (i % print_every == 0 or i == rounds_run - 1):
            print(
                f"  round {i:4d}  train={m['train_huber']:.6f}  val={m['val_huber']:.6f}  "
                f"val_mae={m['val_mae']:.4f}  val_rmse={m['val_rmse']:.4f}"
            )

    res.best_round = best_round if best_round >= 0 else rounds_run - 1
    res.best_val = best_val if best_val != math.inf else (res.history[-1]["val_huber"] if res.history else float("nan"))
    res.model = model
    res.model_path = str(out_dir / f"model.{_MODEL_EXT['gradient_boosting']}")
    with open(res.model_path, "wb") as f:
        pickle.dump(model, f)
    res.final_pred_val_orig = np.asarray(staged_val[res.best_round]).ravel().astype(np.float32)
    return res


# ----- sklearn HistGradientBoostingRegressor ------------------------------------


def _train_hist_gradient_boosting(
    X_train: np.ndarray,
    y_train_orig: np.ndarray,
    X_val: np.ndarray,
    y_val_orig: np.ndarray,
    y_mean_f: float,
    y_std_f: float,
    huber_delta: float,
    args,
    out_dir: Path,
) -> TreeFitResult:
    from sklearn.ensemble import HistGradientBoostingRegressor

    X_train64 = X_train.astype(np.float64, copy=False)
    X_val64 = X_val.astype(np.float64, copy=False)
    y_train64 = y_train_orig.astype(np.float64, copy=False)
    y_val64 = y_val_orig.astype(np.float64, copy=False)

    kwargs = dict(
        loss="squared_error",
        learning_rate=args.lr,
        max_iter=args.n_estimators,
        max_depth=args.max_depth if args.max_depth > 0 else None,
        max_leaf_nodes=args.max_leaf_nodes if args.max_leaf_nodes > 0 else 31,
        l2_regularization=args.reg_lambda,
        random_state=args.seed,
    )
    if args.early_stopping_rounds and args.early_stopping_rounds > 0:
        kwargs.update(
            early_stopping=True,
            n_iter_no_change=int(args.early_stopping_rounds),
            validation_fraction=0.1,
            tol=1e-6,
        )

    res = TreeFitResult()
    res.extra["params"] = kwargs

    start = time.time()
    model = HistGradientBoostingRegressor(**kwargs)
    model.fit(X_train64, y_train64)
    res.elapsed_s = time.time() - start

    best_iter = int(getattr(model, "n_iter_", args.n_estimators))
    res.rounds_run = best_iter

    # HistGB has no staged_predict, so per-round history isn't naturally
    # available. To still produce a curve we score at progressively larger
    # n_iter via a re-fit would invalidate refitting. Instead we report
    # the val curve at a coarse grid: train_pred at 1..best_iter by cloning
    # the model isn't supported, but HistGB exposes ``_predict_iterations``
    # in newer scikit-learn.
    staged_val = None
    staged_train = None
    if hasattr(model, "staged_predict"):
        staged_val = list(model.staged_predict(X_val64))
        staged_train = list(model.staged_predict(X_train64))

    if staged_val is not None and len(staged_val) > 0:
        rounds_run = len(staged_val)
        res.rounds_run = rounds_run
        best_val = math.inf
        best_round = -1
        print_every = max(1, args.validate_every)
        for i, (pred_val, pred_train) in enumerate(zip(staged_val, staged_train)):
            pred_val_arr = np.asarray(pred_val).ravel().astype(np.float32)
            pred_train_arr = np.asarray(pred_train).ravel().astype(np.float32)
            m = _eval_round(
                pred_train_arr, y_train_orig,
                pred_val_arr, y_val_orig,
                y_mean_f, y_std_f, huber_delta,
            )
            res.history.append({"round": i, **m})
            if m["val_huber"] < best_val - args.min_delta:
                best_val = float(m["val_huber"])
                best_round = i
            if args.verbose_history and (i % print_every == 0 or i == rounds_run - 1):
                print(
                    f"  round {i:4d}  train={m['train_huber']:.6f}  val={m['val_huber']:.6f}  "
                    f"val_mae={m['val_mae']:.4f}  val_rmse={m['val_rmse']:.4f}"
                )
        res.best_round = best_round if best_round >= 0 else rounds_run - 1
        res.best_val = best_val if best_val != math.inf else (res.history[-1]["val_huber"] if res.history else float("nan"))
    else:
        # Fall back to a single training-evaluation point.
        pred_train = model.predict(X_train64).ravel().astype(np.float32)
        pred_val = model.predict(X_val64).ravel().astype(np.float32)
        m = _eval_round(
            pred_train, y_train_orig,
            pred_val, y_val_orig,
            y_mean_f, y_std_f, huber_delta,
        )
        m["round"] = best_iter
        res.history.append(m)
        res.best_round = best_iter
        res.best_val = float(m["val_huber"])

    res.model = model
    res.model_path = str(out_dir / f"model.{_MODEL_EXT['hist_gradient_boosting']}")
    with open(res.model_path, "wb") as f:
        pickle.dump(model, f)
    res.final_pred_val_orig = model.predict(X_val64).ravel().astype(np.float32)
    return res


# ----- Random Forest / Extra Trees ---------------------------------------------


def _train_sklearn_forest(
    method: str,
    X_train: np.ndarray,
    y_train_orig: np.ndarray,
    X_val: np.ndarray,
    y_val_orig: np.ndarray,
    y_mean_f: float,
    y_std_f: float,
    huber_delta: float,
    args,
    out_dir: Path,
) -> TreeFitResult:
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        RandomForestRegressor,
    )

    if method == "random_forest":
        ModelCls = RandomForestRegressor
    elif method == "extra_trees":
        ModelCls = ExtraTreesRegressor
    else:
        raise ValueError(method)

    n_estimators = args.n_estimators
    if n_estimators <= 0:
        n_estimators = 300

    kwargs = dict(
        n_estimators=int(n_estimators),
        n_jobs=-1 if args.parallel else 1,
        random_state=args.seed,
    )
    if args.max_features is not None and args.max_features > 0.0:
        kwargs["max_features"] = args.max_features
    else:
        kwargs["max_features"] = 1.0
    if args.max_depth is not None and args.max_depth > 0:
        kwargs["max_depth"] = int(args.max_depth)
    if args.min_samples_leaf is not None and args.min_samples_leaf > 0:
        kwargs["min_samples_leaf"] = int(args.min_samples_leaf)
    if args.max_leaf_nodes is not None and args.max_leaf_nodes > 0:
        kwargs["max_leaf_nodes"] = int(args.max_leaf_nodes)
    if method == "extra_trees":
        # ExtraTrees-compatible kwargs only.
        pass

    res = TreeFitResult()
    res.extra["params"] = kwargs

    X_train64 = X_train.astype(np.float64, copy=False)
    X_val64 = X_val.astype(np.float64, copy=False)
    y_train64 = y_train_orig.astype(np.float64, copy=False)

    start = time.time()
    model = ModelCls(**kwargs)
    model.fit(X_train64, y_train64)
    res.elapsed_s = time.time() - start

    # Aggregated bagging-style loss history: cumulative mean over first k trees.
    estimators = list(model.estimators_)
    n = len(estimators)
    res.rounds_run = n

    pred_val_acc = np.zeros(X_val64.shape[0], dtype=np.float64)
    pred_train_acc = np.zeros(X_train64.shape[0], dtype=np.float64)

    if args.early_stopping_rounds and args.early_stopping_rounds > 0:
        # No native early stopping; treat as "no early stopping" for forests.
        pass

    best_val = math.inf
    best_round = -1
    interval = max(1, n // max(1, args.history_points - 1)) if args.history_points > 1 else n
    sample_idx = sorted({min(n - 1, i) for i in range(0, n, interval)} | {n - 1})
    history_round = 0
    for k in range(1, n + 1):
        pred_val_acc += np.asarray(estimators[k - 1].predict(X_val64), dtype=np.float64)
        pred_train_acc += np.asarray(estimators[k - 1].predict(X_train64), dtype=np.float64)
        if (k == n) or ((k - 1) in sample_idx):
            pred_val_arr = (pred_val_acc / k).astype(np.float32)
            pred_train_arr = (pred_train_acc / k).astype(np.float32)
            m = _eval_round(
                pred_train_arr, y_train_orig,
                pred_val_arr, y_val_orig,
                y_mean_f, y_std_f, huber_delta,
            )
            m["round"] = history_round
            m["trees"] = k
            res.history.append(m)
            if m["val_huber"] < best_val - args.min_delta:
                best_val = float(m["val_huber"])
                best_round = history_round
                res.extra["best_trees_count"] = k
            if args.verbose_history and ((k == n) or ((k - 1) in sample_idx)):
                print(
                    f"  trees={k:4d}  train={m['train_huber']:.6f}  val={m['val_huber']:.6f}  "
                    f"val_mae={m['val_mae']:.4f}  val_rmse={m['val_rmse']:.4f}"
                )
            history_round += 1

    # Default "best" = full-ensemble aggregated mean if no improvement.
    res.best_round = best_round if best_round >= 0 else len(res.history) - 1
    res.best_val = best_val if best_val != math.inf else (res.history[-1]["val_huber"] if res.history else float("nan"))
    res.model = model
    res.model_path = str(out_dir / f"model.{_MODEL_EXT[method]}")
    with open(res.model_path, "wb") as f:
        pickle.dump(model, f)
    pred_val_full = np.asarray(model.predict(X_val64), dtype=np.float32).ravel()
    res.final_pred_val_orig = pred_val_full
    return res


TRAINERS = {
    "xgboost": _train_xgboost,
    "lightgbm": _train_lightgbm,
    "catboost": _train_catboost,
    "gradient_boosting": _train_sklearn_gradient_boosting,
    "hist_gradient_boosting": _train_hist_gradient_boosting,
}


def _train_random_forest(*a, **kw) -> TreeFitResult:
    return _train_sklearn_forest("random_forest", *a, **kw)


def _train_extra_trees(*a, **kw) -> TreeFitResult:
    return _train_sklearn_forest("extra_trees", *a, **kw)


TRAINERS["random_forest"] = _train_random_forest
TRAINERS["extra_trees"] = _train_extra_trees


# ============================================================================
# Main
# ============================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tree / gradient-boosting benchmark across housing + Friedman 1/2/3. "
            "Same data splits, preprocessing, and metrics as the MLP / KNet baselines."
        ),
    )
    parser.add_argument("--method", required=True,
                        choices=sorted(list(TRAINERS.keys())),
                        help="Tree-based method.")
    parser.add_argument("--dataset", required=True,
                        choices=sorted(list(DATASET_LOADERS.keys())),
                        help="Regression task.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: ./output/{method}_{dataset}).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Device 'auto' | 'cpu' | 'cuda' (default: auto-detect). "
                             "'auto' uses CUDA when available. Only xgboost, lightgbm "
                             "and catboost support GPU; sklearn methods always run on "
                             "CPU. A requested 'cuda' falls back to CPU if unavailable.")
    parser.add_argument("--huber-delta", type=float, default=1.0,
                        help="Delta for Huber loss on standardized targets (default: 1.0).")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="Min val-huber improvement to count as 'best round' (default: 1e-4).")
    parser.add_argument("--validate-every", type=int, default=1,
                        help="Record metrics every N rounds (default: 1).")
    parser.add_argument("--history-points", type=int, default=50,
                        help="History sample density for non-boosting methods (default: 50).")
    parser.add_argument("--verbose-history", action="store_true",
                        help="Print per-evaluation-round summary line.")

    # Boosting controls.
    parser.add_argument("--n-estimators", type=int, default=1000,
                        help="Max boosting rounds / total trees (default: 1000).")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Learning rate (eta) for boosting methods (default: 0.05).")
    parser.add_argument("--max-depth", type=int, default=6,
                        help="Tree max depth. Use 0 to let it be unrestricted "
                             "for sklearn (default: 6).")
    parser.add_argument("--max-leaf-nodes", type=int, default=0,
                        help="Cap on leaves per tree (sklearn/LightGBM only). 0 means "
                             "use --max-depth / method default (default: 0).")
    parser.add_argument("--subsample", type=float, default=0.8,
                        help="Row subsample ratio for boosting (default: 0.8).")
    parser.add_argument("--colsample-bytree", type=float, default=0.8,
                        help="Feature subsample ratio for boosting (default: 0.8).")
    parser.add_argument("--reg-alpha", type=float, default=0.1,
                        help="L1 reg weight (default: 0.1).")
    parser.add_argument("--reg-lambda", type=float, default=1.0,
                        help="L2 reg weight (default: 1.0).")
    parser.add_argument("--min-child-weight", type=float, default=5.0,
                        help="Min sum-of-instance-weight per leaf (XGBoost).")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Min loss reduction to make a split (XGBoost).")
    parser.add_argument("--early-stopping-rounds", type=int, default=30,
                        help="Patience for early stopping (0 disables; default: 30).")
    parser.add_argument("--max-features", type=float, default=None,
                        help="Feature subsample ratio per tree. None = let method "
                             "decide; for forests default = 1.0.")

    # Random-forest / sklearn knobs.
    parser.add_argument("--min-samples-leaf", type=int, default=10,
                        help="Min samples per leaf for sklearn ensembles (default: 10).")
    parser.add_argument("--parallel", action="store_true",
                        help="Allow sklearn ensembles to use multi-cores (default: single).")
    return parser


def main():
    args = _build_parser().parse_args()

    if args.output is None:
        args.output = Path(f"./output/{args.method}_{args.dataset}")
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.method not in TRAINERS:
        parser.error(f"unknown method: {args.method}")
    if args.dataset not in DATASET_LOADERS:
        parser.error(f"unknown dataset: {args.dataset}")

    device = _resolve_device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[tree_benchmark] WARNING: --device cuda requested but no CUDA GPU "
              "detected; falling back to CPU")
        device = "cpu"
    args.device = device
    args.gpu_available = device == "cuda"

    X_train, y_train, X_val, y_val, y_mean_f, y_std_f, meta = DATASET_LOADERS[args.dataset]()

    print(
        f"[tree_benchmark] method={args.method} dataset={args.dataset} "
        f"n_train={meta['n_train']} n_val={meta['n_val']} "
        f"n_features={meta['n_features']} device={device} output={out_dir}"
    )

    if device == "cuda" and args.method in (
        "gradient_boosting",
        "hist_gradient_boosting",
        "random_forest",
        "extra_trees",
    ):
        print(
            f"[tree_benchmark] NOTE: {args.method} has no GPU support "
            "(sklearn); running on CPU"
        )

    with open(out_dir / "config_snapshot.txt", "w") as f:
        f.write(f"model: {args.method}\n")
        f.write(f"dataset: {meta['name']} ({meta['split']})\n")
        f.write(f"n_train: {meta['n_train']}  n_val: {meta['n_val']}  n_features: {meta['n_features']}\n")
        f.write(f"target_units: {meta['target_units']}\n")
        f.write(f"target_normalization_for_huber: {meta['target_normalization_for_huber']}\n")
        f.write(f"input_normalization: {meta.get('input_normalization', 'raw features')}\n")
        f.write(f"target_noise_std: {meta.get('target_noise_std', 'n/a')}\n")
        f.write(f"\n# hyperparameters\n")
        keys = [
            "n_estimators", "lr", "max_depth", "max_leaf_nodes",
            "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
            "min_child_weight", "gamma", "early_stopping_rounds",
            "min_samples_leaf", "max_features", "seed", "huber_delta",
            "validate_every", "min_delta", "history_points", "device",
        ]
        for k in keys:
            v = getattr(args, k, None)
            f.write(f"{k}: {v}\n")
        f.write(f"\n# model_kwargs resolved after fit (see below)\n")
        f.write(f"model_kwargs: <filled below>\n")

    trainer = TRAINERS[args.method]
    res = trainer(
        X_train, y_train,
        X_val, y_val,
        y_mean_f, y_std_f, args.huber_delta,
        args, out_dir,
    )

    # Append the resolved model hyperparameters (filled in by the trainer).
    with open(out_dir / "config_snapshot.txt", "a") as f:
        f.write("\n# method_params_resolved\n")
        for k, v in res.extra.items():
            if k == "params":
                f.write("model_kwargs:\n")
                for pk, pv in v.items():
                    f.write(f"  {pk}: {pv}\n")
            else:
                f.write(f"{k}: {v}\n")

    with open(out_dir / "loss_history.txt", "w") as f:
        f.write("round\ttrain_huber\tval_huber\ttrain_mae_orig\tval_mae_orig\t"
                "train_rmse_orig\tval_rmse_orig\tval_mape_orig\n")
        for h in res.history:
            f.write(
                f"{h.get('round', 0)}\t{h.get('train_huber', float('nan')):.6f}\t"
                f"{h.get('val_huber', float('nan')):.6f}\t"
                f"{h.get('train_mae', float('nan')):.6f}\t{h.get('val_mae', float('nan')):.6f}\t"
                f"{h.get('train_rmse', float('nan')):.6f}\t{h.get('val_rmse', float('nan')):.6f}\t"
                f"{h.get('val_mape', float('nan')):.6f}\n"
            )

    plot_loss_curve(
        [h.get("train_huber", float("nan")) for h in res.history],
        [h.get("val_huber", float("nan")) for h in res.history],
        save_path=str(out_dir / "loss_curve.png"),
        title=f"{args.method} (n_est={args.n_estimators}, lr={args.lr}, "
              f"depth={args.max_depth}, leaves={args.max_leaf_nodes}) — {args.dataset}",
    )

    plot_output_fit(
        res.final_pred_val_orig,
        y_val,
        save_path=str(out_dir / "output_fit.png"),
        title=f"{args.method} on {args.dataset} — output fit (best round={res.best_round})",
    )

    n_params = count_tree_params(res.model, args.method)
    best = res.history[res.best_round] if 0 <= res.best_round < len(res.history) else {}

    with open(out_dir / "final_metrics.txt", "w") as f:
        f.write(f"model: {args.method}\n")
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"device: {args.device}\n")
        f.write(f"approx_node_count: {n_params}\n")
        f.write(f"rounds_run: {res.rounds_run}\n")
        f.write(f"best_round: {res.best_round}\n")
        f.write(f"best_val_huber: {best.get('val_huber', float('nan')):.6f}\n")
        f.write(f"best_train_huber: {best.get('train_huber', float('nan')):.6f}\n")
        f.write(f"best_val_mae_orig: {best.get('val_mae', float('nan')):.6f}\n")
        f.write(f"best_val_rmse_orig: {best.get('val_rmse', float('nan')):.6f}\n")
        f.write(f"best_val_mape_orig: {best.get('val_mape', float('nan')):.6f}\n")
        f.write(f"elapsed_seconds: {res.elapsed_s:.2f}\n")
        if res.model_path:
            f.write(f"model_path: {res.model_path}\n")

    print(
        f"[tree_benchmark] best_val_huber={best.get('val_huber', float('nan')):.6f} "
        f"@ round {res.best_round}  val_mae_orig={best.get('val_mae', float('nan')):.4f}  "
        f"val_rmse_orig={best.get('val_rmse', float('nan')):.4f}  "
        f"approx_node_count={n_params}  elapsed={res.elapsed_s:.1f}s"
    )
    print(f"[tree_benchmark] artifacts in {out_dir}")


if __name__ == "__main__":
    main()
