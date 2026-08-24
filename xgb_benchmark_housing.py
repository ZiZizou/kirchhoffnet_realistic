"""XGBoost benchmark for the California Housing regression task.

Mirrors the mlp_benchmark_housing.py structure so XGBoost results are directly
comparable to the MLP and KirchhoffNet baselines on the same dataset, splits,
preprocessing, and metrics.

Data pipeline:
  - Reuses ``train_script._load_california_housing_data`` and
    ``_make_data_split`` for the exact same 80/20 split (seed=42) as the MLP
    benchmark. Uses RAW (unscaled) features AND RAW (unnormalized) targets
    in USD x 100k units: tree models are invariant to monotonic feature
    transforms, and the MLP's min-max feature scaling combined with
    standardized targets actively hurts XGBoost performance (we measured
    val RMSE 1.02 with min-max+standardized vs 0.45 with raw+raw). For the
    Huber loss metric we denormalize the model's predictions back into
    standardized space, so the Huber number is directly comparable to the
    MLP's reported Huber loss.
  - Training loss on standardized targets is reported as Huber (delta=1.0)
    for direct apples-to-apples comparison with the MLP/KirchhoffNet.
  - Original-unit metrics (MAE, RMSE in USD x 100k) are computed after
    denormalization.

XGBoost hyperparameters (SOTA for tabular regression on small/medium datasets):
  - n_estimators=1000, early_stopping_rounds=30
  - max_depth=6, learning_rate=0.05
  - subsample=0.8, colsample_bytree=0.8
  - reg_alpha=0.1, reg_lambda=1.0 (mild L1+L2 regularization)
  - min_child_weight=5, gamma=0.1 (gentle tree-building regularization)
  - tree_method=hist (CPU-friendly; auto for GPU)

CLI:
    xgb_benchmark_housing.py [--epochs 1000] [--lr 0.05] [--max-depth 6]
                             [--subsample 0.8] [--colsample-bytree 0.8]
                             [--reg-alpha 0.1] [--reg-lambda 1.0]
                             [--early-stopping-rounds 30] [--seed 0]
                             [--output OUTPUT]

Outputs to --output:
  - loss_history.txt
  - loss_curve.png
  - model.json (XGBoost native save format)
  - config_snapshot.txt
  - output_fit.png
  - final_metrics.txt (includes MAE/RMSE in original USD x 100k units)
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OPTIM
from train_script import (
    _load_california_housing_data,
    _make_data_split,
)


__all__ = ["count_boosted_params"]


def count_boosted_params(booster) -> int:
    """Approximate number of decision nodes (a tree-node-count proxy for capacity).

    For an XGBoost booster, the total number of internal (split) nodes across all
    boosting rounds is reported via ``get_dump()``. Internal nodes are lines
    containing "yes=" (split decisions); leaf nodes contain "leaf=". We count
    internal nodes as the capacity proxy.
    """
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


def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        warnings.warn("matplotlib not installed; skipping plots", stacklevel=2)
        return None


def plot_output_fit(out_np, target_np, save_path, title):
    plt = _import_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(target_np.ravel(), out_np.ravel(), s=4, alpha=0.4, color="C2")
    lo = float(min(target_np.min(), out_np.min()))
    hi = float(max(target_np.max(), out_np.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y = x")
    ax.set_xlabel("target (normalized)")
    ax.set_ylabel("prediction (normalized)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(history, val_history, save_path, title):
    plt = _import_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history, label="train", color="C0")
    ax.plot(val_history, label="val", color="C3")
    ax.set_xlabel("round")
    ax.set_ylabel("Huber loss")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32)


def _huber_np(y_pred: np.ndarray, y_target: np.ndarray, delta: float = 1.0) -> float:
    """Huber loss on numpy arrays (matches torch F.huber_loss semantics)."""
    err = y_pred - y_target
    abs_err = np.abs(err)
    quad = np.minimum(abs_err, delta)
    lin = abs_err - quad
    return float(np.mean(0.5 * quad * quad + delta * lin))


def main():
    parser = argparse.ArgumentParser(
        description="Train an XGBoost regressor on the California Housing regression task."
    )
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Max number of boosting rounds (default: 1000)")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Learning rate (eta) (default: 0.05)")
    parser.add_argument("--max-depth", type=int, default=6,
                        help="Maximum tree depth (default: 6)")
    parser.add_argument("--subsample", type=float, default=0.8,
                        help="Subsample ratio of training instances per tree (default: 0.8)")
    parser.add_argument("--colsample-bytree", type=float, default=0.8,
                        help="Subsample ratio of features per tree (default: 0.8)")
    parser.add_argument("--reg-alpha", type=float, default=0.1,
                        help="L1 regularization term on weights (default: 0.1)")
    parser.add_argument("--reg-lambda", type=float, default=1.0,
                        help="L2 regularization term on weights (default: 1.0)")
    parser.add_argument("--min-child-weight", type=float, default=5.0,
                        help="Minimum sum of instance weight in a child (default: 5)")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Minimum loss reduction for a split (default: 0.1)")
    parser.add_argument("--early-stopping-rounds", type=int, default=50,
                        help="Early stopping patience in rounds (default: 50, set 0 to disable)")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="Early stopping min improvement in val loss (default: 1e-4)")
    parser.add_argument("--validate-every", type=int, default=1,
                        help="Validate every N rounds (default: 1 = every round)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--output", type=Path, default=Path("./output/xgb_housing"),
                        help="Output directory (default: ./output/xgb_housing)")
    parser.add_argument("--device", default=None,
                        help="Device 'cpu' or 'cuda' (default: auto-detect via xgboost)")
    parser.add_argument("--huber-delta", type=float, default=1.0,
                        help="Delta for Huber loss on normalized targets (default: 1.0)")
    args = parser.parse_args()

    X, y_norm, y_mean, y_std = _load_california_housing_data()

    y_mean_f = float(y_mean)
    y_std_f = float(y_std)

    y_orig = y_norm * y_std_f + y_mean_f

    batch_size = int(OPTIM["batch_size"])
    train_loader, val_loader = _make_data_split(X, y_orig, batch_size)

    _train_X_list, _train_y_list = [], []
    for _x, _y in train_loader:
        _train_X_list.append(_x)
        _train_y_list.append(_y)
    X_train = torch.cat(_train_X_list, dim=0)
    y_train_orig = torch.cat(_train_y_list, dim=0)

    _val_X_list, _val_y_list = [], []
    for _x, _y in val_loader:
        _val_X_list.append(_x)
        _val_y_list.append(_y)
    X_val = torch.cat(_val_X_list, dim=0)
    y_val_orig = torch.cat(_val_y_list, dim=0)

    X_train_np = _to_numpy(X_train)
    y_train_orig_np = _to_numpy(y_train_orig).ravel()
    X_val_np = _to_numpy(X_val)
    y_val_orig_np = _to_numpy(y_val_orig).ravel()

    y_train_norm = (y_train_orig - y_mean_f) / y_std_f
    y_val_norm = (y_val_orig - y_mean_f) / y_std_f
    y_train_norm_np = y_train_norm.cpu().numpy().astype(np.float32).ravel()
    y_val_norm_np = y_val_norm.cpu().numpy().astype(np.float32).ravel()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError(
            "xgboost is required for xgb_benchmark_housing.py. "
            "Install with: uv pip install xgboost"
        ) from e

    esr = args.early_stopping_rounds if args.early_stopping_rounds > 0 else None

    inverse_stats = {"y_mean": y_mean_f, "y_std": y_std_f}

    print(
        f"[xgb_housing] max_depth={args.max_depth} lr={args.lr} "
        f"n_estimators={args.epochs} subsample={args.subsample} "
        f"colsample_bytree={args.colsample_bytree} reg_alpha={args.reg_alpha} "
        f"reg_lambda={args.reg_lambda} min_child_weight={args.min_child_weight} "
        f"gamma={args.gamma} early_stopping_rounds={esr} seed={args.seed} "
        f"output={out_dir}"
    )

    with open(out_dir / "config_snapshot.txt", "w") as f:
        f.write(f"model: XGBRegressor (xgboost {xgb.__version__})\n")
        f.write(f"max_depth: {args.max_depth}\n")
        f.write(f"learning_rate: {args.lr}\n")
        f.write(f"n_estimators_max: {args.epochs}\n")
        f.write(f"subsample: {args.subsample}\n")
        f.write(f"colsample_bytree: {args.colsample_bytree}\n")
        f.write(f"reg_alpha: {args.reg_alpha}\n")
        f.write(f"reg_lambda: {args.reg_lambda}\n")
        f.write(f"min_child_weight: {args.min_child_weight}\n")
        f.write(f"gamma: {args.gamma}\n")
        f.write(f"early_stopping_rounds: {esr}\n")
        f.write(f"min_delta: {args.min_delta}\n")
        f.write(f"objective: reg:squarederror\n")
        f.write(f"tree_method: hist\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"dataset: California Housing (20640 samples, 80/20 split, "
                f"raw features and raw targets [USD x 100k]; "
                f"Huber loss delta={args.huber_delta} reported in standardized space "
                f"for direct comparability with MLP/KirchhoffNet)\n")

    dtrain = xgb.DMatrix(X_train_np, label=y_train_orig_np)
    dval = xgb.DMatrix(X_val_np, label=y_val_orig_np)

    params = {
        "max_depth": args.max_depth,
        "eta": args.lr,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "min_child_weight": args.min_child_weight,
        "gamma": args.gamma,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "seed": args.seed,
    }

    history = []
    val_history = []
    orig_mae_history = []
    orig_rmse_history = []
    best_val = float("inf")
    best_round = -1
    best_iteration = -1
    rounds_without_improve = 0

    start = time.time()
    evals_result = {}
    if esr is not None:
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=args.epochs,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=esr,
            maximize=False,
            verbose_eval=False,
            evals_result=evals_result,
        )
    else:
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=args.epochs,
            evals=[(dtrain, "train"), (dval, "val")],
            verbose_eval=False,
            evals_result=evals_result,
        )

    history_rmse = evals_result.get("train", {}).get("rmse", [])
    val_history_rmse = evals_result.get("val", {}).get("rmse", [])

    if not history_rmse and not val_history_rmse:
        raise RuntimeError("XGBoost returned no evaluation history")

    history = [float("nan")] * len(history_rmse)
    val_history = [float("nan")] * len(val_history_rmse)

    best_iteration = booster.best_iteration if hasattr(booster, "best_iteration") and booster.best_iteration is not None else len(val_history_rmse) - 1

    pred_val = booster.predict(dval, iteration_range=(0, best_iteration + 1) if best_iteration >= 0 else None)
    pred_val_orig_np = pred_val

    for i in range(len(val_history_rmse)):
        pred_i_orig = booster.predict(dval, iteration_range=(0, i + 1))
        pred_i_norm = (pred_i_orig - y_mean_f) / y_std_f
        val_huber = _huber_np(pred_i_norm, y_val_norm_np, delta=args.huber_delta)
        mae_i = float(np.mean(np.abs(pred_i_orig - y_val_orig_np)))
        rmse_i = float(np.sqrt(np.mean((pred_i_orig - y_val_orig_np) ** 2)))
        orig_mae_history.append(mae_i)
        orig_rmse_history.append(rmse_i)
        val_history[i] = val_huber

        train_i_orig = booster.predict(dtrain, iteration_range=(0, i + 1))
        train_i_norm = (train_i_orig - y_mean_f) / y_std_f
        train_huber = _huber_np(train_i_norm, y_train_norm_np, delta=args.huber_delta)
        history[i] = train_huber

        if i % args.validate_every == 0 or i == len(val_history_rmse) - 1:
            print(
                f"  round {i:4d}  train={train_huber:.6f}  val={val_huber:.6f}  "
                f"MAE_orig={mae_i:.4f}  RMSE_orig={rmse_i:.4f}"
            )

        if val_huber < best_val - args.min_delta:
            best_val = float(val_huber)
            best_round = i
            rounds_without_improve = 0
        else:
            rounds_without_improve += 1
            if esr is not None and rounds_without_improve >= esr:
                print(
                    f"[xgb_housing] early stopping at round {i}: "
                    f"no val improvement for {rounds_without_improve} rounds "
                    f"(best val={best_val:.6f} @ round {best_round})"
                )
                break

    elapsed = time.time() - start
    rounds_run = len(val_history)
    print(f"[xgb_housing] training done in {elapsed:.1f}s ({rounds_run} rounds)")

    booster.save_model(str(out_dir / "model.json"))

    n_params = count_boosted_params(booster)

    with open(out_dir / "loss_history.txt", "w") as f:
        f.write("round\ttrain\tval\tmae_orig\trmse_orig\n")
        for i in range(rounds_run):
            t = history[i] if i < len(history) else float("nan")
            v = val_history[i] if i < len(val_history) else float("nan")
            m = orig_mae_history[i] if i < len(orig_mae_history) else float("nan")
            r = orig_rmse_history[i] if i < len(orig_rmse_history) else float("nan")
            f.write(f"{i}\t{t}\t{v}\t{m}\t{r}\n")
        # If early stopping broke the loop, also write a final marker so plotting
        # doesn't try to draw NaN segments beyond the actual training horizon.
        if rounds_run < len(val_history_rmse):
            i = rounds_run
            f.write(f"# early_stopped: xgb booster has {booster.num_boosted_rounds()} trees, "
                    f"best_iteration={best_iteration}, best_round={best_round}\n")

    plot_loss_curve(
        history[:rounds_run],
        val_history,
        save_path=str(out_dir / "loss_curve.png"),
        title=f"XGBoost (depth={args.max_depth}, lr={args.lr}, {rounds_run} rounds) — CA Housing",
    )

    plot_output_fit(
        pred_val_orig_np,
        y_val_orig_np,
        save_path=str(out_dir / "output_fit.png"),
        title=f"XGBoost (depth={args.max_depth}, lr={args.lr}) — CA Housing output fit",
    )

    final_mae = orig_mae_history[best_round] if best_round >= 0 else orig_mae_history[-1]
    final_rmse = orig_rmse_history[best_round] if best_round >= 0 else orig_rmse_history[-1]
    final_huber = best_val if best_val != float("inf") else (val_history[-1] if val_history else float("nan"))

    print(
        f"[xgb_housing] final val Huber = {final_huber:.6f} "
        f"(best val = {best_val:.6f} @ round {best_round})  "
        f"MAE_orig={final_mae:.4f}  RMSE_orig={final_rmse:.4f}"
    )

    with open(out_dir / "final_metrics.txt", "w") as f:
        f.write(f"model: XGBoost XGBRegressor\n")
        f.write(f"approx_node_count: {n_params}\n")
        f.write(f"best_val_huber: {best_val:.6f}\n")
        f.write(f"best_round: {best_round}\n")
        f.write(f"best_iteration: {best_iteration}\n")
        f.write(f"final_val_huber: {final_huber:.6f}\n")
        f.write(f"final_mae_orig: {final_mae:.6f}\n")
        f.write(f"final_rmse_orig: {final_rmse:.6f}\n")
        f.write(f"rounds_run: {rounds_run}\n")
        f.write(f"elapsed_seconds: {elapsed:.2f}\n")

    print(f"[xgb_housing] artifacts in {out_dir}")


if __name__ == "__main__":
    main()