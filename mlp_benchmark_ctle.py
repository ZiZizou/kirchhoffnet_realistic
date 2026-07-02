"""MLP baseline for the CTLE inverse-design regression task.

Mirrors the role of ``mlp_benchmark.py`` (smooth2d) and
``mlp_benchmark_housing.py`` (California Housing) for the CTLE problem that
``train_ctle.py`` solves via knowledge distillation from a pre-trained
``RegimeAwareMoE`` teacher.

The CTLE task maps a 4-dim spec vector ``[power, jitter, height, width]``
to a 7-dim unbounded logit vector ``[fW, current, ind, Rd, Cs, Rs, VDD]``.
The teacher (loaded from a state-dict checkpoint such as
``dagger_student_moe.pt``) labels a synthetic dataset of ``--n-train`` train
specs and ``--n-val`` val specs sampled from ``SPEC_RANGES``. Targets are
per-dimension zero-mean/unit-variance normalized (matching train_ctle.py).

By default (``--q75-input`` on), the raw 4-dim specs are passed through the
teacher's ``scale_input()`` which applies log10 + StandardScaler/Q75
expansion to produce 8 well-conditioned features. This matches the input
space the teacher itself operates in and avoids forcing the MLP to learn the
nonlinear Q75 transformation from scratch. With ``--no-q75-input`` the MLP
receives raw 4-dim specs directly (original behavior).

The MLP regression head shares the same (normalized) target space as the
KirchhoffNet student. Best epoch (val MSE) is restored and final denormalized
metrics are reported alongside per-dim MSE / R² in the original (logit)
space.

CLI:
    mlp_benchmark_ctle.py --teacher-path /path/to/dagger_student_moe.pt \\
        [--hidden-dim 256] [--num-layers 2] [--epochs 800] \\
        [--lr 3e-4] [--output ./output/mlp_ctle] [--device DEVICE]

Outputs to ``--output`` (defaults to ``./output/mlp_ctle``):
  - ``config_snapshot.txt``
  - ``loss_history.txt`` (per-epoch train / val / per-dim MSE / per-dim R²)
  - ``loss_curve.png`` (log-scale train vs val loss)
  - ``per_dim_diagnostics.png`` (per-dim MSE / R² / variance / final-fit scatter)
  - ``output_fit.png`` (predictions vs. targets scatter, one panel per output)
  - ``model.pt``
  - ``final_metrics.txt`` (best / final val MSE, per-dim MSE, per-dim R²)
  - ``noise_metrics.txt`` (when --noise or --noise-aware is set)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analog_noise import (
    AnalogMLPWrapper,
    NoiseConfig,
    evaluate_clean,
    evaluate_with_noise,
)
from config import OPTIM
from train_ctle import (
    PARAM_COLS,
    PARAM_LOG_BOUNDS,
    RegimeAwareMoE,
    generate_ctle_dataset,
    params_from_logits,
    sample_specs,
    seed_everything,
)


__all__ = ["MLPRegressor", "count_parameters"]


_DEFAULT_HIDDEN_DIM = 256
_DEFAULT_NUM_LAYERS = 2


_ACTIVATIONS = {
    "relu": F.relu,
    "tanh": torch.tanh,
}


_ACTIVATION_MODULES = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
}


class MLPRegressor(nn.Module):
    """N-layer feedforward regressor with ``layers`` ModuleList contract.

    Input:  ``(B, in_dim)``
    Output: ``(B, out_dim)``

    Default sizes for the CTLE task: ``in_dim=8`` (when Q75 input scaling is
    enabled, matching the teacher's internal feature space) or ``in_dim=4``
    (raw 4-dim specs when ``--no-q75-input`` is passed), ``hidden_dim=256``,
    ``out_dim=7``, ``num_layers=2`` (Linear -> Act -> Linear -> output, no
    middle block). Parameter count formula:
      ``in_dim * hidden_dim + hidden_dim``
      ``+ (num_layers - 2) * (hidden_dim * hidden_dim + hidden_dim)``
      ``+ hidden_dim * out_dim + out_dim``

    Activation is selectable (``relu`` or ``tanh``). The flat ``layers``
    ModuleList is required by ``AnalogMLPWrapper`` for noise injection.
    """

    def __init__(
        self,
        in_dim: int = 4,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        out_dim: int = len(PARAM_COLS),
        num_layers: int = _DEFAULT_NUM_LAYERS,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"MLPRegressor: activation must be one of {sorted(_ACTIVATIONS)}, "
                f"got {activation!r}"
            )
        if num_layers < 2:
            raise ValueError(
                f"MLPRegressor: num_layers must be >= 2, got {num_layers}"
            )
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.num_layers = int(num_layers)
        self.activation = activation
        act_module = _ACTIVATION_MODULES[activation]

        layers: list[nn.Module] = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act_module)
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_module)
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.fc1 = layers[0]
        self.fc2 = layers[-1]

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        h = u
        for layer in self.layers:
            h = layer(h)
        return h


def count_parameters(net: nn.Module) -> int:
    """Total learnable parameters (weights + biases)."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def _import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        warnings.warn("matplotlib not installed; skipping plots", stacklevel=2)
        return None


def _per_dim_stats(
    preds: torch.Tensor, targets: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Per-dim MSE / R² / target variance from concatenated predictions.

    All inputs are in the same (possibly normalized) target space. R² uses
    the target variance of the same batch as the denominator so it measures
    explained variance on the held-out data.
    """
    preds = preds.detach().to("cpu").double()
    targets = targets.detach().to("cpu").double()
    sumsq = ((preds - targets) ** 2).sum(dim=0)
    target_sumsq = (targets ** 2).sum(dim=0)
    target_sum = targets.sum(dim=0)
    n_samples = targets.shape[0]
    if n_samples == 0:
        z = np.zeros(preds.shape[1], dtype=np.float64)
        return {"per_dim_mse": z, "per_dim_r2": z, "per_dim_var": z}
    per_dim_mse = (sumsq / n_samples).numpy()
    target_mean = (target_sum / n_samples).numpy()
    per_dim_var = (target_sumsq / n_samples).numpy() - target_mean ** 2
    per_dim_var = np.maximum(per_dim_var, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_dim_r2 = 1.0 - per_dim_mse / per_dim_var
    return {
        "per_dim_mse": per_dim_mse,
        "per_dim_r2": per_dim_r2,
        "per_dim_var": per_dim_var,
    }


def validate(net, val_loader, task_fn, device) -> tuple[float, dict[str, np.ndarray]]:
    """Compute mean MSE and per-dim stats over ``val_loader``."""
    net.eval()
    total = 0.0
    n = 0
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for u, target in val_loader:
            u = u.to(device)
            target = target.to(device)
            out = net(u)
            loss = task_fn(out, target)
            total += float(loss.item()) * u.size(0)
            n += u.size(0)
            all_preds.append(out.detach())
            all_targets.append(target.detach())
    net.train()
    avg_mse = total / max(1, n)
    if all_preds:
        stats = _per_dim_stats(torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0))
    else:
        empty = np.zeros(len(PARAM_COLS), dtype=np.float64)
        stats = {"per_dim_mse": empty, "per_dim_r2": empty, "per_dim_var": empty}
    return avg_mse, stats


def denormalize_predictions(
    preds_norm: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    return preds_norm * target_std + target_mean


def compute_denorm_metrics(
    net, val_loader, target_mean, target_std, inverse_stats, device,
    denorm_task_fn,
) -> dict[str, float]:
    """Compute MSE / MAE in the original (un-normalized) logit space.

    ``denorm_task_fn(out_denorm, target_denorm)`` returns a per-sample loss
    tensor. ``inverse_stats`` is the carrier of the teacher's per-dim log
    bounds, used when callers want a more semantically meaningful loss
    (e.g. via ``params_from_logits``). When ``inverse_stats`` is ``None``
    we simply MSE-decode the normalized predictions back to logit space.
    """
    net.eval()
    preds_norm, targets_norm = [], []
    with torch.no_grad():
        for u, t in val_loader:
            u = u.to(device)
            t = t.to(device)
            out = net(u)
            preds_norm.append(out.detach())
            targets_norm.append(t.detach())
    if not preds_norm:
        net.train()
        return {"mse_denorm": float("nan"), "mae_denorm": float("nan")}
    preds_norm = torch.cat(preds_norm, dim=0)
    targets_norm = torch.cat(targets_norm, dim=0)
    preds_denorm = denormalize_predictions(preds_norm, target_mean, target_std)
    targets_denorm = denormalize_predictions(targets_norm, target_mean, target_std)
    mse = float(F.mse_loss(preds_denorm, targets_denorm).item())
    mae = float(F.l1_loss(preds_denorm, targets_denorm).item())
    net.train()
    out = {"mse_denorm": mse, "mae_denorm": mae}
    if inverse_stats is not None:
        try:
            preds_phys = params_from_logits(preds_denorm.to(device))
            targets_phys = params_from_logits(targets_denorm.to(device))
            phys_keys = list(preds_phys.keys())
            mse_phys = float(
                torch.stack(
                    [F.mse_loss(preds_phys[k], targets_phys[k]) for k in phys_keys]
                ).mean().item()
            )
            mae_phys = float(
                torch.stack(
                    [F.l1_loss(preds_phys[k], targets_phys[k]) for k in phys_keys]
                ).mean().item()
            )
            out["mse_phys_paramspace"] = mse_phys
            out["mae_phys_paramspace"] = mae_phys
        except Exception as exc:
            warnings.warn(f"physical-space decode failed: {exc}", stacklevel=2)
    return out


def plot_loss_curve(history, val_history, save_path, title):
    plt = _import_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history, label="train", color="C0")
    ax.plot(val_history, label="val", color="C3")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (normalized targets)")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_per_dim_diagnostics(stats_history, save_path, suptitle):
    """Per-dim MSE / R² / variance summary plot.

    ``stats_history`` is a list of dicts each containing
    ``per_dim_mse``, ``per_dim_r2``, ``per_dim_var`` (and ``epoch``).
    """
    plt = _import_matplotlib()
    if plt is None:
        return
    epochs_h = list(range(len(stats_history)))
    if not epochs_h:
        return
    mse_arr = np.stack([s["per_dim_mse"] for s in stats_history], axis=0)
    r2_arr = np.stack([s["per_dim_r2"] for s in stats_history], axis=0)
    var_arr = stats_history[-1]["per_dim_var"]
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, name in enumerate(PARAM_COLS):
        axes[0, 0].plot(epochs_h, mse_arr[:, i], label=name, color=cmap(i), marker="o", markersize=2)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("per-dim val MSE (normalized)")
    axes[0, 0].set_title("Per-dim validation MSE over epochs")
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 0].grid(True, alpha=0.3)

    for i, name in enumerate(PARAM_COLS):
        axes[0, 1].plot(epochs_h, r2_arr[:, i], label=name, color=cmap(i), marker="o", markersize=2)
    axes[0, 1].axhline(0.0, color="grey", linewidth=0.5, linestyle="--")
    axes[0, 1].axhline(1.0, color="grey", linewidth=0.5, linestyle=":")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("R²")
    axes[0, 1].set_title("Per-dim validation R² over epochs")
    axes[0, 1].legend(fontsize=8, ncol=2)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].bar(PARAM_COLS, var_arr, color=[cmap(i) for i in range(len(PARAM_COLS))])
    axes[1, 0].set_ylabel("target variance (normalized)")
    axes[1, 0].set_title("Per-dim target variance (val set)")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    final_mse = mse_arr[-1]
    final_r2 = r2_arr[-1]
    axes[1, 1].scatter(final_mse, final_r2, c=[cmap(i) for i in range(len(PARAM_COLS))], s=60)
    for i, name in enumerate(PARAM_COLS):
        axes[1, 1].annotate(name, (final_mse[i], final_r2[i]), fontsize=8,
                             xytext=(4, 4), textcoords="offset points")
    axes[1, 1].set_xlabel("final val MSE")
    axes[1, 1].set_ylabel("final val R²")
    axes[1, 1].set_title("Per-dim fit (final epoch)")
    axes[1, 1].axhline(0.0, color="k", linewidth=0.5, linestyle="--")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_output_fit(out, target, save_path, target_mean, target_std):
    plt = _import_matplotlib()
    if plt is None:
        return
    out_np = out.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    out_denorm = out_np * target_std.cpu().numpy() + target_mean.cpu().numpy()
    target_denorm = target_np * target_std.cpu().numpy() + target_mean.cpu().numpy()
    n_cols = 4
    n_rows = int(math.ceil(len(PARAM_COLS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, name in enumerate(PARAM_COLS):
        ax = axes_flat[i]
        ax.scatter(target_denorm[:, i], out_denorm[:, i], s=4, alpha=0.4, color="C0")
        lo = float(min(target_denorm[:, i].min(), out_denorm[:, i].min()))
        hi = float(max(target_denorm[:, i].max(), out_denorm[:, i].max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y = x")
        ax.set_xlabel("target (logit)")
        ax.set_ylabel("prediction (logit)")
        ax.set_title(name)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    for i in range(len(PARAM_COLS), len(axes_flat)):
        axes_flat[i].set_axis_off()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train an N-layer MLP on the CTLE inverse-design regression task "
            "(4-dim spec -> 7-dim teacher logits)."
        )
    )
    parser.add_argument(
        "--teacher-path", type=Path, required=True,
        help="Path to the RegimeAwareMoE state-dict checkpoint "
             "(e.g. dagger_student_moe.pt).",
    )
    parser.add_argument(
        "--scaler-path", type=Path, default=None,
        help="Path to flow_scaler_C.pkl (default: auto-discover next to --teacher-path).",
    )
    parser.add_argument(
        "--teacher-hidden", type=int, default=160,
        help="MoE trunk width (default: 160, matching the published checkpoint).",
    )
    parser.add_argument(
        "--teacher-experts", type=int, default=3,
        help="Number of MoE experts (default: 3).",
    )
    parser.add_argument("--hidden-dim", type=int, default=_DEFAULT_HIDDEN_DIM,
                        help=f"Hidden width (default: {_DEFAULT_HIDDEN_DIM}). "
                             f"Param count depends on input dim: ~3079 params for "
                             f"4 -> {_DEFAULT_HIDDEN_DIM} -> 7 (--no-q75-input), "
                             f"~4103 params for 8 -> {_DEFAULT_HIDDEN_DIM} -> 7 "
                             f"(--q75-input, default).")
    parser.add_argument("--num-layers", type=int, default=_DEFAULT_NUM_LAYERS,
                        help="Number of linear layers, must be >= 2 (default: 2).")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Number of training epochs (default: {OPTIM['epochs']}).")
    parser.add_argument("--lr", type=float, default=None,
                        help=f"Learning rate (default: {OPTIM['lr']}).")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help=f"Weight decay (default: {OPTIM.get('weight_decay', 0.0)}).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: 1024, ~8x smaller than "
                             "OPTIM['batch_size']=4096 which is tuned for the slower "
                             "KirchhoffNet ODE training loop). distill_ctle_kirchhoff.py "
                             "uses 512; 1024 is a comparable choice for the MLP.")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience in epochs (default: 50).")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="Early stopping min improvement in val loss (default: 1e-4).")
    parser.add_argument("--validate-every", type=int, default=5,
                        help="Validate every N epochs (default: 5).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--n-train", type=int, default=100000,
                        help="Number of synthetic training spec samples (default: 100000).")
    parser.add_argument("--n-val", type=int, default=20000,
                        help="Number of synthetic validation spec samples (default: 20000).")
    parser.add_argument("--activation", choices=["relu", "tanh"], default="relu",
                        help="Hidden activation (default: relu).")
    parser.add_argument("--normalize-targets", dest="normalize_targets",
                        action="store_true", default=True,
                        help="Per-dim zero-mean / unit-variance normalize the teacher "
                             "logits before training (default: on, matches train_ctle.py).")
    parser.add_argument("--no-normalize-targets", dest="normalize_targets",
                        action="store_false",
                        help="Disable per-dim normalization.")
    parser.add_argument("--q75-input", dest="q75_input",
                        action="store_true", default=True,
                        help="Apply the teacher's Q75 input scaling (log10 + StandardScaler/Q75 "
                             "normalization) to spec inputs at data-generation time. Transforms "
                             "raw 4-dim specs into 8 well-conditioned features matching the "
                             "teacher's internal scale_input() preprocessing (default: on). "
                             "This matches what distill_ctle_kirchhoff.py does and avoids "
                             "forcing the MLP to learn the nonlinear Q75 transformation from "
                             "scratch.")
    parser.add_argument("--no-q75-input", dest="q75_input",
                        action="store_false",
                        help="Use raw 4-dim spec inputs (original behavior).")
    parser.add_argument("--output", type=Path, default=Path("./output/mlp_ctle"),
                        help="Output directory (default: ./output/mlp_ctle).")
    parser.add_argument("--device", default=None,
                        help="Device 'cpu' or 'cuda' (default: auto-detect).")
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="Max gradient norm for clipping (default: 1.0, "
                             "matching distill_ctle_kirchhoff.py; the KirchhoffNet "
                             "default of 5.0 from OPTIM['grad_clip_norm'] is too "
                             "loose for the MLP). 0 or negative = no clipping.")
    parser.add_argument("--noise", action="store_true",
                        help="Enable realistic hardware noise "
                             "(quantization + circuit noise) on the trained "
                             "model and report Monte Carlo evaluation stats.")
    parser.add_argument("--noise-aware", action="store_true",
                        help="Train under analog noise so the MLP becomes "
                             "robust to it. Implies --noise for final eval.")
    parser.add_argument("--quant-bits", type=int, choices=[4, 6], default=4,
                        help="Bit-width for weight and ADC/DAC quantization "
                             "(default: 4).")
    parser.add_argument("--noise-std", type=float, default=0.05,
                        help="Standard deviation of additive Gaussian circuit "
                             "noise on weights and activations (default: 0.05).")
    parser.add_argument("--mc-trials", type=int, default=20,
                        help="Number of Monte Carlo trials for noisy "
                             "evaluation (default: 20).")
    parser.add_argument("--adc-full-range", type=float, default=3.0,
                        help="Symmetric full-scale range for ADC/DAC "
                             "quantization (default: 3.0).")
    parser.add_argument("--noise-seed", type=int, default=0,
                        help="Seed for noise sampling (default: 0).")
    parser.add_argument("--no-adc", action="store_true",
                        help="Disable per-layer ADC/DAC quantization "
                             "(pure-digital accelerator mode: weight "
                             "quantization + circuit noise only).")
    parser.add_argument("--use-cosine-scheduler", dest="use_cosine",
                        action="store_true", default=True,
                        help="Use CosineAnnealingWarmRestarts scheduler "
                             "(default: on).")
    parser.add_argument("--no-cosine-scheduler", dest="use_cosine",
                        action="store_false",
                        help="Disable LR scheduler (constant LR).")
    args = parser.parse_args()

    epochs = args.epochs if args.epochs is not None else int(OPTIM["epochs"])
    lr = args.lr if args.lr is not None else float(OPTIM["lr"])
    weight_decay = (
        args.weight_decay if args.weight_decay is not None
        else float(OPTIM.get("weight_decay", 0.0))
    )
    batch_size = args.batch_size if args.batch_size is not None else 1024
    grad_clip_norm = args.grad_clip if args.grad_clip is not None else 1.0
    device = args.device if args.device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    device_t = torch.device(device)

    seed_everything(args.seed)

    out_dir = args.output.resolve()
    if out_dir.exists():
        suffix = time.strftime("%Y%m%d_%H%M%S")
        out_dir = out_dir.with_name(f"{out_dir.name}_{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load teacher MoE (mirrors train_ctle.py:1510-1551) ----
    if not args.teacher_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_path}")
    scaler_path = (
        args.scaler_path if args.scaler_path is not None
        else args.teacher_path.parent / "flow_scaler_C.pkl"
    )
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"flow_scaler_C.pkl not found at {scaler_path}. "
            f"Pass --scaler-path to override the auto-discovered location."
        )
    flow_scaler_C = joblib.load(scaler_path)
    scaler_p_mean = float(flow_scaler_C["scaler_y_p"].mean_[0])
    scaler_p_scale = float(flow_scaler_C["scaler_y_p"].scale_[0])
    eye_scale_h = float(flow_scaler_C["eye_scale_h"])
    eye_scale_w = float(flow_scaler_C["eye_scale_w"])
    eye_scale_j = float(flow_scaler_C["eye_scale_j"])
    print(
        f"[mlp_ctle] loaded scaler from {scaler_path} "
        f"(p_mean={scaler_p_mean:.4f}, p_scale={scaler_p_scale:.4f}, "
        f"eye_h={eye_scale_h:.4f}, eye_w={eye_scale_w:.4f}, eye_j={eye_scale_j:.4f})"
    )
    teacher = RegimeAwareMoE(
        trunk_width=args.teacher_hidden,
        trunk_layers=3,
        num_experts=args.teacher_experts,
        input_dim=4,
        output_dim=len(PARAM_COLS),
        param_log_bounds=PARAM_LOG_BOUNDS,
        activation=nn.SiLU,
        use_log_features=True,
        scaler_p_mean=scaler_p_mean,
        scaler_p_scale=scaler_p_scale,
        eye_scale_h=eye_scale_h,
        eye_scale_w=eye_scale_w,
        eye_scale_j=eye_scale_j,
    )
    state = torch.load(args.teacher_path, map_location="cpu")
    teacher.load_state_dict(state)
    teacher.eval()
    teacher.requires_grad_(False)
    teacher.to(device_t)
    n_teacher_params = sum(p.numel() for p in teacher.parameters())
    print(
        f"[mlp_ctle] loaded teacher MoE from {args.teacher_path} "
        f"({n_teacher_params} params, trunk={args.teacher_hidden}, "
        f"experts={args.teacher_experts})"
    )

    # ---- build dataset ----
    train_loader, val_loader, target_mean, target_std = generate_ctle_dataset(
        n_train=args.n_train,
        n_val=args.n_val,
        mlp=teacher,
        device=device_t,
        batch_size=batch_size,
        seed=args.seed,
        normalize=args.normalize_targets,
        q75_input=args.q75_input,
    )
    print(
        f"[mlp_ctle] dataset: n_train={args.n_train} n_val={args.n_val} "
        f"batch_size={batch_size} normalize={args.normalize_targets} "
        f"q75_input={args.q75_input} seed={args.seed}"
    )

    in_dim = 8 if args.q75_input else 4

    # ---- build MLP ----
    net = MLPRegressor(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=len(PARAM_COLS),
        num_layers=args.num_layers,
        activation=args.activation,
    )
    n_params = count_parameters(net)
    net.to(device_t)

    noise_cfg = NoiseConfig(
        quant_bits=args.quant_bits if args.noise or args.noise_aware else None,
        noise_std=args.noise_std if args.noise or args.noise_aware else 0.0,
        mc_trials=args.mc_trials,
        seed=args.noise_seed,
        quantize_input=not args.no_adc,
        quantize_output=not args.no_adc,
        quantize_intermediate=not args.no_adc,
    )
    train_wrapper: AnalogMLPWrapper | None = None
    if args.noise_aware:
        train_wrapper = AnalogMLPWrapper(
            net, noise_cfg, adc_full_range=args.adc_full_range,
        )
        train_wrapper.to(device_t)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler: CosineAnnealingWarmRestarts | None = None
    if args.use_cosine and "scheduler_T_0" in OPTIM:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(OPTIM["scheduler_T_0"]),
            T_mult=int(OPTIM["scheduler_T_mult"]),
            eta_min=float(OPTIM["scheduler_eta_min"]),
        )

    print(
        f"[mlp_ctle] hidden_dim={args.hidden_dim} num_layers={args.num_layers} "
        f"params={n_params} activation={args.activation} "
        f"epochs={epochs} lr={lr} weight_decay={weight_decay} "
        f"batch_size={batch_size} grad_clip_norm={grad_clip_norm} device={device_t} "
        f"output={out_dir}"
    )

    with open(out_dir / "config_snapshot.txt", "w") as f:
        f.write(f"model: MLPRegressor({in_dim} -> {args.hidden_dim} -> 7, "
                f"num_layers={args.num_layers}, activation={args.activation})\n")
        f.write(f"param_count: {n_params}\n")
        f.write(f"teacher_params: {n_teacher_params}\n")
        f.write(f"in_dim: {in_dim} (q75_input={args.q75_input})\n")
        f.write(f"num_layers: {args.num_layers}\n")
        f.write(f"hidden_dim: {args.hidden_dim}\n")
        f.write(f"activation: {args.activation}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"lr: {lr}\n")
        f.write(f"weight_decay: {weight_decay}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"grad_clip_norm: {grad_clip_norm}\n")
        f.write(f"patience: {args.patience}\n")
        f.write(f"min_delta: {args.min_delta}\n")
        f.write(f"validate_every: {args.validate_every}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"device: {device_t}\n")
        f.write(f"teacher_path: {args.teacher_path}\n")
        f.write(f"scaler_path: {scaler_path}\n")
        f.write(f"n_train: {args.n_train}\n")
        f.write(f"n_val: {args.n_val}\n")
        f.write(f"normalize_targets: {args.normalize_targets}\n")
        f.write(f"q75_input: {args.q75_input}\n")
        f.write(f"loss: MSE on {'normalized ' if args.normalize_targets else ''}teacher logits\n")
        if scheduler is not None:
            f.write(
                f"scheduler: CosineAnnealingWarmRestarts("
                f"T_0={OPTIM['scheduler_T_0']}, "
                f"T_mult={OPTIM['scheduler_T_mult']}, "
                f"eta_min={OPTIM['scheduler_eta_min']})\n"
            )
        else:
            f.write("scheduler: constant LR\n")
        f.write(
            f"dataset: CTLE synthetic specs "
            f"(4-dim spec -> {len(PARAM_COLS)}-dim {list(PARAM_COLS)} logits)\n"
        )

    history: list[float] = []
    val_history: list[float] = []
    stats_history: list[dict[str, np.ndarray]] = []
    best_val = float("inf")
    best_epoch = -1
    best_state = None
    epochs_without_improve = 0
    stop_training = False

    start = time.time()
    for epoch in range(epochs):
        if stop_training:
            break
        net.train()
        total_loss = 0.0
        n_batches = 0
        for u, target in train_loader:
            u = u.to(device_t)
            target = target.to(device_t)
            optimizer.zero_grad()
            if train_wrapper is not None:
                out = train_wrapper(u)
            else:
                out = net(u)
            loss = F.mse_loss(out, target)
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += float(loss.item())
            n_batches += 1

        avg_train = total_loss / max(1, n_batches)
        do_validate = (epoch % args.validate_every == 0) or (epoch == epochs - 1)
        if do_validate:
            val_loss, stats = validate(net, val_loader, lambda o, t: F.mse_loss(o, t), device_t)
        else:
            val_loss = val_history[-1] if val_history else avg_train
            stats = stats_history[-1] if stats_history else {
                "per_dim_mse": np.zeros(len(PARAM_COLS), dtype=np.float64),
                "per_dim_r2": np.zeros(len(PARAM_COLS), dtype=np.float64),
                "per_dim_var": np.zeros(len(PARAM_COLS), dtype=np.float64),
            }

        history.append(avg_train)
        val_history.append(val_loss)
        stats_history.append(stats)

        if do_validate:
            if val_loss < best_val - args.min_delta:
                best_val = float(val_loss)
                best_epoch = epoch
                epochs_without_improve = 0
                best_state = {
                    k: v.detach().clone() for k, v in net.state_dict().items()
                }
            else:
                epochs_without_improve += args.validate_every
                if epochs_without_improve >= args.patience:
                    print(
                        f"[mlp_ctle] early stopping at epoch {epoch}: "
                        f"no val improvement for {epochs_without_improve} epochs "
                        f"(best val={best_val:.6f} @ epoch {best_epoch})"
                    )
                    stop_training = True

        if epoch % args.validate_every == 0 or epoch == epochs - 1:
            worst_idx = int(np.argmax(stats["per_dim_mse"])) if stats["per_dim_mse"].size else 0
            worst_name = PARAM_COLS[worst_idx] if stats["per_dim_mse"].size else "?"
            worst_mse = (
                float(stats["per_dim_mse"][worst_idx]) if stats["per_dim_mse"].size else 0.0
            )
            print(
                f"  epoch {epoch:4d}  train={avg_train:.6f}  val={val_loss:.6f}  "
                f"worst_dim={worst_name}({worst_mse:.3e})  agg_r2="
                f"{float(1.0 - np.mean(stats['per_dim_mse'] / np.maximum(stats['per_dim_var'], 1e-12))):.4f}"
                if stats["per_dim_var"].size
                else f"  epoch {epoch:4d}  train={avg_train:.6f}  val={val_loss:.6f}"
            )

    elapsed = time.time() - start
    print(
        f"[mlp_ctle] training done in {elapsed:.1f}s ({len(history)} epochs). "
        f"Best val={best_val:.6f} @ epoch {best_epoch}."
    )

    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"[mlp_ctle] restored best model from epoch {best_epoch}")

    torch.save(net.state_dict(), out_dir / "model.pt")

    with open(out_dir / "loss_history.txt", "w") as f:
        cols = ["epoch", "train_mse", "val_mse"]
        cols += [f"mse_{name}" for name in PARAM_COLS]
        cols += [f"r2_{name}" for name in PARAM_COLS]
        cols += [f"var_{name}" for name in PARAM_COLS]
        f.write("\t".join(cols) + "\n")
        for i, (t, v, s) in enumerate(zip(history, val_history, stats_history)):
            row = [str(i), f"{t:.6e}", f"{v:.6e}"]
            row += [f"{x:.6e}" for x in s["per_dim_mse"]]
            row += [f"{x:.6e}" for x in s["per_dim_r2"]]
            row += [f"{x:.6e}" for x in s["per_dim_var"]]
            f.write("\t".join(row) + "\n")

    plot_loss_curve(
        history, val_history,
        save_path=str(out_dir / "loss_curve.png"),
        title=f"MLP (hidden={args.hidden_dim}, layers={args.num_layers}, "
              f"{args.activation}, {n_params} params) — CTLE spec->logit",
    )
    plot_per_dim_diagnostics(
        stats_history,
        save_path=str(out_dir / "per_dim_diagnostics.png"),
        suptitle=f"MLP CTLE per-dim diagnostics (hidden={args.hidden_dim}, "
                 f"{n_params} params, {len(PARAM_COLS)}-dim output)",
    )

    val_batch = next(iter(val_loader))
    u_val, y_val = val_batch[0][:64], val_batch[1][:64]
    with torch.no_grad():
        out_val = net(u_val.to(device_t))
    plot_output_fit(
        out_val, y_val,
        save_path=str(out_dir / "output_fit.png"),
        target_mean=target_mean.to(device_t),
        target_std=target_std.to(device_t),
    )

    full_val_loss, full_stats = validate(
        net, val_loader, lambda o, t: F.mse_loss(o, t), device_t,
    )
    denorm = compute_denorm_metrics(
        net, val_loader, target_mean, target_std,
        inverse_stats=None, device=device_t,
        denorm_task_fn=lambda o, t: F.mse_loss(o, t),
    )
    print(
        f"[mlp_ctle] final val MSE = {full_val_loss:.6f} "
        f"(best val = {best_val:.6f} @ epoch {best_epoch})  "
        f"denorm MSE/MAE = {denorm['mse_denorm']:.4f}/{denorm['mae_denorm']:.4f}"
    )

    with open(out_dir / "final_metrics.txt", "w") as f:
        f.write(f"param_count: {n_params}\n")
        f.write(f"teacher_params: {n_teacher_params}\n")
        f.write(f"best_val_mse: {best_val:.6f}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"final_val_mse: {full_val_loss:.6f}\n")
        f.write(f"denorm_mse: {denorm['mse_denorm']:.6f}\n")
        f.write(f"denorm_mae: {denorm['mae_denorm']:.6f}\n")
        if "mse_phys_paramspace" in denorm:
            f.write(f"phys_paramspace_mse: {denorm['mse_phys_paramspace']:.6f}\n")
            f.write(f"phys_paramspace_mae: {denorm['mae_phys_paramspace']:.6f}\n")
        f.write(f"epochs_run: {len(history)}\n")
        f.write(f"elapsed_seconds: {elapsed:.2f}\n")
        f.write("per_dim_mse_norm:\n")
        for name, mse in zip(PARAM_COLS, full_stats["per_dim_mse"]):
            f.write(f"  {name}: {mse:.6e}\n")
        f.write("per_dim_r2_norm:\n")
        for name, r2 in zip(PARAM_COLS, full_stats["per_dim_r2"]):
            f.write(f"  {name}: {r2:.6f}\n")
        f.write("per_dim_var_norm:\n")
        for name, var in zip(PARAM_COLS, full_stats["per_dim_var"]):
            f.write(f"  {name}: {var:.6e}\n")

    if args.noise or args.noise_aware:
        print(
            f"[mlp_ctle] running MC noise eval: quant_bits={args.quant_bits} "
            f"noise_std={args.noise_std} trials={args.mc_trials} "
            f"adc_full_range={args.adc_full_range} seed={args.noise_seed} "
            f"adc_enabled={not args.no_adc}"
        )
        eval_cfg = NoiseConfig(
            quant_bits=args.quant_bits,
            noise_std=args.noise_std,
            mc_trials=args.mc_trials,
            seed=args.noise_seed,
            quantize_input=not args.no_adc,
            quantize_output=not args.no_adc,
            quantize_intermediate=not args.no_adc,
        )
        eval_wrapper = AnalogMLPWrapper(
            net, eval_cfg, adc_full_range=args.adc_full_range,
        )
        eval_wrapper.to(device_t)
        clean_loss = evaluate_clean(
            eval_wrapper, val_loader, lambda o, t: F.mse_loss(o, t), device_t,
        )
        noise_result = evaluate_with_noise(
            eval_wrapper, val_loader, lambda o, t: F.mse_loss(o, t),
            eval_cfg, device_t,
        )
        noise_result.clean_loss = clean_loss
        print(
            f"[mlp_ctle] noise eval: clean={clean_loss:.6f} "
            f"noisy_mean={noise_result.mean:.6f} "
            f"noisy_std={noise_result.std:.6f} "
            f"p90={noise_result.p90:.6f} p95={noise_result.p95:.6f}"
        )
        with open(out_dir / "noise_metrics.txt", "w") as f:
            f.write(f"quant_bits: {args.quant_bits}\n")
            f.write(f"noise_std: {args.noise_std}\n")
            f.write(f"mc_trials: {args.mc_trials}\n")
            f.write(f"adc_full_range: {args.adc_full_range}\n")
            f.write(f"noise_seed: {args.noise_seed}\n")
            f.write(f"noise_aware_training: {bool(args.noise_aware)}\n")
            f.write(f"adc_quantization: {not bool(args.no_adc)}\n")
            f.write(f"clean_val_mse: {clean_loss:.6f}\n")
            f.write(f"noisy_mean: {noise_result.mean:.6f}\n")
            f.write(f"noisy_std: {noise_result.std:.6f}\n")
            f.write(f"noisy_p50: {noise_result.p50:.6f}\n")
            f.write(f"noisy_p90: {noise_result.p90:.6f}\n")
            f.write(f"noisy_p95: {noise_result.p95:.6f}\n")
            f.write(f"noisy_best: {noise_result.best:.6f}\n")
            f.write(f"noisy_worst: {noise_result.worst:.6f}\n")
            f.write(f"degradation_mean: {noise_result.mean - clean_loss:.6f}\n")
            f.write("per_trial_losses:\n")
            for i, l in enumerate(noise_result.losses):
                f.write(f"  trial_{i:03d}: {l:.6f}\n")

    print(f"[mlp_ctle] artifacts in {out_dir}")


if __name__ == "__main__":
    main()
