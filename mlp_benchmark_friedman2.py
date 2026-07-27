"""Minimal MLP baseline for the Friedman #2 synthetic regression task.

Purpose: provide a parameter-matched benchmark for the reduced differential
KirchhoffNet on the classic Friedman (1991) MARS test problem #2.

Friedman #2 is a 4-dimensional regression problem. The target is:

    y = sqrt(x1^2 + (x2 * x3 - 1 / (x2 * x4))^2) + noise

with variable-specific ranges:
    x1 ~ U(0, 100)
    x2 ~ U(40*pi, 560*pi)
    x3 ~ U(0, 1)
    x4 ~ U(1, 11)

The canonical noise standard deviation is sigma = 1.0.

CLI:
    mlp_benchmark_friedman2.py [--hidden-dim 100] [--num-layers 2] [--epochs 800]
                               [--lr 1.2e-3] [--target-noise-std 1.0]
                               [--output OUTPUT] [--device DEVICE]

Outputs to --output:
  - loss_history.txt
  - loss_curve.png
  - model.pt
  - config_snapshot.txt
  - output_fit.png
  - final_metrics.txt
  - noise_metrics.txt (only with --noise or --noise-aware)

Defaults match the KirchhoffNet training script (train_script.py):
AdamW (lr=OPTIM.lr, weight_decay=OPTIM.weight_decay), grad_clip_norm=5.0,
batch_size=OPTIM.batch_size, early stopping with patience=50, min_delta=1e-4.
"""

import argparse
import math
import random
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OPTIM
from train_script import _lhs_samples
from analog_noise import (
    AnalogMLPWrapper,
    NoiseConfig,
    evaluate_clean,
    evaluate_with_noise,
)


__all__ = ["MLPRegressor", "count_parameters", "_friedman2", "make_data_friedman2"]


_FRIEDMAN2_IN_DIM = 4
_FRIEDMAN2_PI = math.pi
_FRIEDMAN2_RANGES = [
    (0.0, 100.0),
    (40.0 * _FRIEDMAN2_PI, 560.0 * _FRIEDMAN2_PI),
    (0.0, 1.0),
    (1.0, 11.0),
]


_ACTIVATIONS = {
    "relu": F.relu,
    "tanh": torch.tanh,
}

_ACTIVATION_MODULES = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
}


class MLPRegressor(nn.Module):
    """N-layer feedforward regressor.

    Input: (B, in_dim=4)
    Output: (B, out_dim=1)

    Architecture: num_layers linear layers with activation between them.
      - First layer: in_dim -> hidden_dim
      - Hidden layers (num_layers - 2 of them): hidden_dim -> hidden_dim
      - Final layer: hidden_dim -> out_dim
      - No activation after the final layer (regression head)
    """

    def __init__(
        self,
        in_dim: int = _FRIEDMAN2_IN_DIM,
        hidden_dim: int = 100,
        out_dim: int = 1,
        num_layers: int = 2,
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

        layers = []
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


def _friedman2(x: torch.Tensor) -> torch.Tensor:
    """Friedman #2 target.

    Args:
        x: Tensor of shape (..., 4) with x[..., 0] ~ U(0,100),
           x[..., 1] ~ U(40π, 560π), x[..., 2] ~ U(0,1), x[..., 3] ~ U(1,11).

    Returns:
        Tensor of shape (...) with the deterministic (noise-free) target.
    """
    if x.shape[-1] != _FRIEDMAN2_IN_DIM:
        raise ValueError(
            f"_friedman2 requires exactly {_FRIEDMAN2_IN_DIM} input columns, "
            f"got {x.shape[-1]}"
        )
    x1 = x[..., 0]
    x2 = x[..., 1]
    x3 = x[..., 2]
    x4 = x[..., 3]
    inner = (x2 * x3) - (1.0 / (x2 * x4))
    return torch.sqrt(x1 ** 2 + inner ** 2)


def _minmax_normalize_inputs(
    u_train: torch.Tensor, u_val: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-dim min-max normalize features to [0, 1] using train-set stats.

    Mirrors ``train_script._minmax_normalize_inputs``: stats are computed on
    the training set only, val is normalized with the same scaler, and the
    per-dim min/range are returned for downstream input-distribution
    analysis. Range is clamped to 1e-8 to avoid division by zero when a
    train-set feature is constant.
    """
    u_min = u_train.amin(dim=0, keepdim=True)
    u_max = u_train.amax(dim=0, keepdim=True)
    u_range = (u_max - u_min).clamp(min=1e-8)
    u_train_n = (u_train - u_min) / u_range
    u_val_n = (u_val - u_min) / u_range
    return u_train_n, u_val_n, u_min, u_range


def make_data_friedman2(batch_size: int, noise_std: float = 1.0, val_size: int = 4000,
                       normalize_inputs: bool = True):
    """Build Friedman #2 train/val loaders with target standardization.

    Train inputs are LHS samples scaled to the variable-specific ranges.
    Val inputs are uniform random in the same ranges. When
    ``normalize_inputs=True`` (default), inputs are per-dim min-max scaled
    to [0, 1] using train statistics. Targets are z-scored using train
    mean/std. The returned ``inverse_stats`` carries ``"y_mean"``/
    ``"y_std"`` for target denormalization; when normalized it also
    carries ``"u_min"``/``"u_range"`` for input-distribution analysis.
    """
    n_train = 20000
    u_train_unit = _lhs_samples(n_train, _FRIEDMAN2_IN_DIM, seed=42)
    u_train = _scale_lhs_to_ranges(u_train_unit, _FRIEDMAN2_RANGES)
    y_train = _friedman2(u_train).unsqueeze(1)

    torch.manual_seed(42)
    u_val_unit = torch.rand(val_size, _FRIEDMAN2_IN_DIM)
    u_val = _scale_lhs_to_ranges(u_val_unit, _FRIEDMAN2_RANGES)
    y_val = _friedman2(u_val).unsqueeze(1)

    if noise_std > 0:
        y_train = y_train + noise_std * torch.randn_like(y_train)

    if normalize_inputs:
        u_train, u_val, u_min, u_range = _minmax_normalize_inputs(u_train, u_val)

    y_mean = float(y_train.mean().item())
    y_std = float(y_train.std().clamp(min=1e-6).item())
    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    train_loader = DataLoader(
        TensorDataset(u_train, y_train_n), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(u_val, y_val_n), batch_size=batch_size, shuffle=False
    )
    inverse_stats = {"y_mean": y_mean, "y_std": y_std}
    if normalize_inputs:
        inverse_stats["u_min"] = u_min.squeeze(0).tolist()
        inverse_stats["u_range"] = u_range.squeeze(0).tolist()
    return train_loader, val_loader, F.mse_loss, inverse_stats


def _scale_lhs_to_ranges(u_unit: torch.Tensor, ranges: list[tuple[float, float]]) -> torch.Tensor:
    """Linearly scale [0,1] uniform/LHS samples to per-dim (lo, hi) ranges."""
    out = u_unit.clone()
    for i, (lo, hi) in enumerate(ranges):
        out[..., i] = lo + (hi - lo) * u_unit[..., i]
    return out


def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        warnings.warn("matplotlib not installed; skipping plots", stacklevel=2)
        return None


def validate(net, val_loader, task_fn, device, wrapper=None) -> float:
    net.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for u, target in val_loader:
            u = u.to(device)
            target = target.to(device)
            if wrapper is not None:
                out = wrapper(u)
            else:
                out = net(u)
            loss = task_fn(out, target)
            total += float(loss.item()) * u.size(0)
            n += u.size(0)
    net.train()
    return total / max(1, n)


def compute_orig_metrics(net, val_loader, inverse_stats, device, wrapper=None):
    """Compute MSE/RMSE/MAE/MAPE in original (un-standardized) target units.

    Returns ``(mse_orig, rmse_orig, mae_orig, mape_orig)``. MAPE clips the
    ``|y_true|`` denominator at ``1e-8`` to avoid division by zero.
    """
    net.eval()
    preds, targets = [], []
    with torch.no_grad():
        for u, t in val_loader:
            u = u.to(device)
            t = t.to(device)
            if wrapper is not None:
                preds.append(wrapper(u))
            else:
                preds.append(net(u))
            targets.append(t)
    y_pred_std = torch.cat(preds, dim=0)
    y_true_std = torch.cat(targets, dim=0)
    y_mean = inverse_stats["y_mean"]
    y_std = inverse_stats["y_std"]
    y_pred = y_pred_std * y_std + y_mean
    y_true = y_true_std * y_std + y_mean
    err = y_pred - y_true
    abs_err = err.abs()
    mse = float(F.mse_loss(y_pred, y_true).item())
    rmse = float(torch.sqrt(F.mse_loss(y_pred, y_true)).item())
    mae = float(F.l1_loss(y_pred, y_true).item())
    mape = float((abs_err / y_true.abs().clamp(min=1e-8)).mean().item()) * 100.0
    net.train()
    return mse, rmse, mae, mape


def plot_output_fit(out, target, save_path, title):
    plt = _import_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(target.cpu().numpy().ravel(), out.detach().cpu().numpy().ravel(),
               s=4, alpha=0.4, color="C0")
    lo = float(min(target.min().item(), out.min().item()))
    hi = float(max(target.max().item(), out.max().item()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y = x")
    ax.set_xlabel("target (normalized)")
    ax.set_ylabel("prediction (normalized)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(history, val_history, save_path, title, loss_label: str = "MSE"):
    plt = _import_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history, label="train", color="C0")
    ax.plot(val_history, label="val", color="C3")
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{loss_label} loss")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Train an N-layer MLP on the Friedman #2 synthetic regression task."
    )
    parser.add_argument("--hidden-dim", type=int, default=100,
                        help="Hidden layer width (default: 100)")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of linear layers (default: 2)")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Number of training epochs (default: {OPTIM['epochs']})")
    parser.add_argument("--lr", type=float, default=None,
                        help=f"Learning rate (default: {OPTIM['lr']})")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help=f"Weight decay (default: {OPTIM['weight_decay']})")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Batch size (default: {OPTIM['batch_size']})")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience in epochs (default: 50)")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="Early stopping min improvement in val loss (default: 1e-4)")
    parser.add_argument("--validate-every", type=int, default=5,
                        help="Validate every N epochs (default: 5)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--target-noise-std", type=float, default=1.0,
                        help="Standard deviation of additive Gaussian target "
                             "noise (default: 1.0, the Friedman #2 canonical value).")
    parser.add_argument("--activation", choices=["relu", "tanh"], default="relu",
                        help="Hidden activation (default: relu)")
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse",
                        help="Loss function (default: mse). Huber uses delta=1.0.")
    parser.add_argument("--output", type=Path, default=Path("./output/mlp_friedman2"),
                        help="Output directory (default: ./output/mlp_friedman2)")
    parser.add_argument("--device", default=None,
                        help="Device 'cpu' or 'cuda' (default: auto-detect)")
    parser.add_argument("--noise", action="store_true",
                        help="Enable realistic hardware noise "
                             "(quantization + circuit noise) on the trained "
                             "model and report Monte Carlo evaluation stats.")
    parser.add_argument("--noise-aware", action="store_true",
                        help="Train under analog noise so the MLP becomes "
                             "robust to it. Implies --noise for final eval.")
    parser.add_argument("--quant-bits", type=int, choices=[4, 6], default=4,
                        help="Bit-width for weight and ADC/DAC quantization "
                             "(default: 4). Used when --noise is set.")
    parser.add_argument("--hw-noise-std", type=float, default=0.05,
                        help="Standard deviation of additive Gaussian "
                             "circuit noise on weights and activations "
                             "(default: 0.05). Used when --noise is set.")
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
                             "quantization + circuit noise only, no "
                             "inter-layer converters). Only effective when "
                             "--noise or --noise-aware is set.")
    parser.add_argument("--normalize-inputs", dest="normalize_inputs",
                        action="store_true", default=True,
                        help="Per-dim min-max normalize inputs to [0, 1] "
                             "(default: on). Use --no-normalize-inputs to "
                             "disable for ablation.")
    parser.add_argument("--no-normalize-inputs", dest="normalize_inputs",
                        action="store_false",
                        help="Disable per-dim input normalization.")
    args = parser.parse_args()

    epochs = args.epochs if args.epochs is not None else int(OPTIM["epochs"])
    lr = args.lr if args.lr is not None else float(OPTIM["lr"])
    weight_decay = args.weight_decay if args.weight_decay is not None else float(OPTIM["weight_decay"])
    batch_size = args.batch_size if args.batch_size is not None else int(OPTIM["batch_size"])
    grad_clip_norm = float(OPTIM["grad_clip_norm"])
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    # input-norm-seed/Phase 3: full reproducible seeding (RNG + cuDNN).
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    net = MLPRegressor(
        in_dim=_FRIEDMAN2_IN_DIM,
        hidden_dim=args.hidden_dim,
        out_dim=1,
        num_layers=args.num_layers,
        activation=args.activation,
    )
    n_params = count_parameters(net)
    net.to(device)

    noise_cfg = NoiseConfig(
        quant_bits=args.quant_bits if args.noise or args.noise_aware else None,
        noise_std=args.hw_noise_std if args.noise or args.noise_aware else 0.0,
        mc_trials=args.mc_trials,
        seed=args.noise_seed,
        quantize_input=not args.no_adc,
        quantize_output=not args.no_adc,
        quantize_intermediate=not args.no_adc,
    )
    train_wrapper: AnalogMLPWrapper | None = None
    if args.noise_aware:
        train_wrapper = AnalogMLPWrapper(net, noise_cfg, adc_full_range=args.adc_full_range)
        train_wrapper.to(device)

    train_loader, val_loader, task_fn, inverse_stats = make_data_friedman2(
        batch_size=batch_size, noise_std=args.target_noise_std,
        normalize_inputs=args.normalize_inputs,
    )
    if args.loss == "huber":
        task_fn = lambda o, t: F.huber_loss(o, t, delta=1.0)
    else:
        task_fn = F.mse_loss
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    print(
        f"[mlp_friedman2] hidden_dim={args.hidden_dim} num_layers={args.num_layers} "
        f"params={n_params} activation={args.activation} loss={args.loss} "
        f"epochs={epochs} lr={lr} weight_decay={weight_decay} "
        f"batch_size={batch_size} grad_clip_norm={grad_clip_norm} "
        f"target_noise_std={args.target_noise_std} device={device} output={out_dir}"
    )

    with open(out_dir / "config_snapshot.txt", "w") as f:
        f.write(f"model: MLPRegressor(4 -> {args.hidden_dim} -> 1, "
                f"num_layers={args.num_layers}, activation={args.activation})\n")
        f.write(f"param_count: {n_params}\n")
        f.write(f"num_layers: {args.num_layers}\n")
        f.write(f"hidden_dim: {args.hidden_dim}\n")
        f.write(f"activation: {args.activation}\n")
        f.write(f"loss: {args.loss}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"lr: {lr}\n")
        f.write(f"weight_decay: {weight_decay}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"grad_clip_norm: {grad_clip_norm}\n")
        f.write(f"patience: {args.patience}\n")
        f.write(f"min_delta: {args.min_delta}\n")
        f.write(f"validate_every: {args.validate_every}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"target_noise_std: {args.target_noise_std}\n")
        f.write(f"device: {device}\n")
        f.write(f"dataset: Friedman #2 (4-dim, ranges x1=(0,100) x2=(40π,560π) "
                f"x3=(0,1) x4=(1,11), 20k LHS train / 4k uniform val, "
                f"sigma={args.target_noise_std}, normalized targets)\n")

    loss_label = args.loss.upper()

    history = [] 
    val_history = []
    orig_mse_history = []
    orig_rmse_history = []
    orig_mae_history = []
    orig_mape_history = []
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
            u = u.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            if train_wrapper is not None:
                out = train_wrapper(u)
            else:
                out = net(u)
            loss = task_fn(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        avg_train = total_loss / max(1, n_batches)
        do_validate = (epoch % args.validate_every == 0) or (epoch == epochs - 1)
        if do_validate:
            val_loss = validate(net, val_loader, task_fn, device, wrapper=train_wrapper)
            mse_o, rmse_o, mae_o, mape_o = compute_orig_metrics(
                net, val_loader, inverse_stats, device, wrapper=train_wrapper,
            )
        else:
            val_loss = val_history[-1] if val_history else avg_train
            mse_o = orig_mse_history[-1] if orig_mse_history else 0.0
            rmse_o = orig_rmse_history[-1] if orig_rmse_history else 0.0
            mae_o = orig_mae_history[-1] if orig_mae_history else 0.0
            mape_o = orig_mape_history[-1] if orig_mape_history else 0.0

        history.append(avg_train)
        val_history.append(val_loss)
        orig_mse_history.append(mse_o)
        orig_rmse_history.append(rmse_o)
        orig_mae_history.append(mae_o)
        orig_mape_history.append(mape_o)

        if do_validate:
            if val_loss < best_val - args.min_delta:
                best_val = float(val_loss)
                best_epoch = epoch
                epochs_without_improve = 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                epochs_without_improve += args.validate_every
                if epochs_without_improve >= args.patience:
                    print(
                        f"[mlp_friedman2] early stopping at epoch {epoch}: "
                        f"no val improvement for {epochs_without_improve} epochs "
                        f"(best val={best_val:.6f} @ epoch {best_epoch})"
                    )
                    stop_training = True

        if epoch % args.validate_every == 0 or epoch == epochs - 1:
            print(
                f"  epoch {epoch:4d}  train={avg_train:.6f}  val={val_loss:.6f}  "
                f"RMSE_orig={rmse_o:.4f}  MAE_orig={mae_o:.4f}"
            )

    elapsed = time.time() - start
    print(f"[mlp_friedman2] training done in {elapsed:.1f}s ({len(history)} epochs)")

    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"[mlp_friedman2] restored best model from epoch {best_epoch} (val={best_val:.6f})")

    torch.save(net.state_dict(), out_dir / "model.pt")

    with open(out_dir / "loss_history.txt", "w") as f:
        f.write("epoch\ttrain\tval\tmse_orig\trmse_orig\tmae_orig\tmape_orig\n")
        for i, (t, v, ms, r, m, mp) in enumerate(
            zip(history, val_history, orig_mse_history,
                orig_rmse_history, orig_mae_history, orig_mape_history)
        ):
            f.write(f"{i}\t{t}\t{v}\t{ms:.6f}\t{r:.6f}\t{m:.6f}\t{mp:.6f}\n")

    plot_loss_curve(
        history, val_history,
        save_path=str(out_dir / "loss_curve.png"),
        title=f"MLP (hidden={args.hidden_dim}, layers={args.num_layers}, {args.activation}, {n_params} params) — Friedman #2",
        loss_label=loss_label,
    )

    val_batch = next(iter(val_loader))
    u_val, y_val = val_batch[0][:64], val_batch[1][:64]
    u_val = u_val.to(device)
    y_val = y_val.to(device)
    with torch.no_grad():
        if train_wrapper is not None:
            out = train_wrapper(u_val)
        else:
            out = net(u_val)
    plot_output_fit(
        out, y_val,
        save_path=str(out_dir / "output_fit.png"),
        title=f"MLP (hidden={args.hidden_dim}, layers={args.num_layers}, {args.activation}) — Friedman #2 output fit",
    )

    full_val_loss = validate(net, val_loader, task_fn, device, wrapper=train_wrapper)
    final_mse, final_rmse, final_mae, final_mape = compute_orig_metrics(
        net, val_loader, inverse_stats, device, wrapper=train_wrapper,
    )
    best_mse = orig_mse_history[best_epoch] if 0 <= best_epoch < len(orig_mse_history) else float("nan")
    best_rmse = orig_rmse_history[best_epoch] if 0 <= best_epoch < len(orig_rmse_history) else float("nan")
    best_mae = orig_mae_history[best_epoch] if 0 <= best_epoch < len(orig_mae_history) else float("nan")
    best_mape = orig_mape_history[best_epoch] if 0 <= best_epoch < len(orig_mape_history) else float("nan")
    print(
        f"[mlp_friedman2] final val {loss_label} = {full_val_loss:.6f} "
        f"(best val = {best_val:.6f} @ epoch {best_epoch})  "
        f"MSE_orig={final_mse:.4f}  RMSE_orig={final_rmse:.4f}  MAE_orig={final_mae:.4f}"
    )
    with open(out_dir / "final_metrics.txt", "w") as f:
        f.write(f"param_count: {n_params}\n")
        f.write(f"loss: {args.loss}\n")
        f.write(f"best_val: {best_val:.6f}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"final_val: {full_val_loss:.6f}\n")
        f.write(f"best_mse_orig: {best_mse:.6f}\n")
        f.write(f"best_rmse_orig: {best_rmse:.6f}\n")
        f.write(f"best_mae_orig: {best_mae:.6f}\n")
        f.write(f"best_mape_orig: {best_mape:.6f}\n")
        f.write(f"final_mse_orig: {final_mse:.6f}\n")
        f.write(f"final_rmse_orig: {final_rmse:.6f}\n")
        f.write(f"final_mae_orig: {final_mae:.6f}\n")
        f.write(f"final_mape_orig: {final_mape:.6f}\n")
        f.write(f"epochs_run: {len(history)}\n")
        f.write(f"elapsed_seconds: {elapsed:.2f}\n")

    if args.noise or args.noise_aware:
        print(
            f"[mlp_friedman2] running MC noise eval: quant_bits={args.quant_bits} "
            f"hw_noise_std={args.hw_noise_std} trials={args.mc_trials} "
            f"adc_full_range={args.adc_full_range} seed={args.noise_seed} "
            f"adc_enabled={not args.no_adc}"
        )
        eval_cfg = NoiseConfig(
            quant_bits=args.quant_bits,
            noise_std=args.hw_noise_std,
            mc_trials=args.mc_trials,
            seed=args.noise_seed,
            quantize_input=not args.no_adc,
            quantize_output=not args.no_adc,
            quantize_intermediate=not args.no_adc,
        )
        eval_wrapper = AnalogMLPWrapper(
            net, eval_cfg, adc_full_range=args.adc_full_range,
        )
        eval_wrapper.to(device)
        clean_loss = evaluate_clean(
            eval_wrapper, val_loader, task_fn, device,
        )
        noise_result = evaluate_with_noise(
            eval_wrapper, val_loader, task_fn, eval_cfg, device,
        )
        noise_result.clean_loss = clean_loss
        print(
            f"[mlp_friedman2] noise eval: clean={clean_loss:.6f} "
            f"noisy_mean={noise_result.mean:.6f} "
            f"noisy_std={noise_result.std:.6f} "
            f"p90={noise_result.p90:.6f} p95={noise_result.p95:.6f}"
        )
        with open(out_dir / "noise_metrics.txt", "w") as f:
            f.write(f"quant_bits: {args.quant_bits}\n")
            f.write(f"hw_noise_std: {args.hw_noise_std}\n")
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

    print(f"[mlp_friedman2] artifacts in {out_dir}")


if __name__ == "__main__":
    main()