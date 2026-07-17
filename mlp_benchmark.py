"""Minimal MLP baseline for the smooth2d (Franke) regression task.

Purpose: provide a parameter-matched benchmark for the reduced differential
KirchhoffNet. The smooth2d KirchhoffNet has ~430 learnable parameters
(80 edges * 3 cell logits + 80 raw_mult + 12 raw_leak + 80 edge gates +
12 node gates + 4 sparse-mapper + 2 output-mapper). A 2-layer MLP with
hidden_dim=100 has 401 learnable parameters (2*100 + 100 + 100 + 1),
giving a direct apples-to-apples capacity comparison on the same Franke
function dataset.

CLI:
    mlp_benchmark.py [--hidden-dim 100] [--epochs 2000] [--lr 3e-4]
                     [--output OUTPUT] [--device DEVICE]

Outputs to --output:
  - loss_history.txt
  - loss_curve.png
  - model.pt
  - config_snapshot.txt
  - output_fit.png (scatter + small surface)

Defaults match the KirchhoffNet training script (train_script.py):
AdamW (lr=3e-4, weight_decay=1e-4), grad_clip_norm=5.0, batch_size=128,
early stopping with patience=50, min_delta=1e-4.
"""

import argparse
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
from train_script import make_data_smooth2d
from analog_noise import (
    AnalogMLPWrapper,
    NoiseConfig,
    NoiseBenchmarkResult,
    evaluate_clean,
    evaluate_with_noise,
)


__all__ = ["MLPRegressor", "count_parameters"]


_ACTIVATIONS = {
    "relu": F.relu,
    "tanh": torch.tanh,
}

_ACTIVATION_MODULES = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
}


class MLPRegressor(nn.Module):
    """N-layer feedforward regressor: Linear -> Act -> [Linear -> Act] x (N-2) -> Linear.

    Input: (B, in_dim)
    Output: (B, out_dim)

    Default sizes target ~400 learnable parameters to match the smooth2d
    KirchhoffNet: in_dim=2, hidden_dim=100, out_dim=1, num_layers=2
    -> 2*100 + 100 + 100*1 + 1 = 401.

    For num_layers=2, the middle block is empty (2-layer MLP), preserving
    the original behavior. For num_layers > 2, additional hidden layers
    Linear(hidden_dim, hidden_dim) are inserted.

    Activation is selectable via the ``activation`` argument (one of
    ``"relu"``, ``"tanh"``). Parameter count is identical across activations;
    only the forward-pass nonlinearity changes.
    """

    def __init__(
        self,
        in_dim: int = 2,
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


def plot_loss_curve(history, val_history, save_path, title):
    plt = _import_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history, label="train", color="C0")
    ax.plot(val_history, label="val", color="C3")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Train a minimal MLP on the Franke (smooth2d) regression task."
    )
    parser.add_argument("--hidden-dim", type=int, default=100,
                        help="Hidden layer width (default: 100)")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of linear layers (default: 2)")
    parser.add_argument("--epochs", type=int, default=2000,
                        help=f"Number of training epochs (default: 2000)")
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
    parser.add_argument("--activation", choices=["relu", "tanh"], default="relu",
                        help="Hidden activation (default: relu)")
    parser.add_argument("--output", type=Path, default=Path("./output/mlp_smooth2d"),
                        help="Output directory (default: ./output/mlp_smooth2d)")
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
    parser.add_argument("--noise-std", type=float, default=0.05,
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
    args = parser.parse_args()

    lr = args.lr if args.lr is not None else float(OPTIM["lr"])
    weight_decay = args.weight_decay if args.weight_decay is not None else float(OPTIM["weight_decay"])
    batch_size = args.batch_size if args.batch_size is not None else int(OPTIM["batch_size"])
    grad_clip_norm = float(OPTIM["grad_clip_norm"])
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    net = MLPRegressor(in_dim=2, hidden_dim=args.hidden_dim, out_dim=1,
                       num_layers=args.num_layers, activation=args.activation)
    n_params = count_parameters(net)
    net.to(device)

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
        train_wrapper = AnalogMLPWrapper(net, noise_cfg, adc_full_range=args.adc_full_range)
        train_wrapper.to(device)

    train_loader, val_loader, task_fn = make_data_smooth2d(batch_size=batch_size)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    print(
        f"[mlp] hidden_dim={args.hidden_dim} num_layers={args.num_layers} params={n_params} "
        f"activation={args.activation} "
        f"epochs={args.epochs} lr={lr} weight_decay={weight_decay} "
        f"batch_size={batch_size} grad_clip_norm={grad_clip_norm} device={device} "
        f"output={out_dir}"
    )

    with open(out_dir / "config_snapshot.txt", "w") as f:
        f.write(f"model: MLPRegressor(2 -> {args.hidden_dim} x {args.num_layers} -> 1, activation={args.activation})\n")
        f.write(f"param_count: {n_params}\n")
        f.write(f"num_layers: {args.num_layers}\n")
        f.write(f"activation: {args.activation}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"lr: {lr}\n")
        f.write(f"weight_decay: {weight_decay}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"grad_clip_norm: {grad_clip_norm}\n")
        f.write(f"patience: {args.patience}\n")
        f.write(f"min_delta: {args.min_delta}\n")
        f.write(f"validate_every: {args.validate_every}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"device: {device}\n")
        f.write(f"dataset: smooth2d (Franke, 20k train / 4k val, sigma=0.01, normalized)\n")

    history = []
    val_history = []
    best_val = float("inf")
    best_epoch = -1
    best_state = None
    epochs_without_improve = 0
    stop_training = False

    start = time.time()
    for epoch in range(args.epochs):
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
        do_validate = (epoch % args.validate_every == 0) or (epoch == args.epochs - 1)
        if do_validate:
            val_loss = validate(net, val_loader, task_fn, device, wrapper=train_wrapper)
        else:
            val_loss = val_history[-1] if val_history else avg_train

        history.append(avg_train)
        val_history.append(val_loss)

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
                        f"[mlp] early stopping at epoch {epoch}: "
                        f"no val improvement for {epochs_without_improve} epochs "
                        f"(best val={best_val:.6f} @ epoch {best_epoch})"
                    )
                    stop_training = True

        if epoch % args.validate_every == 0 or epoch == args.epochs - 1:
            print(
                f"  epoch {epoch:4d}  train={avg_train:.6f}  val={val_loss:.6f}"
            )

    elapsed = time.time() - start
    print(f"[mlp] training done in {elapsed:.1f}s ({len(history)} epochs)")

    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"[mlp] restored best model from epoch {best_epoch} (val={best_val:.6f})")

    torch.save(net.state_dict(), out_dir / "model.pt")

    with open(out_dir / "loss_history.txt", "w") as f:
        f.write("epoch\ttrain\tval\n")
        for i, (t, v) in enumerate(zip(history, val_history)):
            f.write(f"{i}\t{t}\t{v}\n")

    plot_loss_curve(
        history, val_history,
        save_path=str(out_dir / "loss_curve.png"),
        title=f"MLP (hidden={args.hidden_dim}, {args.activation}, {n_params} params) — smooth2d",
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
        title=f"MLP (hidden={args.hidden_dim}, {args.activation}) — smooth2d output fit",
    )

    full_val_loss = validate(net, val_loader, task_fn, device, wrapper=train_wrapper)
    print(f"[mlp] final val MSE = {full_val_loss:.6f} (best val = {best_val:.6f} @ epoch {best_epoch})")
    with open(out_dir / "final_metrics.txt", "w") as f:
        f.write(f"param_count: {n_params}\n")
        f.write(f"best_val_mse: {best_val:.6f}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"final_val_mse: {full_val_loss:.6f}\n")
        f.write(f"epochs_run: {len(history)}\n")
        f.write(f"elapsed_seconds: {elapsed:.2f}\n")

    if args.noise or args.noise_aware:
        print(
            f"[mlp] running MC noise eval: quant_bits={args.quant_bits} "
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
        eval_wrapper.to(device)
        clean_loss = evaluate_clean(
            eval_wrapper, val_loader, task_fn, device,
        )
        noise_result = evaluate_with_noise(
            eval_wrapper, val_loader, task_fn, eval_cfg, device,
        )
        noise_result.clean_loss = clean_loss
        print(
            f"[mlp] noise eval: clean={clean_loss:.6f} "
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

    print(f"[mlp] artifacts in {out_dir}")


if __name__ == "__main__":
    main()
