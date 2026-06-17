"""Train a 3-stage grid KirchhoffNet as CTLE inverse design student via 4-phase
knowledge distillation from a pre-trained BoundedMLP teacher.

The BoundedMLP teacher is loaded from a state-dict checkpoint (e.g.
``dagger_student_moe.pt``) and used to label synthetic spec samples for
training the KirchhoffNet. The 4-phase schedule (A: free fit, B1: cell
commitment with KD, B2: edge pruning with KD, C: retrain compact network)
is reused from ``train.py`` with the MLP wrapped as a KirchhoffNet-style
teacher (compatible with ``compute_loss``'s KD path).

Output is unbounded logits matching ``BoundedMLP.forward()``. Physical
parameter conversion via ``params_from_logits()`` is applied post-hoc at
evaluation time only.

Usage:
    python train_ctle.py --teacher-path /path/to/dagger_student_moe.pt \\
        [--grid-size 5] [--epochs 800] [--prune] [--output ./output_ctle]
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CELL_ORDER,
    LAMBDAS,
    OPTIM,
    SCHEDULE_FOUR_PHASE,
    SOLVER,
    TAU,
)
from cell_library import IdealizedCellLibrary
from io_mapper import FanOutInputMapper, SparseInputMapper
from kirchhoff_net import KirchhoffNetWithIO
from topology import build_net_from_config, prune_network
from train import (
    compute_loss,
    compute_solidification_metrics,
    four_phase_boundaries,
    four_phase_kd_active,
    four_phase_lambdas,
    four_phase_tau,
    make_optimizer,
    phase_for_epoch_four,
    prune_readiness_check,
    validate_argmax,
)

# Monkey-patch _compute_regularizers to handle DataParallel wrapping
# (same as train_script.py; DP strips module internals from the wrapper).
import train as _train_module
_orig_compute_regularizers = _train_module._compute_regularizers

def _dp_safe_compute_regularizers(net, trajs, tau, lambdas):
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    return _orig_compute_regularizers(net, trajs, tau, lambdas)

_train_module._compute_regularizers = _dp_safe_compute_regularizers

from visualize import (
    plot_cell_selection,
    plot_network,
    plot_output_fit,
    plot_stage_graph,
)


# =============================================================================
# CTLE problem constants
# =============================================================================

# Spec input column order (must match BoundedMLP.forward).
SPEC_INPUT_COLS = ["power", "jitter", "height", "width"]

# Empirical spec sampling ranges (from ctle_ml_dataset.csv, 83k rows).
# power is log-uniform; others are uniform.
SPEC_RANGES = {
    "power": (0.0012, 0.012),
    "jitter": (1.57, 100.0),
    "height": (0.0, 88.4),
    "width": (0.0, 98.5),
}

# Output parameter column order (matches BoundedMLP's PARAM_LOG_BOUNDS).
PARAM_COLS = ["fW", "current", "ind", "Rd", "Cs", "Rs", "VDD"]

# Log10 bounds for the bounded prediction head.
PARAM_LOG_BOUNDS: dict[str, tuple[float, float]] = {
    "fW":   (math.log10(1e-6),  math.log10(10.0)),
    "current": (math.log10(5e-4), math.log10(2.5)),
    "ind":  (math.log10(1e-12), math.log10(3.0)),
    "Rd":   (math.log10(10),    math.log10(1500)),
    "Cs":   (math.log10(1e-15), math.log10(1e-9)),
    "Rs":   (math.log10(10),    math.log10(1500)),
    "VDD":  (math.log10(0.6),   math.log10(1.2)),
}


def params_from_logits(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert unbounded logits to physical parameter values.

    Applies sigmoid + log-space interpolation in [log_lo, log_hi] per param
    and returns ``10 ** bounded_log`` (the physical value in original units).
    """
    probs = torch.sigmoid(logits)
    out: dict[str, torch.Tensor] = {}
    for i, name in enumerate(PARAM_COLS):
        lo, hi = PARAM_LOG_BOUNDS[name]
        bounded = lo + (hi - lo) * probs[..., i]
        out[name] = torch.pow(10.0, bounded)
    return out


# =============================================================================
# CTLE grid preset
# =============================================================================

def make_ctle_grid_preset(
    grid_size: int = 5,
    num_stages: int = 3,
    num_proj: int = 7,
    write_mode: str | None = None,
) -> dict:
    """Build the ctle_grid preset dict for a 4-spec → 7-logit KirchhoffNet.

    Each stage has ``grid_size**2`` hidden nodes, ``num_proj`` projection
    nodes, and 4 input nodes (one per spec). The write mapping is fan-out
    from 4 inputs to 4 grid quadrants; the read mapping gathers the
    center column + all projection nodes (5 + num_proj features) and
    projects them to 7 logits.

    When ``write_mode`` is provided (one of ``'fan_out'``, ``'dense'``,
    ``'one_to_one'``), it overrides the default write mapping. ``'dense'``
    uses an all-to-all ``InputMapper``; ``'one_to_one'`` uses
    ``SparseInputMapper`` with the first 4 hidden nodes as write targets.
    """
    num_hidden = grid_size * grid_size
    n_stages = max(1, num_stages)
    _stage_cfg = {
        "num_inputs": 4,
        "num_hidden": num_hidden,
        "num_proj": num_proj,
        "num_outputs": 0,
        "hidden_family": "grid",
        "hidden_kwargs": {"height": grid_size, "width": grid_size, "kernel_size": 3},
        "input_pattern": "all_to_all",
        "output_pattern": "all_to_all",
        "proj_pattern": "all_to_all",
        "t_span": SOLVER["t_span"] / n_stages,
        "num_steps": round(SOLVER["num_steps"] / n_stages),
    }

    # Resolve effective write mode.
    eff_write = write_mode if write_mode is not None else "fan_out"

    # Fan-out write: each of 4 inputs writes to 2 corner grid nodes.
    # Layout: input 0 (power)  → top-left, input 1 (jitter) → top-right,
    #         input 2 (height) → bottom-left, input 3 (width) → bottom-right.
    # Use 2 rows per corner (top half = rows 0,1 ; bottom half = rows N-2, N-1).
    top_rows = [0, 1]
    bot_rows = [grid_size - 2, grid_size - 1]
    fan_out = {
        0: [r * grid_size + 0 for r in top_rows],
        1: [r * grid_size + (grid_size - 1) for r in top_rows],
        2: [r * grid_size + 0 for r in bot_rows],
        3: [r * grid_size + (grid_size - 1) for r in bot_rows],
    }

    # Read: center column hidden nodes + all projection nodes.
    center_col = grid_size // 2
    center_nodes = [r * grid_size + center_col for r in range(grid_size)]
    read_idx = center_nodes + list(range(num_hidden, num_hidden + num_proj))

    preset: dict[str, Any] = {
        "stages": [_stage_cfg] * n_stages,
        "use_robust_input": False,
        "loss": "mse",
        "out_dim": len(PARAM_COLS),
        "write_mode": eff_write,
        "read_idx": read_idx,
        "schedule": "four_phase",
        "lambdas": {
            "sparsity": 1e-5,
            "edge_gate": 5e-6,
            "node_gate": 0.0,
            "power": 1e-5,
            "capacitance": 1e-6,
            "rail": 0.1,
        },
        "tau_anneal": True,
    }
    if eff_write == "fan_out":
        preset["write_fan_out"] = fan_out
    return preset


# =============================================================================
# BoundedMLP (teacher) and adapter
# =============================================================================

class BoundedMLP(nn.Module):
    """4-layer MLP that maps 4 specs to 7 unbounded logits.

    Mirrors the architecture in ``generative-distillation.py`` so that the
    pre-trained state-dict (``dagger_student_moe.pt``) loads cleanly.
    ``log_lo`` / ``log_hi`` are non-trainable buffers (they store the
    PARAM_LOG_BOUNDS in log10 space).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 4,
        input_dim: int = 4,
        output_dim: int = 7,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        layers: list[nn.Module] = []
        dims = [input_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.GELU(),
                nn.LayerNorm(dims[i + 1]),
            ])
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, output_dim)

        self.register_buffer("log_lo", torch.zeros(output_dim))
        self.register_buffer("log_hi", torch.zeros(output_dim))
        for i, (lo, hi) in enumerate(PARAM_LOG_BOUNDS.values()):
            self.log_lo[i] = lo
            self.log_hi[i] = hi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class MLPTeacherWrapper(nn.Module):
    """Adapter that exposes a BoundedMLP with KirchhoffNet's forward signature.

    ``compute_loss`` calls ``teacher(u, ctx=..., tau=..., store_trajectory=...,
    cell_mode=...)`` and unpacks ``(y_teacher, _)``. This wrapper ignores the
    KirchhoffNet-specific kwargs and returns ``(mlp(u), None)``.
    """

    def __init__(self, mlp: BoundedMLP) -> None:
        super().__init__()
        self.mlp = mlp

    def forward(
        self,
        u: torch.Tensor,
        ctx: Any = None,
        tau: float | None = None,
        store_trajectory: bool = False,
        cell_mode: str = "soft",
    ) -> tuple[torch.Tensor, None]:
        return self.mlp(u), None


# =============================================================================
# Data generation
# =============================================================================

def sample_specs(n: int, seed: int = 42) -> torch.Tensor:
    """Sample ``n`` spec vectors of shape ``[n, 4]`` from SPEC_RANGES.

    Power is log-uniform (covers one decade); the other three specs are
    uniform within their empirical bounds. Uses numpy's legacy RandomState
    so the dataset is fully reproducible regardless of torch's RNG state.
    """
    rng = np.random.RandomState(seed)
    p_lo, p_hi = SPEC_RANGES["power"]
    j_lo, j_hi = SPEC_RANGES["jitter"]
    h_lo, h_hi = SPEC_RANGES["height"]
    w_lo, w_hi = SPEC_RANGES["width"]

    log_p = rng.uniform(math.log10(p_lo), math.log10(p_hi), size=n)
    power = 10.0 ** log_p
    jitter = rng.uniform(j_lo, j_hi, size=n)
    height = rng.uniform(h_lo, h_hi, size=n)
    width = rng.uniform(w_lo, w_hi, size=n)

    specs = np.stack([power, jitter, height, width], axis=1).astype(np.float32)
    return torch.from_numpy(specs)


def generate_ctle_dataset(
    n_train: int,
    n_val: int,
    mlp: BoundedMLP,
    device: torch.device,
    batch_size: int,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders of (specs, mlp_logits) pairs.

    Generates ``n_train + n_val`` spec samples, labels them via the MLP,
    then takes the first ``n_train`` as training and the remaining
    ``n_val`` as validation. Both subsets are deterministic via ``seed``.
    """
    mlp.eval()
    mlp.to(device)

    n_total = n_train + n_val
    specs = sample_specs(n_total, seed=seed).to(device)
    with torch.no_grad():
        all_logits: list[torch.Tensor] = []
        for i in range(0, n_total, batch_size):
            batch = specs[i:i + batch_size]
            all_logits.append(mlp(batch).detach().to("cpu"))
        logits = torch.cat(all_logits, dim=0)

    train_ds = TensorDataset(specs[:n_train].to("cpu"), logits[:n_train])
    val_ds = TensorDataset(specs[n_train:].to("cpu"), logits[n_train:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


# =============================================================================
# Validation helpers
# =============================================================================

def validate(
    net: KirchhoffNetWithIO,
    val_loader: DataLoader,
    task_fn,
    device: torch.device,
) -> float:
    """Mean task loss over the val loader (no ctx — we use a static no-op)."""
    net.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for u, target in val_loader:
            u = u.to(device)
            target = target.to(device)
            out, _ = net(u, ctx=None, store_trajectory=False, cell_mode="soft")
            loss = task_fn(out, target)
            total += float(loss.item()) * u.size(0)
            n += u.size(0)
    net.train()
    return total / max(1, n)


# =============================================================================
# I/O mapper transfer helpers (copied from train_script to keep this file
# self-contained; same logic, no copy/paste from the train_script namespace).
# =============================================================================

def _remap_indices(idx_list, remap):
    out = []
    for i in idx_list:
        if i in remap:
            out.append(remap[i])
    return out


def _transfer_input_mapper(raw_mapper, raw_write_idx, stage0_remap, pruned_first_n, in_dim):
    """Rebuild the InputMapper for the pruned stage, transferring weights.

    Supports FanOutInputMapper (the only mode we use for ctle_grid).
    """
    from io_mapper import FanOutInputMapper

    if isinstance(raw_mapper, FanOutInputMapper):
        new_fan_out: dict[int, list[int]] = {}
        for inp, targets in raw_mapper.fan_out_map.items():
            new_fan_out[inp] = _remap_indices(targets, stage0_remap)
        new_mapper = FanOutInputMapper(
            in_dim=raw_mapper.in_dim,
            out_dim=pruned_first_n,
            fan_out_map=new_fan_out,
            x_max=raw_mapper.x_max,
        )
        with torch.no_grad():
            # Only copy gain/bias for targets that survived pruning.
            old_flat = raw_mapper._flat_targets.tolist()
            surviving = torch.tensor(
                [t in stage0_remap for t in old_flat], dtype=torch.bool
            )
            new_mapper.gain.data.copy_(raw_mapper.gain.data[surviving])
            new_mapper.bias.data.copy_(raw_mapper.bias.data[surviving])
        return new_mapper, None

    raise TypeError(
        f"transfer_input_mapper: unsupported mapper type {type(raw_mapper).__name__}"
    )


def _transfer_output_mapper(raw_mapper, raw_read_idx, last_remap, pruned_last_n, out_dim):
    """Rebuild the OutputMapper for the pruned stage, transferring weights.

    Uses sparse read mode (read_idx is always set in ctle_grid).
    """
    from io_mapper import OutputMapper

    if raw_read_idx is not None:
        new_read_idx = _remap_indices(raw_read_idx, last_remap)
        new_mapper = OutputMapper(
            node_dim=pruned_last_n, out_dim=out_dim, read_idx=new_read_idx,
        )
        surviving_old_positions = [
            i for i, idx in enumerate(raw_read_idx) if idx in last_remap
        ]
        with torch.no_grad():
            new_mapper.proj.weight.data.copy_(
                raw_mapper.proj.weight.data[:, surviving_old_positions]
            )
            new_mapper.proj.bias.data.copy_(raw_mapper.proj.bias.data)
        return new_mapper, new_read_idx

    raise TypeError("transfer_output_mapper: ctle_grid requires read_idx")


# =============================================================================
# Solidification log helper
# =============================================================================

def _log_solidification(log_path: Path, epoch: int, metrics: dict) -> None:
    """Append one row of solidification metrics to ``log_path``."""
    sorted_keys = sorted(metrics.keys())
    if not log_path.exists() or log_path.stat().st_size == 0:
        with log_path.open("w") as f:
            f.write("epoch\t" + "\t".join(sorted_keys) + "\n")
    with log_path.open("a") as f:
        row = [str(epoch)]
        for k in sorted_keys:
            v = metrics[k]
            row.append(f"{v:.6e}" if isinstance(v, float) else str(v))
        f.write("\t".join(row) + "\n")


# =============================================================================
# Gradient norm helpers (ported from train_script.py)
# =============================================================================


def collect_gradient_norms(raw_net):
    """Walk ``raw_net`` (assumed unwrapped from DataParallel) and collect
    per-group L2 gradient norms.

    Returns: dict with keys
      stage{i}_logits, stage{i}_raw_mult, stage{i}_raw_leak,
      stage{i}_z_logits, stage{i}_u_logits  (one entry per stage)
      stage_transfer
      in_mapper
      out_mapper
    Each value is either a float (the L2 norm) or None if the group has
    no parameter or no gradient.
    """
    stage_sq = {}
    stage_components = ("logits", "raw_mult", "raw_leak", "z_logits", "u_logits")
    transfer_sq = 0.0
    transfer_found = False
    in_sq = 0.0
    in_found = False
    out_sq = 0.0
    out_found = False

    for name, p in raw_net.named_parameters():
        if ".stages." in name:
            for comp in stage_components:
                if name.endswith("." + comp):
                    stage_idx = int(name.split(".stages.")[1].split(".")[0])
                    stage_sq.setdefault(f"stage{stage_idx}_{comp}", 0.0)

    for name, p in raw_net.named_parameters():
        if p.grad is None:
            continue
        gnorm_sq = float(p.grad.norm().item()) ** 2
        if ".stages." in name:
            for comp in stage_components:
                if name.endswith("." + comp):
                    stage_idx = int(name.split(".stages.")[1].split(".")[0])
                    key = f"stage{stage_idx}_{comp}"
                    stage_sq[key] = stage_sq.get(key, 0.0) + gnorm_sq
        if "transfers" in name or "stage_transfer" in name:
            transfer_sq += gnorm_sq
            transfer_found = True
        if "input_mapper" in name:
            in_sq += gnorm_sq
            in_found = True
        elif "output_mapper" in name:
            out_sq += gnorm_sq
            out_found = True

    out = {}
    for k, v in stage_sq.items():
        out[k] = v ** 0.5 if v > 0 else None
    out["stage_transfer"] = transfer_sq ** 0.5 if transfer_found else None
    out["in_mapper"] = in_sq ** 0.5 if in_found else None
    out["out_mapper"] = out_sq ** 0.5 if out_found else None
    return out


def _grad_norm_keys(norms):
    """Deterministic key ordering for gradient norm output."""
    return sorted(
        [k for k in norms.keys() if k.startswith("stage")]
        + [k for k in ("stage_transfer", "in_mapper", "out_mapper") if k in norms]
    )


def log_gradient_norms(grad_log_path, epoch, raw_net, *, retrain=False, optimizer=None):
    """Append one row of per-group L2 gradient norms to ``grad_log_path``.

    On the first call, also writes the header row.
    If an ``optimizer`` with multiple param groups is provided, appends the
    per-group learning rates as additional columns (``lr0``, ``lr1``, ...).
    """
    raw = raw_net.module if isinstance(raw_net, torch.nn.DataParallel) else raw_net
    norms = collect_gradient_norms(raw)
    ordered_keys = _grad_norm_keys(norms)

    lr_keys = []
    if optimizer is not None and len(optimizer.param_groups) > 1:
        lr_keys = [f"lr{i}" for i in range(len(optimizer.param_groups))]

    if not grad_log_path.exists():
        cols = ordered_keys + lr_keys
        with open(grad_log_path, "w") as f:
            f.write("epoch\t" + "\t".join(cols) + "\n")

    prefix = "retrain_" if retrain else ""
    row_parts = [f"{prefix}{epoch}"]
    for k in ordered_keys:
        v = norms.get(k)
        row_parts.append(f"{v:.6e}" if v is not None else "-")
    if lr_keys:
        for g in optimizer.param_groups:
            row_parts.append(f"{g['lr']:.6e}")
    with open(grad_log_path, "a") as f:
        f.write("\t".join(row_parts) + "\n")


# =============================================================================
# Argmax cell mode helpers (4-phase training uses STE in B/C; we implement
# STE inline via train.compute_loss(cell_mode='ste') which already exists).
# =============================================================================

def resolve_cell_mode(cli_value: str, phase: str) -> str:
    """Resolve the cell selection mode for the current epoch.

    'soft' uses softmax-weighted mixture of cells per edge.
    'ste' uses one cell per edge in forward + straight-through soft grads.
    'auto' returns 'ste' for phases B1, B2, C; 'soft' for phase A.
    """
    if cli_value in ("soft", "ste"):
        return cli_value
    return "ste" if phase in ("B1", "B2", "C") else "soft"


# =============================================================================
# Main training entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a 3-stage grid KirchhoffNet as CTLE inverse design student via 4-phase KD from a pre-trained BoundedMLP teacher."
    )
    parser.add_argument(
        "--teacher-path", type=Path, required=True,
        help="Path to the BoundedMLP state-dict checkpoint (e.g. dagger_student_moe.pt).",
    )
    parser.add_argument(
        "--grid-size", type=int, default=5,
        help="Hidden grid height/width (default: 5, total hidden = grid_size^2 per stage).",
    )
    parser.add_argument(
        "--num-stages", type=int, default=3,
        help="Number of ODE stages (default: 3).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("./output_ctle"),
        help="Output directory (default: ./output_ctle).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=f"Total training epochs (default: {OPTIM['epochs']}).",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help=f"Base learning rate (default: {OPTIM['lr']}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help=f"Batch size (default: {OPTIM['batch_size']}).",
    )
    parser.add_argument(
        "--n-train", type=int, default=100000,
        help="Number of synthetic training samples to generate (default: 100000).",
    )
    parser.add_argument(
        "--n-val", type=int, default=20000,
        help="Number of synthetic validation samples (default: 20000).",
    )
    parser.add_argument(
        "--prune", dest="prune", action="store_true", default=False,
        help="Run gate-based pruning at the B2→C boundary and retrain.",
    )
    parser.add_argument(
        "--no-prune", dest="prune", action="store_false",
        help="Skip pruning (train only, no retrain).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device 'cpu' or 'cuda' (default: auto-detect).",
    )
    parser.add_argument(
        "--validate-every", type=int, default=5,
        help="Validate every N epochs (default: 5).",
    )
    parser.add_argument(
        "--early-stop", dest="early_stop", action="store_true", default=True,
        help="Enable early stopping in Phase A+B1+B2 (default: on).",
    )
    parser.add_argument(
        "--no-early-stop", dest="early_stop", action="store_false",
        help="Disable early stopping.",
    )
    parser.add_argument(
        "--patience", type=int, default=500,
        help="Early stopping patience in epochs (default: 500).",
    )
    parser.add_argument(
        "--min-delta", type=float, default=1e-4,
        help="Early stopping min improvement in val loss (default: 1e-4).",
    )
    parser.add_argument(
        "--cell-mode", choices=["soft", "ste", "auto"], default="auto",
        help="Cell selection mode. 'auto' uses 'ste' for B1/B2/C, 'soft' for A.",
    )
    parser.add_argument(
        "--retrain-lr", type=float, default=None,
        help="Learning rate for Phase C retrain (default: same as --lr).",
    )
    parser.add_argument(
        "--retrain-epochs", type=int, default=None,
        help="Epochs for Phase C retrain (default: same as --epochs, capped at half).",
    )
    parser.add_argument(
        "--stage-lr-scale", type=float, default=1.0,
        help="Per-stage geometric LR multiplier (default: 1.0).",
    )
    parser.add_argument(
        "--mapper-lr-scale", type=float, default=1.0,
        help="LR multiplier for I/O mapper params (default: 1.0).",
    )
    parser.add_argument(
        "--compile", dest="compile", action="store_true", default=None,
        help="Enable torch.compile on hot paths (default: on when CUDA).",
    )
    parser.add_argument(
        "--no-compile", dest="compile", action="store_false",
        help="Disable torch.compile.",
    )
    parser.add_argument(
        "--parallel", dest="parallel", action="store_true", default=None,
        help="Enable DataParallel across multiple GPUs (default: on when >=2 GPUs).",
    )
    parser.add_argument(
        "--no-parallel", dest="parallel", action="store_false",
        help="Disable DataParallel.",
    )
    parser.add_argument(
        "--write-mode", choices=["fan_out", "dense", "one_to_one"], default=None,
        help="Input write mapping (default: from preset, typically 'fan_out'). "
             "'fan_out' uses FanOutInputMapper with preset-defined targets; "
             "'dense' uses all-to-all InputMapper; "
             "'one_to_one' uses SparseInputMapper with first 4 hidden nodes.",
    )
    parser.add_argument(
        "--grad-log", dest="grad_log", action="store_true", default=False,
        help="Enable per-epoch gradient norm logging.",
    )
    parser.add_argument(
        "--grad-log-every", type=int, default=10,
        help="Log gradient norms every N epochs (default: 10, used when --grad-log is enabled).",
    )
    args = parser.parse_args()

    # ---- resolve config ----
    epochs = args.epochs if args.epochs is not None else int(OPTIM["epochs"])
    lr = args.lr if args.lr is not None else float(OPTIM["lr"])
    batch_size = args.batch_size if args.batch_size is not None else int(OPTIM["batch_size"])
    if args.retrain_lr is not None:
        retrain_lr = args.retrain_lr
    else:
        retrain_lr = lr

    out_dir = args.output.resolve()
    if out_dir.exists():
        suffix = time.strftime("%Y%m%d_%H%M%S")
        out_dir = out_dir.with_name(f"{out_dir.name}_{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device is not None:
        device = args.device
    device = torch.device(device)

    n_gpus = torch.cuda.device_count() if str(device).startswith("cuda") else 0
    compile_enabled = args.compile if args.compile is not None else (n_gpus >= 1)
    parallel_enabled = args.parallel if args.parallel is not None else (n_gpus >= 2)

    print(f"[train_ctle] device={device} epochs={epochs} lr={lr} batch_size={batch_size} "
          f"grid_size={args.grid_size} num_stages={args.num_stages} "
          f"compile={compile_enabled} parallel={parallel_enabled} ({n_gpus} GPUs) "
          f"output={out_dir}")

    # ---- load teacher MLP ----
    if not args.teacher_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_path}")
    mlp = BoundedMLP(hidden_dim=64, num_layers=4, input_dim=4, output_dim=7)
    state = torch.load(args.teacher_path, map_location="cpu")
    mlp.load_state_dict(state)
    mlp.eval()
    mlp.requires_grad_(False)
    mlp.to(device)
    n_teacher_params = sum(p.numel() for p in mlp.parameters())
    print(f"[train_ctle] loaded teacher MLP from {args.teacher_path} "
          f"({n_teacher_params} params)")

    # ---- build KirchhoffNet ----
    cell_lib = IdealizedCellLibrary()
    preset = make_ctle_grid_preset(
        grid_size=args.grid_size, num_stages=args.num_stages,
        write_mode=args.write_mode,
    )
    net: KirchhoffNetWithIO = build_net_from_config(preset, cell_lib=cell_lib)
    net.to(device)
    n_kirchhoff_params = sum(p.numel() for p in net.parameters())
    in_mapper_name = type(net.input_mapper).__name__
    out_mapper_name = type(net.output_mapper).__name__
    print(f"[train_ctle] built KirchhoffNet: in_dim=4 out_dim=7 "
          f"hid={net.hid_count} proj={net.proj_count} stages={len(net.core.stages)} "
          f"input_mapper={in_mapper_name} output_mapper={out_mapper_name} "
          f"({n_kirchhoff_params} params)")
    print(f"[train_ctle] write_idx={list(net.write_idx) if net.write_idx is not None else None} "
          f"read_idx={list(net.read_idx) if net.read_idx is not None else None}")

    # Resolve mapper names for later use.
    _effective_write_mode = (
        "fan_out" if isinstance(net.input_mapper, FanOutInputMapper)
        else "one_to_one" if isinstance(net.input_mapper, SparseInputMapper)
        else "dense"
    )
    if args.prune and _effective_write_mode != "fan_out":
        raise ValueError(
            f"--prune with --write-mode={_effective_write_mode} is not supported; "
            f"pruning requires fan_out write mode (for _transfer_input_mapper compatibility). "
            f"Use --write-mode fan_out (default) or omit --prune."
        )

    # ---- torch.compile (hot paths) ----
    if parallel_enabled and n_gpus >= 2:
        compile_enabled = False
        print("[train_ctle] disabling compile for DataParallel compatibility (use single GPU for compile)")

    if compile_enabled and str(device).startswith("cuda"):
        try:
            cell_lib.compile_forward()
            for stage in net.core.stages:
                if stage.num_edges() > 0:
                    stage.compile_rhs()
            print("[train_ctle] torch.compile enabled on cell_lib.forward and stage.rhs")
        except Exception as e:
            print(f"[train_ctle] torch.compile setup failed: {e}; continuing without compile")
            compile_enabled = False

    # ---- DataParallel ----
    if parallel_enabled and n_gpus >= 2:
        try:
            net = torch.nn.DataParallel(net, device_ids=list(range(n_gpus)))
            print(f"[train_ctle] DataParallel enabled on {n_gpus} GPUs")
        except Exception as e:
            print(f"[train_ctle] DataParallel setup failed: {e}; continuing single-GPU")

    def _unwrap(m):
        return m.module if isinstance(m, torch.nn.DataParallel) else m

    raw_net = _unwrap(net)

    # Save config snapshot
    snapshot_path = out_dir / "config_snapshot.txt"
    with snapshot_path.open("w") as f:
        f.write(f"teacher_path: {args.teacher_path}\n")
        f.write(f"teacher_params: {n_teacher_params}\n")
        f.write(f"grid_size: {args.grid_size}\n")
        f.write(f"num_stages: {args.num_stages}\n")
        f.write(f"hid_count: {net.hid_count}\n")
        f.write(f"proj_count: {net.proj_count}\n")
        f.write(f"write_idx: {list(net.write_idx) if net.write_idx is not None else None}\n")
        f.write(f"read_idx: {list(net.read_idx) if net.read_idx is not None else None}\n")
        f.write(f"kirchhoff_params: {n_kirchhoff_params}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"lr: {lr}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"stage_lr_scale: {args.stage_lr_scale}\n")
        f.write(f"mapper_lr_scale: {args.mapper_lr_scale}\n")
        f.write(f"write_mode: {_effective_write_mode}\n")
        f.write(f"compile: {compile_enabled}\n")
        f.write(f"parallel: {parallel_enabled}\n")
        f.write(f"grad_log: {args.grad_log}\n")
        f.write(f"grad_log_every: {args.grad_log_every}\n")
        f.write(f"prune: {args.prune}\n")
        f.write(f"\nLAMBDAS (effective):\n")
        for k, v in preset["lambdas"].items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nSCHEDULE_FOUR_PHASE fractions: {SCHEDULE_FOUR_PHASE['frac_a']}/"
                f"{SCHEDULE_FOUR_PHASE['frac_b1']}/{SCHEDULE_FOUR_PHASE['frac_b2']}/"
                f"{SCHEDULE_FOUR_PHASE['frac_c']}\n")

    # ---- generate dataset ----
    print(f"[train_ctle] generating {args.n_train + args.n_val} synthetic spec samples "
          f"(train={args.n_train}, val={args.n_val})...")
    train_loader, val_loader = generate_ctle_dataset(
        n_train=args.n_train,
        n_val=args.n_val,
        mlp=mlp,
        device=device,
        batch_size=batch_size,
    )
    print(f"[train_ctle] train batches={len(train_loader)} val batches={len(val_loader)}")

    task_fn = F.mse_loss

    # ---- save init stage graphs ----
    for i, stage in enumerate(raw_net.core.stages):
        plot_stage_graph(
            stage,
            save_path=str(out_dir / f"stage{i + 1}_graph_init.png"),
            title=f"ctle_grid — Stage {i + 1} (init)",
        )

    # ---- schedule boundaries ----
    a_end, b1_end, b2_end, c_end = four_phase_boundaries(epochs)
    print(f"[train_ctle] four_phase schedule: A=[0,{a_end}) B1=[{a_end},{b1_end}) "
          f"B2=[{b1_end},{b2_end}) C=[{b2_end},{c_end})")
    ab_total = b2_end
    c_total = max(1, epochs - b2_end)
    if args.retrain_epochs is not None:
        c_total = args.retrain_epochs

    # ---- 4-phase training state ----
    teacher_net = MLPTeacherWrapper(mlp).to(device)

    optimizer = make_optimizer(
        net, lr=lr,
        stage_lr_scale=args.stage_lr_scale,
        mapper_lr_scale=args.mapper_lr_scale,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, ab_total), eta_min=float(OPTIM["scheduler_eta_min"]),
    )

    grad_clip = float(OPTIM["grad_clip_norm"])
    history: list[float] = []
    val_history: list[float] = []
    val_argmax_history: list[float] = []

    best_val = float("inf")
    best_epoch = -1
    best_state: dict | None = None
    best_metric_name = "val"
    ewop = 0
    stop_training = False

    solid_log_path = out_dir / "solidification_metrics.txt"
    grad_log_path: Path | None = (
        out_dir / "gradient_norms.txt" if args.grad_log else None
    )
    if grad_log_path is not None:
        print(f"[train_ctle] gradient logging enabled (every {args.grad_log_every} epochs) -> {grad_log_path}")
    val_v_history: list[float] = []
    val_argmax_v_history: list[float] = []
    solid_metrics_history: list[dict] = []

    readiness_prune_fired = False
    readiness_prune_epoch = -1

    # ---- Phase A + B1 + B2 training loop ----
    print("[train_ctle] starting 4-phase training loop")
    for epoch in range(ab_total):
        if stop_training:
            break
        net.train()
        tau = four_phase_tau(epoch, epochs)
        phase = phase_for_epoch_four(epoch, epochs)
        eff_lambdas = four_phase_lambdas(epoch, epochs, preset["lambdas"])
        cell_mode = resolve_cell_mode(args.cell_mode, phase)

        # Stop early if readiness triggered (handled at validate step, but
        # we also break here if the flag was set in a previous iteration).
        if phase == "B2" and readiness_prune_fired:
            print(f"[four_phase] readiness pruning triggered at epoch {epoch} "
                  f"(B2 cut short, was scheduled for {b2_end})")
            break

        # ---- one epoch ----
        total_loss = 0.0
        n_batches = 0
        for u_b, tgt_b in train_loader:
            optimizer.zero_grad()
            u_b = u_b.to(device)
            tgt_b = tgt_b.to(device)

            # Teacher KD is active in B1/B2 only.
            kd_teacher = teacher_net if (phase in ("B1", "B2")) else None
            kd_lambda = float(SCHEDULE_FOUR_PHASE.get("lambda_kd", 1.0)) if kd_teacher is not None else 0.0

            loss_task, loss_structural, _ = compute_loss(
                net, u_b, tgt_b, ctx=None, task_fn=task_fn,
                lambdas=eff_lambdas, tau=tau, return_parts=True,
                amp=False, reg_scale=1.0, cell_mode=cell_mode,
                teacher=kd_teacher, lambda_kd=kd_lambda, teacher_tau=1.0,
                teacher_cell_mode="soft",
            )
            loss_task.backward(retain_graph=True)
            loss_structural.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
            optimizer.step()
            total_loss += float((loss_task + loss_structural).item())
            n_batches += 1
        avg_train = total_loss / max(1, n_batches)

        # ---- gradient logging ----
        if grad_log_path is not None and epoch % args.grad_log_every == 0:
            log_gradient_norms(grad_log_path, epoch, raw_net, optimizer=optimizer)

        # ---- validation ----
        do_validate = (epoch % args.validate_every == 0) or (epoch == ab_total - 1)
        if do_validate:
            val_loss = validate(net, val_loader, task_fn, device)
            val_v_history.append(val_loss)
            val_history.append(val_loss)
            val_arg = validate_argmax(net, val_loader, task_fn, lambda b: None, device)
            val_argmax_history.append(val_arg)
            val_argmax_v_history.append(val_arg)

            if phase in ("A", "B1", "B2"):
                metrics = compute_solidification_metrics(raw_net, tau=tau)
                _log_solidification(solid_log_path, epoch, metrics)
                solid_metrics_history.append(metrics)

            # Phase B2 readiness check
            if (
                phase == "B2"
                and len(val_argmax_v_history) >= 10
                and len(val_v_history) >= 10
                and len(solid_metrics_history) >= 10
            ):
                is_ready, ready_details = prune_readiness_check(
                    val_v_history, val_argmax_v_history, solid_metrics_history,
                )
                _log_solidification(
                    solid_log_path, epoch,
                    {"ready_ratio": ready_details.get("ratio", -1.0),
                     "ready_prob": ready_details.get("max_cell_prob", -1.0),
                     "ready_stability": ready_details.get("stability", -1.0),
                     "ready_improvement": ready_details.get("improvement_rate", -1.0),
                     "all_ready": 1.0 if ready_details.get("all_ready", False) else 0.0},
                )
                if is_ready and not readiness_prune_fired:
                    readiness_prune_fired = True
                    readiness_prune_epoch = epoch
                    print(f"[four_phase] READINESS TRIGGERED at epoch {epoch}: "
                          f"ratio={ready_details['ratio']:.3f}, "
                          f"prob={ready_details['max_cell_prob']:.3f}, "
                          f"stability={ready_details['stability']:.4f}, "
                          f"improvement={ready_details['improvement_rate']:.6f}")
        else:
            val_loss = val_history[-1] if val_history else avg_train
            val_history.append(val_loss)
            if val_argmax_history:
                val_argmax_history.append(val_argmax_history[-1])

        history.append(avg_train)

        # ---- checkpointing (use argmax val for B1/B2; soft for A) ----
        if do_validate:
            use_argmax_ckpt = phase in ("B1", "B2") and val_argmax_history and len(val_argmax_history) > 0
            if use_argmax_ckpt:
                sel_metric = float(val_argmax_history[-1])
                sel_name = "val_argmax"
            else:
                sel_metric = float(val_loss)
                sel_name = "val"
            if sel_metric < best_val - args.min_delta:
                best_val = sel_metric
                best_epoch = epoch
                best_metric_name = sel_name
                ewop = 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                ewop += args.validate_every
                if args.early_stop and ewop >= args.patience:
                    print(f"[train_ctle] early stop at epoch {epoch}: "
                          f"no {best_metric_name} improvement for {ewop} epochs "
                          f"(best {best_metric_name}={best_val:.4f} @ epoch {best_epoch})")
                    stop_training = True

        scheduler.step()
        print(
            f"  epoch {epoch:4d} [{phase}]  train={avg_train:.4f}  "
            f"val={val_loss:.4f}  val_argmax={val_argmax_history[-1] if val_argmax_history else 0:.4f}  "
            f"tau={tau:.3f}  lr={optimizer.param_groups[0]['lr']:.2e}"
        )

    # ---- end of Phase A+B1+B2 ----
    if not stop_training and not readiness_prune_fired:
        readiness_prune_fired = True
        readiness_prune_epoch = b2_end
        print(f"[train_ctle] A+B1+B2 complete, no readiness trigger — "
              f"fallback prune at epoch {b2_end}")

    # ---- restore best pre-prune state ----
    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"[train_ctle] restored best pre-prune state "
              f"(epoch {best_epoch}, {best_metric_name}={best_val:.4f})")

    # ---- save pre-prune artifacts ----
    loss_history_path = out_dir / "loss_history.txt"
    with loss_history_path.open("w") as f:
        f.write("epoch\ttrain\tval\tval_argmax\tphase\n")
        for i, (t, v) in enumerate(zip(history, val_history)):
            p = phase_for_epoch_four(i, epochs)
            va = val_argmax_history[i] if i < len(val_argmax_history) else float("nan")
            f.write(f"{i}\t{t:.6e}\t{v:.6e}\t{va:.6e}\t{p}\n")

    # loss curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    fig, ax = _plt.subplots(figsize=(8, 4.5))
    ax.plot(history, label="train", color="C0")
    ax.plot(val_history, label="val (soft)", color="C3")
    if val_argmax_history:
        ax.plot(val_argmax_history, label="val (argmax)", color="C2", linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (logit space)")
    ax.set_title(f"ctle_grid (KD from MLP) — 4-phase training (total {epochs} epochs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=120, bbox_inches="tight")
    _plt.close(fig)

    # output fit (logit space)
    with torch.no_grad():
        val_batch = next(iter(val_loader))
        u_v = val_batch[0][:64].to(device)
        y_v = val_batch[1][:64].to(device)
        out_v, _ = net(u_v, ctx=None, store_trajectory=False, cell_mode="soft")
    plot_output_fit(
        out_v, y_v, loss_name="mse",
        title="Pre-prune output fit (logit space, 7 params flattened)",
        save_path=str(out_dir / "output_fit.png"),
    )

    raw_net = _unwrap(net)
    for i, stage in enumerate(raw_net.core.stages):
        plot_stage_graph(
            stage,
            save_path=str(out_dir / f"stage{i + 1}_graph_trained.png"),
            title=f"ctle_grid — Stage {i + 1} (trained, {stage.num_edges()} edges)",
        )
        plot_cell_selection(
            stage.logits, cell_order=CELL_ORDER,
            save_path=str(out_dir / f"stage{i + 1}_cell_selection_trained.png"),
            title=f"ctle_grid — Stage {i + 1} cell selection (trained)",
        )

    torch.save(net.state_dict(), out_dir / "model.pt")
    print(f"[train_ctle] saved pre-prune model to {out_dir / 'model.pt'}")

    # ---- pruning + retrain (Phase C) ----
    if args.prune:
        edge_thresh = float(SCHEDULE_FOUR_PHASE.get("prune_edge_threshold", 0.05))
        node_thresh = float(SCHEDULE_FOUR_PHASE.get("prune_node_threshold", 0.05))
        prune_nodes_by_gate = bool(SCHEDULE_FOUR_PHASE.get("prune_nodes_by_gate", False))
        pre_edges = sum(s.num_edges() for s in raw_net.core.stages)
        pre_nodes = sum(s.num_nodes for s in raw_net.core.stages)
        print(f"[prune] pre-prune: {pre_edges} edges, {pre_nodes} nodes "
              f"(edge_thresh={edge_thresh}, prune_nodes_by_gate={prune_nodes_by_gate})")

        pruned_core, stage_remaps = prune_network(
            raw_net.core,
            edge_threshold=edge_thresh,
            node_threshold=node_thresh,
            transfer_params=True,
            write_idx=list(raw_net.write_idx) if raw_net.write_idx is not None else None,
            read_idx=list(raw_net.read_idx) if raw_net.read_idx is not None else None,
            prune_nodes_by_gate=prune_nodes_by_gate,
        )
        post_edges = sum(s.num_edges() for s in pruned_core.stages)
        post_nodes = sum(s.num_nodes for s in pruned_core.stages)
        print(f"[prune] post-prune: {post_edges} edges, {post_nodes} nodes "
              f"(removed {pre_edges - post_edges} edges, {pre_nodes - post_nodes} nodes)")

        # Rebuild I/O mappers with weight transfer
        stage0_remap = stage_remaps[0]
        last_remap = stage_remaps[-1]
        pruned_first_n = pruned_core.stages[0].num_nodes
        pruned_last_n = pruned_core.stages[-1].num_nodes
        in_dim = 4
        out_dim = len(PARAM_COLS)
        raw_write_idx = list(raw_net.write_idx) if raw_net.write_idx is not None else None
        raw_read_idx = list(raw_net.read_idx) if raw_net.read_idx is not None else None

        input_mapper_pruned, _ = _transfer_input_mapper(
            raw_net.input_mapper, raw_write_idx, stage0_remap, pruned_first_n, in_dim,
        )
        output_mapper_pruned, pruned_read_idx = _transfer_output_mapper(
            raw_net.output_mapper, raw_read_idx, last_remap, pruned_last_n, out_dim,
        )

        pruned_net = KirchhoffNetWithIO(
            input_mapper_pruned, pruned_core, output_mapper_pruned,
            hid_count=pruned_first_n, proj_count=0,
            final_hid_count=pruned_last_n, final_proj_count=0,
            write_idx=None,  # fan_out mode uses fan_out_map
            read_idx=pruned_read_idx,
        )
        pruned_net.to(device)
        n_pruned = sum(p.numel() for p in pruned_net.parameters())
        print(f"[prune] pruned network: {n_pruned} params "
              f"({100.0 * n_pruned / max(1, n_kirchhoff_params):.1f}% of pre-prune)")

        # ---- Phase C retrain ----
        retrain_optimizer = make_optimizer(
            pruned_net, lr=retrain_lr,
            stage_lr_scale=1.0, mapper_lr_scale=1.0,
        )
        retrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            retrain_optimizer, T_max=max(1, c_total),
            eta_min=float(OPTIM["scheduler_eta_min"]),
        )
        retrain_history: list[float] = []
        retrain_val_history: list[float] = []
        retrain_val_argmax: list[float] = []
        best_val_pruned = float("inf")
        best_epoch_pruned = -1
        best_state_pruned: dict | None = None
        ewop_p = 0
        for repoch in range(c_total):
            pruned_net.train()
            global_epoch = b2_end + repoch
            tau_r = four_phase_tau(global_epoch, epochs)
            eff_c_lambdas = four_phase_lambdas(global_epoch, epochs, preset["lambdas"])
            cell_mode_c = resolve_cell_mode(args.cell_mode, "C")

            tot = 0.0
            nb = 0
            for u_b, tgt_b in train_loader:
                retrain_optimizer.zero_grad()
                u_b = u_b.to(device)
                tgt_b = tgt_b.to(device)
                loss_task, loss_structural, _ = compute_loss(
                    pruned_net, u_b, tgt_b, ctx=None, task_fn=task_fn,
                    lambdas=eff_c_lambdas, tau=tau_r, return_parts=True,
                    amp=False, reg_scale=1.0, cell_mode=cell_mode_c,
                )
                loss_task.backward(retain_graph=True)
                loss_structural.backward()
                torch.nn.utils.clip_grad_norm_(pruned_net.parameters(), max_norm=grad_clip)
                retrain_optimizer.step()
                tot += float((loss_task + loss_structural).item())
                nb += 1
            avg = tot / max(1, nb)
            retrain_history.append(avg)

            if grad_log_path is not None and (repoch % args.grad_log_every == 0 or repoch == c_total - 1):
                log_gradient_norms(grad_log_path, global_epoch, pruned_net,
                                   retrain=True, optimizer=retrain_optimizer)

            retrain_scheduler.step()

            if repoch % args.validate_every == 0 or repoch == c_total - 1:
                val_r = validate(pruned_net, val_loader, task_fn, device)
                retrain_val_history.append(val_r)
                val_arg_r = validate_argmax(pruned_net, val_loader, task_fn, lambda b: None, device)
                retrain_val_argmax.append(val_arg_r)

                if repoch % args.validate_every == 0:
                    c_metrics = compute_solidification_metrics(pruned_net, tau=tau_r)
                    _log_solidification(solid_log_path, global_epoch, c_metrics)

                # Phase C: deployable is argmax; use it for checkpoint selection.
                sel_metric = float(val_arg_r)
                if sel_metric < best_val_pruned - args.min_delta:
                    best_val_pruned = sel_metric
                    best_epoch_pruned = repoch
                    ewop_p = 0
                    best_state_pruned = {k: v.detach().clone() for k, v in pruned_net.state_dict().items()}
                else:
                    ewop_p += args.validate_every
                    if args.early_stop and ewop_p >= args.patience:
                        print(f"[prune] retrain early stop at epoch {repoch} "
                              f"(best val_argmax={best_val_pruned:.4f})")
                        break
            else:
                retrain_val_history.append(retrain_val_history[-1] if retrain_val_history else avg)
                if retrain_val_argmax:
                    retrain_val_argmax.append(retrain_val_argmax[-1])

            print(
                f"  phase-C epoch {repoch:4d}  train={avg:.4f}  "
                f"val={retrain_val_history[-1]:.4f}  val_argmax={retrain_val_argmax[-1]:.4f}  "
                f"tau={tau_r:.3f}"
            )

        if best_state_pruned is not None:
            pruned_net.load_state_dict(best_state_pruned)
            print(f"[prune] restored best pruned state "
                  f"(epoch {best_epoch_pruned}, val_argmax={best_val_pruned:.4f})")

        # Save pruned artifacts
        torch.save(pruned_net.state_dict(), out_dir / "model_pruned.pt")
        print(f"[prune] saved pruned model to {out_dir / 'model_pruned.pt'}")

        for i, stage in enumerate(pruned_core.stages):
            plot_stage_graph(
                stage,
                save_path=str(out_dir / f"stage{i + 1}_graph_pruned.png"),
                title=f"ctle_grid — Stage {i + 1} (pruned, {stage.num_edges()} edges, {stage.num_nodes} nodes)",
            )
            plot_cell_selection(
                stage.logits, cell_order=CELL_ORDER,
                save_path=str(out_dir / f"stage{i + 1}_cell_selection_pruned.png"),
                title=f"ctle_grid — Stage {i + 1} cell selection (pruned)",
            )

        with torch.no_grad():
            out_pruned, _ = pruned_net(u_v, ctx=None, store_trajectory=False, cell_mode="soft")
        plot_output_fit(
            out_pruned, y_v, loss_name="mse",
            title="Pruned output fit (logit space, 7 params flattened)",
            save_path=str(out_dir / "output_fit_pruned.png"),
        )

        # Pruning summary
        with (out_dir / "prune_summary.txt").open("w") as f:
            f.write(f"edge_threshold: {edge_thresh}\n")
            f.write(f"node_threshold: {node_thresh}\n")
            f.write(f"prune_nodes_by_gate: {prune_nodes_by_gate}\n")
            f.write(f"pre_edges: {pre_edges}\n")
            f.write(f"post_edges: {post_edges}\n")
            f.write(f"pre_nodes: {pre_nodes}\n")
            f.write(f"post_nodes: {post_nodes}\n")
            f.write(f"edges_removed: {pre_edges - post_edges}\n")
            f.write(f"nodes_removed: {pre_nodes - post_nodes}\n")
            f.write(f"pre_params: {n_kirchhoff_params}\n")
            f.write(f"post_params: {n_pruned}\n")
            f.write(f"retrain_epochs: {c_total}\n")
            f.write(f"best_val_pruned: {best_val_pruned:.6f}\n")
            f.write(f"best_epoch_pruned: {best_epoch_pruned}\n")

    # ---- physical-domain evaluation ----
    print("[train_ctle] physical-domain evaluation (relative error per param)...")
    eval_net = pruned_net if (args.prune and best_state_pruned is not None) else net
    eval_net.eval()
    eval_n = min(2000, len(val_loader.dataset))
    eval_specs = sample_specs(eval_n, seed=123).to(device)
    with torch.no_grad():
        eval_logits_mlp = mlp(eval_specs)
        eval_logits_kirchhoff, _ = eval_net(eval_specs, ctx=None, store_trajectory=False, cell_mode="soft")
    params_mlp = params_from_logits(eval_logits_mlp)
    params_kirchhoff = params_from_logits(eval_logits_kirchhoff)

    eval_path = out_dir / "physical_eval.txt"
    with eval_path.open("w") as f:
        f.write(f"physical_param_relative_error (eval_n={eval_n})\n")
        f.write(f"{'param':<10} {'median':>10} {'p90':>10} {'max':>10}\n")
        print(f"  {'param':<10} {'median':>10} {'p90':>10} {'max':>10}")
        for name in PARAM_COLS:
            p_mlp = params_mlp[name].cpu().numpy()
            p_kh = params_kirchhoff[name].cpu().numpy()
            denom = np.maximum(np.abs(p_mlp), 1e-12)
            rel_err = np.abs(p_kh - p_mlp) / denom
            med = float(np.median(rel_err))
            p90 = float(np.percentile(rel_err, 90))
            mx = float(np.max(rel_err))
            f.write(f"{name:<10} {med:>10.4f} {p90:>10.4f} {mx:>10.4f}\n")
            print(f"  {name:<10} {med:>10.4f} {p90:>10.4f} {mx:>10.4f}")

    print(f"[train_ctle] done — artifacts in {out_dir}")


if __name__ == "__main__":
    main()
