"""Train a 3-stage grid KirchhoffNet as CTLE inverse design student via 4-phase
knowledge distillation from a pre-trained RegimeAwareMoE teacher.

The RegimeAwareMoE teacher is loaded from a state-dict checkpoint (e.g.
``dagger_student_moe.pt``) and used to label synthetic spec samples for
training the KirchhoffNet. The companion scaler file ``flow_scaler_C.pkl``
is auto-discovered from the same directory as the checkpoint; override
with ``--scaler-path`` if stored elsewhere. The 4-phase schedule (A: free
fit, B1: cell commitment with KD, B2: edge pruning with KD, C: retrain
compact network) is reused from ``train.py`` with the MoE wrapped as a
KirchhoffNet-style teacher (compatible with ``compute_loss``'s KD path).

Output is unbounded logits matching ``RegimeAwareMoE.forward()``. Physical
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

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DEGREE_BUDGET,
    LAMBDAS,
    OPTIM,
    PRUNE,
    SCHEDULE_FOUR_PHASE,
    SCHEDULE_THREE_PHASE,
    SOLVER,
    TAU,
)
from cell_library import IdealizedCellLibrary, make_cell_library, SimpleEdgeLibrary
from io_mapper import FanOutInputMapper, SparseInputMapper
from kirchhoff_net import KirchhoffNetWithIO
from topology import build_net_from_config, prune_network
from train import (
    budget_frac_for_epoch,
    budget_temperature_for_epoch,
    compute_loss,
    compute_solidification_metrics,
    four_phase_boundaries,
    four_phase_kd_active,
    four_phase_lambdas,
    four_phase_tau,
    make_optimizer,
    phase_boundaries,
    phase_for_epoch,
    phase_for_epoch_four,
    prune_readiness_check,
    reg_schedule,
    tau_for_epoch,
    three_phase_lambdas,
    three_phase_tau,
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

# Spec input column order (must match RegimeAwareMoE.forward).
SPEC_INPUT_COLS = ["power", "jitter", "height", "width"]

# Empirical spec sampling ranges (from ctle_ml_dataset.csv, 83k rows).
# power is log-uniform; others are uniform.
SPEC_RANGES = {
    "power": (0.0012, 0.012),
    "jitter": (1.57, 100.0),
    "height": (0.0, 88.4),
    "width": (0.0, 98.5),
}

# Output parameter column order (matches RegimeAwareMoE's PARAM_LOG_BOUNDS).
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
# CTLE preset factory (grid + cluster families)
# =============================================================================

def make_ctle_preset(
    family: str = "grid",
    grid_size: int = 5,
    num_hidden: int | None = None,
    num_stages: int = 3,
    num_proj: int | None = None,
    write_mode: str | None = None,
    bidirectional: bool = False,
    cluster_edge_prob: float = 1.0,
    cluster_seed: int | None = None,
    q75_input: bool = False,
    edge_repeats: int = 2,
    nodes_per_target: int = 0,
    readout_offset: int = 0,
) -> dict:
    """Build a 4-spec → 7-logit CTLE KirchhoffNet preset for a given family.

    ``family='grid'`` (default, backward compatible):
        Each stage has ``grid_size**2`` hidden nodes laid out on a 2D grid
        with 3x3 convolution-style neighborhood edges, plus ``num_proj``
        projection nodes and 4 input nodes. Write mapping is fan-out from
        4 inputs to 4 grid corners; read mapping gathers the center column
        + all projection nodes (grid_size + num_proj features).

    ``family='cluster'``:
        Each stage has ``num_hidden`` hidden nodes wired as a fully (or
        partially) connected Erdos-Renyi graph via
        ``cluster_graph(num_hidden, edge_prob=cluster_edge_prob)``. Since
        every node connects to every other, ``num_proj=0`` (no structural
        benefit for projection nodes) and ``read_idx=list(range(num_hidden))``
        reads every hidden node. Write mapping defaults to ``dense``
        (all-to-all ``InputMapper``).

    When ``q75_input=True``, ``family='cluster'``, and ``num_inputs`` becomes
    8 (teacher's scale_input(): log10 + StandardScaler/Q75 expansion).
    Caller must apply the same scaling to training inputs.

    When ``nodes_per_target > 0``, group the 7 regression targets into per-target
    readout windows of ``nodes_per_target`` consecutive state nodes each. The
    network auto-sizes so ``grid_size**2`` (grid) or ``num_hidden`` (cluster)
    accommodates ``nodes_per_target * 7 + readout_offset`` state nodes. In
    this mode ``num_proj`` is forced to 0 (no projection nodes; heads read
    directly from hidden state) and ``--prune`` must be disabled at the CLI.

    Args:
        family: 'grid' or 'cluster'.
        grid_size: Square grid side length (grid family).
        num_hidden: Hidden node count (cluster family). Required when family='cluster'.
        num_stages: Number of ODE stages.
        num_proj: Projection node count (grid family default 7; cluster always 0).
        write_mode: 'fan_out' | 'dense' | 'one_to_one' | None (default per family).
        bidirectional: Emit two directed edges per node pair.
        cluster_edge_prob: Edge probability for cluster family (default 1.0 = fully connected).
        cluster_seed: RNG seed for cluster edge sampling.
        q75_input: When True and family='cluster', set num_inputs=8 (Q75-scaled features).
        edge_repeats: Number of parallel edges per hidden node pair (default 2,
            range 1-8). Composes multiplicatively with ``bidirectional``. Each
            repeated edge gets independent cell-type logits, gate, and
            multiplier. I/O and projection edges are NOT repeated.
        nodes_per_target: If > 0, enable grouped per-target readout with this
            many state nodes per target. Auto-sizes the network.
        readout_offset: Starting state index for the first target's window
            (only used when ``nodes_per_target > 0``).
    """
    if edge_repeats < 1 or edge_repeats > 8:
        raise ValueError(f"edge_repeats must be in [1, 8], got {edge_repeats}")
    if nodes_per_target < 0:
        raise ValueError(f"nodes_per_target must be >= 0, got {nodes_per_target}")
    if readout_offset < 0:
        raise ValueError(f"readout_offset must be >= 0, got {readout_offset}")
    n_stages = max(1, num_stages)

    grouped = nodes_per_target > 0
    required_state_dim = (
        readout_offset + nodes_per_target * len(PARAM_COLS) if grouped else 0
    )

    if family == "grid":
        if grouped:
            # Auto-size: find smallest grid_size whose square >= required.
            import math as _math
            grid_size = max(grid_size, _math.ceil(_math.sqrt(required_state_dim)))
            num_proj = 0
            print(f"[make_ctle_preset] grouped readout: nodes_per_target={nodes_per_target}, "
                  f"offset={readout_offset} -> grid_size auto-sized to {grid_size} "
                  f"({grid_size ** 2} hidden >= {required_state_dim} required), "
                  f"num_proj forced to 0")
        if num_proj is None:
            num_proj = 7
        n_hidden = grid_size * grid_size
        if num_hidden is not None and num_hidden != n_hidden and not grouped:
            print(f"[make_ctle_preset] note: grid mode ignores num_hidden={num_hidden}; "
                  f"using grid_size**2={n_hidden}")
        if q75_input:
            print(f"[make_ctle_preset] note: --q75-input is ignored for grid family "
                  f"(only supported with cluster). num_inputs stays at 4.")
        _stage_cfg = {
            "num_inputs": 4,
            "num_hidden": n_hidden,
            "num_proj": num_proj,
            "num_outputs": 0,
            "hidden_family": "grid",
            "hidden_kwargs": {"height": grid_size, "width": grid_size,
                              "kernel_size": 3, "bidirectional": bidirectional},
            "edge_repeats": edge_repeats,
            "input_pattern": "all_to_all",
            "output_pattern": "all_to_all",
            "proj_pattern": "all_to_all",
            "t_span": SOLVER["t_span"] / n_stages,
            "num_steps": round(SOLVER["num_steps"] / n_stages),
        }
        if grouped:
            # GroupedOutputMapper reads directly from hidden state; no projection
            # nodes, no center-column selection.
            eff_write = write_mode if write_mode is not None else "fan_out"
            top_rows = [0, 1]
            bot_rows = [grid_size - 2, grid_size - 1]
            fan_out = {
                0: [r * grid_size + 0 for r in top_rows],
                1: [r * grid_size + (grid_size - 1) for r in top_rows],
                2: [r * grid_size + 0 for r in bot_rows],
                3: [r * grid_size + (grid_size - 1) for r in bot_rows],
            }
            read_idx = None  # GroupedOutputMapper handles its own windowing.
        else:
            eff_write = write_mode if write_mode is not None else "fan_out"
            top_rows = [0, 1]
            bot_rows = [grid_size - 2, grid_size - 1]
            fan_out = {
                0: [r * grid_size + 0 for r in top_rows],
                1: [r * grid_size + (grid_size - 1) for r in top_rows],
                2: [r * grid_size + 0 for r in bot_rows],
                3: [r * grid_size + (grid_size - 1) for r in bot_rows],
            }
            center_col = grid_size // 2
            center_nodes = [r * grid_size + center_col for r in range(grid_size)]
            read_idx = center_nodes + list(range(n_hidden, n_hidden + num_proj))

    elif family == "cluster":
        if grouped:
            if num_hidden is None:
                num_hidden = 0  # will be overwritten below
            num_proj = 0
            num_hidden = required_state_dim
            print(f"[make_ctle_preset] grouped readout: nodes_per_target={nodes_per_target}, "
                  f"offset={readout_offset} -> num_hidden auto-sized to {num_hidden}")
        else:
            if num_hidden is None:
                raise ValueError(
                    "make_ctle_preset(family='cluster') requires num_hidden to be set. "
                    "Pass --num-hidden N on the CLI or num_hidden=N to the factory."
                )
            if num_hidden < 2:
                raise ValueError(f"num_hidden must be >= 2 for cluster family (got {num_hidden})")
            if num_proj is not None and num_proj != 0:
                print(f"[make_ctle_preset] note: cluster mode forces num_proj=0 "
                      f"(was {num_proj}); all hidden nodes are already fully connected.")
        n_hidden = int(num_hidden)
        num_proj = 0
        n_inputs = 8 if q75_input else 4
        if cluster_seed is None:
            # Default to 0 if no seed was provided (caller didn't derive one).
            cluster_seed = 0
        _stage_cfg = {
            "num_inputs": n_inputs,
            "num_hidden": n_hidden,
            "num_proj": num_proj,
            "num_outputs": 0,
            "hidden_family": "cluster",
            "hidden_kwargs": {"edge_prob": cluster_edge_prob,
                              "seed": cluster_seed,
                              "bidirectional": bidirectional},
            "edge_repeats": edge_repeats,
            "input_pattern": "all_to_all",
            "output_pattern": "all_to_all",
            "proj_pattern": "all_to_all",
            "t_span": SOLVER["t_span"] / n_stages,
            "num_steps": round(SOLVER["num_steps"] / n_stages),
        }
        if write_mode == "fan_out":
            print(f"[make_ctle_preset] note: cluster mode has no spatial grid corners; "
                  f"falling back from 'fan_out' to 'dense' write mode.")
            eff_write = "dense"
        else:
            eff_write = write_mode if write_mode is not None else "dense"
        # Grouped: read_idx=None (mapper handles windowing).
        # Non-grouped: read every hidden node.
        read_idx = None if grouped else list(range(n_hidden))
        fan_out = None

    else:
        raise ValueError(f"Unknown family: {family!r} (expected 'grid' or 'cluster')")

    preset: dict[str, Any] = {
        "stages": [_stage_cfg] * n_stages,
        "use_robust_input": False,
        "loss": "mse",
        "out_dim": len(PARAM_COLS),
        "write_mode": eff_write,
        "read_idx": read_idx,
        "schedule": "four_phase",
        "tau_anneal": True,
    }
    if eff_write == "fan_out" and fan_out is not None:
        preset["write_fan_out"] = fan_out
    if grouped:
        preset["grouped_readout"] = {
            "nodes_per_target": nodes_per_target,
            "offset": readout_offset,
        }
    return preset


def make_ctle_grid_preset(
    grid_size: int = 5,
    num_stages: int = 3,
    num_proj: int = 7,
    write_mode: str | None = None,
    bidirectional: bool = False,
    q75_input: bool = False,
    edge_repeats: int = 2,
    nodes_per_target: int = 0,
    readout_offset: int = 0,
) -> dict:
    """Backward-compatible thin wrapper for the grid CTLE preset.

    Equivalent to ``make_ctle_preset(family='grid', grid_size=grid_size, ...)``.
    Note: q75_input has no effect with grid family (only supported with cluster).
    """
    return make_ctle_preset(
        family="grid",
        grid_size=grid_size,
        num_stages=num_stages,
        num_proj=num_proj,
        write_mode=write_mode,
        bidirectional=bidirectional,
        q75_input=q75_input,
        edge_repeats=edge_repeats,
        nodes_per_target=nodes_per_target,
        readout_offset=readout_offset,
    )


# =============================================================================
# RegimeAwareMoE (teacher) and adapter
# =============================================================================

class RegimeAwareMoE(nn.Module):
    """Mixture-of-Experts MLP teacher that maps 4 raw specs to 7 unbounded logits.

    Mirrors the architecture in ``generative-distillation-improved-dagger-nuance.py``
    so that the pre-trained state-dict (``dagger_student_moe.pt``) loads cleanly.
    Performs internal Q75 + StandardScaler input preprocessing via
    ``scale_input()``; raw spec tensors ``[N, 4]`` can be passed directly to
    ``forward()``. ``log_lo`` / ``log_hi`` are non-trainable buffers (they
    store PARAM_LOG_BOUNDS in log10 space).
    """

    def __init__(
        self,
        trunk_width: int = 160,
        trunk_layers: int = 3,
        num_experts: int = 3,
        input_dim: int = 4,
        output_dim: int = 7,
        param_log_bounds: dict[str, tuple[float, float]] | None = None,
        activation: type[nn.Module] = nn.SiLU,
        use_log_features: bool = True,
        scaler_p_mean: float = 0.0,
        scaler_p_scale: float = 1.0,
        eye_scale_h: float = 1.0,
        eye_scale_w: float = 1.0,
        eye_scale_j: float = 1.0,
    ) -> None:
        super().__init__()
        self.trunk_width = int(trunk_width)
        self.trunk_layers = int(trunk_layers)
        self.num_experts = int(num_experts)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.activation = activation
        self.use_log_features = bool(use_log_features)
        if param_log_bounds is None:
            param_log_bounds = PARAM_LOG_BOUNDS

        trunk_input_dim = input_dim * 2 if use_log_features else input_dim
        trunk_dims = [trunk_input_dim] + [trunk_width] * trunk_layers

        trunk_layers_list: list[nn.Module] = []
        for i in range(trunk_layers):
            trunk_layers_list.append(nn.Linear(trunk_dims[i], trunk_dims[i + 1]))
            trunk_layers_list.append(activation())
        self.trunk = nn.Sequential(*trunk_layers_list)

        self.gate = nn.Linear(trunk_input_dim, num_experts, bias=False)
        self.regime_classifier = nn.Linear(trunk_input_dim, num_experts, bias=False)

        self.experts = nn.ModuleList([
            nn.Linear(trunk_width, output_dim) for _ in range(num_experts)
        ])

        self.log_lo = nn.Parameter(torch.zeros(output_dim), requires_grad=False)
        self.log_hi = nn.Parameter(torch.zeros(output_dim), requires_grad=False)
        for i, (lo, hi) in enumerate(param_log_bounds.values()):
            self.log_lo.data[i] = lo
            self.log_hi.data[i] = hi

        self.scaler_p_scale = float(scaler_p_scale)
        self.scaler_p_mean = float(scaler_p_mean)
        self._eye_scale_j = float(eye_scale_j)
        self._eye_scale_h = float(eye_scale_h)
        self._eye_scale_w = float(eye_scale_w)

    def scale_input(self, x: torch.Tensor) -> torch.Tensor:
        power_log = torch.log10(x[..., 0].clamp(min=1e-12))
        power_scaled = (power_log - self.scaler_p_mean) / self.scaler_p_scale
        jitter_scaled = torch.log10(x[..., 1].clamp(min=1e-12)) / self._eye_scale_j
        height_scaled = torch.log10(x[..., 2].clamp(min=1e-12)) / self._eye_scale_h
        width_scaled = torch.log10(x[..., 3].clamp(min=1e-12)) / self._eye_scale_w
        linear_scaled = torch.stack([power_scaled, jitter_scaled, height_scaled, width_scaled], dim=-1)
        if self.use_log_features:
            log_scaled = torch.stack([power_log, torch.log10(x[..., 1].clamp(min=1e-12)),
                                      torch.log10(x[..., 2].clamp(min=1e-12)),
                                      torch.log10(x[..., 3].clamp(min=1e-12))], dim=-1)
            return torch.cat([linear_scaled, log_scaled], dim=-1)
        return linear_scaled

    def forward(self, x: torch.Tensor, return_regime_loss: bool = False):
        x_s = self.scale_input(x)
        h = self.trunk(x_s)
        gate_weights = F.softmax(self.gate(x_s), dim=-1)
        expert_outputs = torch.stack([expert(h) for expert in self.experts], dim=-1)
        logits = (expert_outputs * gate_weights.unsqueeze(-2)).sum(dim=-1)

        if return_regime_loss:
            return logits, torch.tensor(0.0, device=logits.device)
        return logits


class MLPTeacherWrapper(nn.Module):
    """Adapter that exposes a RegimeAwareMoE with KirchhoffNet's forward signature.

    ``compute_loss`` calls ``teacher(u, ctx=..., tau=..., store_trajectory=...,
    cell_mode=...)`` and unpacks ``(y_teacher, _)``. This wrapper ignores the
    KirchhoffNet-specific kwargs and returns ``(mlp(u), None)``.
    """

    def __init__(self, mlp: RegimeAwareMoE) -> None:
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


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    Seeding covers CPU and CUDA. cuDNN auto-tuner (``cudnn.benchmark``) and
    deterministic mode (``cudnn.deterministic``) are intentionally left at
    their defaults: this is "partial" determinism (bit-identical weights and
    data sampling, but slight non-determinism from cuDNN's algorithm choice
    under varying batch sizes). For full bit-exact reproducibility across
    runs, set ``torch.backends.cudnn.deterministic = True`` manually.

    Args:
        seed: Integer seed value.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seeded_worker_init_fn(worker_id: int) -> None:
    """DataLoader worker_init_fn that seeds each worker's RNG deterministically.

    Uses ``base_seed + worker_id`` so multi-worker loading is reproducible
    when ``base_seed`` is fixed. ``base_seed`` is set as a module-level
    attribute by ``seed_everything``-style callers via the closure below.
    """
    base = getattr(_seeded_worker_init_fn, "_base_seed", 0)
    import random
    seed = (base + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_ctle_dataset(
    n_train: int,
    n_val: int,
    mlp: RegimeAwareMoE,
    device: torch.device,
    batch_size: int,
    seed: int = 42,
    normalize: bool = True,
    q75_input: bool = False,
) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """Build train/val DataLoaders of (specs, mlp_logits) pairs.

    Generates ``n_train + n_val`` spec samples, labels them via the MLP,
    then takes the first ``n_train`` as training and the remaining
    ``n_val`` as validation. Both subsets are deterministic via ``seed``.

    When ``normalize`` is True (default), the per-dim mean and std of the
    teacher logits are computed on the **training set** and used to
    normalize both train and val targets to zero mean / unit variance.
    The student learns to predict normalized logits; callers should
    denormalize at eval time via ``logits = output * std + mean``.

    When ``q75_input`` is True, the input specs are preprocessed through
    the teacher's ``scale_input()`` (log10 + StandardScaler/Q75 expansion,
    4-dim → 8-dim). The student then sees well-conditioned 8-feature inputs
    and its InputMapper becomes ``Linear(8, num_hidden)``.

    Returns:
      (train_loader, val_loader, target_mean, target_std).
      When ``normalize`` is False, ``target_mean`` is zeros and
      ``target_std`` is ones (identity transform).
    """
    mlp.eval()
    mlp.to(device)

    n_total = n_train + n_val
    specs_raw = sample_specs(n_total, seed=seed).to(device)
    if q75_input:
        with torch.no_grad():
            specs = mlp.scale_input(specs_raw).detach()
    else:
        specs = specs_raw

    with torch.no_grad():
        all_logits: list[torch.Tensor] = []
        for i in range(0, n_total, batch_size):
            batch = specs_raw[i:i + batch_size]
            all_logits.append(mlp(batch).detach().to("cpu"))
        logits = torch.cat(all_logits, dim=0)

    train_logits_raw = logits[:n_train]
    if normalize:
        target_mean = train_logits_raw.mean(dim=0)
        target_std = train_logits_raw.std(dim=0).clamp(min=1e-6)
    else:
        target_mean = torch.zeros(train_logits_raw.shape[1], dtype=train_logits_raw.dtype)
        target_std = torch.ones(train_logits_raw.shape[1], dtype=train_logits_raw.dtype)

    train_targets = (train_logits_raw - target_mean) / target_std
    val_targets = (logits[n_train:] - target_mean) / target_std

    train_ds = TensorDataset(specs[:n_train].to("cpu"), train_targets)
    val_ds = TensorDataset(specs[n_train:].to("cpu"), val_targets)

    # Seeded DataLoader for reproducible shuffling across runs.
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
        generator=g, worker_init_fn=_seeded_worker_init_fn,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, target_mean, target_std


# =============================================================================
# Validation helpers
# =============================================================================

def compute_per_dim_stats(
    net: KirchhoffNetWithIO,
    loader: DataLoader,
    device: torch.device,
    *,
    tau: float | None = None,
    cell_mode: str = "soft",
) -> dict[str, np.ndarray | float]:
    """Compute per-dimension MSE, R², and target variance over ``loader``.

    Returns a dict with keys:
      agg_mse       — scalar, mean MSE across all dims and samples
      per_dim_mse   — np.ndarray of shape [D], per-dim MSE in normalized space
      per_dim_r2    — np.ndarray of shape [D], per-dim R² vs val-set variance
      per_dim_var   — np.ndarray of shape [D], val-set target variance per dim

    All metrics are computed in the (possibly normalized) target space. R² uses
    the validation set's own target variance as the denominator so it measures
    explained variance on held-out data.
    """
    net.eval()
    sumsq = None
    target_sumsq = None
    target_sum = None
    n_samples = 0
    with torch.no_grad():
        for u, target in loader:
            u = u.to(device)
            target = target.to(device)
            out, _ = net(u, ctx=None, store_trajectory=False,
                         cell_mode=cell_mode, tau=tau)
            sq = (out - target) ** 2
            if sumsq is None:
                sumsq = sq.sum(dim=0).detach().to("cpu").double()
                target_sumsq = (target ** 2).sum(dim=0).detach().to("cpu").double()
                target_sum = target.sum(dim=0).detach().to("cpu").double()
            else:
                sumsq += sq.sum(dim=0).detach().to("cpu").double()
                target_sumsq += (target ** 2).sum(dim=0).detach().to("cpu").double()
                target_sum += target.sum(dim=0).detach().to("cpu").double()
            n_samples += u.size(0)
    net.train()
    if sumsq is None or n_samples == 0:
        empty = np.zeros(len(PARAM_COLS), dtype=np.float64)
        return {
            "agg_mse": float("nan"),
            "per_dim_mse": empty,
            "per_dim_r2": empty,
            "per_dim_var": empty,
        }
    per_dim_mse = (sumsq / n_samples).numpy()
    target_mean = (target_sum / n_samples).numpy()
    per_dim_var = (target_sumsq / n_samples).numpy() - target_mean ** 2
    per_dim_var = np.maximum(per_dim_var, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_dim_r2 = 1.0 - per_dim_mse / per_dim_var
    agg_mse = float(per_dim_mse.mean())
    return {
        "agg_mse": agg_mse,
        "per_dim_mse": per_dim_mse,
        "per_dim_r2": per_dim_r2,
        "per_dim_var": per_dim_var,
    }


def _per_dim_stats_header() -> list[str]:
    cols = ["epoch", "phase", "agg_mse"]
    for name in PARAM_COLS:
        cols.append(f"mse_{name}")
    for name in PARAM_COLS:
        cols.append(f"r2_{name}")
    for name in PARAM_COLS:
        cols.append(f"var_{name}")
    return cols


def _log_per_dim_stats(log_path: Path, epoch: int, phase: str,
                       stats: dict) -> None:
    """Append one row of per-dimension stats to ``log_path`` (TSV)."""
    cols = _per_dim_stats_header()
    if not log_path.exists() or log_path.stat().st_size == 0:
        with log_path.open("w") as f:
            f.write("\t".join(cols) + "\n")
    row = [str(epoch), phase, f"{stats['agg_mse']:.6e}"]
    for v in stats["per_dim_mse"]:
        row.append(f"{v:.6e}")
    for v in stats["per_dim_r2"]:
        row.append(f"{v:.6e}")
    for v in stats["per_dim_var"]:
        row.append(f"{v:.6e}")
    with log_path.open("a") as f:
        f.write("\t".join(row) + "\n")


def _worst_dim_label(stats: dict) -> tuple[str, float]:
    """Return (param_name, mse_value) for the worst-fit dimension this epoch."""
    idx = int(np.argmax(stats["per_dim_mse"]))
    return PARAM_COLS[idx], float(stats["per_dim_mse"][idx])


def _log_drive_diagnostics(
    net: torch.nn.Module, val_loader, device: torch.device,
    log_path: Path, epoch: int,
) -> None:
    """Log persistent drive metrics per stage to ``log_path``."""
    if not hasattr(net, 'enable_drive') or not net.enable_drive:
        return
    if not log_path.exists():
        with log_path.open("w") as f:
            f.write("epoch\tstage\tmean_drive_error\tmean_abs_i_drive\tfrac_near_rail\n")
    net.eval()
    with torch.no_grad():
        val_batch = next(iter(val_loader))
        u, _ = val_batch
        u = u[:4].to(device)  # small batch
        hidden_drives = [dm(u) for dm in net.drive_mappers]
        full_drives = [net._make_full_drive(hd) for hd in hidden_drives]
        y, trajs = net(u, ctx=None, tau=1.0, store_trajectory=True)
    with log_path.open("a") as f:
        for s_idx, stage in enumerate(net.core.stages):
            if not stage._has_drive:
                continue
            x_drive = full_drives[s_idx]
            traj = trajs[s_idx]  # [B, N, steps+1]
            x_final = traj[:, :, -1]  # [B, N]
            # Mean absolute drive error at final state.
            err = (x_final[:, stage._drive_idx] - x_drive[:, stage._drive_idx]).abs().mean().item()
            # Expected drive current magnitude at final state.
            g_in = F.softplus(stage.raw_drive_g).unsqueeze(0)
            err_i = x_drive[:, stage._drive_idx] - x_final[:, stage._drive_idx]
            i_drive = stage.drive_isat * torch.tanh(g_in * err_i / stage.drive_isat)
            mean_i = i_drive.abs().mean().item()
            # Fraction of driven nodes near rail.
            frac_rail = (x_final[:, stage._drive_idx].abs() > 0.9 * stage.x_max).float().mean().item()
            f.write(f"{epoch}\t{s_idx}\t{err:.6e}\t{mean_i:.6e}\t{frac_rail:.6e}\n")


def _plot_per_dim_diagnostics(history: list[dict], save_path: Path,
                               norm_label: str = "(normalized)",
                               suptitle: str = "Per-dimension diagnostics") -> None:
    """Generate a 2×2 summary plot from a list of per-dim stats dicts.

    ``history`` entries must have keys ``epoch``, ``per_dim_mse``, ``per_dim_r2``,
    ``per_dim_var`` (the last entry's ``per_dim_var`` is used for the variance bar
    chart). Saves to ``save_path``.
    """
    epochs_h = [h["epoch"] for h in history]
    mse_arr = np.stack([h["per_dim_mse"] for h in history], axis=0)
    r2_arr = np.stack([h["per_dim_r2"] for h in history], axis=0)
    var_arr = history[-1]["per_dim_var"]

    fig, axes = _plt.subplots(2, 2, figsize=(12, 8))
    cmap = _plt.get_cmap("tab10")
    for i, name in enumerate(PARAM_COLS):
        axes[0, 0].plot(epochs_h, mse_arr[:, i], label=name, color=cmap(i), marker="o", markersize=2)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel(f"per-dim MSE {norm_label}")
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
    axes[1, 0].set_ylabel(f"target variance {norm_label}")
    axes[1, 0].set_title("Per-dim target variance (val set)")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    final_mse = mse_arr[-1]
    final_r2 = r2_arr[-1]
    axes[1, 1].scatter(final_mse, final_r2,
                       c=[cmap(i) for i in range(len(PARAM_COLS))], s=60)
    for i, name in enumerate(PARAM_COLS):
        axes[1, 1].annotate(name, (final_mse[i], final_r2[i]),
                             fontsize=8, xytext=(4, 4), textcoords="offset points")
    axes[1, 1].set_xlabel("final val MSE")
    axes[1, 1].set_ylabel("final val R²")
    axes[1, 1].set_title("Per-dim fit (final epoch)")
    axes[1, 1].axhline(0.0, color="k", linewidth=0.5, linestyle="--")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=120, bbox_inches="tight")
    _plt.close(fig)


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

    raise TypeError("transfer_output_mapper: ctle preset requires read_idx")


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
    # Per-edge device parameter suffixes: covers SimpleEdgeLibrary.param
    # (I=ReLU/tanh(p0*Vsrc+p1*Vdest+p2)), RealisticTanhLibrary
    # (alpha_raw, bias_raw), and RealisticTanhUpgradeLibrary (alpha_raw,
    # gm_raw, isat_raw, bias_raw). All contribute to the same `device_param`
    # gradient-norm metric per stage.
    device_param_suffixes = ("param", "alpha_raw", "bias_raw", "gm_raw", "isat_raw")
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
            if any(name.endswith(".cell_lib." + s) for s in device_param_suffixes):
                stage_idx = int(name.split(".stages.")[1].split(".")[0])
                stage_sq.setdefault(f"stage{stage_idx}_device_param", 0.0)

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
            if any(name.endswith(".cell_lib." + s) for s in device_param_suffixes):
                stage_idx = int(name.split(".stages.")[1].split(".")[0])
                key = f"stage{stage_idx}_device_param"
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

def resolve_cell_mode(cli_value: str, phase: str, schedule_mode: str = "four_phase") -> str:
    """Resolve the cell selection mode for the current epoch
    (four-phase-redesign/Phase 2b).

    Behavior:
    - ``cli_value == 'soft'`` or ``'ste'``: honor the explicit override.
    - ``cli_value == 'auto'`` (default): use ``'ste'`` for compressed
      phases (B/B1/B2/C under a phased schedule) and ``'soft'`` for the
      free-fit Phase A. Outside of a phased schedule, always ``'soft'``.

    Args:
        cli_value: The CLI value of ``--cell-mode`` ('soft', 'ste', or 'auto').
        phase: Active phase name ('A', 'B', 'B1', 'B2', 'C', or '').
        schedule_mode: 'legacy', 'three_phase', or 'four_phase'.

    Returns:
        Resolved cell_mode: 'soft' or 'ste'.
    """
    if cli_value in ("soft", "ste"):
        return cli_value
    if schedule_mode in ("three_phase", "four_phase") and phase in ("B", "C", "B1", "B2"):
        return "ste"
    return "soft"


# =============================================================================
# Schedule resolver (mirrors train_script._resolve_schedule).
# =============================================================================

def _resolve_schedule(preset: dict, cli_value: str | None) -> str:
    """Resolve the active schedule mode (three-phase-schedule, four-phase-redesign).

    Precedence: explicit CLI flag > preset['schedule'] > 'four_phase' (CTLE default).
    """
    if cli_value is not None:
        return cli_value
    preset_val = preset.get("schedule")
    if preset_val in ("legacy", "three_phase", "four_phase"):
        return preset_val
    return "four_phase"


# =============================================================================
# Main training entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a 3-stage grid KirchhoffNet as CTLE inverse design student via 4-phase KD from a pre-trained RegimeAwareMoE teacher."
    )
    parser.add_argument(
        "--teacher-path", type=Path, required=True,
        help="Path to the RegimeAwareMoE state-dict checkpoint (e.g. dagger_student_moe.pt).",
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
    parser.add_argument(
        "--grid-size", type=int, default=5,
        help="Hidden grid height/width (default: 5, total hidden = grid_size^2 per stage). "
             "Used only when --hidden-family=grid.",
    )
    parser.add_argument(
        "--hidden-family", choices=["grid", "cluster"], default="grid",
        dest="hidden_family",
        help="Hidden graph family (default: grid). 'cluster' uses "
             "Erdos-Renyi cluster_graph with --num-hidden nodes; 'grid' uses "
             "2D grid_graph with --grid-size. Cluster mode reads from all "
             "hidden nodes and uses dense write mode.",
    )
    parser.add_argument(
        "--num-hidden", type=int, default=None, dest="num_hidden",
        help="Number of hidden nodes per stage (cluster family only). "
             "Required when --hidden-family=cluster; ignored for grid.",
    )
    parser.add_argument(
        "--cluster-edge-prob", type=float, default=1.0, dest="cluster_edge_prob",
        help="Erdos-Renyi edge probability for cluster family (default: 1.0 = fully connected).",
    )
    parser.add_argument(
        "--cluster-seed", type=int, default=None, dest="cluster_seed",
        help="RNG seed for cluster_graph edge sampling (default: derived "
             "from --seed if not provided).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, dest="seed",
        help="Global RNG seed for Python, NumPy, PyTorch (CPU+CUDA), and "
             "data generation (default: 42). Use the same value across "
             "runs to reproduce model initialization and training data. "
             "cuDNN auto-tuner remains enabled (partial determinism).",
    )
    parser.add_argument(
        "--num-stages", type=int, default=3,
        help="Number of ODE stages (default: 3).",
    )
    parser.add_argument(
        "--cell-library", type=str, default=None, dest="cell_library",
        choices=["legacy", "v15", "v2", "relu", "tanh", "tanh_realistic", "tanh_realistic_upgrade"],
        help="Cell library: 'legacy' (L,S,P,Z, default), 'v15' (O_weak,O_hard,P0,N0,D1,Z), "
             "'relu' (I=ReLU(p0*Vsrc+p1*Vdest+p2)), 'tanh' (I=tanh(p0*Vsrc+p1*Vdest+p2)), "
             "'tanh_realistic' (I=tanh(A*Vsrc - B*Vdest + C), A,B>0, A+B=1), "
             "'tanh_realistic_upgrade' (I=Isat*tanh(gm*(A*Vsrc - B*Vdest) + C), "
             "bounded gm/Isat per-edge).",
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
        "--normalize-targets", dest="normalize_targets", action="store_true", default=True,
        help="Per-dim zero-mean/unit-var normalize teacher logits before training (default: on).",
    )
    parser.add_argument(
        "--no-normalize-targets", dest="normalize_targets", action="store_false",
        help="Disable per-dim normalization (train on raw teacher logits).",
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
        "--mapper-lr-scale", type=float, default=0.1,
        help="LR multiplier for I/O mapper params (default: 0.1).",
    )
    parser.add_argument(
        "--struct-lr-scale", type=float, default=2.0,
        help="LR multiplier for structural core params (z_logits, logits, "
             "raw_mult). Default 2.0. When != 1.0 uses flat groups and "
             "ignores --stage-lr-scale.",
    )
    parser.add_argument(
        "--dyn-lr-scale", type=float, default=1.0,
        help="LR multiplier for sensitive dynamical params (raw_leak, "
             "raw_drive_g). Default 1.0. When != 1.0 uses flat groups and "
             "ignores --stage-lr-scale.",
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
        "--amp", dest="amp", action="store_true", default=None,
        help="Enable mixed precision (AMP) (default: on when CUDA).",
    )
    parser.add_argument(
        "--no-amp", dest="amp", action="store_false",
        help="Disable mixed precision.",
    )
    parser.add_argument(
        "--amp-dtype", choices=["float16", "bfloat16"], default="float16",
        help="AMP autocast dtype (default: float16; bfloat16 needs Ampere+).",
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
    parser.add_argument(
        "--bidirectional", dest="bidirectional", action="store_true", default=False,
        help="Emit two directed edges per unique node pair in the hidden graph "
             "(i->j AND j->i). Doubles the hidden edge count and gives "
             "asymmetric cells (P/rectifier) true bidirectional capability. "
             "Default: off (single edge per pair).",
    )
    parser.add_argument(
        "--no-bidirectional", dest="bidirectional", action="store_false",
        help="Disable dual edges per node pair (default).",
    )
    parser.add_argument(
        "--edge-repeats", type=int, default=2, choices=range(1, 9),
        help="Number of parallel edges per hidden node pair (1-8, default: 2). "
             "Each repeated edge gets independent cell-type logits, gate, and "
             "multiplier. Composes multiplicatively with --bidirectional. "
             "I/O and projection edges are not repeated. Set to 1 for the "
             "previous single-edge behavior.",
    )
    parser.add_argument(
        "--persistent-drive", dest="persistent_drive", action="store_true", default=False,
        help="Enable persistent bounded drive current in all stages. Each stage "
             "receives a tanh-bounded source current pulling driven hidden nodes "
             "toward an input-derived target pattern. Requires write_mode='fan_out'. "
             "Drive scale decays [1.0, 0.5, 0.25] across stages.",
    )
    parser.add_argument(
        "--q75-input", dest="q75_input", action="store_true", default=False,
        help="Apply teacher's Q75 input scaling (log10 + StandardScaler/Q75 normalization) "
             "to spec inputs at data-generation time. Transforms 4 raw specs into 8 "
             "well-conditioned features, matching the RegimeAwareMoE's scale_input() "
             "preprocessing. Cluster family only (grid fan_out assumes 4 inputs).",
    )
    parser.add_argument(
        "--no-q75-input", dest="q75_input", action="store_false",
        help="Use raw 4-dim spec inputs (default).",
    )
    parser.add_argument(
        "--nodes-per-target", type=int, default=0, dest="nodes_per_target",
        help="If > 0, each of the 7 regression targets reads from this many "
             "consecutive state nodes via an independent Linear head. Auto-sizes "
             "the network (grid: smallest grid_size with grid_size**2 >= "
             "nodes_per_target*7 + readout_offset; cluster: num_hidden = "
             "nodes_per_target*7 + readout_offset). Replaces the monolithic "
             "OutputMapper with a GroupedOutputMapper. 0 = use default "
             "OutputMapper (default). Forces --num_proj=0 and disables --prune.",
    )
    parser.add_argument(
        "--readout-offset", type=int, default=0, dest="readout_offset",
        help="Starting state index for the first target's readout window "
             "(only used with --nodes-per-target > 0). Default: 0.",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=None, dest="weight_decay",
        help=f"AdamW weight decay (default: {OPTIM.get('weight_decay', 0.0)}). "
             f"Pass 1e-4 to match distill_ctle_kirchhoff.py.",
    )
    parser.add_argument(
        "--grad-clip", type=float, default=None, dest="grad_clip",
        help=f"Max gradient norm for clipping (default: {OPTIM.get('grad_clip_norm', 0.0)}). "
             f"Pass 1.0 to match distill_ctle_kirchhoff.py. 0 or negative = no clipping.",
    )
    parser.add_argument(
        "--schedule", choices=["legacy", "three_phase", "four_phase"], default=None,
        help="Training schedule mode (default: from preset['schedule'], "
             "fallback 'four_phase'). 'three_phase' uses the fit-compress-prune "
             "pipeline (Phase A: fit, Phase B: compress via gate penalties, "
             "Phase C: auto-prune + retrain). 'four_phase' splits B into B1 "
             "(cell commitment) and B2 (edge pruning, readiness-gated) and "
             "adds KD-anchored retrain. 'legacy' uses reg_schedule warmup "
             "with no phased lambdas/tau.",
    )
    parser.add_argument(
        "--budget", action="store_true", default=False,
        dest="budget",
        help="Enable degree-budget edge competition. Each destination "
             "(or source) node keeps a fraction of its incoming edges open "
             "via temperature-scaled softmax renormalization of z_logits. "
             "Replaces the L1 edge_gate pressure with explicit competition.",
    )
    parser.add_argument(
        "--budget-frac-start", type=float, default=None,
        dest="budget_frac_start",
        help="Initial budget fraction per group (permissive, 1.0 = no "
             "restriction). Default: 1.0.",
    )
    parser.add_argument(
        "--budget-frac-end", type=float, default=None,
        dest="budget_frac_end",
        help="Final budget fraction per group (restrictive, 0.0 = disables "
             "budget, 0.75 = keep 75% of edges). Default: 0.75.",
    )
    parser.add_argument(
        "--budget-temp-start", type=float, default=None,
        dest="budget_temp_start",
        help="Initial softmax temperature (soft). Default: 1.0.",
    )
    parser.add_argument(
        "--budget-temp-end", type=float, default=None,
        dest="budget_temp_end",
        help="Final softmax temperature (sharp, approaches hard top-k_eff). "
             "Default: 0.1.",
    )
    parser.add_argument(
        "--budget-axis", choices=["dst", "src", "both"], default=None,
        dest="budget_axis",
        help="Competition axis: 'dst' (per-destination, default), 'src' "
             "(per-source), or 'both' (multiplicative).",
    )
    args = parser.parse_args()

    # ---- seed global RNGs early (before any model/data construction) ----
    seed_everything(args.seed)
    # Derive cluster_seed from global seed when not explicitly overridden.
    if args.cluster_seed is None:
        args.cluster_seed = args.seed
    # Set base_seed for DataLoader worker_init_fn.
    _seeded_worker_init_fn._base_seed = args.seed

    # ---- early validation: cluster requires num_hidden ----
    if args.hidden_family == "cluster":
        if args.nodes_per_target > 0:
            # Grouped mode auto-sizes num_hidden; explicit value is ignored.
            if args.num_hidden is not None:
                print(f"[train_ctle] note: --num-hidden is ignored when "
                      f"--nodes-per-target > 0 (auto-sized to "
                      f"{args.nodes_per_target}*7+{args.readout_offset})")
        elif args.num_hidden is None:
            raise ValueError(
                "--num-hidden is required when --hidden-family=cluster. "
                "Pass an integer (e.g. --num-hidden 25) to specify the number "
                "of hidden nodes per stage."
            )
        if args.num_hidden is not None and args.num_hidden < 2:
            raise ValueError(
                f"--num-hidden must be >= 2 for cluster family (got {args.num_hidden})"
            )

    # ---- early validation: --q75-input is cluster-only ----
    if args.q75_input and args.hidden_family != "cluster":
        raise ValueError(
            "--q75-input is only supported with --hidden-family=cluster. "
            "The grid family's fan_out mapping hard-codes 4-input-to-corner "
            "assumptions that break under the 8-feature Q75 expansion. "
            "Either switch to --hidden-family cluster or omit --q75-input."
        )

    if args.q75_input and float(SCHEDULE_FOUR_PHASE.get("lambda_kd", 0.0)) > 0.0:
        print("[train_ctle] warning: --q75-input is incompatible with teacher KD; "
              "KD is disabled for B1/B2 phases (the teacher network expects raw "
              "4-dim specs, not pre-scaled 8-dim features).")

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
    amp_enabled = args.amp if args.amp is not None else torch.cuda.is_available()
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    compile_enabled = args.compile if args.compile is not None else (n_gpus >= 1)
    parallel_enabled = args.parallel if args.parallel is not None else (n_gpus >= 2)

    print(f"[train_ctle] device={device} epochs={epochs} lr={lr} batch_size={batch_size} "
          f"hidden_family={args.hidden_family} "
          f"grid_size={args.grid_size} num_hidden={args.num_hidden} "
          f"num_stages={args.num_stages} "
          f"q75_input={args.q75_input} "
          f"nodes_per_target={args.nodes_per_target} readout_offset={args.readout_offset} "
          f"weight_decay={args.weight_decay if args.weight_decay is not None else OPTIM['weight_decay']} "
          f"grad_clip={args.grad_clip if args.grad_clip is not None else OPTIM['grad_clip_norm']} "
          f"compile={compile_enabled} parallel={parallel_enabled} "
          f"amp={amp_enabled} amp_dtype={args.amp_dtype} ({n_gpus} GPUs) "
          f"seed={args.seed} cluster_seed={args.cluster_seed} "
          f"output={out_dir}")

    # ---- load teacher MoE ----
    if not args.teacher_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_path}")
    scaler_path = args.scaler_path if args.scaler_path is not None else args.teacher_path.parent / "flow_scaler_C.pkl"
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
    print(f"[train_ctle] loaded scaler from {scaler_path} "
          f"(p_mean={scaler_p_mean:.4f}, p_scale={scaler_p_scale:.4f}, "
          f"eye_h={eye_scale_h:.4f}, eye_w={eye_scale_w:.4f}, eye_j={eye_scale_j:.4f})")
    mlp = RegimeAwareMoE(
        trunk_width=args.teacher_hidden,
        trunk_layers=3,
        num_experts=args.teacher_experts,
        input_dim=4,
        output_dim=7,
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
    mlp.load_state_dict(state)
    mlp.eval()
    mlp.requires_grad_(False)
    mlp.to(device)
    n_teacher_params = sum(p.numel() for p in mlp.parameters())
    print(f"[train_ctle] loaded teacher MoE from {args.teacher_path} "
          f"({n_teacher_params} params, trunk={args.teacher_hidden}, "
          f"experts={args.teacher_experts})")

    # ---- build KirchhoffNet ----
    lib_name = args.cell_library if args.cell_library is not None else "legacy"
    cell_lib = make_cell_library(lib_name)
    preset = make_ctle_preset(
        family=args.hidden_family,
        grid_size=args.grid_size,
        num_hidden=args.num_hidden,
        num_stages=args.num_stages,
        write_mode=args.write_mode,
        bidirectional=args.bidirectional,
        cluster_edge_prob=args.cluster_edge_prob,
        cluster_seed=args.cluster_seed,
        q75_input=args.q75_input,
        edge_repeats=args.edge_repeats,
        nodes_per_target=args.nodes_per_target,
        readout_offset=args.readout_offset,
    )

    # Inject base lambdas from the config module-level LAMBDAS dict
    # (single source of truth for base regularizer weights). Schedule-specific
    # overrides (lambdas_b, lambdas_c, etc.) are applied per-epoch by
    # three_phase_lambdas / four_phase_lambdas during training.
    preset["lambdas"] = dict(LAMBDAS)

    # ---- resolve schedule mode (CLI override > preset default > 'four_phase') ----
    schedule_mode = _resolve_schedule(preset, args.schedule)
    print(f"[train_ctle] schedule_mode={schedule_mode}")

    # ---- resolve degree-budget config (degree-budget-topk plan) ----
    # CLI override > DEGREE_BUDGET defaults. --budget enables the master switch.
    budget_enabled = bool(args.budget) or bool(DEGREE_BUDGET.get("enabled", False))
    budget_frac_start = float(
        args.budget_frac_start if args.budget_frac_start is not None
        else DEGREE_BUDGET["frac_start"]
    )
    budget_frac_end = float(
        args.budget_frac_end if args.budget_frac_end is not None
        else DEGREE_BUDGET["frac_end"]
    )
    budget_temp_start = float(
        args.budget_temp_start if args.budget_temp_start is not None
        else DEGREE_BUDGET["temperature_start"]
    )
    budget_temp_end = float(
        args.budget_temp_end if args.budget_temp_end is not None
        else DEGREE_BUDGET["temperature_end"]
    )
    budget_axis = (
        args.budget_axis if args.budget_axis is not None
        else DEGREE_BUDGET["axis"]
    )
    budget_anneal_frac = float(DEGREE_BUDGET["anneal_frac"])
    if budget_enabled:
        if not (0.0 <= budget_frac_end <= 1.0):
            print(f"[train_ctle] WARNING: budget_frac_end={budget_frac_end} must be in [0,1], "
                  f"clamping to [0,1]")
            budget_frac_end = max(0.0, min(1.0, budget_frac_end))
        if not (0.0 <= budget_frac_start <= 1.0):
            print(f"[train_ctle] WARNING: budget_frac_start={budget_frac_start} must be in [0,1], "
                  f"clamping to [0,1]")
            budget_frac_start = max(0.0, min(1.0, budget_frac_start))
        print(
            f"[train_ctle] budget=enabled axis={budget_axis} "
            f"frac: {budget_frac_start:.2f}->{budget_frac_end:.2f} "
            f"temp: {budget_temp_start:.2f}->{budget_temp_end:.2f} "
            f"anneal_frac={budget_anneal_frac}"
        )
        # Check base lambdas and schedule-specific overrides for edge_gate.
        # With LAMBDAS as base (edge_gate=0) this typically won't fire;
        # the schedule check catches any non-zero edge_gate in the schedule
        # dict's lambdas_b / lambdas_b1 / lambdas_b2 (the phases where
        # budget competition is active).
        _base_edge = float(preset["lambdas"].get("edge_gate", 0.0))
        _sched_edge = 0.0
        if schedule_mode == "three_phase":
            _sched_edge = float(SCHEDULE_THREE_PHASE.get("lambdas_b", {}).get("edge_gate", 0.0))
        elif schedule_mode == "four_phase":
            _sched_edge = float(SCHEDULE_FOUR_PHASE.get("lambdas_b1", {}).get("edge_gate", 0.0))
            _sched_edge += float(SCHEDULE_FOUR_PHASE.get("lambdas_b2", {}).get("edge_gate", 0.0))
        if _base_edge > 0.0 or _sched_edge > 0.0:
            print(
                f"[train_ctle] WARNING: budget enabled but edge_gate may be active "
                f"(base={_base_edge}, schedule_phases={_sched_edge}). "
                f"The edge_gate regularizer (L1 on sigma(z_logits)) may fight the budget. "
                f"Consider setting edge_gate to 0 in all lambda sources."
            )
    net: KirchhoffNetWithIO = build_net_from_config(
        preset, cell_lib=cell_lib, enable_drive=args.persistent_drive,
    )
    net.to(device)
    n_kirchhoff_params = sum(p.numel() for p in net.parameters())
    in_mapper_name = type(net.input_mapper).__name__
    out_mapper_name = type(net.output_mapper).__name__
    _in_dim = 8 if args.q75_input else 4
    if args.nodes_per_target > 0:
        readout_mode_str = (
            f"GroupedOutputMapper(nodes_per_target={args.nodes_per_target}, "
            f"offset={args.readout_offset})"
        )
    else:
        readout_mode_str = "OutputMapper (default)"
    print(f"[train_ctle] readout mode: {readout_mode_str}")
    print(f"[train_ctle] built KirchhoffNet: in_dim={_in_dim} out_dim=7 "
          f"hid={net.hid_count} proj={net.proj_count} stages={len(net.core.stages)} "
          f"input_mapper={in_mapper_name} output_mapper={out_mapper_name} "
          f"({n_kirchhoff_params} params)")
    print(f"[train_ctle] write_idx={list(net.write_idx) if net.write_idx is not None else None} "
          f"read_idx={list(net.read_idx) if net.read_idx is not None else None}"
          f" persistent_drive={args.persistent_drive}")
    topo_label = f"ctle_{args.hidden_family}"
    if args.bidirectional or args.edge_repeats > 1:
        edges_per_stage = [s.num_edges() for s in net.core.stages]
        mult = 2 if args.bidirectional else 1
        mult *= args.edge_repeats
        print(f"[train_ctle] bidirectional={args.bidirectional} edge_repeats={args.edge_repeats}: "
              f"{edges_per_stage} edges per stage ({mult}× single-edge baseline)")

    # Resolve mapper names for later use.
    _effective_write_mode = (
        "fan_out" if isinstance(net.input_mapper, FanOutInputMapper)
        else "one_to_one" if isinstance(net.input_mapper, SparseInputMapper)
        else "dense"
    )

    # Cluster-specific guards must fire before the generic fan_out check below,
    # because cluster always uses dense write mode — the generic check would
    # raise a confusing ValueError instead of a graceful warning.

    if args.prune and args.nodes_per_target > 0:
        print(f"[train_ctle] WARNING: --prune is not supported with --nodes-per-target > 0 "
              f"(grouped readout uses GroupedOutputMapper, incompatible with "
              f"_transfer_output_mapper's OutputMapper-only path). Disabling --prune.")
        args.prune = False

    if args.prune and args.hidden_family == "cluster":
        print(f"[train_ctle] WARNING: --prune is not supported with --hidden-family=cluster "
              f"(cluster uses dense write mode, incompatible with fan_out-based "
              f"transfer_input_mapper). Disabling --prune.")
        args.prune = False

    if args.persistent_drive and args.hidden_family == "cluster":
        print(f"[train_ctle] WARNING: --persistent-drive is not supported with "
              f"--hidden-family=cluster (cluster uses dense write mode; "
              f"persistent_drive requires fan_out write mode). Disabling --persistent-drive.")
        args.persistent_drive = False

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

    # Effective num_hidden (args.num_hidden for cluster, grid_size**2 for grid).
    _eff_num_hidden = args.num_hidden if args.num_hidden is not None else (args.grid_size ** 2)

    # Save config snapshot
    snapshot_path = out_dir / "config_snapshot.txt"
    with snapshot_path.open("w") as f:
        f.write(f"teacher_path: {args.teacher_path}\n")
        f.write(f"scaler_path: {scaler_path}\n")
        f.write(f"teacher_params: {n_teacher_params}\n")
        f.write(f"teacher_hidden: {args.teacher_hidden}\n")
        f.write(f"teacher_experts: {args.teacher_experts}\n")
        f.write(f"hidden_family: {args.hidden_family}\n")
        f.write(f"grid_size: {args.grid_size}\n")
        f.write(f"num_hidden: {_eff_num_hidden}\n")
        f.write(f"seed: {args.seed}\n")
        if args.hidden_family == "cluster":
            f.write(f"cluster_edge_prob: {args.cluster_edge_prob}\n")
            f.write(f"cluster_seed: {args.cluster_seed}\n")
        f.write(f"num_stages: {args.num_stages}\n")
        f.write(f"hid_count: {net.hid_count}\n")
        f.write(f"proj_count: {net.proj_count}\n")
        f.write(f"write_idx: {list(net.write_idx) if net.write_idx is not None else None}\n")
        f.write(f"read_idx: {list(net.read_idx) if net.read_idx is not None else None}\n")
        f.write(f"persistent_drive: {args.persistent_drive}\n")
        f.write(f"kirchhoff_params: {n_kirchhoff_params}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"lr: {lr}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"stage_lr_scale: {args.stage_lr_scale}\n")
        f.write(f"mapper_lr_scale: {args.mapper_lr_scale}\n")
        f.write(f"struct_lr_scale: {args.struct_lr_scale}\n")
        f.write(f"dyn_lr_scale: {args.dyn_lr_scale}\n")
        f.write(f"write_mode: {_effective_write_mode}\n")
        f.write(f"compile: {compile_enabled}\n")
        f.write(f"parallel: {parallel_enabled}\n")
        f.write(f"amp: {amp_enabled}\n")
        f.write(f"amp_dtype: {args.amp_dtype}\n")
        f.write(f"grad_log: {args.grad_log}\n")
        f.write(f"grad_log_every: {args.grad_log_every}\n")
        f.write(f"prune: {args.prune}\n")
        f.write(f"normalize_targets: {args.normalize_targets}\n")
        f.write(f"q75_input: {args.q75_input}\n")
        f.write(f"nodes_per_target: {args.nodes_per_target}\n")
        f.write(f"readout_offset: {args.readout_offset}\n")
        f.write(f"grouped_readout: {args.nodes_per_target > 0}\n")
        f.write(f"weight_decay: {args.weight_decay if args.weight_decay is not None else OPTIM['weight_decay']}\n")
        f.write(f"grad_clip: {args.grad_clip if args.grad_clip is not None else OPTIM['grad_clip_norm']}\n")
        f.write(f"\nLAMBDAS (effective):\n")
        for k, v in preset["lambdas"].items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nSCHEDULE_MODE: {schedule_mode}\n")
        if schedule_mode == "four_phase":
            f.write(f"SCHEDULE_FOUR_PHASE fractions: {SCHEDULE_FOUR_PHASE['frac_a']}/"
                    f"{SCHEDULE_FOUR_PHASE['frac_b1']}/{SCHEDULE_FOUR_PHASE['frac_b2']}/"
                    f"{SCHEDULE_FOUR_PHASE['frac_c']}\n")
        elif schedule_mode == "three_phase":
            f.write(f"SCHEDULE_THREE_PHASE fractions: {SCHEDULE_THREE_PHASE['frac_a']}/"
                    f"{SCHEDULE_THREE_PHASE['frac_b']}/{SCHEDULE_THREE_PHASE['frac_c']}\n")
        f.write(f"budget_enabled: {budget_enabled}\n")
        if budget_enabled:
            f.write(f"budget: frac {budget_frac_start}->{budget_frac_end}, "
                    f"temp {budget_temp_start}->{budget_temp_end}, axis={budget_axis}, "
                    f"anneal_frac={budget_anneal_frac}\n")

    # ---- generate dataset ----
    print(f"[train_ctle] generating {args.n_train + args.n_val} synthetic spec samples "
          f"(train={args.n_train}, val={args.n_val}, "
          f"normalize_targets={args.normalize_targets}, q75_input={args.q75_input})...")
    train_loader, val_loader, target_mean, target_std = generate_ctle_dataset(
        n_train=args.n_train,
        n_val=args.n_val,
        mlp=mlp,
        device=device,
        batch_size=batch_size,
        seed=args.seed,
        normalize=args.normalize_targets,
        q75_input=args.q75_input,
    )
    print(f"[train_ctle] train batches={len(train_loader)} val batches={len(val_loader)}")
    if args.normalize_targets:
        norm_msg = ", ".join(
            f"{n}={float(m):+.2f}/{float(s):.2f}"
            for n, m, s in zip(PARAM_COLS, target_mean, target_std)
        )
        print(f"[train_ctle] per-dim target stats (mean/std): {norm_msg}")
        torch.save(
            {"mean": target_mean, "std": target_std, "params": PARAM_COLS},
            out_dir / "target_norm_stats.pt",
        )

    task_fn = F.mse_loss

    # ---- save init stage graphs ----
    for i, stage in enumerate(raw_net.core.stages):
        plot_stage_graph(
            stage,
            save_path=str(out_dir / f"stage{i + 1}_graph_init.png"),
            title=f"{topo_label} — Stage {i + 1} (init)",
        )

    # ---- schedule boundaries ----
    # Branch on schedule_mode to support three_phase and legacy in addition
    # to the default four_phase. For three_phase, ab_total=b_end (no B1/B2
    # split, no readiness gating); for legacy, ab_total=epochs and retrain
    # only runs if --prune is explicitly set.
    if schedule_mode == "four_phase":
        a_end, b1_end, b2_end, c_end = four_phase_boundaries(epochs)
        print(f"[train_ctle] four_phase schedule: A=[0,{a_end}) B1=[{a_end},{b1_end}) "
              f"B2=[{b1_end},{b2_end}) C=[{b2_end},{c_end})")
        ab_total = b2_end
        c_total = max(1, epochs - b2_end)
        needs_prune = True
    elif schedule_mode == "three_phase":
        a_end, b_end, _ = phase_boundaries(epochs)
        print(f"[train_ctle] three_phase schedule: A=[0,{a_end}) B=[{a_end},{b_end}) "
              f"C=[{b_end},{epochs})")
        ab_total = b_end
        c_total = max(1, epochs - b_end)
        needs_prune = True
    else:  # legacy
        print(f"[train_ctle] legacy schedule: single-phase over {epochs} epochs")
        a_end = b1_end = b2_end = b_end = epochs
        ab_total = epochs
        c_total = 0
        needs_prune = args.prune
    if args.retrain_epochs is not None:
        c_total = args.retrain_epochs

    # ---- 4-phase training state ----
    teacher_net = MLPTeacherWrapper(mlp).to(device)

    optimizer = make_optimizer(
        net, lr=lr,
        weight_decay=args.weight_decay,
        stage_lr_scale=args.stage_lr_scale,
        mapper_lr_scale=args.mapper_lr_scale,
        struct_lr_scale=args.struct_lr_scale,
        dyn_lr_scale=args.dyn_lr_scale,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, ab_total), eta_min=float(OPTIM["scheduler_eta_min"]),
    )

    grad_clip = float(args.grad_clip) if args.grad_clip is not None else float(OPTIM["grad_clip_norm"])
    if grad_clip <= 0.0:
        grad_clip = float("inf")
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

    per_dim_log_path = out_dir / "per_dim_stats.txt"
    per_dim_history: list[dict] = []
    print(f"[train_ctle] per-dimension stats logging enabled -> {per_dim_log_path}")



    readiness_prune_fired = False
    readiness_prune_epoch = -1

    # ---- AMP scaler ----
    scaler = (
        torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    )

    # ---- Phase A + (B1 + B2 | B) training loop ----
    if schedule_mode == "four_phase":
        loop_desc = "train_ctle [A+B1+B2]"
    elif schedule_mode == "three_phase":
        loop_desc = "train_ctle [A+B]"
    else:
        loop_desc = "train_ctle [legacy]"
    print(f"[train_ctle] starting {schedule_mode} training loop ({loop_desc})")
    for epoch in range(ab_total):
        if stop_training:
            break
        net.train()

        # Degree budget / fraction competition (degree-budget-topk plan).
        # Recompute frac and temperature for this epoch and push to all
        # stages. Done once per epoch (global schedule); the per-batch
        # budget gate is recomputed inside rhs() from each stage's
        # current budget_frac / budget_temperature.
        if budget_enabled:
            _b_frac = budget_frac_for_epoch(
                epoch, ab_total, budget_frac_start, budget_frac_end, budget_anneal_frac,
            )
            _b_T = budget_temperature_for_epoch(
                epoch, ab_total, budget_temp_start, budget_temp_end, budget_anneal_frac,
            )
            for _stage in raw_net.core.stages:
                _stage.set_budget_frac(_b_frac, _b_T)
                _stage.budget_axis = budget_axis

        # Schedule-aware tau / phase / lambdas dispatch.
        if schedule_mode == "four_phase":
            tau = four_phase_tau(epoch, epochs)
            phase = phase_for_epoch_four(epoch, epochs)
            eff_lambdas = four_phase_lambdas(epoch, epochs, preset["lambdas"])
            reg_scale = 1.0  # four_phase_lambdas already includes warmup
            # Stop early if readiness triggered (handled at validate step,
            # but we also break here if the flag was set in a previous
            # iteration).
            if phase == "B2" and readiness_prune_fired:
                print(f"[four_phase] readiness pruning triggered at epoch {epoch} "
                      f"(B2 cut short, was scheduled for {b2_end})")
                break
        elif schedule_mode == "three_phase":
            tau = three_phase_tau(epoch, epochs)
            phase = phase_for_epoch(epoch, epochs)
            eff_lambdas = three_phase_lambdas(epoch, epochs, preset["lambdas"])
            reg_scale = 1.0  # three_phase_lambdas already includes warmup
        else:  # legacy
            tau = tau_for_epoch(epoch, total_epochs=epochs)
            phase = ""
            reg_scale = reg_schedule(epoch)
            eff_lambdas = dict(preset["lambdas"])
        cell_mode = resolve_cell_mode(args.cell_mode, phase, schedule_mode)

        # ---- one epoch ----
        total_loss = 0.0
        n_batches = 0
        for u_b, tgt_b in train_loader:
            optimizer.zero_grad()
            u_b = u_b.to(device)
            tgt_b = tgt_b.to(device)

            # Teacher KD is active in B1/B2 only.
            # KD is incompatible with --q75-input because the teacher network
            # expects raw 4-dim specs, not pre-scaled 8-dim features.
            kd_teacher = (teacher_net if (phase in ("B1", "B2") and not args.q75_input) else None)
            kd_lambda = float(SCHEDULE_FOUR_PHASE.get("lambda_kd", 1.0)) if kd_teacher is not None else 0.0

            loss_task, loss_structural, _ = compute_loss(
                net, u_b, tgt_b, ctx=None, task_fn=task_fn,
                lambdas=eff_lambdas, tau=tau, return_parts=True,
                amp=amp_enabled, amp_dtype=amp_dtype, reg_scale=reg_scale,
                cell_mode=cell_mode,
                teacher=kd_teacher, lambda_kd=kd_lambda, teacher_tau=1.0,
                teacher_cell_mode="soft",
            )
            if scaler is not None and scaler._enabled:
                scaler.scale(loss_task + loss_structural).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                (loss_task + loss_structural).backward()
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
            stats = compute_per_dim_stats(net, val_loader, device,
                                          tau=tau, cell_mode=cell_mode)
            val_loss = stats["agg_mse"]
            val_v_history.append(val_loss)
            val_history.append(val_loss)
            _log_per_dim_stats(per_dim_log_path, epoch, phase, stats)
            per_dim_history.append({"epoch": epoch, "phase": phase, **stats})
            _log_drive_diagnostics(raw_net, val_loader, device,
                                   out_dir / "drive_stats.txt", epoch)

            val_arg = validate_argmax(net, val_loader, task_fn, lambda b, device=None: None, device)
            val_argmax_history.append(val_arg)
            val_argmax_v_history.append(val_arg)

            if (
                (schedule_mode == "three_phase" and phase in ("A", "B"))
                or (schedule_mode == "four_phase" and phase in ("A", "B1", "B2"))
            ):
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
            use_argmax_ckpt = (
                (schedule_mode == "three_phase" and phase == "B")
                or (schedule_mode == "four_phase" and phase in ("B1", "B2"))
            ) and val_argmax_history and len(val_argmax_history) > 0
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
        if do_validate and per_dim_history:
            worst_name, worst_val = _worst_dim_label(per_dim_history[-1])
            worst_str = f"  worst_dim={worst_name}({worst_val:.3f})"
        else:
            worst_str = ""
        print(
            f"  epoch {epoch:4d} [{phase}]  train={avg_train:.4f}  "
            f"val={val_loss:.4f}  val_argmax={val_argmax_history[-1] if val_argmax_history else 0:.4f}  "
            f"tau={tau:.3f}  lr={optimizer.param_groups[0]['lr']:.2e}{worst_str}"
        )

    # ---- end of overcomplete phase(s) ----
    if schedule_mode == "four_phase":
        if not stop_training and not readiness_prune_fired:
            readiness_prune_fired = True
            readiness_prune_epoch = b2_end
            print(f"[train_ctle] A+B1+B2 complete, no readiness trigger — "
                  f"fallback prune at epoch {b2_end}")
    elif schedule_mode == "three_phase":
        # three_phase always prunes at B->C (no readiness gating).
        if not stop_training:
            print(f"[train_ctle] A+B complete at epoch {b_end}, "
                  f"pruning at B->C boundary")
    else:  # legacy
        # legacy only prunes when --prune is explicitly set.
        needs_prune = bool(args.prune)
        if needs_prune and not stop_training:
            print(f"[train_ctle] legacy schedule complete at epoch {epochs}, "
                  f"pruning (--prune was set)")

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
            if schedule_mode == "four_phase":
                p = phase_for_epoch_four(i, epochs)
            elif schedule_mode == "three_phase":
                p = phase_for_epoch(i, epochs)
            else:
                p = "A"
            va = val_argmax_history[i] if i < len(val_argmax_history) else float("nan")
            f.write(f"{i}\t{t:.6e}\t{v:.6e}\t{va:.6e}\t{p}\n")

    # loss curve
    fig, ax = _plt.subplots(figsize=(8, 4.5))
    ax.plot(history, label="train", color="C0")
    ax.plot(val_history, label="val (soft)", color="C3")
    if val_argmax_history:
        ax.plot(val_argmax_history, label="val (argmax)", color="C2", linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (logit space)")
    ax.set_title(f"{topo_label} (KD from MLP) — {schedule_mode} training (total {epochs} epochs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=120, bbox_inches="tight")
    _plt.close(fig)

    # per-dimension summary plot
    if per_dim_history:
        _plot_per_dim_diagnostics(
            per_dim_history, out_dir / "per_dim_stats.png",
            norm_label="(normalized)" if args.normalize_targets else "(logit space)",
            suptitle=f"Per-dimension diagnostics ({topo_label}) — {loop_desc}",
        )

    # output fit (logit space)

    # output fit (logit space)
    with torch.no_grad():
        val_batch = next(iter(val_loader))
        u_v = val_batch[0][:64].to(device)
        y_v = val_batch[1][:64].to(device)
        out_v, _ = net(u_v, ctx=None, store_trajectory=False,
                       cell_mode="soft", tau=0.001)
    fit_title = (
        "Pre-prune output fit (normalized logit space, 7 params flattened)"
        if args.normalize_targets
        else "Pre-prune output fit (logit space, 7 params flattened)"
    )
    plot_output_fit(
        out_v, y_v, loss_name="mse",
        title=fit_title,
        save_path=str(out_dir / "output_fit.png"),
    )

    raw_net = _unwrap(net)
    for i, stage in enumerate(raw_net.core.stages):
        plot_stage_graph(
            stage,
            save_path=str(out_dir / f"stage{i + 1}_graph_trained.png"),
            title=f"{topo_label} — Stage {i + 1} (trained, {stage.num_edges()} edges)",
        )
        if stage.logits is not None:
            plot_cell_selection(
                stage.logits, cell_order=stage.cell_lib._cell_order,
                save_path=str(out_dir / f"stage{i + 1}_cell_selection_trained.png"),
                title=f"{topo_label} — Stage {i + 1} cell selection (trained)",
            )

    torch.save(net.state_dict(), out_dir / "model.pt")
    print(f"[train_ctle] saved pre-prune model to {out_dir / 'model.pt'}")

    # ---- pruning + retrain (Phase C) ----
    if needs_prune:
        if schedule_mode == "four_phase":
            edge_thresh = float(SCHEDULE_FOUR_PHASE.get("prune_edge_threshold", 0.05))
            node_thresh = float(SCHEDULE_FOUR_PHASE.get("prune_node_threshold", 0.05))
        elif schedule_mode == "three_phase":
            edge_thresh = float(SCHEDULE_THREE_PHASE.get("prune_edge_threshold", 0.1))
            node_thresh = float(SCHEDULE_THREE_PHASE.get("prune_node_threshold", 0.05))
        else:  # legacy
            edge_thresh = float(PRUNE["edge_threshold"])
            node_thresh = float(PRUNE["node_threshold"])
        # DEPRECATED (deprecate-node-gates): prune_nodes_by_gate is always False.
        prune_nodes_by_gate = False
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

        # Rebuild drive mappers for the pruned network when persistent drive was active.
        drive_mappers_pruned = None
        enable_drive_pruned = False
        if raw_net.enable_drive:
            pruned_widths = [s.num_nodes for s in pruned_core.stages]
            if len(set(pruned_widths)) == 1:
                pruned_fan_out = input_mapper_pruned.fan_out_map
                drive_mappers_pruned = [
                    FanOutInputMapper(
                        in_dim=in_dim, out_dim=pruned_first_n, fan_out_map=pruned_fan_out,
                    )
                    for _ in range(len(pruned_core.stages))
                ]
                enable_drive_pruned = True
            else:
                print(f"[prune] WARNING: stages pruned to different widths {pruned_widths}; "
                      f"disabling persistent drive for retrain")

        pruned_net = KirchhoffNetWithIO(
            input_mapper_pruned, pruned_core, output_mapper_pruned,
            hid_count=pruned_first_n, proj_count=0,
            final_hid_count=pruned_last_n, final_proj_count=0,
            write_idx=None,  # fan_out mode uses fan_out_map
            read_idx=pruned_read_idx,
            enable_drive=enable_drive_pruned,
            drive_mappers=drive_mappers_pruned,
        )
        pruned_net.to(device)
        n_pruned = sum(p.numel() for p in pruned_net.parameters())
        print(f"[prune] pruned network: {n_pruned} params "
              f"({100.0 * n_pruned / max(1, n_kirchhoff_params):.1f}% of pre-prune)")

        # ---- Phase C retrain ----
        retrain_optimizer = make_optimizer(
            pruned_net, lr=retrain_lr,
            weight_decay=args.weight_decay,
            stage_lr_scale=1.0, mapper_lr_scale=1.0,
            struct_lr_scale=args.struct_lr_scale,
            dyn_lr_scale=args.dyn_lr_scale,
        )
        retrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            retrain_optimizer, T_max=max(1, c_total),
            eta_min=float(OPTIM["scheduler_eta_min"]),
        )
        retrain_history: list[float] = []
        retrain_val_history: list[float] = []
        retrain_val_argmax: list[float] = []
        retrain_per_dim_history: list[dict] = []
        best_val_pruned = float("inf")
        best_epoch_pruned = -1
        best_state_pruned: dict | None = None
        retrain_scaler = (
            torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
        )
        ewop_p = 0
        for repoch in range(c_total):
            pruned_net.train()
            # Degree budget (degree-budget-topk plan): in Phase C the
            # budget is disabled (frac=0.0) — the compact network is
            # already pruned, so per-edge competition has no work.
            if budget_enabled:
                for _stage in pruned_net.core.stages:
                    _stage.set_budget_frac(0.0, budget_temp_end)
                    _stage.budget_axis = budget_axis
            # Schedule-aware Phase C retrain: schedule's tau/lambdas at the
            # appropriate global epoch so the retrain picks up where the
            # overcomplete training left off.
            if schedule_mode == "four_phase":
                global_epoch = b2_end + repoch
                tau_r = four_phase_tau(global_epoch, epochs)
                eff_c_lambdas = four_phase_lambdas(global_epoch, epochs, preset["lambdas"])
                reg_scale_r = 1.0
            elif schedule_mode == "three_phase":
                global_epoch = b_end + repoch
                tau_r = three_phase_tau(global_epoch, epochs)
                eff_c_lambdas = three_phase_lambdas(global_epoch, epochs, preset["lambdas"])
                reg_scale_r = 1.0
            else:  # legacy
                global_epoch = repoch
                tau_r = tau_for_epoch(repoch, total_epochs=c_total)
                eff_c_lambdas = dict(preset["lambdas"])
                reg_scale_r = reg_schedule(repoch)
            cell_mode_c = resolve_cell_mode(args.cell_mode, "C", schedule_mode)

            tot = 0.0
            nb = 0
            for u_b, tgt_b in train_loader:
                retrain_optimizer.zero_grad()
                u_b = u_b.to(device)
                tgt_b = tgt_b.to(device)
                loss_task, loss_structural, _ = compute_loss(
                    pruned_net, u_b, tgt_b, ctx=None, task_fn=task_fn,
                    lambdas=eff_c_lambdas, tau=tau_r, return_parts=True,
                    amp=amp_enabled, amp_dtype=amp_dtype, reg_scale=reg_scale_r,
                    cell_mode=cell_mode_c,
                )
                if retrain_scaler is not None and retrain_scaler._enabled:
                    retrain_scaler.scale(loss_task + loss_structural).backward()
                    retrain_scaler.unscale_(retrain_optimizer)
                    torch.nn.utils.clip_grad_norm_(pruned_net.parameters(), max_norm=grad_clip)
                    retrain_scaler.step(retrain_optimizer)
                    retrain_scaler.update()
                else:
                    (loss_task + loss_structural).backward()
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
                c_stats = compute_per_dim_stats(pruned_net, val_loader, device,
                                                tau=tau_r, cell_mode=cell_mode_c)
                retrain_val_history.append(c_stats["agg_mse"])
                val_arg_r = validate_argmax(pruned_net, val_loader, task_fn, lambda b, device=None: None, device)
                retrain_val_argmax.append(val_arg_r)
                _log_per_dim_stats(per_dim_log_path, global_epoch, "C", c_stats)
                retrain_per_dim_history.append({"epoch": global_epoch, "phase": "C", **c_stats})
                _log_drive_diagnostics(pruned_net, val_loader, device,
                                       out_dir / "drive_stats.txt", global_epoch)

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

            _on_retrain_val = (repoch % args.validate_every == 0 or repoch == c_total - 1)
            if _on_retrain_val and retrain_per_dim_history:
                wn, wv = _worst_dim_label(retrain_per_dim_history[-1])
                worst_c = f"  worst_dim={wn}({wv:.3f})"
            else:
                worst_c = ""
            print(
                f"  phase-C epoch {repoch:4d}  train={avg:.4f}  "
                f"val={retrain_val_history[-1]:.4f}  val_argmax={retrain_val_argmax[-1]:.4f}  "
                f"tau={tau_r:.3f}{worst_c}"
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
                title=f"{topo_label} — Stage {i + 1} (pruned, {stage.num_edges()} edges, {stage.num_nodes} nodes)",
            )
            if stage.logits is not None:
                plot_cell_selection(
                    stage.logits, cell_order=stage.cell_lib._cell_order,
                    save_path=str(out_dir / f"stage{i + 1}_cell_selection_pruned.png"),
                    title=f"{topo_label} — Stage {i + 1} cell selection (pruned)",
                )

        with torch.no_grad():
            out_pruned, _ = pruned_net(u_v, ctx=None, store_trajectory=False,
                                       cell_mode="soft", tau=0.001)
        pruned_fit_title = (
            "Pruned output fit (normalized logit space, 7 params flattened)"
            if args.normalize_targets
            else "Pruned output fit (logit space, 7 params flattened)"
        )
        plot_output_fit(
            out_pruned, y_v, loss_name="mse",
            title=pruned_fit_title,
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

        # Combined per-dim plot (pre-prune + retrain)
        if retrain_per_dim_history:
            _plot_per_dim_diagnostics(
                per_dim_history + retrain_per_dim_history,
                out_dir / "per_dim_stats.png",
                norm_label="(normalized)" if args.normalize_targets else "(logit space)",
                suptitle=f"Per-dimension diagnostics ({topo_label}) — all phases",
            )

    # ---- physical-domain evaluation ----
    print("[train_ctle] physical-domain evaluation (relative error per param)...")
    eval_net = pruned_net if (args.prune and best_state_pruned is not None) else net
    eval_net.eval()
    eval_n = min(2000, len(val_loader.dataset))
    eval_specs = sample_specs(eval_n, seed=123).to(device)
    with torch.no_grad():
        eval_logits_mlp = mlp(eval_specs)
        # When q75_input is active, the student expects 8-dim Q75-scaled
        # features (the same transformation applied during training,
        # see make_ctle_data mid-loop). The teacher handles raw 4-dim
        # internally via its own scale_input() in forward().
        eval_specs_student = mlp.scale_input(eval_specs) if args.q75_input else eval_specs
        eval_logits_kirchhoff, _ = eval_net(eval_specs_student, ctx=None,
                                            store_trajectory=False,
                                            cell_mode="soft", tau=0.001)
    if args.normalize_targets:
        mean_d = target_mean.to(eval_logits_kirchhoff.device)
        std_d = target_std.to(eval_logits_kirchhoff.device)
        eval_logits_kirchhoff = eval_logits_kirchhoff * std_d + mean_d
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
