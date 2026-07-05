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
    TAU)
from io_mapper import FanOutInputMapper, SparseInputMapper
from kirchhoff_net import KirchhoffNetWithIO
from topology import build_net_from_config, prune_network
from train import (
    budget_frac_for_epoch,
    budget_temperature_for_epoch,
    compute_loss,
    four_phase_boundaries,
    four_phase_kd_active,
    four_phase_lambdas,
    four_phase_tau,
    make_optimizer,
    phase_boundaries,
    phase_for_epoch,
    phase_for_epoch_four,
    reg_schedule,
    tau_for_epoch,
    three_phase_lambdas,
    three_phase_tau)

# Monkey-patch _compute_regularizers to handle DataParallel wrapping
# (same as train_script.py; DP strips module internals from the wrapper).
import train as _train_module
_orig_compute_regularizers = _train_module._compute_regularizers

def _dp_safe_compute_regularizers(net, trajs, lambdas):
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    return _orig_compute_regularizers(net, trajs, lambdas)

_train_module._compute_regularizers = _dp_safe_compute_regularizers

from visualize import (
    plot_network,
    plot_output_fit,
    plot_stage_graph)

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
    "width": (0.0, 98.5)}

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
    "VDD":  (math.log10(0.6),   math.log10(1.2))}

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
# CTLE preset factory (grid family)
# =============================================================================

def make_ctle_preset(
    family: str = "grid",
    grid_size: int = 5,
    num_hidden: int | None = None,
    num_stages: int = 3,
    num_proj: int | None = None,
    write_mode: str | None = None,
    bidirectional: bool = False,
    q75_input: bool = False,
    edge_repeats: int = 2,
    nodes_per_target: int = 0,
    readout_offset: int = 0,
    leak_mode: str = "programmable",
    leak_constant: float | None = None) -> dict:
    """Build a 4-spec → 7-logit CTLE KirchhoffNet preset for the grid family.

    ``family='grid'`` (default, backward compatible):
        Each stage has ``grid_size**2`` hidden nodes laid out on a 2D grid
        with 3x3 convolution-style neighborhood edges, plus ``num_proj``
        projection nodes and 4 input nodes (8 when ``q75_input=True``).
        Write mapping is fan-out from inputs to grid corners; read mapping
        gathers the center column + all projection nodes (grid_size +
        num_proj features).

    When ``q75_input=True``, ``num_inputs`` becomes 8 (teacher's
    ``scale_input()``: log10 + StandardScaler/Q75 expansion). Caller must
    apply the same scaling to training inputs.

    When ``nodes_per_target > 0``, group the 7 regression targets into per-target
    readout windows of ``nodes_per_target`` consecutive state nodes each. The
    network auto-sizes so ``grid_size**2`` accommodates
    ``nodes_per_target * 7 + readout_offset`` state nodes. In
    this mode ``num_proj`` is forced to 0 (no projection nodes; heads read
    directly from hidden state) and ``--prune`` must be disabled at the CLI.

    Args:
        family: Must be 'grid'.
        grid_size: Square grid side length (grid family).
        num_hidden: Ignored for grid family (uses ``grid_size**2``).
        num_stages: Number of ODE stages.
        num_proj: Projection node count (grid family default 7).
        write_mode: 'fan_out' | 'dense' | 'one_to_one' | None (default per family).
        bidirectional: Emit two directed edges per node pair.
        q75_input: When True, set ``num_inputs=8`` (Q75-scaled features).
        edge_repeats: Number of parallel edges per hidden node pair (default 2,
            range 1-8). Composes multiplicatively with ``bidirectional``.
            I/O and projection edges are NOT repeated.
        nodes_per_target: If > 0, enable grouped per-target readout with this
            many state nodes per target. Auto-sizes the network.
        readout_offset: Starting state index for the first target's window
            (only used when ``nodes_per_target > 0``).
        leak_mode: ``"programmable"`` (default) or ``"non-programmable"``.
            Stored in the preset dict so :func:`build_net_from_config` can
            thread it through to :class:`DifferentialStage`.
        leak_constant: Fixed leak value used when ``leak_mode="non-programmable"``.
            Stored in the preset dict.
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
        _stage_cfg = {
            "num_inputs": 8 if q75_input else 4,
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
            "num_steps": round(SOLVER["num_steps"] / n_stages)}
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
                3: [r * grid_size + (grid_size - 1) for r in bot_rows]}
            read_idx = None  # GroupedOutputMapper handles its own windowing.
        else:
            eff_write = write_mode if write_mode is not None else "fan_out"
            top_rows = [0, 1]
            bot_rows = [grid_size - 2, grid_size - 1]
            fan_out = {
                0: [r * grid_size + 0 for r in top_rows],
                1: [r * grid_size + (grid_size - 1) for r in top_rows],
                2: [r * grid_size + 0 for r in bot_rows],
                3: [r * grid_size + (grid_size - 1) for r in bot_rows]}
            center_col = grid_size // 2
            center_nodes = [r * grid_size + center_col for r in range(grid_size)]
            read_idx = center_nodes + list(range(n_hidden, n_hidden + num_proj))

    elif family == "cluster":
        raise ValueError(
            "make_ctle_preset(family='cluster') is no longer supported. "
            "Use the grid family (default) instead."
        )

    else:
        raise ValueError(f"Unknown family: {family!r} (expected 'grid')")

    preset: dict[str, Any] = {
        "stages": [_stage_cfg] * n_stages,
        "use_robust_input": False,
        "loss": "mse",
        "out_dim": len(PARAM_COLS),
        "write_mode": eff_write,
        "read_idx": read_idx,
        "schedule": "four_phase",
        "tau_anneal": True}
    if eff_write == "fan_out" and fan_out is not None:
        preset["write_fan_out"] = fan_out
    if grouped:
        preset["grouped_readout"] = {
            "nodes_per_target": nodes_per_target,
            "offset": readout_offset}
    if leak_mode != "programmable" or leak_constant is not None:
        preset["leak_mode"] = leak_mode
        if leak_constant is not None:
            preset["leak_constant"] = leak_constant
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
    leak_mode: str = "programmable",
    leak_constant: float | None = None) -> dict:
    """Backward-compatible thin wrapper for the grid CTLE preset.

    Equivalent to ``make_ctle_preset(family='grid', grid_size=grid_size, ...)``.
    """
    return make_ctle_preset(
        family="grid",
        grid_size=grid_size,
        num_stages=num_stages,
        num_proj=num_proj,
        write_mode=write_mode,
        bidirectional=bidirectional,
        q75_input=q75_input,
        leak_mode=leak_mode,
        leak_constant=leak_constant,
        edge_repeats=edge_repeats,
        nodes_per_target=nodes_per_target,
        readout_offset=readout_offset)

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
        eye_scale_j: float = 1.0) -> None:
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
    ...)`` and unpacks ``(y_teacher, _)``. This wrapper ignores the
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
        store_trajectory: bool = False) -> tuple[torch.Tensor, None]:
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
    q75_input: bool = False) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
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
        generator=g, worker_init_fn=_seeded_worker_init_fn)
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
    tau: float | None = None) -> dict[str, np.ndarray | float]:
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
                         )
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
            "per_dim_var": empty}
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
        "per_dim_var": per_dim_var}

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
    log_path: Path, epoch: int) -> None:
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
            x_max=raw_mapper.x_max)
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
            node_dim=pruned_last_n, out_dim=out_dim, read_idx=new_read_idx)
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
      stage{i}_raw_leak,
      stage{i}_z_logits, stage{i}_u_logits  (one entry per stage)
      stage_transfer
      in_mapper
      out_mapper
    Each value is either a float (the L2 norm) or None if the group has
    no parameter or no gradient.
    """
    stage_sq = {}
    stage_components = ("raw_leak", "z_logits", "u_logits")
    # Per-edge device parameter suffixes: covers SimpleEdgeLibrary.param
    # (I=ReLU/tanh(p0*Vsrc+p1*Vdest+p2)), RealisticTanhLibrary
    # (alpha_raw, bias_raw), RealisticTanhUpgradeLibrary (alpha_raw,
    # gm_raw, isat_raw, bias_raw), and FreeTanhLibrary (a_raw, b_raw, s_raw,
    # gm_raw, isat_raw, theta_raw). All contribute to the same `device_param`
    # gradient-norm metric per stage.
    device_param_suffixes = (
        "param", "alpha_raw", "bias_raw", "gm_raw", "isat_raw",
        "a_raw", "b_raw", "s_raw", "theta_raw")
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
# (STE handled inline.)
# =============================================================================

