"""CTLE MoE teacher + synthetic spec dataset helpers for the MLP/MoE baselines.

Extracted verbatim from the former ``train_ctle.py`` (removed in surgery 4/6):
only the pieces ``mlp_benchmark_ctle.py`` (and its tests) still need.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SPEC_INPUT_COLS = ["power", "jitter", "height", "width"]

SPEC_RANGES = {
    "power": (0.0012, 0.012),
    "jitter": (1.57, 100.0),
    "height": (0.0, 88.4),
    "width": (0.0, 98.5)}

PARAM_COLS = ["fW", "current", "ind", "Rd", "Cs", "Rs", "VDD"]

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
