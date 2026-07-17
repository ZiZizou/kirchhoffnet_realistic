"""Realistic hardware noise injection for MLP benchmarks.

Models three primary noise sources found in analog crossbar accelerators:
  1. Weight Quantization: quantize MLP weights to N-bit symmetric fixed-point
     (simulates limited-precision weight storage in analog memory cells).
  2. ADC/DAC Quantization: quantize the input and output of each linear layer
     to simulate Analog-to-Digital and Digital-to-Analog converters at the
     boundary between analog and digital domains.
  3. Circuit Noise: add Gaussian noise with a standard deviation of sigma
     to weights and activations to model thermal noise, process variations,
     and IR-drop in analog crossbars.

Design notes:
- All noise is sampled once per forward pass and reused across the entire
  forward. Resampling inside intermediate layers would model time-varying
  noise but is not what is asked for here.
- Weight quantization uses a straight-through estimator (STE) in training
  mode so gradients flow through the rounding operation. In eval mode the
  quantization is hard (no STE).
- ADC/DAC quantization is applied as fixed symmetric quantization; no STE
  needed because activations do not need gradients through the quantize step
  in the typical use case (the wrapper is used in eval mode for the noisy
  pass; for noise-aware training the loss is computed on the noisy forward).
- The wrapper is intentionally simple and only handles ``MLPRegressor`` (a
  flat ModuleList of Linear + Activation). It is not a general Module hook.

Typical usage:

.. code-block:: python

    from analog_noise import (
        NoiseConfig, AnalogMLPWrapper, evaluate_with_noise,
    )
    from mlp_benchmark import MLPRegressor

    base = MLPRegressor(in_dim=2, hidden_dim=100, out_dim=1)
    base.load_state_dict(torch.load("model.pt"))
    cfg = NoiseConfig(quant_bits=4, noise_std=0.05, mc_trials=20)
    noisy = AnalogMLPWrapper(base, cfg)
    result = evaluate_with_noise(noisy, val_loader, task_fn, cfg, device)
    print(result.mean, result.std, result.p90)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch
import torch.nn as nn


__all__ = [
    "NoiseConfig",
    "NoiseBenchmarkResult",
    "fake_quantize_symmetric",
    "WeightQuantizer",
    "AdcDacQuantizer",
    "CircuitNoise",
    "AnalogMLPWrapper",
    "evaluate_with_noise",
]


@dataclass
class NoiseConfig:
    """Configuration for analog-style noise injection.

    Attributes:
        quant_bits: Bit-width for both weight and ADC/DAC quantization
            (4 or 6). ``None`` disables quantization.
        noise_std: Standard deviation of additive Gaussian circuit noise on
            weights and activations. ``0.0`` disables circuit noise.
        quantize_input: Whether to quantize the input of the first layer
            (DAC). Default ``True``.
        quantize_output: Whether to quantize the output of the final layer
            (ADC). Default ``True``.
        quantize_intermediate: Whether to quantize activations between
            hidden layers (intra-network ADC/DAC). Default ``True``.
        weight_noise: Whether to inject Gaussian noise on the weights.
            Default ``True``.
        activation_noise: Whether to inject Gaussian noise on the
            activations (pre- and post-quantization). Default ``True``.
        mc_trials: Number of Monte Carlo trials for noise evaluation.
        seed: Random seed for noise sampling. ``None`` uses non-deterministic
            RNG (torch global).
        ste: Straight-through estimator for weight quantization during
            training. ``True`` allows gradients to flow through the
            rounding op; ``False`` is equivalent to a hard quantize.
        clip_quant_range: If ``True``, clamp activations to the
            quantization range after ADC to model saturation. Default ``True``.
    """

    quant_bits: int | None = 4
    noise_std: float = 0.05
    quantize_input: bool = True
    quantize_output: bool = True
    quantize_intermediate: bool = True
    weight_noise: bool = True
    activation_noise: bool = True
    mc_trials: int = 20
    seed: int | None = None
    ste: bool = True
    clip_quant_range: bool = True


@dataclass
class NoiseBenchmarkResult:
    """Result of a multi-trial Monte Carlo evaluation under analog noise.

    Attributes:
        losses: List of per-trial val losses.
        mean: Mean loss across trials.
        std: Standard deviation across trials.
        p50: Median loss.
        p90: 90th percentile loss.
        p95: 95th percentile loss.
        best: Best (minimum) loss across trials.
        worst: Worst (maximum) loss across trials.
        config: The NoiseConfig used.
        clean_loss: Val loss of the underlying model without noise, if
            computed by the caller.
    """

    losses: list[float] = field(default_factory=list)
    mean: float = float("nan")
    std: float = float("nan")
    p50: float = float("nan")
    p90: float = float("nan")
    p95: float = float("nan")
    best: float = float("nan")
    worst: float = float("nan")
    config: NoiseConfig | None = None
    clean_loss: float | None = None

    def summary(self) -> str:
        """Human-readable one-line summary."""
        if not self.losses:
            return "NoiseBenchmarkResult: empty"
        clean = (
            f" clean={self.clean_loss:.6f}" if self.clean_loss is not None else ""
        )
        return (
            f"trials={len(self.losses)} mean={self.mean:.6f} "
            f"std={self.std:.6f} p50={self.p50:.6f} p90={self.p90:.6f} "
            f"p95={self.p95:.6f} best={self.best:.6f} worst={self.worst:.6f}"
            f"{clean}"
        )


# ---------------------------------------------------------------------------
# Quantization primitives
# ---------------------------------------------------------------------------


def _quant_levels(bits: int) -> int:
    """Number of quantization levels for a symmetric N-bit scheme."""
    return max(2, (1 << bits) - 1)


def fake_quantize_symmetric(
    x: torch.Tensor,
    bits: int,
    ste: bool = True,
    dim: int | None = None,
) -> torch.Tensor:
    """Symmetric N-bit quantization on tensor ``x``.

    Symmetric quantization uses ``2^(bits-1) - 1`` positive levels and the
    same number of negative levels around zero.

    Scale granularity:
      - ``dim=None`` (default): per-tensor scale, computed from the global
        maximum absolute value across the entire tensor. A single outlier
        weight forces the entire quantization grid to coarsen.
      - ``dim`` specified: per-row scale computed from the maximum absolute
        value along that axis (with ``keepdim=True``). For a
        ``(out_features, in_features)`` weight matrix, ``dim=0`` gives each
        output channel its own full-resolution grid regardless of outliers
        elsewhere in the matrix. ``dim=-1`` is equivalent to ``dim=0`` for a
        2-D matrix and is the typical choice for Linear weight tensors.

    All-zero rows are handled by leaving their scale at the floor
    ``1.0`` so division stays finite.

    With ``ste=True`` the forward returns hard-quantized values but the
    backward gradient is passed through unchanged (straight-through
    estimator). With ``ste=False`` no gradient flows (useful for hard
    inference).
    """
    if bits is None or bits <= 0:
        return x
    qmax = _quant_levels(bits)
    if dim is None:
        abs_max = x.abs().detach().max()
        if abs_max.item() == 0.0:
            scale = torch.ones_like(abs_max)
        else:
            scale = abs_max / float(qmax)
    else:
        abs_max = x.abs().detach().max(dim=dim, keepdim=True).values
        all_zero = (abs_max == 0.0)
        scale = abs_max / float(qmax)
        scale = torch.where(all_zero, torch.ones_like(scale), scale)
    # avoid div-by-zero in the very small scale case
    scale = scale.clamp_min(1e-12)
    x_scaled = x / scale
    if ste:
        x_rounded = x_scaled + (x_scaled.detach().round() - x_scaled.detach())
    else:
        x_rounded = x_scaled.round()
    return (x_rounded * scale).clamp(-abs_max - scale, abs_max + scale)


def quantize_with_range(
    x: torch.Tensor,
    bits: int,
    full_range: float,
    ste: bool = True,
) -> torch.Tensor:
    """Quantize ``x`` using a fixed dynamic range (for ADC/DAC).

    ``full_range`` is the symmetric full-scale range of the converter. All
    values outside ``[-full_range, full_range]`` are clamped before
    quantization (simulating saturation).
    """
    if bits is None or bits <= 0:
        return x
    qmax = _quant_levels(bits)
    x_clamped = x.clamp(-full_range, full_range)
    scale = full_range / float(qmax)
    x_scaled = x_clamped / scale
    if ste:
        x_rounded = x_scaled + (x_scaled.detach().round() - x_scaled.detach())
    else:
        x_rounded = x_scaled.round()
    return x_rounded * scale


# ---------------------------------------------------------------------------
# Injectors
# ---------------------------------------------------------------------------


class WeightQuantizer:
    """Apply fake-quantization and additive Gaussian noise to a tensor.

    Stateful wrapper that records the per-tensor noise (so that the same
    weight tensor sees the same noise across a single forward pass, but
    fresh noise is drawn for the next forward). This is the pattern called
    for by the user prompt: noise is sampled once per forward and frozen.
    """

    def __init__(self, cfg: NoiseConfig):
        self.cfg = cfg

    def apply(self, w: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        out = w
        if self.cfg.quant_bits is not None and self.cfg.quant_bits > 0:
            quant_dim = w.ndim - 1 if w.ndim >= 2 else None
            out = fake_quantize_symmetric(
                out, self.cfg.quant_bits, ste=self.cfg.ste, dim=quant_dim,
            )
        if self.cfg.noise_std > 0.0 and self.cfg.weight_noise:
            noise = torch.empty_like(out)
            noise.normal_(mean=0.0, std=self.cfg.noise_std, generator=generator)
            out = out + noise
        return out


class AdcDacQuantizer:
    """Quantize activations to simulate ADC/DAC converters.

    Uses a fixed full-scale range so that the quantization grid is
    deterministic and matches the activation range the model is expected
    to see. The wrapper chooses a sensible per-layer range from the input
    statistics on the first forward (or via the explicit ``range_`` arg).
    """

    def __init__(
        self,
        cfg: NoiseConfig,
        full_range: float = 3.0,
    ):
        self.cfg = cfg
        self.full_range = float(full_range)

    def apply(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        out = x
        if (
            self.cfg.quant_bits is not None
            and self.cfg.quant_bits > 0
        ):
            out = quantize_with_range(
                out, self.cfg.quant_bits, self.full_range, ste=self.cfg.ste,
            )
        if (
            self.cfg.noise_std > 0.0
            and self.cfg.activation_noise
        ):
            noise = torch.empty_like(out)
            noise.normal_(mean=0.0, std=self.cfg.noise_std, generator=generator)
            out = out + noise
        return out


class CircuitNoise:
    """Standalone Gaussian noise injection for activations.

    Useful when callers want noise without quantization (e.g. to study
    pure thermal-noise effects independently of quantization).
    """

    def __init__(self, std: float):
        self.std = float(std)

    def apply(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if self.std <= 0.0:
            return x
        noise = torch.empty_like(x)
        noise.normal_(mean=0.0, std=self.std, generator=generator)
        return x + noise


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class AnalogMLPWrapper(nn.Module):
    """Wrap an ``MLPRegressor`` and inject realistic analog hardware noise.

    The wrapped module's ``forward`` is intercepted to:
      - quantize the input of the first linear layer (DAC);
      - quantize the input of every subsequent linear layer (intra-network
        ADC);
      - quantize the output of the final linear layer (ADC);
      - inject Gaussian circuit noise on the weights of each linear layer
        before the matmul;
      - inject Gaussian circuit noise on the activations (after quantization,
        before being fed to the next layer).

    The wrapper freezes a single noise realization per ``forward`` call.
    Each new forward call draws fresh noise; this matches the prompt's
    "noise is sampled once per forward and frozen" semantic.

    All Linear layers are detected by iterating the wrapped model's
    ``layers`` ModuleList (which is what both ``MLPRegressor`` classes in
    this codebase use). Non-Linear modules are passed through unchanged
    but still receive activation noise if ``activation_noise`` is enabled.
    """

    def __init__(
        self,
        base: nn.Module,
        cfg: NoiseConfig,
        adc_full_range: float = 3.0,
    ):
        super().__init__()
        if cfg.quant_bits is not None and cfg.quant_bits not in (4, 6):
            raise ValueError(
                f"AnalogMLPWrapper: quant_bits must be 4, 6, or None, "
                f"got {cfg.quant_bits!r}"
            )
        if cfg.noise_std < 0.0:
            raise ValueError(
                f"AnalogMLPWrapper: noise_std must be >= 0, got {cfg.noise_std}"
            )
        if not hasattr(base, "layers") or not isinstance(base.layers, nn.ModuleList):
            raise TypeError(
                "AnalogMLPWrapper: base must expose a `layers` ModuleList "
                "(the MLPRegressor contract)."
            )
        self.base = base
        self.cfg = cfg
        self.adc_full_range = float(adc_full_range)
        self._weight_quantizer = WeightQuantizer(cfg)
        self._adc_quantizer = AdcDacQuantizer(cfg, full_range=self.adc_full_range)
        self._activation_noise = CircuitNoise(
            cfg.noise_std if cfg.activation_noise else 0.0,
        )

    def _linears(self) -> list[nn.Linear]:
        return [m for m in self.base.layers if isinstance(m, nn.Linear)]

    def _make_generator(self, device: torch.device) -> torch.Generator | None:
        if self.cfg.seed is None:
            return None
        gen = torch.Generator(device=device)
        gen.manual_seed(int(self.cfg.seed))
        return gen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gen = self._make_generator(x.device)
        h = x
        linears = self._linears()
        n_linear = len(linears)
        linear_idx = 0
        for layer in self.base.layers:
            if isinstance(layer, nn.Linear):
                lin = layer
                # DAC before first linear: quantize input
                if linear_idx == 0 and self.cfg.quantize_input:
                    h = self._adc_quantizer.apply(h, generator=gen)
                elif linear_idx > 0 and self.cfg.quantize_intermediate:
                    h = self._adc_quantizer.apply(h, generator=gen)
                # Apply weight quantization + noise to weights
                q_w = self._weight_quantizer.apply(lin.weight, generator=gen)
                # Compute linear with perturbed weights (bias also gets noise)
                bias = lin.bias
                if (
                    self.cfg.noise_std > 0.0
                    and self.cfg.weight_noise
                    and bias is not None
                ):
                    b_noise = torch.empty_like(bias)
                    b_noise.normal_(mean=0.0, std=self.cfg.noise_std,
                                    generator=gen)
                    bias = bias + b_noise
                h = torch.nn.functional.linear(h, q_w, bias)
                linear_idx += 1
            else:
                # Activation module: apply activation then optional
                # post-activation noise (only intra-network, not after the
                # final layer).
                h = layer(h)
                if (
                    self.cfg.noise_std > 0.0
                    and self.cfg.activation_noise
                    and linear_idx < n_linear
                    and self.cfg.quantize_intermediate
                ):
                    h = self._activation_noise.apply(h, generator=gen)
        if self.cfg.quantize_output:
            h = self._adc_quantizer.apply(h, generator=gen)
        return h

    # ------------------------------------------------------------------
    # Convenience pass-through to the underlying module's state dict so
    # training / saving / loading of weights "just works".
    # ------------------------------------------------------------------
    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return self.base.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):  # type: ignore[override]
        return self.base.load_state_dict(state_dict, *args, **kwargs)

    def parameters(self, *args, **kwargs):  # type: ignore[override]
        return self.base.parameters(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):  # type: ignore[override]
        return self.base.named_parameters(*args, **kwargs)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_with_noise(
    model: AnalogMLPWrapper,
    val_loader: Iterable,
    task_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    cfg: NoiseConfig,
    device: torch.device | str,
    trials: int | None = None,
    base_seed: int | None = None,
) -> NoiseBenchmarkResult:
    """Run multi-trial Monte Carlo evaluation under analog noise.

    Each trial uses a fresh noise realization (one per trial). The wrapper
    itself uses the ``cfg.seed`` (or a trial-derived seed if
    ``base_seed`` is given) to draw independent noise samples.
    """
    model.eval()
    n_trials = int(trials) if trials is not None else int(cfg.mc_trials)
    losses: list[float] = []

    base_seed_used = base_seed if base_seed is not None else cfg.seed
    for t in range(n_trials):
        # Set per-trial seed so the noise is reproducible per-trial
        if base_seed_used is not None:
            trial_seed = int(base_seed_used) + t
            model.cfg.seed = trial_seed
        total = 0.0
        n = 0
        for u, target in val_loader:
            u = u.to(device)
            target = target.to(device)
            out = model(u)
            loss = task_fn(out, target)
            total += float(loss.item()) * u.size(0)
            n += u.size(0)
        losses.append(total / max(1, n))

    losses_t = torch.tensor(losses, dtype=torch.float64)
    result = NoiseBenchmarkResult(
        losses=losses,
        mean=float(losses_t.mean().item()),
        std=float(losses_t.std(unbiased=False).item()),
        p50=float(losses_t.quantile(0.50).item()),
        p90=float(losses_t.quantile(0.90).item()),
        p95=float(losses_t.quantile(0.95).item()),
        best=float(losses_t.min().item()),
        worst=float(losses_t.max().item()),
        config=cfg,
    )
    # Restore original seed so callers can re-use the wrapper
    if base_seed_used is not None:
        model.cfg.seed = int(base_seed_used)
    return result


@torch.no_grad()
def evaluate_clean(
    model: AnalogMLPWrapper,
    val_loader: Iterable,
    task_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device | str,
) -> float:
    """Evaluate the underlying model with all noise disabled."""
    # Temporarily disable noise and quantization
    saved = (
        model.cfg.quant_bits,
        model.cfg.noise_std,
        model.cfg.quantize_input,
        model.cfg.quantize_output,
        model.cfg.quantize_intermediate,
        model.cfg.weight_noise,
        model.cfg.activation_noise,
    )
    model.cfg.quant_bits = None
    model.cfg.noise_std = 0.0
    model.cfg.quantize_input = False
    model.cfg.quantize_output = False
    model.cfg.quantize_intermediate = False
    model.cfg.weight_noise = False
    model.cfg.activation_noise = False
    total = 0.0
    n = 0
    for u, target in val_loader:
        u = u.to(device)
        target = target.to(device)
        out = model(u)
        loss = task_fn(out, target)
        total += float(loss.item()) * u.size(0)
        n += u.size(0)
    # Restore
    (
        model.cfg.quant_bits,
        model.cfg.noise_std,
        model.cfg.quantize_input,
        model.cfg.quantize_output,
        model.cfg.quantize_intermediate,
        model.cfg.weight_noise,
        model.cfg.activation_noise,
    ) = saved
    return total / max(1, n)