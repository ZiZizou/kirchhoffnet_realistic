"""Realistic hardware noise injection for the KirchhoffNet.

Analog of ``analog_noise.AnalogMLPWrapper`` but for the KirchhoffNet pipeline
(write/evolve/read). Models three noise sources that mirror the MLP benchmark
noise model where applicable:

  1. ADC/DAC Quantization: quantize the digital input before the
     InputMapper (DAC) and the analog output after the OutputMapper (ADC).
     The mappers themselves are linear (resistor-network-style weighted sum
     implementations); quantization lives at the digital boundaries.
  2. Circuit Noise: additive Gaussian noise on the state voltages after
     each ODE stage (thermal noise / IR-drop on the analog rails) plus
     on the output mapper output before ADC quantization.
  3. Process Variation is intentionally NOT included here; it lives in
     ``sim_context.SimContext`` and the ``--variation`` flag in
     ``train_script.py``. Circuit noise and process variation compose
     naturally: variation perturbs cell parameters, circuit noise perturbs
     state voltages.

Weight quantization is intentionally NOT implemented because KirchhoffNet
does not learn continuous weight matrices per edge -- it selects from a
small set of fixed OTA designs (L/S/P/Z or v15/v2) via per-edge logits.
The effective "weight" is the gm of the chosen cell, not a learned real.

Design notes:
- All noise is sampled once per forward pass and reused across the entire
  forward. Resampling inside intermediate layers would model time-varying
  noise (stochastic ODE) which the Heun solver does not support.
- ADC/DAC quantization uses a fixed symmetric full-scale range so the
  quantization grid is deterministic and matches the activation range the
  model is expected to see.
- The wrapper delegates ``parameters()``, ``named_parameters()``,
  ``state_dict()``, and ``load_state_dict()`` to the underlying model so
  training "just works" (optimizers target the base model directly).
- ``stage_noise_std`` is threaded through ``KirchhoffNetWithIO.forward()``
  into ``KirchhoffNet.forward()`` where one fresh Gaussian sample per
  ODE stage is added to the state vector. Within a stage, the sample is
  frozen across all Heun substeps (does not break the deterministic ODE
  assumption; matches ``noise_notes.md`` guidance).

Typical usage:

.. code-block:: python

    from kirchhoff_noise import KirchhoffNetNoiseWrapper, evaluate_kirchhoff_with_noise
    from analog_noise import NoiseConfig

    base = build_kirchhoff_net(...)
    base.load_state_dict(torch.load("model.pt"))
    cfg = NoiseConfig(quant_bits=4, noise_std=0.05, mc_trials=20)
    noisy = KirchhoffNetNoiseWrapper(base, cfg)
    result = evaluate_kirchhoff_with_noise(
        noisy, val_loader, task_fn, ctx_factory, cfg, device,
    )
    print(result.mean, result.std, result.p90)
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

import torch
import torch.nn as nn

from analog_noise import (
    AdcDacQuantizer,
    NoiseBenchmarkResult,
    NoiseConfig,
)


__all__ = [
    "KirchhoffNetNoiseWrapper",
    "evaluate_kirchhoff_with_noise",
    "evaluate_kirchhoff_clean",
]


class KirchhoffNetNoiseWrapper(nn.Module):
    """Wrap a ``KirchhoffNetWithIO`` and inject analog hardware noise.

    The wrapped module's ``forward`` is intercepted to:
      - quantize the digital input ``u`` before the InputMapper (DAC);
      - thread ``stage_noise_std`` into the KirchhoffNet core so each ODE
        stage's final state vector has additive Gaussian noise
        (post-stage circuit noise on the analog voltage rails);
      - add Gaussian circuit noise to the OutputMapper output;
      - quantize the analog output ``y`` to digital (ADC).

    The wrapper freezes a single noise realization per ``forward`` call.
    Each new forward call draws fresh noise (stage-noise stays frozen for
    all Heun substeps within a stage).

    The wrapper does NOT modify the cell parameter variation (SimContext);
    variation and noise are orthogonal and compose.
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
                f"KirchhoffNetNoiseWrapper: quant_bits must be 4, 6, or None, "
                f"got {cfg.quant_bits!r}"
            )
        if cfg.noise_std < 0.0:
            raise ValueError(
                f"KirchhoffNetNoiseWrapper: noise_std must be >= 0, "
                f"got {cfg.noise_std}"
            )
        self.base = base
        self.cfg = cfg
        self.adc_full_range = float(adc_full_range)
        self._adc_quantizer = AdcDacQuantizer(cfg, full_range=self.adc_full_range)
        # stage_noise_std drives the per-stage Gaussian injection inside
        # the KirchhoffNet core; we mirror cfg.noise_std here.
        self._stage_noise_std = float(
            cfg.noise_std if cfg.activation_noise else 0.0
        )
        # output_noise is the additive Gaussian on the output mapper output
        # BEFORE ADC quantization. Mirrors cfg.noise_std / weight_noise flag.
        self._output_noise_std = float(
            cfg.noise_std if cfg.weight_noise else 0.0
        )

    def _make_generator(self, device: torch.device) -> torch.Generator | None:
        """Build a per-call torch.Generator seeded from ``self.cfg.seed``.

        Mirrors the ``AnalogMLPWrapper`` contract: when ``cfg.seed`` is
        ``None`` we fall back to the global RNG (non-deterministic). When
        it is set, every ``forward`` call draws independent Gaussian
        samples from a Generator seeded with that value, which lets MC
        trials be independently reproducible (set ``wrapper.cfg.seed = N``
        before each ``forward`` to realize trial ``N``).
        """
        if self.cfg.seed is None:
            return None
        gen = torch.Generator(device=device)
        gen.manual_seed(int(self.cfg.seed))
        return gen

    def forward(
        self,
        u: torch.Tensor,
        ctx,
        tau: float | None = None,
        store_trajectory: bool = False,
        cell_mode: str = "soft",
        solver: str = "heun",
        deq_cfg: dict | None = None,
    ):
        gen = self._make_generator(u.device)
        h = u
        # DAC: quantize digital input before analog conversion. The
        # AdcDacQuantizer also injects activation noise internally if
        # cfg.noise_std > 0 and cfg.activation_noise; pass gen so that
        # noise is drawn from the same seeded stream.
        if (
            self.cfg.quant_bits is not None
            and self.cfg.quant_bits > 0
            and self.cfg.quantize_input
        ):
            h = self._adc_quantizer.apply(h, generator=gen)
        # Run base KirchhoffNetWithIO with stage_noise_std threaded in.
        # The base injects one Gaussian sample per ODE stage when
        # stage_noise_std > 0; we draw those samples from the same
        # seeded generator so the trial is reproducible end-to-end.
        y, trajs = self.base(
            h, ctx=ctx, tau=tau, store_trajectory=store_trajectory,
            cell_mode=cell_mode, solver=solver, deq_cfg=deq_cfg,
            stage_noise_std=self._stage_noise_std,
            stage_noise_generator=gen,
        )
        # Output circuit noise before ADC.
        if self._output_noise_std > 0.0:
            noise = torch.empty_like(y)
            noise.normal_(mean=0.0, std=self._output_noise_std, generator=gen)
            y = y + noise
        # ADC: quantize analog output to digital.
        if (
            self.cfg.quant_bits is not None
            and self.cfg.quant_bits > 0
            and self.cfg.quantize_output
        ):
            y = self._adc_quantizer.apply(y, generator=gen)
        return y, trajs

    # ------------------------------------------------------------------
    # Convenience pass-through to the underlying module's state dict so
    # training / saving / loading of weights "just works". This mirrors
    # the AnalogMLPWrapper contract.
    # ------------------------------------------------------------------
    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return self.base.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):  # type: ignore[override]
        return self.base.load_state_dict(state_dict, *args, **kwargs)

    def parameters(self, *args, **kwargs):  # type: ignore[override]
        return self.base.parameters(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):  # type: ignore[override]
        return self.base.named_parameters(*args, **kwargs)


def _kirchhoff_forward(
    wrapper: KirchhoffNetNoiseWrapper,
    u: torch.Tensor,
    ctx,
) -> torch.Tensor:
    """Run the wrapped KirchhoffNet forward once with the current wrapper state.

    Returns just the output tensor (discards trajectories).
    """
    y, _ = wrapper(u, ctx=ctx)
    return y


@torch.no_grad()
def evaluate_kirchhoff_with_noise(
    wrapper: KirchhoffNetNoiseWrapper,
    val_loader: Iterable,
    task_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ctx_factory: Callable,
    cfg: NoiseConfig,
    device: torch.device | str,
    trials: int | None = None,
    base_seed: int | None = None,
) -> NoiseBenchmarkResult:
    """Run multi-trial Monte Carlo evaluation under KirchhoffNet analog noise.

    Each trial uses a fresh noise realization (one per trial). Process
    variation (SimContext) is re-sampled per batch from ``ctx_factory``;
    circuit noise is re-sampled per trial via ``wrapper.cfg.seed``.

    Args:
        wrapper: KirchhoffNetNoiseWrapper around a trained KirchhoffNetWithIO.
        val_loader: DataLoader yielding (u, target) batches.
        task_fn: Loss function ``task_fn(out, target)`` returning a scalar.
        ctx_factory: Callable ``ctx_factory(batch_size, device=...)`` returning
            a SimContext for the batch. Use ``make_static_ctx_factory()`` for
            clean (no-variation) evaluation, or a variation-aware factory for
            combined variation+noise robustness studies.
        cfg: NoiseConfig controlling quant/noise magnitudes.
        device: Device for tensors.
        trials: Number of MC trials (defaults to ``cfg.mc_trials``).
        base_seed: If provided, each trial uses ``base_seed + trial_idx`` as
            noise seed for reproducibility. Defaults to ``cfg.seed``.

    Returns:
        ``NoiseBenchmarkResult`` with per-trial losses plus summary stats.
    """
    wrapper.eval()
    n_trials = int(trials) if trials is not None else int(cfg.mc_trials)
    losses: list[float] = []

    base_seed_used = base_seed if base_seed is not None else cfg.seed
    original_seed = wrapper.cfg.seed
    for t in range(n_trials):
        if base_seed_used is not None:
            wrapper.cfg.seed = int(base_seed_used) + t
        total = 0.0
        n = 0
        for u, target in val_loader:
            u = u.to(device)
            target = target.to(device)
            ctx = ctx_factory(u.size(0), device=device)
            out = _kirchhoff_forward(wrapper, u, ctx)
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
    if base_seed_used is not None:
        wrapper.cfg.seed = int(original_seed) if original_seed is not None else original_seed
    return result


@torch.no_grad()
def evaluate_kirchhoff_clean(
    wrapper: KirchhoffNetNoiseWrapper,
    val_loader: Iterable,
    task_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ctx_factory: Callable,
    device: torch.device | str,
) -> float:
    """Evaluate the underlying KirchhoffNet with all noise disabled.

    Uses a clean SimContext (if ctx_factory is variation-aware, callers
    should pass a clean-context factory). Temporarily disables quantization
    and circuit noise on the wrapper, runs a single pass, and restores
    the wrapper's original config.
    """
    saved = (
        wrapper.cfg.quant_bits,
        wrapper.cfg.noise_std,
        wrapper.cfg.quantize_input,
        wrapper.cfg.quantize_output,
        wrapper.cfg.quantize_intermediate,
        wrapper.cfg.weight_noise,
        wrapper.cfg.activation_noise,
    )
    wrapper.cfg.quant_bits = None
    wrapper.cfg.noise_std = 0.0
    wrapper.cfg.quantize_input = False
    wrapper.cfg.quantize_output = False
    wrapper.cfg.quantize_intermediate = False
    wrapper.cfg.weight_noise = False
    wrapper.cfg.activation_noise = False
    # Re-derive internal noise stds from disabled config.
    wrapper._stage_noise_std = 0.0
    wrapper._output_noise_std = 0.0
    total = 0.0
    n = 0
    for u, target in val_loader:
        u = u.to(device)
        target = target.to(device)
        ctx = ctx_factory(u.size(0), device=device)
        out = _kirchhoff_forward(wrapper, u, ctx)
        loss = task_fn(out, target)
        total += float(loss.item()) * u.size(0)
        n += u.size(0)
    (
        wrapper.cfg.quant_bits,
        wrapper.cfg.noise_std,
        wrapper.cfg.quantize_input,
        wrapper.cfg.quantize_output,
        wrapper.cfg.quantize_intermediate,
        wrapper.cfg.weight_noise,
        wrapper.cfg.activation_noise,
    ) = saved
    wrapper._stage_noise_std = float(
        saved[1] if saved[6] else 0.0,
    )
    wrapper._output_noise_std = float(
        saved[1] if saved[5] else 0.0,
    )
    return total / max(1, n)