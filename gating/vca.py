"""VCA (voltage-controlled amplifier) edge gain.

gain_e(u) = 2 * sigmoid((u @ W) @ v_e.T [+ b_e])   shape [B, E]

``W`` is the shared (in_dim, rank) projection, ``v_e`` the (E, rank) per-edge embedding and
``b_e`` an optional per-edge bias. With W = 0 (the init) the gain is exactly 1.

Hardware (Phase 3a 3.10): ``bits`` quantises the logit ``zeta_e`` on ``[-zeta_max, zeta_max]``
with a straight-through estimator (the DAC of the bus mixer) and ``isat_eff_scale`` keeps the
effective saturation current ``Isat * sigma(z) * g_vca`` inside the hardware ceiling.
"""
from __future__ import annotations

import torch

from hw.constraints import ISAT_HW_MAX, ZETA_MAX, vca_quantize

__all__ = ["vca_edge_gain", "isat_eff_scale"]


def vca_edge_gain(u: torch.Tensor, W: torch.Tensor, v_e: torch.Tensor,
                  bias: torch.Tensor | None = None,
                  bits: int | None = None,
                  zeta_max: float = ZETA_MAX) -> torch.Tensor:
    u_proj = u @ W                 # [B, rank]
    logits = u_proj @ v_e.T        # [B, E]
    if bias is not None:
        logits = logits + bias
    if bits is not None:
        # 3.10: the bus mixer drives a DAC -> quantise zeta (straight-through).
        logits = vca_quantize(logits, bits, zeta_max)
    return 2.0 * torch.sigmoid(logits)


def isat_eff_scale(isat: torch.Tensor, sigma_z: torch.Tensor,
                   g_vca: torch.Tensor | None = None,
                   hi: float = ISAT_HW_MAX) -> torch.Tensor:
    """Scale that keeps ``Isat_eff = Isat * sigma(z) * g_vca`` at or below ``hi``.

    Returns ``clamp(Isat_eff, 0, hi) / Isat_eff`` (1 when nothing has to be clipped), so the
    caller can multiply the edge current by it.  ``isat`` is ``[E]``, ``sigma_z`` is ``[E]`` and
    ``g_vca`` is ``[B, E]`` (or ``None``).
    """
    eff = isat * sigma_z
    if g_vca is not None:
        eff = eff * g_vca
    eff = eff.clamp_min(1e-12)
    return eff.clamp(max=float(hi)) / eff
