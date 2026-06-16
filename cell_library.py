"""Idealized tanh-based cell library for the reduced differential KirchhoffNet.

Provides IdealizedCellLibrary that computes edge currents from differential
drive variable u = x_src - rho * x_dst, with soft library selection over
L (weak linear), S (saturating tanh), P (smooth bounded rectifier), and
Z (disabled) cell families.

Standard cell formula (L, S, Z):
    i(u) = isat * tanh((gm * u + bias) / isat) + gleak * u

Rectifier cell formula (P):
    i_P(u) = isat * tanh(gm * softplus((u - theta) / beta) / isat)
This gives directionality, thresholding, bounded current, differentiability,
and physical plausibility as an active rail-powered branch.

Cell parameters come from config.CELL_LIBRARY via cells_to_tensor_dict().
Compliance gating turns the edge off as |x_src| or |x_dst] approaches x_max.
PVT + mismatch from SimContext is injected multiplicatively on gm.

Multiplicity (R3.3): m = softplus(raw_mult), not 1 + softplus(raw_mult).
A fully Z-selected edge thus has m → 0 in addition to p_Z → 1, so the
weighted power/area proxy in train.py can be driven to zero by both
factors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    CELL_LIBRARY,
    CELL_ORDER,
    NUM_CELLS,
    PHYS,
    cells_to_tensor_dict,
)


__all__ = ["IdealizedCellLibrary"]


class IdealizedCellLibrary(nn.Module):
    """Tanh-surrogate edge cell library with smooth rectifier support.

    Buffers: gm, isat, rho, gleak, bias, theta, beta, each shape [Q].
    theta/beta are only used by rectifier cells; other cells carry neutral
    dummy values. The _is_rect buffer flags which cells use the rectifier
    formula vs the standard tanh formula.
    """

    def __init__(
        self,
        cell_overrides: dict | None = None,
        beta_softness: float | None = None,
    ) -> None:
        super().__init__()
        tensors = cells_to_tensor_dict()
        if cell_overrides is not None:
            for cell_name, overrides in cell_overrides.items():
                idx = CELL_ORDER.index(cell_name)
                for k, v in overrides.items():
                    tensors[k][idx] = float(v)
        for k, t in tensors.items():
            self.register_buffer(k, t)
        if beta_softness is None:
            beta_softness = PHYS["beta_softness"]
        self.beta_softness = float(beta_softness)

        # Per-cell rectifier mask (True for cells that use the smooth bounded
        # rectifier formula). Derived from cell name: cells containing 'P' or
        # 'N' (case-insensitive) are treated as rectifiers.
        is_rect = torch.tensor(
            [c.upper() in ("P", "N") for c in CELL_ORDER],
            dtype=torch.bool,
        )
        self.register_buffer("_is_rect", is_rect)
        self._has_rect = bool(is_rect.any().item())

    @property
    def num_cells(self) -> int:
        return int(self.gm.numel())

    def gm_values(self) -> torch.Tensor:
        """Return per-cell gm (transconductance) values, shape [Q]."""
        return self.gm.detach()

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        logits: torch.Tensor,
        raw_mult: torch.Tensor,
        x_max: float,
        ctx,  # SimContext
        tau: float = 1.0,
        cell_mode: str = "soft",
    ) -> torch.Tensor:
        """Compute edge currents for one stage call.

        Args:
            x_src: [batch, num_edges] source-side differential voltages.
            x_dst: [batch, num_edges] destination-side differential voltages.
            logits: [num_edges, Q] library logits.
            raw_mult: [num_edges] raw multiplicity pre-softplus.
            x_max: Differential rail limit (V).
            ctx: SimContext with optional mismatch.
            tau: Soft library selection temperature.
            cell_mode: Cell selection mode. ``'soft'`` (default) uses
                standard softmax weighting — a mixture of cells per edge.
                ``'ste'`` (straight-through estimator, four-phase-redesign)
                uses one cell per edge in the forward pass
                (``one_hot(argmax(softmax(logits/tau)))``) and routes the
                backward pass through the soft weights via
                ``w_hard + w_soft - w_soft.detach()``. This produces a
                discrete deployable model while preserving the smooth
                gradient signal needed for training.

        Returns:
            i_edge: [batch, num_edges] edge currents in uA.
        """
        if cell_mode not in ("soft", "ste"):
            raise ValueError(
                f"cell_mode must be 'soft' or 'ste', got {cell_mode!r}"
            )

        batch, num_edges = x_src.shape
        Q = self.num_cells

        w_soft = F.softmax(logits / tau, dim=-1)  # [E, Q]

        # four-phase-redesign/Phase 2a: straight-through hard selection.
        # Forward uses a one-hot hard selection; backward sees soft grads.
        if cell_mode == "ste":
            idx = w_soft.argmax(dim=-1)  # [E]
            w_hard = F.one_hot(idx, num_classes=Q).to(w_soft.dtype)  # [E, Q]
            weights = w_hard + w_soft - w_soft.detach()  # STE: hard fwd, soft bwd
        else:
            weights = w_soft

        mult = F.softplus(raw_mult)  # [E] (R3.3: m → 0 as raw_mult → -∞)
        mult = mult.unsqueeze(0).unsqueeze(-1)  # [1, E, 1]

        gm = self.gm.view(1, 1, Q)  # [1, 1, Q]
        isat = self.isat.view(1, 1, Q).clamp_min(1e-6)  # [1, 1, Q]
        rho = self.rho.view(1, 1, Q)  # [1, 1, Q]
        gleak = self.gleak.view(1, 1, Q)  # [1, 1, Q]
        bias = self.bias.view(1, 1, Q)  # [1, 1, Q]

        if ctx is not None and ctx.edge_mismatch is not None:
            em = ctx.edge_mismatch
            if em.device != gm.device:
                em = em.to(gm.device)
            gm = gm * torch.exp(em).unsqueeze(0)  # [1, E, Q]
        if ctx is not None and getattr(ctx, "global_gain_shift", 0.0) != 0.0:
            shift = torch.tensor(ctx.global_gain_shift, dtype=gm.dtype, device=gm.device)
            gm = gm * torch.exp(shift)

        u = x_src.unsqueeze(-1) - rho * x_dst.unsqueeze(-1)  # [B, E, Q]

        # Standard tanh formula (L, S, Z): i = isat*tanh((gm*u+bias)/isat) + gleak*u
        i_standard = isat * torch.tanh((gm * u + bias) / isat) + gleak * u

        if self._has_rect:
            # Smooth bounded rectifier formula (P):
            # i_P = isat * tanh(gm * softplus((u - theta) / beta) / isat)
            theta = self.theta.view(1, 1, Q)  # [1, 1, Q]
            beta = self.beta.view(1, 1, Q).clamp_min(1e-3)  # [1, 1, Q]
            inner = F.softplus((u - theta) / beta)
            i_rect = isat * torch.tanh(gm * inner / isat)
            is_rect = self._is_rect.view(1, 1, Q)
            i_cell = torch.where(is_rect, i_rect, i_standard)
        else:
            i_cell = i_standard

        gate_src = torch.sigmoid((x_max - x_src.abs()) / self.beta_softness)  # [B, E]
        gate_dst = torch.sigmoid((x_max - x_dst.abs()) / self.beta_softness)  # [B, E]
        gate = (gate_src * gate_dst).unsqueeze(-1)  # [B, E, 1]

        weights_b = weights.unsqueeze(0)  # [1, E, Q]
        i_edge = (mult * weights_b * gate * i_cell).sum(dim=-1)  # [B, E]
        return i_edge

    def compile_forward(self, backend: str | None = None):
        """Compile `forward` with `torch.compile` for kernel fusion.

        Fuses softmax/tanh/sigmoid/softplus/elementwise math for ~1.3-2×
        throughput on T4 Tensor Cores.
        """
        self.forward = torch.compile(self.forward, backend=backend)


def make_default_library() -> IdealizedCellLibrary:
    """Convenience constructor that uses config defaults verbatim."""
    return IdealizedCellLibrary()
