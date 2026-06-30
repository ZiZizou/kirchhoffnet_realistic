"""Idealized tanh-based cell library for the reduced differential KirchhoffNet.

Provides IdealizedCellLibrary that computes edge currents from differential
drive variable u = x_src - rho * x_dst, with soft library selection over a
set of physically motivated bounded branch macros.

Four formula types:
  - standard:     i = isat * tanh((gm * u + bias) / isat) + gleak * u
  - pos_rect:     i = isat * tanh(gm * softplus((u - theta) / beta) / isat)
  - neg_rect:     i = -isat * tanh(gm * softplus((-u - theta) / beta) / isat)
  - dead_zone:    i = isat * tanh(gm * softplus((u - theta)/beta) / isat)
                    - isat * tanh(gm * softplus((-u - theta)/beta) / isat)

Legacy library:  L (standard, low gm), S (standard, high gm), P (pos_rect), Z (off)
v1.5 library:    O_weak (standard, low gm), O_hard (standard, high gm),
                 P0 (pos_rect), N0 (neg_rect), D1 (dead_zone), Z (off)

Cell parameters come from config via cells_to_tensor_dict(library_name).
Compliance gating turns the edge off as |x_src| or |x_dst| approaches x_max.
PVT + mismatch from SimContext is injected multiplicatively on gm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    CELL_LIBRARIES,
    PHYS,
    TANH_REALISTIC_GM_MAX,
    TANH_REALISTIC_GM_MIN,
    TANH_REALISTIC_ISAT_MAX,
    TANH_REALISTIC_ISAT_MIN,
    cells_to_tensor_dict,
)


__all__ = [
    "IdealizedCellLibrary",
    "RealisticTanhLibrary",
    "RealisticTanhUpgradeLibrary",
    "FreeTanhLibrary",
    "SimpleEdgeLibrary",
    "make_cell_library",
]


class IdealizedCellLibrary(nn.Module):
    """Edge cell library with formula dispatch for standard, pos/neg rectifier,
    and dead-zone cell types.

    Buffers: gm, isat, gleak, bias, theta, beta, cell_type_code, each
    shape [Q]. cell_type_code stores integer codes for formula dispatch.

    Preactivation coefficients (one of, not both):
      - ``rho`` (legacy/v15): single destination gain; u = x_src - rho * x_dst.
      - ``src_gain`` / ``dst_gain`` (v2): per-cell mix; u = src_gain*x_src - dst_gain*x_dst.

    The presence of the ``src_gain`` buffer (set by ``cells_to_tensor_dict``
    based on the library name) selects mix mode. ``_use_mix`` is True iff
    src_gain/dst_gain are present.
    """

    def __init__(
        self,
        cell_overrides: dict | None = None,
        beta_softness: float | None = None,
        library_name: str = "legacy",
    ) -> None:
        super().__init__()
        self.library_name = library_name
        lib_cfg = CELL_LIBRARIES[library_name]
        self._cell_order = list(lib_cfg["cell_order"])
        self.z_index = len(self._cell_order) - 1

        tensors = cells_to_tensor_dict(library_name=library_name)
        if cell_overrides is not None:
            for cell_name, overrides in cell_overrides.items():
                idx = self._cell_order.index(cell_name)
                for k, v in overrides.items():
                    if k in tensors:
                        tensors[k][idx] = float(v)
        for k, t in tensors.items():
            self.register_buffer(k, t)
        if beta_softness is None:
            beta_softness = PHYS["beta_softness"]
        self.beta_softness = float(beta_softness)

        # Preactivation mix mode: v2 libraries carry src_gain/dst_gain
        # buffers (per-cell mix); legacy/v15 carry a single rho buffer.
        self._use_mix = "src_gain" in tensors
        if self._use_mix and "rho" in tensors:
            raise ValueError(
                f"Library {library_name!r} emitted both src_gain and rho; "
                "cells_to_tensor_dict should produce exactly one set."
            )

        # Per-cell type masks for formula dispatch (bool, derived from
        # cell_type_code). Registered as persistent=False buffers so they
        # move with .to(device) but don't pollute state_dict.
        ctc = self.cell_type_code  # [Q]
        self.register_buffer("_is_std", ctc == 0, persistent=False)
        self.register_buffer("_is_pos_rect", ctc == 1, persistent=False)
        self.register_buffer("_is_neg_rect", ctc == 2, persistent=False)
        self.register_buffer("_is_dead_zone", ctc == 3, persistent=False)
        self.register_buffer("_is_rect", (ctc == 1) | (ctc == 2) | (ctc == 3))

    @property
    def num_cells(self) -> int:
        return int(self.gm.numel())

    @property
    def has_z_cell(self) -> bool:
        """Check whether the Z (disabled) cell exists (gm == 0 at z_index)."""
        return self.gm[self.z_index].item() == 0.0

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
                standard softmax weighting. ``'ste'`` uses one cell per edge
                in the forward pass with straight-through soft gradients.

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

        if cell_mode == "ste":
            idx = w_soft.argmax(dim=-1)  # [E]
            w_hard = F.one_hot(idx, num_classes=Q).to(w_soft.dtype)  # [E, Q]
            weights = w_hard + w_soft - w_soft.detach()
        else:
            weights = w_soft

        mult = F.softplus(raw_mult)  # [E]
        mult = mult.unsqueeze(0).unsqueeze(-1)  # [1, E, 1]

        gm = self.gm.view(1, 1, Q)  # [1, 1, Q]
        isat = self.isat.view(1, 1, Q)  # [1, 1, Q]
        gleak = self.gleak.view(1, 1, Q)  # [1, 1, Q]
        bias = self.bias.view(1, 1, Q)  # [1, 1, Q]

        if ctx is not None and ctx.edge_mismatch is not None:
            em = ctx.edge_mismatch
            if em.device != gm.device or em.dtype != gm.dtype:
                em = em.to(device=gm.device, dtype=gm.dtype)
            gm = gm * torch.exp(em).unsqueeze(0)  # [1, E, Q]
        if ctx is not None and getattr(ctx, "global_gain_shift", 0.0) != 0.0:
            shift = torch.tensor(
                ctx.global_gain_shift,
                dtype=gm.dtype,
                device=gm.device,
            )
            gm = gm * torch.exp(shift)
        if (
            ctx is not None
            and getattr(ctx, "edge_isat_mismatch", None) is not None
        ):
            em_i = ctx.edge_isat_mismatch
            if em_i.device != isat.device or em_i.dtype != isat.dtype:
                em_i = em_i.to(device=isat.device, dtype=isat.dtype)
            isat = isat * torch.exp(em_i).unsqueeze(0)  # [1, E, Q]
        if ctx is not None and getattr(ctx, "global_isat_shift", 0.0) != 0.0:
            shift_i = torch.tensor(
                ctx.global_isat_shift,
                dtype=isat.dtype,
                device=isat.device,
            )
            isat = isat * torch.exp(shift_i)

        isat = isat.clamp_min(1e-6)

        # Preactivation: v2 uses per-cell src_gain/dst_gain mix; legacy/v15
        # use a single rho destination gain. v2 cells have gleak=0 and bias=0
        # so the standard-formula additive terms are no-ops (correct for v2).
        if self._use_mix:
            src_gain = self.src_gain.view(1, 1, Q)  # [1, 1, Q]
            dst_gain = self.dst_gain.view(1, 1, Q)  # [1, 1, Q]
            u = src_gain * x_src.unsqueeze(-1) - dst_gain * x_dst.unsqueeze(-1)
        else:
            rho = self.rho.view(1, 1, Q)  # [1, 1, Q]
            u = x_src.unsqueeze(-1) - rho * x_dst.unsqueeze(-1)  # [B, E, Q]

        # ---- Compute all formula variants in parallel ----

        # 1. Standard formula: isat * tanh((gm*u + bias) / isat) + gleak*u
        i_standard = isat * torch.tanh((gm * u + bias) / isat) + gleak * u

        # Shared theta/beta for rectifier and dead-zone variants.
        theta = self.theta.view(1, 1, Q)  # [1, 1, Q]
        beta = self.beta.view(1, 1, Q).clamp_min(1e-3)  # [1, 1, Q]

        # 2. Positive rectifier: isat * tanh(gm * softplus((u - theta) / beta) / isat)
        inner_p = F.softplus((u - theta) / beta)
        i_pos_rect = isat * torch.tanh(gm * inner_p / isat)

        # 3. Negative rectifier: -isat * tanh(gm * softplus((-u - theta) / beta) / isat)
        inner_n = F.softplus((-u - theta) / beta)
        i_neg_rect = -isat * torch.tanh(gm * inner_n / isat)

        # 4. Dead zone: pos_rect(u) - neg_rect(u)  (both use the cell's own theta/beta)
        i_dead_zone = isat * torch.tanh(gm * inner_p / isat) - isat * torch.tanh(gm * inner_n / isat)

        # ---- Dispatch via per-cell type masks (inline from buffer for device safety) ----
        ctc = self.cell_type_code.view(1, 1, Q)
        i_cell = torch.where(ctc == 3, i_dead_zone,
                   torch.where(ctc == 2, i_neg_rect,
                   torch.where(ctc == 1, i_pos_rect, i_standard)))

        gate_src = torch.sigmoid((x_max - x_src.abs()) / self.beta_softness)  # [B, E]
        gate_dst = torch.sigmoid((x_max - x_dst.abs()) / self.beta_softness)  # [B, E]
        gate = (gate_src * gate_dst).unsqueeze(-1)  # [B, E, 1]

        weights_b = weights.unsqueeze(0)  # [1, E, Q]
        i_edge = (mult * weights_b * gate * i_cell).sum(dim=-1)  # [B, E]
        return i_edge

    def compile_forward(self, backend: str = "inductor"):
        """Compile `forward` with `torch.compile` for kernel fusion."""
        self.forward = torch.compile(self.forward, backend=backend)


class SimpleEdgeLibrary(nn.Module):
    """Single-cell edge device with per-edge learnable parameters.

    Two modes:
      - ``mode="relu"``:  I = ReLU(p0 * Vsrc + p1 * Vdest + p2)
      - ``mode="tanh"``:  I = tanh(p0 * Vsrc + p1 * Vdest + p2)

    Holds a single ``nn.Parameter`` of shape ``[3, E]`` for per-edge
    learnable weights.  No cell selection, no multiplicity, no Z cell.
    Compliance gating (src/dst rail clamp) is applied.
    """

    def __init__(self, num_edges: int, mode: str) -> None:
        super().__init__()
        if mode not in ("relu", "tanh"):
            raise ValueError(f"SimpleEdgeLibrary mode must be 'relu' or 'tanh', got {mode!r}")
        self._mode = mode
        self._act = torch.relu if mode == "relu" else torch.tanh
        self.param = nn.Parameter(torch.randn(3, num_edges))
        self._cell_order = ["S"]
        self.z_index = 0
        self._beta_softness = float(PHYS["beta_softness"])

    @property
    def num_cells(self) -> int:
        return 1

    @property
    def has_z_cell(self) -> bool:
        """No Z (disabled) cell in this library."""
        return False

    def gm_values(self) -> torch.Tensor:
        """Per-cell gm proxy: shape [1] (Q=1). Returns 0 — no per-cell-type
        gm concept for simple devices; the actual per-edge conductance is
        captured by ``self.param``."""
        return torch.zeros(1, device=self.param.device)

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        logits: torch.Tensor,
        raw_mult: torch.Tensor,
        x_max: float,
        ctx,
        tau: float = 1.0,
        cell_mode: str = "soft",
    ) -> torch.Tensor:
        """Compute edge currents.

        ``logits``, ``raw_mult``, ``tau``, ``cell_mode``, and ``ctx`` are
        ignored (no cell selection / multiplicity for this library).
        """
        del logits, raw_mult, tau, cell_mode, ctx
        batch, num_edges = x_src.shape
        u = (self.param[0].unsqueeze(0) * x_src
             + self.param[1].unsqueeze(0) * x_dst
             + self.param[2].unsqueeze(0))
        i_cell = self._act(u)

        # Compliance gating (same as IdealizedCellLibrary).
        gate_src = torch.sigmoid((x_max - x_src.abs()) / self._beta_softness)
        gate_dst = torch.sigmoid((x_max - x_dst.abs()) / self._beta_softness)
        gate = gate_src * gate_dst

        return gate * i_cell

    def compile_forward(self, backend: str = "inductor"):
        pass


class RealisticTanhLibrary(nn.Module):
    """Per-edge constrained-differential tanh device.

    Per-edge formula::

        I = tanh(A * Vsrc - B * Vdest + C)

    where ``A = sigmoid(alpha_raw)``, ``B = 1 - A`` (so ``A, B > 0`` and
    ``A + B = 1`` exactly), and ``C = bias_raw`` when ``bias_enabled=True``
    (otherwise ``C = 0`` exactly, no parameter, no gradient).

    Holds one or two learnable parameters of shape ``[E]``:
      - ``alpha_raw``: unconstrained source/destination mix coefficient (always present).
      - ``bias_raw``:  additive pre-tanh bias, init=0; only present when
        ``bias_enabled=True``.

    ``bias_enabled`` defaults to ``False``. Set via ``config.py`` library
    entry ``BIAS_ENABLED`` or by passing the kwarg to the constructor.

    Multi-edge parameters (multiplicity / cell logits) are not used:
    ``DifferentialStage`` skips them for simple-edge libraries. Compliance
    gating (src/dst rail clamp) is applied after the tanh evaluation.
    """

    def __init__(self, num_edges: int, bias_enabled: bool = False) -> None:
        super().__init__()
        self._bias_enabled = bool(bias_enabled)
        self.alpha_raw = nn.Parameter(torch.randn(num_edges))
        if self._bias_enabled:
            self.bias_raw = nn.Parameter(torch.zeros(num_edges))
        self._cell_order = ["S"]
        self.z_index = 0
        self._beta_softness = float(PHYS["beta_softness"])

    @property
    def num_cells(self) -> int:
        return 1

    @property
    def has_z_cell(self) -> bool:
        return False

    def gm_values(self) -> torch.Tensor:
        """No per-cell gm concept for this device; returns zeros."""
        return torch.zeros(1, device=self.alpha_raw.device)

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        logits: torch.Tensor,
        raw_mult: torch.Tensor,
        x_max: float,
        ctx,
        tau: float = 1.0,
        cell_mode: str = "soft",
    ) -> torch.Tensor:
        """Compute edge currents.

        ``logits``, ``raw_mult``, ``tau``, ``cell_mode``, and ``ctx`` are
        ignored (single-cell device; no library selection or multiplicity).
        When ``bias_enabled=False`` the bias term is identically zero and
        ``self.bias_raw`` is not referenced (no gradient flows).
        """
        del logits, raw_mult, tau, cell_mode, ctx
        A = torch.sigmoid(self.alpha_raw).unsqueeze(0)  # [1, E]
        B = 1.0 - A
        bias = self.bias_raw.unsqueeze(0) if self._bias_enabled else 0.0
        u = A * x_src - B * x_dst + bias
        i_cell = torch.tanh(u)

        gate_src = torch.sigmoid((x_max - x_src.abs()) / self._beta_softness)
        gate_dst = torch.sigmoid((x_max - x_dst.abs()) / self._beta_softness)
        gate = gate_src * gate_dst

        return gate * i_cell

    def compile_forward(self, backend: str = "inductor"):
        pass


class RealisticTanhUpgradeLibrary(nn.Module):
    """Per-edge saturated realistic tanh device with bounded gain/saturation.

    Per-edge formula::

        I = Isat * tanh(gm * (A * Vsrc - B * Vdest) + C)

    Parameterization (all per-edge, shape ``[E]``):
      - ``alpha_raw`` → A = sigmoid(alpha_raw), B = 1 - A (so A, B > 0, A+B=1).
      - ``gm_raw``    → gm = gm_min + (gm_max - gm_min) * sigmoid(gm_raw).
      - ``isat_raw``  → Isat = isat_min + (isat_max - isat_min) * sigmoid(isat_raw).
      - ``bias_raw``  → C; only present when ``bias_enabled=True`` (init=0).
        When disabled, ``C = 0`` exactly and ``bias_raw`` does not exist.

    Boundary defaults come from ``config.TANH_REALISTIC_*`` and can be
    overridden per-instance via constructor kwargs. Initial ``gm`` and
    ``Isat`` sit at the geometric midpoint (sigmoid(0)=0.5 → (min+max)/2).

    ``bias_enabled`` defaults to ``False``. Set via ``config.py`` library
    entry ``BIAS_ENABLED`` or by passing the kwarg to the constructor.

    No cell selection / multiplicity. Compliance gating applied after tanh.
    """

    def __init__(
        self,
        num_edges: int,
        gm_min: float | None = None,
        gm_max: float | None = None,
        isat_min: float | None = None,
        isat_max: float | None = None,
        bias_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.gm_min = float(gm_min if gm_min is not None else TANH_REALISTIC_GM_MIN)
        self.gm_max = float(gm_max if gm_max is not None else TANH_REALISTIC_GM_MAX)
        self.isat_min = float(isat_min if isat_min is not None else TANH_REALISTIC_ISAT_MIN)
        self.isat_max = float(isat_max if isat_max is not None else TANH_REALISTIC_ISAT_MAX)
        if self.gm_max <= self.gm_min:
            raise ValueError(
                f"RealisticTanhUpgradeLibrary requires gm_max > gm_min, "
                f"got [{self.gm_min}, {self.gm_max}]"
            )
        if self.isat_max <= self.isat_min:
            raise ValueError(
                f"RealisticTanhUpgradeLibrary requires isat_max > isat_min, "
                f"got [{self.isat_min}, {self.isat_max}]"
            )

        self._bias_enabled = bool(bias_enabled)
        self.alpha_raw = nn.Parameter(torch.randn(num_edges))
        self.gm_raw = nn.Parameter(torch.full((num_edges,), -2.3))
        self.isat_raw = nn.Parameter(torch.full((num_edges,), -2.3))
        if self._bias_enabled:
            self.bias_raw = nn.Parameter(torch.zeros(num_edges))

        self._cell_order = ["S"]
        self.z_index = 0
        self._beta_softness = float(PHYS["beta_softness"])

    @property
    def num_cells(self) -> int:
        return 1

    @property
    def has_z_cell(self) -> bool:
        return False

    def gm_values(self) -> torch.Tensor:
        """No per-cell gm concept for this device; returns zeros."""
        return torch.zeros(1, device=self.alpha_raw.device)

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        logits: torch.Tensor,
        raw_mult: torch.Tensor,
        x_max: float,
        ctx,
        tau: float = 1.0,
        cell_mode: str = "soft",
    ) -> torch.Tensor:
        """Compute edge currents.

        ``logits``, ``raw_mult``, ``tau``, ``cell_mode``, and ``ctx`` are
        ignored (single-cell device; no library selection, multiplicity, or
        PVT/mismatch support). When ``bias_enabled=False`` the bias term is
        identically zero and ``self.bias_raw`` is not referenced.
        """
        del logits, raw_mult, tau, cell_mode, ctx
        sig_alpha = torch.sigmoid(self.alpha_raw)
        A = sig_alpha.unsqueeze(0)  # [1, E]
        B = 1.0 - A
        sig_gm = torch.sigmoid(self.gm_raw)
        gm = self.gm_min + (self.gm_max - self.gm_min) * sig_gm  # [E]
        sig_isat = torch.sigmoid(self.isat_raw)
        isat = (self.isat_min + (self.isat_max - self.isat_min) * sig_isat).clamp_min(1e-6)  # [E]
        bias = self.bias_raw.unsqueeze(0) if self._bias_enabled else 0.0
        u = (A * x_src - B * x_dst) * gm.unsqueeze(0) + bias
        i_cell = isat.unsqueeze(0) * torch.tanh(u)

        gate_src = torch.sigmoid((x_max - x_src.abs()) / self._beta_softness)
        gate_dst = torch.sigmoid((x_max - x_dst.abs()) / self._beta_softness)
        gate = gate_src * gate_dst

        return gate * i_cell

    def compile_forward(self, backend: str = "inductor"):
        pass


class FreeTanhLibrary(nn.Module):
    """Per-edge signed-realistic tanh device with independent A/B and STE sign.

    Per-edge formula::

        I = I_sat * tanh(gm * (s * (A * Vsrc - B * Vdest) + theta))

    Parameterization (all per-edge, shape ``[E]``):
      - ``a_raw``   → A = softplus(a_raw), A >= 0, no upper bound, independent of B.
      - ``b_raw``   → B = softplus(b_raw), B >= 0, no upper bound, no sum constraint
                       (A + B is not required to equal 1).
      - ``s_raw``   → s = sign(s_raw) (forward discrete ±1; backward uses a
                       straight-through estimator so grad flows through s_raw).
      - ``gm_raw``  → gm = gm_min + (gm_max - gm_min) * sigmoid(gm_raw).
      - ``isat_raw``→ Isat = clamp_min(isat_min + (isat_max - isat_min) * sigmoid(isat_raw), 1e-6).
      - ``theta_raw``→ theta; only present when ``bias_enabled=True`` (init=0).
                       When disabled, ``theta = 0`` exactly and ``theta_raw`` does
                       not exist (no parameter, no gradient).

    All six raw tensors are registered as ``nn.Parameter`` so the optimizer
    tracks and updates them.

    Boundary defaults come from ``config.TANH_REALISTIC_*`` and can be
    overridden per-instance via constructor kwargs. Initial ``gm`` and
    ``Isat`` sit at the geometric midpoint (sigmoid(0)=0.5 → (min+max)/2).

    ``bias_enabled`` defaults to ``False``. Set via ``config.py`` library
    entry ``BIAS_ENABLED`` or by passing the kwarg to the constructor.

    No cell selection / multiplicity. Compliance gating applied after tanh.
    """

    def __init__(
        self,
        num_edges: int,
        gm_min: float | None = None,
        gm_max: float | None = None,
        isat_min: float | None = None,
        isat_max: float | None = None,
        bias_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.gm_min = float(gm_min if gm_min is not None else TANH_REALISTIC_GM_MIN)
        self.gm_max = float(gm_max if gm_max is not None else TANH_REALISTIC_GM_MAX)
        self.isat_min = float(isat_min if isat_min is not None else TANH_REALISTIC_ISAT_MIN)
        self.isat_max = float(isat_max if isat_max is not None else TANH_REALISTIC_ISAT_MAX)
        if self.gm_max <= self.gm_min:
            raise ValueError(
                f"FreeTanhLibrary requires gm_max > gm_min, "
                f"got [{self.gm_min}, {self.gm_max}]"
            )
        if self.isat_max <= self.isat_min:
            raise ValueError(
                f"FreeTanhLibrary requires isat_max > isat_min, "
                f"got [{self.isat_min}, {self.isat_max}]"
            )

        self._bias_enabled = bool(bias_enabled)
        self.a_raw = nn.Parameter(torch.randn(num_edges))
        self.b_raw = nn.Parameter(torch.randn(num_edges))
        self.s_raw = nn.Parameter(torch.randn(num_edges))
        self.gm_raw = nn.Parameter(torch.full((num_edges,), -2.3))
        self.isat_raw = nn.Parameter(torch.full((num_edges,), -2.3))
        if self._bias_enabled:
            self.theta_raw = nn.Parameter(torch.zeros(num_edges))

        self._cell_order = ["S"]
        self.z_index = 0
        self._beta_softness = float(PHYS["beta_softness"])

    @property
    def num_cells(self) -> int:
        return 1

    @property
    def has_z_cell(self) -> bool:
        return False

    def gm_values(self) -> torch.Tensor:
        """No per-cell gm concept for this device; returns zeros."""
        return torch.zeros(1, device=self.a_raw.device)

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        logits: torch.Tensor,
        raw_mult: torch.Tensor,
        x_max: float,
        ctx,
        tau: float = 1.0,
        cell_mode: str = "soft",
    ) -> torch.Tensor:
        """Compute edge currents.

        ``logits``, ``raw_mult``, ``tau``, ``cell_mode``, and ``ctx`` are
        ignored (single-cell device; no library selection, multiplicity, or
        PVT/mismatch support). When ``bias_enabled=False`` the theta term
        is identically zero and ``self.theta_raw`` is not referenced.
        """
        del logits, raw_mult, tau, cell_mode, ctx
        A = torch.nn.functional.softplus(self.a_raw).unsqueeze(0)  # [1, E]
        B = torch.nn.functional.softplus(self.b_raw).unsqueeze(0)  # [1, E]
        s = torch.sign(self.s_raw)  # [E]
        s_ste = s + self.s_raw - self.s_raw.detach()  # STE: forward ±1, backward grad=1
        sig_gm = torch.sigmoid(self.gm_raw)
        gm = self.gm_min + (self.gm_max - self.gm_min) * sig_gm  # [E]
        sig_isat = torch.sigmoid(self.isat_raw)
        isat = (self.isat_min + (self.isat_max - self.isat_min) * sig_isat).clamp_min(1e-6)  # [E]
        theta = self.theta_raw.unsqueeze(0) if self._bias_enabled else 0.0
        u = (s_ste.unsqueeze(0) * (A * x_src - B * x_dst) + theta) * gm.unsqueeze(0)
        i_cell = isat.unsqueeze(0) * torch.tanh(u)

        gate_src = torch.sigmoid((x_max - x_src.abs()) / self._beta_softness)
        gate_dst = torch.sigmoid((x_max - x_dst.abs()) / self._beta_softness)
        gate = gate_src * gate_dst

        return gate * i_cell

    def compile_forward(self, backend: str = "inductor"):
        pass


def make_cell_library(
    library_name: str, num_edges: int | None = None
) -> IdealizedCellLibrary | SimpleEdgeLibrary | RealisticTanhLibrary | RealisticTanhUpgradeLibrary | FreeTanhLibrary:
    """Factory: returns the appropriate edge-library class.

    Simple per-edge devices (``relu``, ``tanh``, ``tanh_realistic``,
    ``tanh_realistic_upgrade``, ``tanh_free``) return ``SimpleEdgeLibrary``,
    ``RealisticTanhLibrary``, ``RealisticTanhUpgradeLibrary``, or
    ``FreeTanhLibrary`` respectively. ``legacy``/``v15``/``v2`` (and any
    other bounded-macro library registered in ``CELL_LIBRARIES``) return
    ``IdealizedCellLibrary``.

    The factory reads ``BIAS_ENABLED`` from the corresponding
    ``CELL_LIBRARIES[library_name]`` entry (default ``False``) and passes
    it to ``RealisticTanhLibrary`` / ``RealisticTanhUpgradeLibrary`` /
    ``FreeTanhLibrary``.

    When ``num_edges`` is ``None`` (template), the returned simple library
    is created with ``num_edges=1`` and must be replaced with a per-stage
    instance (matching the actual edge count) before use.
    """
    n = num_edges if num_edges is not None else 1
    if library_name in ("relu", "tanh"):
        return SimpleEdgeLibrary(num_edges=n, mode=library_name)
    if library_name == "tanh_realistic":
        bias_enabled = bool(CELL_LIBRARIES[library_name].get("BIAS_ENABLED", False))
        return RealisticTanhLibrary(num_edges=n, bias_enabled=bias_enabled)
    if library_name == "tanh_realistic_upgrade":
        bias_enabled = bool(CELL_LIBRARIES[library_name].get("BIAS_ENABLED", False))
        return RealisticTanhUpgradeLibrary(num_edges=n, bias_enabled=bias_enabled)
    if library_name == "tanh_free":
        bias_enabled = bool(CELL_LIBRARIES[library_name].get("BIAS_ENABLED", False))
        return FreeTanhLibrary(num_edges=n, bias_enabled=bias_enabled)
    return IdealizedCellLibrary(library_name=library_name)


def make_default_library() -> IdealizedCellLibrary:
    """Convenience constructor that uses config defaults verbatim (legacy)."""
    return IdealizedCellLibrary()
