"""Per-edge current source library for the reduced differential KirchhoffNet.

Provides simple per-edge devices that compute edge currents directly from
the source and destination node voltages without cell-type selection:

  - SimpleEdgeLibrary:  I = ReLU(p0*Vsrc + p1*Vdest + p2)  or  I = tanh(...)
  - RealisticTanhLibrary:  I = tanh(A*Vsrc - B*Vdest + C), A+B=1
  - RealisticTanhUpgradeLibrary:  I = Isat * tanh(gm*(A*Vsrc - B*Vdest) + C)
  - FreeTanhLibrary:  I = Isat * tanh(gm*(s*(A*Vsrc - B*Vdest) + theta))

All devices are single-cell (no library selection). Compliance gating
turns the edge off as |x_src| or |x_dst| approaches x_max.
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
    "SimpleEdgeLibrary",
    "RealisticTanhLibrary",
    "RealisticTanhUpgradeLibrary",
    "FreeTanhLibrary",
    "make_cell_library",
]


class SimpleEdgeLibrary(nn.Module):
    """Single-cell edge device with per-edge learnable parameters.

    Two modes:
      - ``mode="relu"``:  I = ReLU(p0 * Vsrc + p1 * Vdest + p2)
      - ``mode="tanh"``:  I = tanh(p0 * Vsrc + p1 * Vdest + p2)

    Holds a single ``nn.Parameter`` of shape ``[3, E]`` for per-edge
    learnable weights. Compliance gating (src/dst rail clamp) is applied.
    """

    def __init__(self, num_edges: int, mode: str) -> None:
        super().__init__()
        if mode not in ("relu", "tanh"):
            raise ValueError(f"SimpleEdgeLibrary mode must be 'relu' or 'tanh', got {mode!r}")
        self._mode = mode
        self._act = torch.relu if mode == "relu" else torch.tanh
        self.param = nn.Parameter(torch.randn(3, num_edges))
        self._beta_softness = float(PHYS["beta_softness"])

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        x_max: float,
    ) -> torch.Tensor:
        batch, num_edges = x_src.shape
        u = (self.param[0].unsqueeze(0) * x_src
             + self.param[1].unsqueeze(0) * x_dst
             + self.param[2].unsqueeze(0))
        i_cell = self._act(u)

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

    Compliance gating (src/dst rail clamp) is applied after the tanh evaluation.
    """

    def __init__(self, num_edges: int, bias_enabled: bool = False) -> None:
        super().__init__()
        self._bias_enabled = bool(bias_enabled)
        self.alpha_raw = nn.Parameter(torch.randn(num_edges))
        if self._bias_enabled:
            self.bias_raw = nn.Parameter(torch.zeros(num_edges))
        self._beta_softness = float(PHYS["beta_softness"])

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        x_max: float,
    ) -> torch.Tensor:
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

    Compliance gating applied after tanh.
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

        self._beta_softness = float(PHYS["beta_softness"])

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        x_max: float,
    ) -> torch.Tensor:
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
      - ``b_raw``   → B = softplus(b_raw), B >= 0, no upper bound, no sum constraint.
      - ``s_raw``   → s = sign(s_raw) (forward discrete ±1; backward uses STE).
      - ``gm_raw``  → gm = gm_min + (gm_max - gm_min) * sigmoid(gm_raw).
      - ``isat_raw``→ Isat = clamp_min(isat_min + (isat_max - isat_min) * sigmoid(isat_raw), 1e-6).
      - ``theta_raw``→ theta; only present when ``bias_enabled=True`` (init=0).

    Boundary defaults come from ``config.TANH_REALISTIC_*``.
    Compliance gating applied after tanh.
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

        self._beta_softness = float(PHYS["beta_softness"])

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        x_max: float,
    ) -> torch.Tensor:
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
) -> SimpleEdgeLibrary | RealisticTanhLibrary | RealisticTanhUpgradeLibrary | FreeTanhLibrary:
    """Factory: returns the appropriate edge-library class.

    ``relu`` / ``tanh`` → ``SimpleEdgeLibrary``.
    ``tanh_realistic`` → ``RealisticTanhLibrary``.
    ``tanh_realistic_upgrade`` → ``RealisticTanhUpgradeLibrary``.
    ``tanh_free`` → ``FreeTanhLibrary``.

    The factory reads ``BIAS_ENABLED`` from the corresponding
    ``CELL_LIBRARIES[library_name]`` entry (default ``False``) and passes
    it to ``RealisticTanhLibrary`` / ``RealisticTanhUpgradeLibrary`` /
    ``FreeTanhLibrary``.

    When ``num_edges`` is ``None`` (template), the returned library
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
    raise ValueError(
        f"Unknown cell library: {library_name!r}. "
        f"Available: 'relu', 'tanh', 'tanh_realistic', "
        f"'tanh_realistic_upgrade', 'tanh_free'."
    )
