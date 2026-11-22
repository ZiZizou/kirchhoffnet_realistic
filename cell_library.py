"""Per-edge current source library for the reduced differential KirchhoffNet.

Only ``FreeTanhLibrary`` (``tanh_free``) is supported::

    I = Isat * (tanh(u) + ALPHA*u) + G*(Vsrc - Vdest),
    u = gm * (s*(A*Vsrc - B*Vdest) + theta)

The device is single-cell (no library selection). Compliance gating turns the
tanh branch off as |x_src| or |x_dst| approaches x_max. The legacy libraries
(SimpleEdge, RealisticTanh*, LinearOTA, AntiParallel and the L/S/P/Z/v15/v2
cell tables) were removed in surgery 5/6; see the ``pre-surgery`` tag.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
)
from hw.constraints import (
    G_MIN,
    THETA_MAX,
    attenuation,
    ic_factor,
    ratio_ab,
    slope_bound,
    tanh_ste,
    theta_bound,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from hw import HWConfig
    from hw.constraints import MismatchState


__all__ = [
    "FreeTanhLibrary",
    "make_cell_library",
]








class FreeTanhLibrary(nn.Module):
    """Per-edge signed-realistic tanh device with independent A/B and STE sign,
    in parallel with an optimizable resistive shunt.

    Per-edge formula (leaky tanh + linear shunt)::

        u = gm * (s * (A * Vsrc - B * Vdest) + theta)
        I_tanh      = I_sat * (tanh(u) + ALPHA * u)
        I_resistive = G * (Vsrc - Vdest),   G = softplus(g_resistive_raw) > 0
        I_total     = I_tanh + I_resistive

    The ``ALPHA * u`` linear term guarantees a non-zero gradient even when
    ``tanh(u)`` is saturated (gradient floor = ``ALPHA``), so the cell never
    becomes a flat plateau for backward signal.

    The resistive shunt is exposed via a dedicated method so that, under
    ``--freeze-read``, the tanh contribution can be frozen at the initial state
    while the resistive term keeps routing current from evolving voltages.

    Parameterization (all per-edge, shape ``[E]``):
      - ``a_raw``   → A = softplus(a_raw), A >= 0, no upper bound, independent of B.
      - ``b_raw``   → B = softplus(b_raw), B >= 0, no upper bound, no sum constraint.
      - ``s_raw``   → s = sign(s_raw) (forward discrete ±1; backward uses STE).
      - ``gm_raw``  → gm = gm_min + (gm_max - gm_min) * sigmoid(gm_raw).
      - ``isat_raw``→ Isat = clamp_min(isat_min + (isat_max - isat_min) * sigmoid(isat_raw), 1e-6).
      - ``theta_raw``→ theta; only present when ``bias_enabled=True`` (init=0).
      - ``g_resistive_raw``→ G = softplus(g_resistive_raw), positive shunt
        conductance (init=-5 → G ≈ 0.0067).

    Boundary defaults come from ``config.TANH_REALISTIC_*``.
    Compliance gating applied after tanh.
    """

    # Leaky tanh slope: output = Isat * (tanh(u) + ALPHA * u). Sets the
    # minimum gradient magnitude so backward signal never vanishes in
    # saturation. Mirrors the LeakyReLU default of 0.01.
    ALPHA = 0.01

    def __init__(
        self,
        num_edges: int,
        gm_min: float | None = None,
        gm_max: float | None = None,
        isat_min: float | None = None,
        isat_max: float | None = None,
        bias_enabled: bool = False,
        hw: "HWConfig | None" = None,
        mismatch: "MismatchState | None" = None,
        family: str = "core",
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
        # Initialize a_raw, b_raw near 0 so softplus gives small values (~0.7)
        # instead of randn() which can cause huge A,B and gradient explosion.
        self.a_raw = nn.Parameter(torch.zeros(num_edges))
        self.b_raw = nn.Parameter(torch.zeros(num_edges))
        # 3.1 (hw_ratio_ab): the edge then uses ONE ratio parameter (``r_raw``)
        # instead of the two independent ``a_raw`` / ``b_raw`` (which stay in
        # the module but are unused).  ``r_raw = 0`` -> ``A = B = 1``.  The
        # parameter exists ONLY when the flag is on, so the flags-off
        # state_dict stays byte-identical to the pre-constraint one.
        if hw is not None and hw.constraints.hw_ratio_ab:
            self.r_raw = nn.Parameter(torch.zeros(num_edges))
        # Initialize s_raw to small random values so sign() gives ±1, not 0.
        # STE allows gradients to flow; forward needs non-zero sign for tanh path.
        self.s_raw = nn.Parameter(torch.randn(num_edges) * 0.1)
        # Initialize gm_raw, isat_raw more negative so initial gm, isat are near
        # minimum (0.01) to avoid initial gradient explosion in evolving core.
        # -5.0 -> sigmoid(-5) ≈ 0.0067 -> gm ≈ 0.01 + 9.99*0.0067 ≈ 0.077
        self.gm_raw = nn.Parameter(torch.full((num_edges,), -5.0))
        self.isat_raw = nn.Parameter(torch.full((num_edges,), -5.0))
        if self._bias_enabled:
            self.theta_raw = nn.Parameter(torch.zeros(num_edges))
        # This branch is always active, including when ``freeze_read`` is
        # enabled.  Start it in the same low-current regime as the OTA path:
        # softplus(-5) ≈ 0.0067, rather than a stiff 0.693 per-edge shunt.
        self.g_resistive_raw = nn.Parameter(torch.full((num_edges,), -5.0))

        self._beta_softness = float(PHYS["beta_softness"])

        # Hardware configuration (``hw.HWConfig``) or ``None``.
        # ``None`` => no hardware constraints: every hook below keeps the
        # pre-constraint behaviour bit-for-bit.  Not a parameter/buffer, so
        # the state_dict (and the golden checkpoints) are unaffected.
        self.hw = hw
        # 3.9 mismatch: the shared per-slot draw state (one physical fabric,
        # so every stage sees the same epsilon for the same slot) and the
        # edge family this library belongs to ("core", "boundary", "sense",
        # "output_ode").  ``None`` => no mismatch.
        self.mismatch = mismatch
        self.family = str(family)

    # ------------------------------------------------------------------
    @property
    def hw_constraints(self):
        """The ``HWConstraints`` of ``self.hw`` or ``None`` when the hardware layer is off."""
        hw = getattr(self, "hw", None)
        return None if hw is None else hw.constraints

    def current_gm(self) -> torch.Tensor:
        """Current gm values ``[E]`` from the bounded sigmoid map.

        This is the GLN *base* ``gm0``: the static gm the cell would use
        without rails. GLN multiplies it by ``exp(delta)`` and re-clamps to
        ``[gm_min, gm_max]`` (see ``gln_rails.GLNRails``).
        """
        sig_gm = torch.sigmoid(self.gm_raw)
        return self.gm_min + (self.gm_max - self.gm_min) * sig_gm        # [E]

    def isat_values(self) -> torch.Tensor:
        """Saturation current per edge ``[E]`` from the bounded sigmoid map.

        Same expression as :meth:`_tanh_core` uses, exposed so the hardware layer
        (``hw/constraints.py``) can read Isat without duplicating the parameterisation.
        """
        sig_isat = torch.sigmoid(self.isat_raw)
        return (self.isat_min + (self.isat_max - self.isat_min) * sig_isat).clamp_min(1e-6)  # [E]

    # ------------------------------------------------------------------
    # hardware-constraint views (Phase 3a 3.1 / 3.2 / 3.3 / 3.6)
    # ------------------------------------------------------------------
    def ratio_ab_values(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The 3.1 ratio form ``(A, B)`` with ``A + B == 2``, or ``None`` when the flag is off.

        ``r = exp(ln(r_max) tanh(r_raw))``, ``A = 2r/(1+r)``, ``B = 2/(1+r)``.
        """
        c = self.hw_constraints
        if c is None or not c.hw_ratio_ab:
            return None
        return ratio_ab(self.r_raw)

    def ab_values(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``(A, B)`` input attenuations the device uses, ``[E]`` each.

        Legacy: ``softplus(a_raw)`` / ``softplus(b_raw)`` (unbounded).
        3.1 (``hw_ratio_ab``): the ratio form (``A + B = 2``).
        3.1 + 3.2: the pair is normalised to ``max = 1`` (the common gain is
        folded into the slope factor of 3.2, so the per-input slope stays
        bounded by ``slope_max``).
        """
        c = self.hw_constraints
        if c is None or not c.hw_ratio_ab:
            return (torch.nn.functional.softplus(self.a_raw),
                    torch.nn.functional.softplus(self.b_raw))
        A, B = ratio_ab(self.r_raw)
        if c.hw_slope_bound:
            m = torch.maximum(A, B)
            A, B = A / m, B / m
        return A, B

    def gm_eff_values(self) -> torch.Tensor:
        """The slope the device uses, ``[E]`` (the GLN override is not included).

        Legacy: ``gm_min + (gm_max - gm_min) sigmoid(gm_raw)``.
        3.2 / 3.3: ``s_max * f_IC(Isat) * g`` with ``g = g_min + (1-g_min) sigmoid(gm_raw)``.
        """
        c = self.hw_constraints
        if c is None or not (c.hw_slope_bound or c.hw_ic_coupling):
            return self.current_gm()
        return self._gm_hw(c, None)

    def theta_values(self) -> torch.Tensor | None:
        """The input-referred offset ``theta`` ``[E]`` (``None`` when the cell has no bias)."""
        if not self._bias_enabled:
            return None
        c = self.hw_constraints
        if c is not None and c.hw_theta_bound:
            return theta_bound(self.theta_raw, THETA_MAX)
        return self.theta_raw

    def _mismatch_factor(self, name: str, n: int, ref: torch.Tensor):
        """The 3.9 mismatch factor of ``name`` for the first ``n`` slots (or ``None``).

        Multiplicative for ``isat`` / ``a`` / ``b`` / ``g`` (``exp(sigma eps)``),
        additive for ``theta`` (``sigma eps``).  ``None`` when the mismatch is
        off, when that sigma is 0, or in eval mode without ``hw_mismatch_eval``.
        """
        mm = self.mismatch
        if mm is None:
            return None
        return mm.get(self.family, name, n, dtype=ref.dtype, device=ref.device,
                      training=self.training)

    def _gm_hw(self, c, gm_override: torch.Tensor | None) -> torch.Tensor:
        """The hardware slope of 3.2 / 3.3: ``gm_eff = s_max * f_IC(Isat) * g``.

        ``g = g_min + (1 - g_min) sigmoid(gm_raw)`` is the attenuation of the
        input pair.  When the GLN modulates the slope (legacy ``gm_override``,
        a gm tensor), its *relative* modulation is applied to ``g`` and
        re-clamped to ``[g_min, 1]``, so the bound ``gm_eff <= slope_max``
        still holds for the GLN-gated families too.
        """
        units = self.hw.units
        g = attenuation(self.gm_raw, G_MIN)                       # [E]
        if gm_override is not None:
            rel = gm_override / self.current_gm()                 # [B, E] or [1, E]
            g = torch.clamp(g.unsqueeze(0) * rel, G_MIN, 1.0)
        f_ic = None
        if c.hw_ic_coupling:
            f_ic = ic_factor(self.isat_values(), units)
        gm = units.slope_max * g
        if f_ic is not None:
            gm = gm * f_ic
        return gm

    def _tanh_core(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        gm_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Internal: leaky-tanh current (no compliance gating).

        Returns ``I_sat * (tanh(u) + ALPHA * u)`` with shape ``[B, E]`` —
        broadcastable against ``x_src``/``x_dst`` which are ``[B, E]``.

        ``gm_override`` (GLN path): caller-supplied modulated gm, shape
        ``[B, E]`` (or ``[1, E]``), used instead of the static sigmoid map.
        The stage calls this with the GLN-modulated gm for boundary /
        readout-sense families only; the core family always uses the static
        map (GLN never touches core edges in v1).

        Every hardware constraint is a flag-gated branch: with the flag off
        (or with no ``hw`` at all) the legacy expressions are used unchanged,
        so the forward stays bit-identical to the pre-constraint model.
        """
        c = self.hw_constraints
        if c is not None and c.hw_ratio_ab:                          # 3.1
            A, B = self.ab_values()                                 # [E]
            A = A.unsqueeze(0)                                      # [1, E]
            B = B.unsqueeze(0)                                      # [1, E]
        else:
            A = torch.nn.functional.softplus(self.a_raw).unsqueeze(0)        # [1, E]
            B = torch.nn.functional.softplus(self.b_raw).unsqueeze(0)        # [1, E]
        if self.mismatch is not None:                                     # 3.9
            n_edges = A.shape[-1]
            f_a = self._mismatch_factor("a", n_edges, A)
            if f_a is not None:
                A = A * f_a
            f_b = self._mismatch_factor("b", n_edges, B)
            if f_b is not None:
                B = B * f_b
        s = torch.sign(self.s_raw)                                       # [E]
        s_ste = s + self.s_raw - self.s_raw.detach()                     # STE: [E]
        if c is not None and (c.hw_slope_bound or c.hw_ic_coupling):      # 3.2 / 3.3
            gm = self._gm_hw(c, gm_override)                              # [E] / [B, E]
        elif gm_override is not None:
            gm = gm_override                                             # [B, E] or [1, E]
        else:
            sig_gm = torch.sigmoid(self.gm_raw)
            gm = self.gm_min + (self.gm_max - self.gm_min) * sig_gm      # [E]
            gm = gm.unsqueeze(0)                                         # [1, E]
        sig_isat = torch.sigmoid(self.isat_raw)
        isat = (self.isat_min + (self.isat_max - self.isat_min) * sig_isat).clamp_min(1e-6)  # [E]
        if self.mismatch is not None:                                     # 3.9
            f_i = self._mismatch_factor("isat", isat.shape[-1], isat)
            if f_i is not None:
                isat = isat * f_i
        if c is not None and c.hw_theta_bound and self._bias_enabled:     # 3.6
            theta = theta_bound(self.theta_raw, THETA_MAX).unsqueeze(0)
        else:
            theta = self.theta_raw.unsqueeze(0) if self._bias_enabled else 0.0
        if self.mismatch is not None and self._bias_enabled:              # 3.9
            f_t = self._mismatch_factor("theta", theta.shape[-1], theta)
            if f_t is not None:
                theta = theta + f_t
        u = (s_ste.unsqueeze(0) * (A * x_src - B * x_dst) + theta) * gm
        if c is not None and (c.hw_no_alpha or c.hw_alpha_bwd > 0.0):      # 3.5
            t = tanh_ste(u, c.hw_alpha_bwd)                              # forward: tanh(u)
        else:
            t = torch.tanh(u) + self.ALPHA * u
        return isat.unsqueeze(0) * t

    def forward_tanh(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        x_max: float,
        gm_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Leaky-tanh current with compliance gating, no resistive shunt.

        Use this when the resistive term must be excluded (e.g. ``--freeze-read``
        wants to freeze only the tanh contribution at ``x0``).
        Returns ``[batch, E]`` gated by ``gate_src * gate_dst``.

        ``gm_override`` (GLN path): modulated gm ``[B, E]`` from
        :meth:`GLNRails.modulate_gm_z`, used instead of the static sigmoid
        map (see :meth:`_tanh_core`).
        """
        i_cell = self._tanh_core(x_src, x_dst, gm_override=gm_override)  # [1, E] or [B, E]
        i_cell = i_cell.expand(x_src.shape[0], -1)                      # [B, E]
        gate_src = torch.sigmoid((x_max - x_src.abs()) / self._beta_softness)
        gate_dst = torch.sigmoid((x_max - x_dst.abs()) / self._beta_softness)
        return (gate_src * gate_dst) * i_cell

    def resistive_current(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
    ) -> torch.Tensor:
        """Resistive shunt current ``G * (Vsrc - Vdest)``, no compliance gating.

        Returns shape ``[batch, E]``. Always uses evolving voltages (intended
        to bypass ``--freeze-read``).
        """
        G = torch.nn.functional.softplus(self.g_resistive_raw).unsqueeze(0)  # [1, E]
        if self.mismatch is not None:                                     # 3.9
            f_g = self._mismatch_factor("g", G.shape[-1], G)
            if f_g is not None:
                G = G * f_g
        return G * (x_src - x_dst)

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        x_max: float,
    ) -> torch.Tensor:
        """Full cell current: leaky tanh + resistive shunt.

        Only the tanh contribution is compliance-gated; the resistive shunt
        is a physical conductor that always routes current based on the
        voltage difference (intentional — see ``resistive_current``).
        Stage-level edge_mask / budget_gate are applied separately in
        ``DifferentialStage.rhs()``.
        """
        i_tanh = self.forward_tanh(x_src, x_dst, x_max)
        i_res = self.resistive_current(x_src, x_dst)
        return i_tanh + i_res

    def compile_forward(self, backend: str = "inductor"):
        pass






def make_cell_library(
    library_name: str,
    num_edges: int | None = None,
    gm_min: float | None = None,
    gm_max: float | None = None,
    isat_min: float | None = None,
    isat_max: float | None = None,
    hw: "HWConfig | None" = None,
    mismatch: "MismatchState | None" = None,
    family: str = "core",
) -> FreeTanhLibrary:
    """Factory: returns a ``FreeTanhLibrary``. Only ``'tanh_free'`` is supported.

    ``gm_min`` / ``gm_max`` / ``isat_min`` / ``isat_max`` override the
    bounded sigmoid parameterization rails (defaults from config
    ``TANH_REALISTIC_*``); the constructor validates ``max > min``.

    The factory reads ``BIAS_ENABLED`` from ``CELL_LIBRARIES['tanh_free']``.

    ``hw`` is the hardware configuration (``hw.HWConfig``) or ``None``.  It is
    stored on the library (``lib.hw``) and read by the constraint hooks in
    :meth:`FreeTanhLibrary._tanh_core`.  ``None`` means "no hardware
    constraints": the forward pass is bit-identical to the pre-constraint
    model.  It is not a parameter or a buffer, so the state_dict is unchanged.

    When ``num_edges`` is ``None`` (template), the returned library
    is created with ``num_edges=1`` and must be replaced with a per-stage
    instance (matching the actual edge count) before use.
    """
    if library_name != "tanh_free":
        raise ValueError(
            f"Unknown cell library: {library_name!r}. Only 'tanh_free' is "
            f"supported (the legacy 'relu', 'tanh', 'tanh_realistic', "
            f"'tanh_realistic_upgrade', 'linear_ota', 'tanh_anti', 'legacy', "
            f"'v15' and 'v2' libraries were removed)."
        )
    n = num_edges if num_edges is not None else 1
    bias_enabled = bool(CELL_LIBRARIES[library_name].get("BIAS_ENABLED", False))
    return FreeTanhLibrary(
        num_edges=n,
        bias_enabled=bias_enabled,
        gm_min=gm_min, gm_max=gm_max,
        isat_min=isat_min, isat_max=isat_max,
        hw=hw,
        mismatch=mismatch,
        family=family,
    )
