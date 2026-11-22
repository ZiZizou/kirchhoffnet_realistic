"""HWConstraints (flag dataclass) and the pure-torch constraint functions.

One function per Phase 3a section 3 constraint (Phase 3b section 2.1 table).  Nothing here
touches module state: every function takes tensors plus (where it needs physical numbers) an
``HWUnits``.  The model hooks live in ``cell_library.py`` / ``differential_stage.py`` /
``gln_rails.py`` / ``gating`` / ``stage_transfer.py`` / ``train.py`` and call these.

Flag names and defaults are exactly those of Phase 3a section 3.11 (all off).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .units import HWUnits

# ---------------------------------------------------------------------------
# design constants of Phase 3a (the flag list fixes only the switches; these are the numbers
# the sections quote and are module constants, not flags)
# ---------------------------------------------------------------------------
R_MAX = 4.0              # 3.1  A/B ratio range [1/R_MAX, R_MAX]
G_MIN = 1.0 / 16.0       # 3.2  attenuation floor
THETA_MAX = 1.0          # 3.6  |theta| <= THETA_MAX
W_MAX = 4.0              # 3.7  crossbar mirror ratio bound
KAPPA = 3.0              # 3.9  tail + both mirror legs + bias overhead
LAMBDA_V = 0.15          # 3.4  channel-length modulation [1/V]
COMPLIANCE_K = 2000.0    # 3.4  softplus_k sharpness (see compliance notes below)
ZETA_MAX = 4.0           # 3.10 range of the VCA logit DAC
ISAT_HW_MAX = 16.0       # 3.10 hardware ceiling of Isat_eff = Isat*sigma(z)*g_vca
POWER_REF_W = 10e-6      # 3.9  P_ref of L_pow = lambda_P * P_avg / P_ref
C_GATE, C_LEG, C_WIRE = 2e-15, 1e-15, 4e-15   # 3.8 (22FDX assumptions)
STIFFNESS_LIMIT = 2.0    # 3.3  Heun is unstable above dt*lambda = 2

MISMATCH_DEFAULT = {"isat": 0.05, "g": 0.10, "a": 0.01, "b": 0.01, "theta": 0.1}
MISMATCH_KEYS = tuple(MISMATCH_DEFAULT)

XBAR_MODES = ("signed", "offset_binary", "differential")
C_NODE_MODES = (None, "uniform", "physical")
GLN_MODES = (None, "tail", "slope")


# ---------------------------------------------------------------------------
# flags
# ---------------------------------------------------------------------------
@dataclass
class HWConstraints:
    hw_ratio_ab: bool = False                       # 3.1
    hw_slope_bound: bool = False                    # 3.2
    hw_ic_coupling: bool = False                    # 3.3
    hw_compliance: bool = False                     # 3.4
    hw_clamp_transfer: bool = False                 # 3.4
    hw_no_alpha: bool = False                       # 3.5
    hw_alpha_bwd: float = 0.0                       # 3.5
    hw_vref_symmetric: bool = False                 # 3.6
    hw_vref_bits: Optional[int] = None              # 3.6
    hw_theta_bound: bool = False                    # 3.6
    hw_xbar_bound: bool = False                     # 3.7
    hw_xbar_bits: Optional[int] = None              # 3.7
    hw_xbar_mode: str = "signed"                    # 3.7
    hw_c_node: Optional[str] = None                 # 3.8
    hw_power_lambda: float = 0.0                    # 3.9
    hw_mismatch_sigma: Optional[dict] = None        # 3.9
    hw_mismatch_eval: bool = False                  # 3.9
    hw_vca_bits: Optional[int] = None               # 3.10
    hw_isat_eff_clamp: bool = False                 # 3.10
    hw_gln_mode: Optional[str] = None               # 3.10
    hw_gln_delta_max: float = math.log(4.0)         # 3.10
    hw_check_stiffness: bool = False                # 3.3 (diagnostic)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.hw_xbar_mode not in XBAR_MODES:
            raise ValueError(f"hw_xbar_mode must be one of {XBAR_MODES}, got {self.hw_xbar_mode!r}")
        if self.hw_c_node not in C_NODE_MODES:
            raise ValueError(f"hw_c_node must be one of {C_NODE_MODES}, got {self.hw_c_node!r}")
        if self.hw_gln_mode not in GLN_MODES:
            raise ValueError(f"hw_gln_mode must be one of {GLN_MODES}, got {self.hw_gln_mode!r}")
        for name in ("hw_vref_bits", "hw_xbar_bits", "hw_vca_bits"):
            v = getattr(self, name)
            if v is not None and (int(v) != v or int(v) < 1):
                raise ValueError(f"{name} must be None or an int >= 1, got {v!r}")
        if self.hw_alpha_bwd < 0 or self.hw_power_lambda < 0:
            raise ValueError("hw_alpha_bwd and hw_power_lambda must be >= 0")
        if self.hw_gln_delta_max <= 0:
            raise ValueError("hw_gln_delta_max must be > 0")
        if self.hw_mismatch_sigma is not None:
            if not isinstance(self.hw_mismatch_sigma, dict):
                raise ValueError("hw_mismatch_sigma must be None or a dict")
            bad = set(self.hw_mismatch_sigma) - set(MISMATCH_KEYS)
            if bad:
                raise ValueError(f"hw_mismatch_sigma: unknown keys {sorted(bad)}; allowed {MISMATCH_KEYS}")

    # ------------------------------------------------------------------
    def is_off(self) -> bool:
        """True only when every flag is at its default."""
        return self == HWConstraints()

    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = dict(v) if isinstance(v, dict) else v
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "HWConstraints":
        names = {f.name for f in fields(cls)}
        kw = {}
        for k, v in (d or {}).items():
            key = k if k in names else ("hw_" + k if ("hw_" + k) in names else None)
            if key is None:
                raise KeyError(f"HWConstraints: unknown key {k!r}; valid keys: {sorted(names)}")
            if key in kw:
                raise KeyError(f"HWConstraints: key {k!r} given twice")
            kw[key] = v
        return cls(**kw)

    # which model families a flag set touches (used by the hooks)
    def mismatch_active(self) -> bool:
        return self.hw_mismatch_sigma is not None and any(
            float(v) != 0.0 for v in self.hw_mismatch_sigma.values())

    def affects_cell(self) -> bool:
        return bool(self.hw_ratio_ab or self.hw_slope_bound or self.hw_ic_coupling
                    or self.hw_no_alpha or self.hw_alpha_bwd > 0.0 or self.hw_theta_bound
                    or self.mismatch_active() or self.hw_gln_mode == "tail"
                    or self.hw_compliance)


# ---------------------------------------------------------------------------
# generic helper
# ---------------------------------------------------------------------------
def quantize_ste(x: torch.Tensor, bits: Optional[int], lo: float, hi: float) -> torch.Tensor:
    """Uniform ``bits``-bit quantiser on [lo, hi] with a straight-through gradient.

    Forward: nearest of the 2**bits codes (``2**bits - 1`` intervals).  Backward: identity inside
    [lo, hi] (the clamp has zero gradient outside).  ``bits=None`` returns ``x`` unchanged.
    """
    if bits is None:
        return x
    levels = float(2 ** int(bits) - 1)
    span = float(hi) - float(lo)
    xc = torch.clamp(x, float(lo), float(hi))
    q = torch.round((xc - lo) / span * levels) / levels * span + lo
    return xc + (q - xc).detach()


# ---------------------------------------------------------------------------
# 3.1 bounded A/B (ratio form)
# ---------------------------------------------------------------------------
def ratio_ab(r_raw: torch.Tensor, r_max: float = R_MAX):
    """``r = exp(ln(r_max) tanh(r_raw))``; ``A = 2r/(1+r)``, ``B = 2/(1+r)`` so A + B = 2, A/B = r."""
    r = torch.exp(math.log(r_max) * torch.tanh(r_raw))
    return 2.0 * r / (1.0 + r), 2.0 / (1.0 + r)


# ---------------------------------------------------------------------------
# 3.2 gm*A bound / 3.3 IC coupling
# ---------------------------------------------------------------------------
def attenuation(g_raw: torch.Tensor, g_min: float = G_MIN) -> torch.Tensor:
    """``g = g_min + (1 - g_min) sigmoid(g_raw)`` in [g_min, 1]."""
    return g_min + (1.0 - g_min) * torch.sigmoid(g_raw)


def ic_factor(isat: torch.Tensor, units: HWUnits, i_spec: Optional[float] = None) -> torch.Tensor:
    """``f(IC) = 2 / (1 + sqrt(1 + 4 IC))``, ``IC = Isat*Iu / (2 I_spec)`` (1 in weak inversion)."""
    i_spec = units.i_spec if i_spec is None else i_spec
    ic = isat * units.iu / (2.0 * i_spec)
    return 2.0 / (1.0 + torch.sqrt(1.0 + 4.0 * ic))


def slope_bound(g_raw: torch.Tensor, units: HWUnits, g_min: float = G_MIN,
                A: Optional[torch.Tensor] = None, B: Optional[torch.Tensor] = None,
                f_ic: Optional[torch.Tensor] = None):
    """Hardware slope ``gm_eff = s_max * f_IC * g`` and the normalised inputs.

    Returns ``(gm_eff, A_hat, B_hat)`` with ``A_hat = A/max(A,B)``, ``B_hat = B/max(A,B)`` (so
    ``max(A_hat, B_hat) = 1`` and ``gm_eff * max(A_hat, B_hat) <= slope_max``).  ``A_hat/B_hat``
    are ``None`` when A, B are not given.
    """
    gm = units.slope_max * attenuation(g_raw, g_min)
    if f_ic is not None:
        gm = gm * f_ic
    if A is None or B is None:
        return gm, None, None
    m = torch.maximum(A, B)
    return gm, A / m, B / m


# ---------------------------------------------------------------------------
# 3.4 output compliance (leg-wise)
# ---------------------------------------------------------------------------
def leg_factors(x_n: torch.Tensor, units: HWUnits, rail: bool = True,
                lambda_v: float = LAMBDA_V, k: float = COMPLIANCE_K, v_off: float = 0.0):
    """Multipliers ``(f_top, f_bot)`` of the P leg (sources from VDD) and N leg (sinks to ground)
    of an output at node voltage ``x_n`` [units].

    ``f_top = (1 - lambda_v*Vu*x_n) * c_top(x_n)``, ``f_bot = (1 + lambda_v*Vu*x_n) * c_bot(x_n)`` with
    ``c_top = 1 - exp(-softplus_k((x_hard - x_n)*Vu/U_T - v_off))`` and the mirror for ``c_bot``.
    Deviation from 3a 3.4 (so the 3b 2.4 assertions hold): both factors are divided by their value
    at x_n = 0 (legs equal ``Isat (1 +- tanh u)/2`` at Vcm to ~1e-6), the channel-length term is
    referenced to Vcm (``1 -/+ lambda_v Vu x_n`` instead of ``1 + lambda_v (x_hard -/+ x_n) Vu``), and
    ``softplus_k`` uses a sharp ``k = COMPLIANCE_K`` so the outward leg is < 1e-3 of nominal at
    ``x_n = +-x_hard`` (k = 1 leaves 0.5 there).  The at-rest output conductance is exactly
    ``lambda_v*Vu*Isat`` (3.5 ``g_out``).  The knee uses the 85 C ``U_T`` of ``units``.
    ``rail=False`` drops the compliance roll-off (only the g_out term remains).
    """
    lam = lambda_v * units.vu
    f_top = (1.0 - lam * x_n).clamp_min(0.0)
    f_bot = (1.0 + lam * x_n).clamp_min(0.0)
    if rail:
        scale = units.vu / units.u_t
        y_top = (units.x_hard - x_n) * scale - v_off
        y_bot = (units.x_hard + x_n) * scale - v_off
        # normalised by the value at x_n = 0 so the factors are exactly 1 at Vcm
        y0 = units.x_hard * scale - v_off
        c0 = float(1.0 - torch.exp(-F.softplus(torch.tensor(y0, dtype=torch.float64), beta=k)))
        f_top = f_top * (1.0 - torch.exp(-F.softplus(y_top, beta=k))) / c0
        f_bot = f_bot * (1.0 - torch.exp(-F.softplus(y_bot, beta=k))) / c0
    return f_top, f_bot


def compliance_legs(i_p: torch.Tensor, i_n: torch.Tensor, x_d: torch.Tensor,
                    x_s: Optional[torch.Tensor], units: HWUnits, rail: bool = True, **kw):
    """Current into the destination and source node of one OTA edge after compliance.

    ``i_p = Isat (1 + t)/2`` and ``i_n = Isat (1 - t)/2`` are the pre-compliance branch currents
    (``t = tanh(u)``).  The destination's P leg carries ``i_p`` and its N leg ``i_n``; the source node's
    legs are the complement (its N leg sinks ``i_p``, its P leg sources ``i_n``), so
    ``into_dst = i_p f_top(x_d) - i_n f_bot(x_d)`` and ``into_src = i_n f_top(x_s) - i_p f_bot(x_s)``.
    With both legs in compliance ``into_dst = -into_src = Isat t`` (charge conserved).
    ``x_s=None`` (an ideal source terminal) returns ``(into_dst, None)``.
    """
    ftd, fbd = leg_factors(x_d, units, rail, **kw)
    into_d = i_p * ftd - i_n * fbd
    if x_s is None:
        return into_d, None
    fts, fbs = leg_factors(x_s, units, rail, **kw)
    return into_d, i_n * fts - i_p * fbs


# ---------------------------------------------------------------------------
# 3.5 g_out and straight-through floor
# ---------------------------------------------------------------------------
def g_out_term(isat_eff: torch.Tensor, x_n: torch.Tensor, units: HWUnits,
               lambda_v: float = LAMBDA_V) -> torch.Tensor:
    """Output-conductance current of one edge end: ``-lambda_v*Vu*Isat_eff*x_n`` (to Vcm)."""
    return -(lambda_v * units.vu) * isat_eff * x_n


def tanh_ste(u: torch.Tensor, alpha_bwd: float = 0.0) -> torch.Tensor:
    """Forward ``tanh(u)`` exactly; backward ``1 - tanh(u)^2 + alpha_bwd``."""
    t = torch.tanh(u)
    if alpha_bwd:
        t = t + alpha_bwd * (u - u.detach())
    return t


# ---------------------------------------------------------------------------
# 3.6 Vref, theta
# ---------------------------------------------------------------------------
def vref_map(raw: torch.Tensor, x_max: float, bits: Optional[int] = None,
             symmetric: bool = True) -> torch.Tensor:
    """Symmetric ``x_max*tanh(raw)`` (range [-x_max, x_max], 0 at raw = 0) or the legacy
    ``x_max*sigmoid(raw)`` (range [0, x_max]), optionally ``bits``-bit quantised (STE)."""
    if symmetric:
        v = x_max * torch.tanh(raw)
        return quantize_ste(v, bits, -x_max, x_max)
    v = torch.sigmoid(raw) * x_max
    return quantize_ste(v, bits, 0.0, x_max)


def theta_bound(theta_raw: torch.Tensor, theta_max: float = THETA_MAX) -> torch.Tensor:
    return theta_max * torch.tanh(theta_raw)


# ---------------------------------------------------------------------------
# 3.7 crossbar
# ---------------------------------------------------------------------------
def xbar_map(W: torch.Tensor, mode: str = "signed", bits: Optional[int] = None,
             bound: bool = False, w_max: float = W_MAX) -> torch.Tensor:
    """Effective crossbar weights: optional bound ``w_max*tanh(W/w_max)`` then optional quantisation.

    ``signed``: sign-magnitude (uniform STE grid on [-w_max, w_max]).  ``offset_binary``:
    ``W = m - w_max`` with the code ``m`` in [0, 2 w_max] quantised.  ``differential``:
    ``W = W+ - W-`` with both unipolar parts quantised on [0, w_max].  Without quantisation all three
    are the identical ideal matrix."""
    if mode not in XBAR_MODES:
        raise ValueError(f"unknown xbar mode {mode!r}")
    if bound:
        W = w_max * torch.tanh(W / w_max)
    if bits is None:
        return W
    if mode == "signed":
        return quantize_ste(W, bits, -w_max, w_max)
    if mode == "offset_binary":
        return quantize_ste(W + w_max, bits, 0.0, 2.0 * w_max) - w_max
    wp = quantize_ste(torch.relu(W), bits, 0.0, w_max)
    wn = quantize_ste(torch.relu(-W), bits, 0.0, w_max)
    return wp - wn


# ---------------------------------------------------------------------------
# 3.8 node capacitance
# ---------------------------------------------------------------------------
def c_node_vector(num_nodes: int, src, dst, units: HWUnits, mode: str, *,
                  boundary_dst=(), sense_src=(), mesh_src=(), mesh_dst=(),
                  acc_start: Optional[int] = None, n_xbar_cols: int = 0,
                  xbar_mode: str = "signed", c_gate: float = C_GATE, c_leg: float = C_LEG,
                  c_wire: float = C_WIRE, dtype=torch.float32) -> torch.Tensor:
    """Per-node capacitance in units of ``C_u``, shape ``[1, N]``.

    Terminals (3.8): every core edge puts one input gate and one output leg on both of its nodes;
    a boundary edge one gate + one leg on its destination; a sense OTA one gate on its (hidden)
    source; a mesh readout edge a gate on the source and a gate + leg on the destination.  Nodes
    ``>= acc_start`` are crossbar accumulators: ``n_xbar_cols`` legs (twice with ``differential``).
    ``C_par = n_gate*c_gate + n_leg*c_leg + c_wire``.

    ``"uniform"``: the fixed cap pads every node to exactly ``C_u`` (all ones); raises if some
    ``C_par`` exceeds ``C_u``.  ``"physical"``: one fixed cap per node class (hidden / accumulator),
    sized so the class-mean node is ``C_u``; nodes differ by their own terminal count.
    """
    if mode not in ("uniform", "physical"):
        raise ValueError(f"c_node mode must be 'uniform' or 'physical', got {mode!r}")
    c_par, start = node_parasitics(num_nodes, src, dst, boundary_dst=boundary_dst, sense_src=sense_src,
                                   mesh_src=mesh_src, mesh_dst=mesh_dst, acc_start=acc_start,
                                   n_xbar_cols=n_xbar_cols, xbar_mode=xbar_mode, c_gate=c_gate,
                                   c_leg=c_leg, c_wire=c_wire)
    return _c_node_from_parasitics(c_par, start, num_nodes, units, mode, dtype)


def node_parasitics(num_nodes: int, src, dst, *, boundary_dst=(), sense_src=(), mesh_src=(),
                    mesh_dst=(), acc_start: Optional[int] = None, n_xbar_cols: int = 0,
                    xbar_mode: str = "signed", c_gate: float = C_GATE, c_leg: float = C_LEG,
                    c_wire: float = C_WIRE):
    """Terminal-count parasitic capacitance per node [F] (float64 ``[N]``) and the accumulator start."""
    gates = torch.zeros(num_nodes, dtype=torch.float64)
    legs = torch.zeros(num_nodes, dtype=torch.float64)

    def _add(t, idx):
        for i in idx:
            t[int(i)] += 1.0

    _add(gates, src), _add(gates, dst), _add(legs, src), _add(legs, dst)
    _add(gates, boundary_dst), _add(legs, boundary_dst)
    _add(gates, sense_src)
    _add(gates, mesh_src), _add(gates, mesh_dst), _add(legs, mesh_dst)
    start = num_nodes if acc_start is None else int(acc_start)
    if n_xbar_cols and start < num_nodes:
        legs[start:] += float(n_xbar_cols) * (2.0 if xbar_mode == "differential" else 1.0)
    return gates * c_gate + legs * c_leg + c_wire, start


def _c_node_from_parasitics(c_par, start, num_nodes, units, mode, dtype):
    cu = float(units.cu)
    if mode == "uniform":
        if bool((c_par > cu).any()):
            raise ValueError(
                f"hw_c_node='uniform': node parasitic {float(c_par.max())*1e15:.1f} fF exceeds "
                f"C_u = {cu*1e15:.1f} fF (fixed cap would be negative)")
        c = torch.full((num_nodes,), 1.0, dtype=torch.float64)
    else:
        c = torch.empty(num_nodes, dtype=torch.float64)
        for lo, hi in ((0, start), (start, num_nodes)):
            if hi <= lo:
                continue
            fixed = cu - float(c_par[lo:hi].mean())
            if fixed < 0:
                raise ValueError("hw_c_node='physical': class-mean parasitic exceeds C_u")
            c[lo:hi] = (fixed + c_par[lo:hi]) / cu
    return c.to(dtype).view(1, -1)


# ---------------------------------------------------------------------------
# 3.9 power and mismatch
# ---------------------------------------------------------------------------
def _stage_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    return list(x.unbind(0)) if x.dim() > 1 else [x]


def power_penalty(isat, sigma_z, g_vca=None, units: Optional[HWUnits] = None,
                  kappa: float = KAPPA) -> torch.Tensor:
    """``P_avg = kappa * VDD * Iu * mean_s sum_e Isat_{s,e} sigma(z_{s,e}) g_vca`` [W].

    ``isat`` / ``sigma_z`` / ``g_vca``: ``[S, E]`` tensors or lists of per-stage 1-D tensors
    (stages may have different edge counts).  ``g_vca=None`` means 1."""
    units = HWUnits() if units is None else units
    isat_l, z_l = _stage_list(isat), _stage_list(sigma_z)
    g_l = [None] * len(isat_l) if g_vca is None else _stage_list(g_vca)
    tot = 0.0
    for i, z, g in zip(isat_l, z_l, g_l):
        e = i * z
        if g is not None:
            e = e * g
        tot = tot + e.sum()
    return kappa * units.vdd * units.iu * tot / float(len(isat_l))


def mismatch_draw(shape, sigma: float, gen: Optional[torch.Generator] = None,
                  additive: bool = False, dtype=None, device=None) -> torch.Tensor:
    """Log-normal multiplier ``exp(sigma*eps)`` (or additive ``sigma*eps``), ``eps ~ N(0,1)``."""
    eps = torch.randn(tuple(shape), generator=gen, dtype=dtype or torch.float64,
                      device=None if gen is not None else device)
    out = sigma * eps if additive else torch.exp(sigma * eps)
    return out.to(device) if device is not None else out


def apply_mismatch(value: torch.Tensor, factor: Optional[torch.Tensor],
                   additive: bool = False) -> torch.Tensor:
    if factor is None:
        return value
    return value + factor if additive else value * factor


class MismatchState:
    """Per-slot mismatch draws shared by every stage (one physical fabric).

    ``get(family, name, n, ...)`` returns the factor of the first ``n`` slots of ``(family, name)``;
    all stages ask the same cache, so the same slot gets the same epsilon in every stage.  ``new_batch()``
    (called by stage 0 at the start of a forward) clears the cache, i.e. a new draw per batch.  Returns
    ``None`` when that sigma is 0 or when the model is in eval mode and ``eval_on`` is False.
    """

    def __init__(self, sigma: dict, eval_on: bool = False, seed: Optional[int] = None) -> None:
        self.sigma = {k: float(sigma.get(k, 0.0)) for k in MISMATCH_KEYS}
        self.eval_on = bool(eval_on)
        self.seed = seed
        self._gen: Optional[torch.Generator] = None
        self._cache: dict = {}

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_gen"] = None
        d["_cache"] = {}
        return d

    def reseed(self, seed: Optional[int]) -> None:
        self.seed = seed
        self._gen = None
        self._cache.clear()

    def _generator(self):
        if self.seed is None:
            return None
        if self._gen is None:
            self._gen = torch.Generator()
            self._gen.manual_seed(int(self.seed))
        return self._gen

    def new_batch(self) -> None:
        self._cache.clear()

    def enabled(self, training: bool) -> bool:
        return training or self.eval_on

    def get(self, family: str, name: str, n: int, dtype=None, device=None,
            training: bool = True) -> Optional[torch.Tensor]:
        sig = self.sigma.get(name, 0.0)
        if sig == 0.0 or not self.enabled(training):
            return None
        key = (family, name)
        cur = self._cache.get(key)
        if cur is None or cur.numel() < n:
            extra = n if cur is None else n - cur.numel()
            new = mismatch_draw((extra,), sig, self._generator(), additive=(name == "theta"),
                                dtype=torch.float64)
            cur = new if cur is None else torch.cat([cur, new])
            self._cache[key] = cur
        out = cur[:n]
        return out.to(device=device, dtype=dtype) if (dtype is not None or device is not None) else out


# ---------------------------------------------------------------------------
# 3.10 GLN tail and VCA x Isat
# ---------------------------------------------------------------------------
def gln_tail(delta: torch.Tensor, delta_max: float) -> torch.Tensor:
    """Multiplier on Isat: ``exp(clamp(delta, -delta_max, delta_max))`` (so in [1/4, 4] for ln 4)."""
    return torch.exp(torch.clamp(delta, -delta_max, delta_max))


def isat_eff_clamp(isat: torch.Tensor, sigma_z: torch.Tensor, g_vca: Optional[torch.Tensor] = None,
                   hi: float = ISAT_HW_MAX, lo: float = 0.0) -> torch.Tensor:
    """Hardware-feasible ``Isat_eff = clamp(Isat * sigma(z) * g_vca, lo, hi)``."""
    eff = isat * sigma_z
    if g_vca is not None:
        eff = eff * g_vca
    return torch.clamp(eff, lo, hi)


def vca_quantize(logits: torch.Tensor, bits: Optional[int], zeta_max: float = ZETA_MAX) -> torch.Tensor:
    """STE quantisation of the VCA logits zeta_e on [-zeta_max, zeta_max]."""
    return quantize_ste(logits, bits, -zeta_max, zeta_max)


# ---------------------------------------------------------------------------
# 3.3 diagnostic
# ---------------------------------------------------------------------------
def stiffness_dt_lambda(dt: float, lam: torch.Tensor) -> torch.Tensor:
    """``dt * lambda`` per node (Heun is stable for dt*lambda <= 2)."""
    return dt * lam
