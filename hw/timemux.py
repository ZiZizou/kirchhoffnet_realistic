"""Time-multiplexed fabric model (HR-KNet hardware refactor, Phase 3b, W3).

This module is the *self-contained* part of the hardware layer: it implements
the sample-and-hold (S&H) boundary of a time-multiplexed fabric, the
deterministic hardware errors of that boundary (charge injection, gain error,
droop, kT/C noise), the bias-settling model of a frozen core OTA, and the
interstage-transfer variants.

It deliberately imports **only** ``torch``, ``dataclasses`` and ``math`` so
that it can be imported before (or entirely without) ``hw/units.py`` and
``hw/constraints.py``.  The physical unit quantities it needs are taken either
as plain floats or from a duck-typed ``units`` object that exposes the
attributes of ``hw.units.HWUnits`` (Phase 3a section 2.5):

    vu, cu, temp_c, ktc_rms_units, x_hard          (aliases accepted)

Reference: ``kn_hwdocs/phase3b_fabric_impl_retrain.md`` sections 1.1-1.5
(physics), 2.1 (this module) and 2.4(a) (the tests).

Integration notes for the model hooks (workstream W4):

* ``StageBoundary`` must be inserted between a stage call and the
  ``StageTransfer`` call: ``x_hold, x = boundary.capture(x, stage_idx)``; the
  next stage reads ``x_hold`` (frozen core read), the transfer sees ``x``.
* The per-node sign buffers are registered with ``persistent=False`` so that
  attaching a ``StageBoundary`` never changes ``state_dict().keys()`` (the
  "flags off == golden" guarantee of Phase 3b section 2.4(b)).  The signs are
  regenerated deterministically from ``seed``.
* With ``config.enable == False`` (the default) ``capture`` is an *exact*
  identity and returns the input tensor unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
import torch.nn as nn


__all__ = [
    "K_B",
    "Q_E",
    "DEFAULT_VDD",
    "DEFAULT_VCM",
    "DEFAULT_VU",
    "DEFAULT_CU",
    "DEFAULT_TEMP_C",
    "DEFAULT_N_SLOPE",
    "XFER_MODES",
    "TimeMuxConfig",
    "StageBoundary",
    "quantize_ste",
    "settle_factor",
    "transfer_full",
    "transfer_shared_vth",
    "transfer_quant",
    "transfer_kick",
    "transfer_no_tanh",
    "transfer_no_relu",
    "transfer_apply",
]


# --------------------------------------------------------------------------
# physical constants and defaults (Phase 3a section 2.4/2.5)
# --------------------------------------------------------------------------

K_B = 1.380649e-23       # J/K
Q_E = 1.602176634e-19    # C

DEFAULT_VDD = 0.8        # V
DEFAULT_VCM = 0.4        # V
DEFAULT_VU = 0.07        # V per unit
DEFAULT_CU = 50e-15      # F, physical capacitance of one unit
DEFAULT_TEMP_C = 85.0    # degC (worst case for the slope bound)
DEFAULT_N_SLOPE = 1.3    # subthreshold slope factor n

#: Interstage transfer variants (Phase 3b section 1.4).
XFER_MODES = ("full", "shared_vth", "quant", "kick", "no_tanh", "no_relu")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _lookup(obj, names, default=None):
    """Return the first present attribute of ``obj`` among ``names``."""
    if obj is None:
        return default
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _relu(x):
    """``F.relu`` without importing ``torch.nn.functional``."""
    return torch.clamp(x, min=0.0)


# --------------------------------------------------------------------------
# straight-through quantiser
# --------------------------------------------------------------------------

def quantize_ste(x, bits, lo=None, hi=None):
    """Uniform straight-through quantiser.

    Forward: the value is rounded onto a grid of ``2**bits`` levels that
    spans ``[lo, hi]``; the value is clipped to the range first.
    Backward: the gradient is the identity (1), so the quantiser is a no-op
    for optimisation (this is the "STE gradient is 1" requirement of Phase 3b
    section 2.4(a)).

    Args:
        x: tensor to quantise.
        bits: number of bits (>= 1).  ``None`` or ``<= 0`` returns ``x``
            unchanged.
        lo: lower end of the quantiser range.  If ``lo`` and ``hi`` are both
            ``None`` the range is taken from ``x`` itself (min/max, detached).
        hi: upper end of the quantiser range.

    Returns:
        A tensor with the same shape/dtype/device as ``x`` whose values lie on
        the quantiser grid and whose gradient w.r.t. ``x`` is 1.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"quantize_ste expects a tensor, got {type(x)!r}")
    if bits is None:
        return x
    bits = int(bits)
    if bits <= 0:
        return x

    if lo is None or hi is None:
        if x.numel() == 0:
            return x
        lo = x.detach().min().to(dtype=x.dtype, device=x.device)
        hi = x.detach().max().to(dtype=x.dtype, device=x.device)
    lo = float(lo)
    hi = float(hi)
    if hi <= lo:
        return x

    n_levels = float(2 ** bits)
    step = (hi - lo) / (n_levels - 1.0)
    xc = torch.clamp(x, lo, hi)
    q = torch.round((xc - lo) / step) * step + lo
    # forward value is exactly q, backward gradient is exactly 1
    return q.detach() + (x - x.detach())


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

@dataclass
class TimeMuxConfig:
    """Configuration of the time-multiplexed fabric model.

    Mirrors Phase 3b section 2.1 (field names and defaults).  Everything is
    OFF by default: ``enable=False`` means the legacy (ideal) code path.

    Attributes:
        enable: master switch of the time-multiplexing model.
        c_hold_ff: hold capacitance per sampled node, in fF.
        inject_mv: residual (post dummy-switch) charge-injection offset in mV,
            the constant part.  Its *sign* is a fixed per-node property.
        inject_gain_err: signal-dependent part of the injection error, a
            fraction (``x -> x * (1 + inject_gain_err)``).
        droop_mv: worst-case droop over one stage, in mV.  Deterministic, the
            sign is fixed per node.
        ktc: add kT/C noise at each sampling event (training only).
        settle: enable the bias-settle model on the frozen core currents
            (needs the previous stage's currents).
        c_int_ff: internal node capacitance used for tau_e, in fF.
        isat_floor: floor on the effective Isat (0 = off, 0.5 recommended
            together with ``settle``).
        xfer_mode: one of :data:`XFER_MODES`.
        param_bits: QAT bit width for the stage parameters (Isat, g, ratio,
            theta, G, W, transfer).  ``None`` disables parameter quantisation.
        node_quant_bits: bit width of the node-voltage quantiser at the
            boundary (``None`` = off; the doc's optional 8-bit quantiser is
            ``node_quant_bits=8``).
    """

    enable: bool = False
    c_hold_ff: float = 100.0
    inject_mv: float = 1.0
    inject_gain_err: float = 0.0
    droop_mv: float = 1.0
    ktc: bool = True
    settle: bool = False
    c_int_ff: float = 3.0
    isat_floor: float = 0.0
    xfer_mode: str = "full"
    param_bits: int | None = None
    node_quant_bits: int | None = None

    def __post_init__(self):
        self.enable = bool(self.enable)
        self.ktc = bool(self.ktc)
        self.settle = bool(self.settle)
        self.c_hold_ff = float(self.c_hold_ff)
        self.inject_mv = float(self.inject_mv)
        self.inject_gain_err = float(self.inject_gain_err)
        self.droop_mv = float(self.droop_mv)
        self.c_int_ff = float(self.c_int_ff)
        self.isat_floor = float(self.isat_floor)
        self.xfer_mode = str(self.xfer_mode)
        if self.xfer_mode not in XFER_MODES:
            raise ValueError(
                f"xfer_mode must be one of {XFER_MODES}, got {self.xfer_mode!r}"
            )
        if self.param_bits is not None:
            self.param_bits = int(self.param_bits)
            if self.param_bits <= 0:
                raise ValueError(f"param_bits must be >= 1, got {self.param_bits}")
        if self.node_quant_bits is not None:
            self.node_quant_bits = int(self.node_quant_bits)
            if self.node_quant_bits <= 0:
                raise ValueError(
                    f"node_quant_bits must be >= 1, got {self.node_quant_bits}"
                )

    # -- (de)serialisation --------------------------------------------------

    @classmethod
    def from_dict(cls, data):
        """Build a config from a (partial) mapping.

        Unknown keys are ignored, so a JSON file written for a newer revision
        of the hardware layer still loads.  Missing keys keep their default.
        """
        if data is None:
            return cls()
        if isinstance(data, cls):
            return cls(**data.to_dict())
        if not hasattr(data, "items"):
            raise TypeError(f"expected a mapping, got {type(data)!r}")
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)

    def to_dict(self):
        """Return the config as a plain dict (JSON-serialisable)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def is_off(self):
        """True when the time-multiplexing model does nothing at all."""
        return not self.enable

    def copy(self, **overrides):
        """Return a copy with ``overrides`` applied."""
        data = self.to_dict()
        data.update(overrides)
        return TimeMuxConfig.from_dict(data)


# --------------------------------------------------------------------------
# interstage transfer variants (pure functions)
# --------------------------------------------------------------------------
#
# The reference form is the one implemented by ``stage_transfer.StageTransfer``
# with ``activation="residual-relu-tanh"``:
#
#     x' = w1 * x + w2 * tanh(x) + w3 * relu(x - vth)
#
# All helpers below take the same five arguments so that they are
# interchangeable; ``w1, w2, w3, vth`` are scalars or per-node tensors that
# broadcast against ``x`` of shape (batch, nodes).

def transfer_full(x, w1, w2, w3, vth):
    """Reference transfer: ``w1*x + w2*tanh(x) + w3*relu(x - vth)``.

    Numerically identical to ``StageTransfer(activation="residual-relu-tanh")``.
    """
    return w1 * x + w2 * torch.tanh(x) + w3 * _relu(x - vth)


def transfer_shared_vth(x, w1, w2, w3, vth):
    """Transfer with a single ``vth`` shared by all nodes of the boundary.

    ``vth`` may be a scalar (one DAC for the whole boundary) or a per-node
    vector; in the latter case it is collapsed to its mean, which is the value
    a single shared DAC would have to be programmed to.
    """
    if isinstance(vth, torch.Tensor) and vth.numel() > 1:
        vth = vth.mean()
    return transfer_full(x, w1, w2, w3, vth)


def transfer_quant(x, w1, w2, w3, vth, bits=5, ranges=None):
    """Transfer with straight-through quantisation of the parameters.

    ``w1, w2, w3, vth`` are quantised onto ``2**bits`` levels before the
    reference transfer is applied.  ``ranges`` is an optional mapping
    ``{"w1": (lo, hi), ...}``; the default range is ``(-2, 2)`` for every
    parameter (the trial_0011 checkpoint spans w1 0.43..1.57, w2 -0.52..0.58,
    w3 -1.36..1.00, vth -1.26..0.92).
    """
    if bits is None or int(bits) <= 0:
        return transfer_full(x, w1, w2, w3, vth)
    ranges = ranges or {}
    out = []
    for name, value in (("w1", w1), ("w2", w2), ("w3", w3), ("vth", vth)):
        lo, hi = ranges.get(name, (-2.0, 2.0))
        out.append(quantize_ste(value, int(bits), lo, hi))
    qw1, qw2, qw3, qvth = out
    return transfer_full(x, qw1, qw2, qw3, qvth)


def transfer_kick(x, w1, w2, w3, vth, t_x=None, pulse_fraction=1.0):
    """Transfer realised as a short current pulse onto the existing node cap.

    The pulse realises ``delta = (w1-1)*x + w2*tanh(x) + w3*relu(x-vth)`` as
    ``x' = x + delta``; the circuit difference with :func:`transfer_full` is
    that no unity-buffer write-back path is needed (Phase 3b section 1.4).

    In the idealised model the pulse delivers the whole ``delta`` (the current
    simply scales as ``1/t_x``), so the numeric result is identical to
    :func:`transfer_full` -- this is the ``t_x -> 0`` limit that Phase 3b
    section 2.4(a) asks for.  ``t_x`` is accepted (and used by the hardware
    cost model) so that the signature stays the one of the doc; it does not
    change the value.  ``pulse_fraction`` (< 1) models a finite-current
    penalty: the pulse delivers only that fraction of ``delta``.
    """
    delta = (w1 - 1.0) * x + w2 * torch.tanh(x) + w3 * _relu(x - vth)
    return x + float(pulse_fraction) * delta


def transfer_no_tanh(x, w1, w2, w3, vth):
    """Transfer without the tanh term (``w2 == 0``): one OTA less per node."""
    return w1 * x + w3 * _relu(x - vth)


def transfer_no_relu(x, w1, w2, w3, vth):
    """Transfer without the rectifier term (``w3 == 0``)."""
    return w1 * x + w2 * torch.tanh(x)


def transfer_apply(x, w1, w2, w3, vth, mode="full", **kwargs):
    """Dispatch to the transfer variant named by ``mode``."""
    mode = str(mode)
    if mode == "full":
        return transfer_full(x, w1, w2, w3, vth)
    if mode == "shared_vth":
        return transfer_shared_vth(x, w1, w2, w3, vth)
    if mode == "quant":
        return transfer_quant(x, w1, w2, w3, vth, **kwargs)
    if mode == "kick":
        return transfer_kick(x, w1, w2, w3, vth, **kwargs)
    if mode == "no_tanh":
        return transfer_no_tanh(x, w1, w2, w3, vth)
    if mode == "no_relu":
        return transfer_no_relu(x, w1, w2, w3, vth)
    raise ValueError(f"unknown xfer_mode {mode!r}; expected one of {XFER_MODES}")


# --------------------------------------------------------------------------
# bias settling
# --------------------------------------------------------------------------

def settle_factor(tau_e, t_stage):
    """Fraction of the *old* current that survives the boundary.

    ``f = (tau_e / T) * (1 - exp(-T / tau_e))`` with ``f -> 0`` for a fast
    settling OTA (the slot reaches its new current) and ``f -> 1`` for an OTA
    that does not settle at all (it keeps its old current).
    """
    te = float(tau_e)
    t = float(t_stage)
    if t <= 0.0:
        return 0.0
    if te <= 0.0:
        return 0.0
    if math.isinf(te):
        return 1.0
    ratio = te / t
    return ratio * (-math.expm1(-1.0 / ratio))


# --------------------------------------------------------------------------
# the boundary module
# --------------------------------------------------------------------------

class StageBoundary(nn.Module):
    """Sample-and-hold boundary of a time-multiplexed fabric.

    ``capture`` models one boundary of the fabric:

    1. **kT/C noise** -- a Gaussian with sigma = ``ktc_rms_units`` is added to
       the sampled node voltages (training only, unless ``noise_eval``).
    2. **Charge injection** -- a constant offset ``inject_mv / vu`` is added;
       its *sign* is a fixed per-node property (a seeded buffer), because the
       charge injected by the sampling switch is always the same polarity.
    3. **Gain error** -- the signal-dependent part of the injection error:
       ``x -> x * (1 + inject_gain_err)``.
    4. **Droop** -- the hold capacitor leaks during the stage; the droop is a
       deterministic offset with a fixed sign per node.  It is applied to
       ``x_hold`` only, because the hold cap is what droops; the transfer
       reads the node, not the hold cap.
    5. **Quantisation** -- optional 8-bit straight-through quantisation of the
       node voltage over ``[-x_hard, x_hard]``.

    The returned pair is ``(x_hold, x_to_transfer)``:

    * ``x_hold`` is what the next stage sees through its frozen core read.
    * ``x_to_transfer`` is what the interstage transfer is applied to.

    Args:
        config: a :class:`TimeMuxConfig` (or a mapping that is converted to
            one).  ``None`` means the default (disabled) configuration.
        n_nodes: number of nodes that cross the boundary.
        units: optional duck-typed unit object (``hw.units.HWUnits`` or a
            stand-in) exposing ``vu``, ``cu``, ``temp_c``, ``ktc_rms_units``
            and ``x_hard``.
        vu: voltage per unit [V]; overrides ``units``.
        cu: physical capacitance of one unit [F]; overrides ``units``.
        temp_c: temperature [degC]; overrides ``units``.
        ktc_rms_units: kT/C rms noise in units; overrides ``units``.
        x_hard: hard rail in units; overrides ``units``.
        n_slope: subthreshold slope factor n (used for tau_e).
        quant_bits: bit width of the node-voltage quantiser (``None`` = off,
            the doc's optional 8-bit quantiser is ``quant_bits=8``).  When it
            is not given, ``config.node_quant_bits`` is used.
        noise_eval: keep the stochastic noise on in eval mode.
        seed: seed of the per-node sign buffers.
    """

    def __init__(
        self,
        config=None,
        n_nodes=18,
        units=None,
        *,
        vu=None,
        cu=None,
        temp_c=None,
        ktc_rms_units=None,
        x_hard=None,
        n_slope=DEFAULT_N_SLOPE,
        quant_bits=None,
        noise_eval=False,
        seed=0,
    ):
        super().__init__()

        if config is None:
            config = TimeMuxConfig()
        elif not isinstance(config, TimeMuxConfig):
            config = TimeMuxConfig.from_dict(config)
        self.config = config

        self.n_nodes = int(n_nodes)
        self.noise_eval = bool(noise_eval)
        if quant_bits is None:
            # the config owns the node-quantiser width (timemux.node_quant_bits)
            quant_bits = getattr(config, "node_quant_bits", None)
        self.quant_bits = None if quant_bits is None else int(quant_bits)
        self.n_slope = float(n_slope)
        self.seed = int(seed)

        # -- unit quantities (plain floats or from a duck-typed object) -----
        self.vu = float(_lookup(units, ("vu", "Vu"), DEFAULT_VU if vu is None else vu))
        self.cu = float(_lookup(units, ("cu", "Cu"), DEFAULT_CU if cu is None else cu))
        self.temp_c = float(
            _lookup(units, ("temp_c", "temperature", "temp"),
                    DEFAULT_TEMP_C if temp_c is None else temp_c)
        )
        self.x_hard = float(
            _lookup(units, ("x_hard",),
                    (DEFAULT_VDD / 2.0 / self.vu) if x_hard is None else x_hard)
        )
        ktcu = _lookup(units, ("ktc_rms_units", "ktc_rms"),
                       None if ktc_rms_units is None else ktc_rms_units)
        if ktcu is None:
            ktcu = math.sqrt(K_B * (self.temp_c + 273.15) / self.cu) / self.vu
        self.ktc_rms_units = float(ktcu)
        self.u_t = K_B * (self.temp_c + 273.15) / Q_E

        # -- per-node, fixed signs of the deterministic errors --------------
        # persistent=False: attaching this module must not change the
        # state_dict of the model (Phase 3b section 2.4(b)).
        self.register_buffer(
            "inject_sign", self._sign_buffer(self.seed), persistent=False
        )
        self.register_buffer(
            "droop_sign", self._sign_buffer(self.seed + 1), persistent=False
        )

        # -- the generator used when the caller does not supply one -------
        self._gen = torch.Generator()
        self._gen.manual_seed(self.seed)
        self.last_stage_idx = -1

    # -- construction helpers ---------------------------------------------

    def _sign_buffer(self, seed):
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        signs = torch.randint(0, 2, (self.n_nodes,), generator=gen, dtype=torch.float32)
        return signs * 2.0 - 1.0

    # -- derived quantities ------------------------------------------------

    @property
    def inject_units(self):
        """Charge-injection offset in units (mV -> units)."""
        return self.config.inject_mv * 1e-3 / self.vu

    @property
    def droop_units(self):
        """Droop offset in units (mV -> units)."""
        return self.config.droop_mv * 1e-3 / self.vu

    @property
    def kappa_int(self):
        """``kappa_int = n * U_T * C_int / (C_u * V_u)`` in model time units.

        With the defaults (3 fF, n = 1.3, 85 degC, 50 fF, 0.07 V) this is
        0.036 model time units = 2.7 ns, i.e. ``tau_e = kappa_int / Isat_eff``
        (Phase 3b section 1.5).
        """
        return (
            self.n_slope
            * self.u_t
            * self.config.c_int_ff
            * 1e-15
            / (self.cu * self.vu)
        )

    def effective_isat(self, isat):
        """Apply the Isat floor (``config.isat_floor``; 0 = off)."""
        floor = self.config.isat_floor
        if not floor:
            return isat
        if isinstance(isat, torch.Tensor):
            return torch.clamp(isat, min=float(floor))
        return max(float(isat), float(floor))

    def tau_e_units(self, isat_eff):
        """Bias-settling time constant in model time units.

        ``tau_e = kappa_int / Isat_eff`` (Phase 3b section 1.5).  ``inf`` is
        returned for a zero current (an off OTA never settles).
        """
        isat = float(self.effective_isat(float(isat_eff)))
        if isat <= 0.0:
            return math.inf
        return self.kappa_int / isat

    # -- the boundary ------------------------------------------------------

    def _randn(self, x, generator):
        gen = generator if generator is not None else self._gen
        try:
            noise = torch.randn(
                x.shape, generator=gen, dtype=x.dtype, device="cpu"
            )
        except (RuntimeError, TypeError):
            noise = torch.randn(x.shape, dtype=x.dtype, device="cpu")
        return noise.to(x.device)

    def capture(self, x_pre, stage_idx=0, generator=None):
        """Sample the nodes at a stage boundary.

        Args:
            x_pre: node voltages before the boundary, shape (batch, nodes).
            stage_idx: index of the boundary (0-based); recorded for
                bookkeeping and for callers that drive their own generator.
            generator: optional ``torch.Generator`` for the noise.  When it is
                ``None`` the module's own (seeded) generator is used.

        Returns:
            ``(x_hold, x_to_transfer)``, both of shape (batch, nodes).
        """
        if not isinstance(x_pre, torch.Tensor):
            raise TypeError(f"capture expects a tensor, got {type(x_pre)!r}")
        if x_pre.dim() < 1 or x_pre.shape[-1] != self.n_nodes:
            raise ValueError(
                f"expected the last dimension to be {self.n_nodes}, "
                f"got {tuple(x_pre.shape)}"
            )
        self.last_stage_idx = int(stage_idx)

        cfg = self.config
        if not cfg.enable:
            # exact identity: the legacy path is byte-for-byte unchanged
            return x_pre, x_pre

        x = x_pre

        # 1. kT/C noise (training only unless noise_eval)
        if cfg.ktc and (self.training or self.noise_eval):
            x = x + self._randn(x, generator) * self.ktc_rms_units

        # 2. constant charge-injection offset, fixed sign per node
        if cfg.inject_mv != 0.0:
            x = x + self.inject_sign.to(dtype=x.dtype, device=x.device) * self.inject_units

        # 3. signal-dependent part of the injection error
        if cfg.inject_gain_err != 0.0:
            x = x * (1.0 + cfg.inject_gain_err)

        # 4. droop: the hold cap leaks, the node does not
        x_to_transfer = x
        if cfg.droop_mv != 0.0:
            x_hold = x + self.droop_sign.to(dtype=x.dtype, device=x.device) * self.droop_units
        else:
            x_hold = x

        # 5. optional node-voltage quantisation
        if self.quant_bits:
            x_hold = quantize_ste(x_hold, self.quant_bits, -self.x_hard, self.x_hard)
            x_to_transfer = quantize_ste(
                x_to_transfer, self.quant_bits, -self.x_hard, self.x_hard
            )

        return x_hold, x_to_transfer

    # -- bias settling -----------------------------------------------------

    @staticmethod
    def settle_core(i_new, i_prev, tau_e, t_stage):
        """Effective frozen core current after a bias swap.

        ``I_new + (I_prev - I_new) * (tau_e / T) * (1 - exp(-T / tau_e))``

        ``i_new`` is the current the slot is programmed to, ``i_prev`` the
        current it carried during the previous stage, ``tau_e`` the bias
        settling time constant and ``t_stage`` the length of a stage, all in
        the same (model) time unit.
        """
        f = settle_factor(tau_e, t_stage)
        return i_new + (i_prev - i_new) * f

    # -- transfer ----------------------------------------------------------

    def apply_transfer(self, x, w1, w2, w3, vth):
        """Apply the interstage transfer selected by ``config.xfer_mode``."""
        mode = self.config.xfer_mode
        if mode == "quant" and self.config.param_bits:
            return transfer_quant(x, w1, w2, w3, vth, bits=self.config.param_bits)
        return transfer_apply(x, w1, w2, w3, vth, mode=mode)

    def extra_repr(self) -> str:
        return (
            f"n_nodes={self.n_nodes}, enable={self.config.enable}, "
            f"xfer_mode={self.config.xfer_mode!r}, "
            f"quant_bits={self.quant_bits}, noise_eval={self.noise_eval}"
        )
