"""HWUnits: the single source of truth for the model-unit <-> physical-unit mapping.

Phase 3a section 2.5 (frozen dataclass, primary knobs + derived properties) with the two
additions of Phase 3b section 2.1 (``t_ovh_per_boundary`` / ``tu_effective``) and the
``i_spec`` design-class field of Phase 3a section 3.3.

This module imports nothing from the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

K_B, Q_E = 1.380649e-23, 1.602176634e-19

__all__ = ["HWUnits", "K_B", "Q_E"]


@dataclass(frozen=True)
class HWUnits:
    # --- primary (the only knobs) ---
    vdd: float = 0.8                  # V
    vcm: float = 0.4                  # V
    vu: float = 0.07                  # V per unit
    cu: float = 50e-15                # F, physical capacitance of C_eff = 1
    t_total_phys: float = 350e-9      # s, whole inference
    t_total_units: float = 4.660583   # model t_span (circuit_model_report.md:19)
    n_slope: float = 1.3              # subthreshold slope factor (ASSUMPTION)
    temp_c: float = 85.0              # worst case for the slope bound
    x_max: float = 4.0                # soft / design rail in units
    # --- Phase 3a 3.3: specific current of the OTA input pair (design-class field) ---
    i_spec: float = 2e-6              # A (W/L giving I_spec = 2 uA, 3a 3.3 "Numbers")
    # --- Phase 3b 2.1: time-multiplexing overhead (0 => Tu unchanged) ---
    t_ovh_per_boundary: float = 0.0   # s of non-integrating time per stage boundary
    n_boundaries: int = 6             # 7 stages => 6 boundaries (the "6" of 3b 2.1)

    # --- derived ---
    @property
    def tu(self) -> float:
        return self.t_total_phys / self.t_total_units

    @property
    def tu_effective(self) -> float:
        """Seconds per model time unit once the boundary overheads are subtracted."""
        return (self.t_total_phys - self.n_boundaries * self.t_ovh_per_boundary) / self.t_total_units

    @property
    def iu(self) -> float:
        return self.cu * self.vu / self.tu

    @property
    def gu(self) -> float:            # = iu / vu
        return self.cu / self.tu

    @property
    def u_t(self) -> float:
        return K_B * (self.temp_c + 273.15) / Q_E

    @property
    def slope_max(self) -> float:
        """Max tanh slope per unit: vu / (2 n U_T). 0.872 at vu=.07, 85C, n=1.3."""
        return self.vu / (2.0 * self.n_slope * self.u_t)

    @property
    def x_hard(self) -> float:
        return self.vdd / 2.0 / self.vu

    @property
    def x_swing_v(self) -> float:
        return self.x_max * self.vu

    @property
    def ktc_rms_units(self) -> float:
        return math.sqrt(K_B * (self.temp_c + 273.15) / self.cu) / self.vu

    # --- conversions ---
    def v_to_x(self, v):
        return (v - self.vcm) / self.vu

    def x_to_v(self, x):
        return self.vcm + self.vu * x

    def isat_to_amps(self, i):
        return i * self.iu

    def g_to_siemens(self, g):
        return g * self.gu

    def t_to_s(self, t):
        return t * self.tu

    def power_w(self, isat_sum_units: float) -> float:
        return isat_sum_units * self.iu * self.vdd
