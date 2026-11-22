"""GLN (input-conditioned log-gm rails) as a per-family gm override.

``GLNRails.modulate_gm_z`` returns the modulated gm ``[B, E]`` for a family; this thin wrapper
returns ``None`` when GLN does not gate that family so callers fall back to the static gm.

Phase 3a 3.10 "tail" mode: the GLN modulates the tail current (Isat) instead of the tanh slope.
In that mode the override is an ``hw.constraints.IsatScale`` marker (a ``[B, E]`` multiplier on
Isat) and the slope stays at the cell's static (hardware) value.
"""
from __future__ import annotations

import torch

from hw.constraints import IsatScale

__all__ = ["gln_gm_override"]


def gln_gm_override(rails, family: str, cell_lib, z: torch.Tensor | None,
                    mode: str | None = None,
                    delta_max: float | None = None) -> torch.Tensor | None:
    if rails is None or z is None or family not in rails.families:
        return None
    if mode == "tail":
        # 3.10 option 1: Isat_e' = Isat_e * exp(clamp(delta, -delta_max, delta_max)).
        dm = float(delta_max) if delta_max is not None else rails_delta_max_default()
        return IsatScale(rails.modulate_isat_z(z, family, dm))
    return rails.modulate_gm_z(cell_lib.current_gm(), z, family)


def rails_delta_max_default() -> float:
    """The default ``delta_max`` of the tail mode (ln 4, the 3.10 number)."""
    from hw.constraints import GLN_DELTA_MAX_DEFAULT
    return GLN_DELTA_MAX_DEFAULT
