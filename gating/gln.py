"""GLN (input-conditioned log-gm rails) as a per-family gm override.

``GLNRails.modulate_gm_z`` returns the modulated gm ``[B, E]`` for a family; this thin wrapper
returns ``None`` when GLN does not gate that family so callers fall back to the static gm.
"""
from __future__ import annotations

import torch

__all__ = ["gln_gm_override"]


def gln_gm_override(rails, family: str, cell_lib, z: torch.Tensor | None) -> torch.Tensor | None:
    if rails is None or z is None or family not in rails.families:
        return None
    return rails.modulate_gm_z(cell_lib.current_gm(), z, family)
