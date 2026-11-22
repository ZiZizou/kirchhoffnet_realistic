"""Input-conditioned edge gating (VCA gain, GLN gm modulation) behind two small interfaces.

    vca.vca_edge_gain(u, W, v_e, bias) -> [B, E]   multiplicative gain on edge currents
    gln.gln_gm_override(rails, family, cell_lib, z) -> [B, E] | None   gm fed into the tanh

``DifferentialStage.edge_gain(family, u)`` and ``DifferentialStage.edge_gm_scale(family,
cell_lib, z)`` wrap them so the stage right-hand side applies one gain multiply and one
gm override per edge family. Parameter attributes stay on the stage (state_dict keys unchanged).
"""
from gating.vca import vca_edge_gain
from gating.gln import gln_gm_override

__all__ = ["vca_edge_gain", "gln_gm_override"]
