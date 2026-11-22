"""VCA (voltage-controlled amplifier) edge gain.

gain_e(u) = 2 * sigmoid((u @ W) @ v_e.T [+ b_e])   shape [B, E]

``W`` is the shared (in_dim, rank) projection, ``v_e`` the (E, rank) per-edge embedding and
``b_e`` an optional per-edge bias. With W = 0 (the init) the gain is exactly 1.
"""
from __future__ import annotations

import torch

__all__ = ["vca_edge_gain"]


def vca_edge_gain(u: torch.Tensor, W: torch.Tensor, v_e: torch.Tensor,
                  bias: torch.Tensor | None = None) -> torch.Tensor:
    u_proj = u @ W                 # [B, rank]
    logits = u_proj @ v_e.T        # [B, E]
    if bias is not None:
        logits = logits + bias
    return 2.0 * torch.sigmoid(logits)
