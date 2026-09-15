"""Low-rank causal tap rails for the Phase-8 tied write/retain probe.

The module deliberately has no knowledge of cells or integration.  Given the
eight causal delay-bank voltages for one sample, it returns one multiplier per
dynamic node.  The stage applies that multiplier to its complete KCL
accumulator *and* leak, leaving the rail clip outside the multiplier.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TapRails", "tap_rails_param_count"]


def _inv_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


class TapRails(nn.Module):
    """Shared causal rails and dense per-node tied gate mixes.

    ``r=tanh(softplus(alpha_raw)*(u @ a.T + c))`` and
    ``m=2*(1-sigmoid(beta + r @ w.T))``.  ``w=beta=0`` is an exact identity
    multiplier, while nonzero ``a`` prevents the rail path from being a dead
    saddle when ``w`` is first optimized.
    """

    def __init__(self, n_taps: int, n_nodes: int, B: int = 2,
                 alpha_init: float = 1.0) -> None:
        super().__init__()
        if n_taps < 1 or n_nodes < 1 or B not in (2, 3, 4):
            raise ValueError("TapRails requires positive dimensions and B in {2, 3, 4}")
        self.n_taps, self.n_nodes, self.B = int(n_taps), int(n_nodes), int(B)
        self.a = nn.Parameter(torch.randn(B, n_taps) * 0.01)
        self.c = nn.Parameter(torch.zeros(B))
        self.alpha_raw = nn.Parameter(torch.full((B,), _inv_softplus(alpha_init)))
        # Identity is carried by the per-node mix and bias, not the rails.
        self.w = nn.Parameter(torch.zeros(n_nodes, B))
        self.beta = nn.Parameter(torch.zeros(n_nodes))

    def rails(self, taps: torch.Tensor) -> torch.Tensor:
        if taps.dim() != 2 or taps.shape[1] != self.n_taps:
            raise ValueError(f"expected taps [batch, {self.n_taps}], got {tuple(taps.shape)}")
        alpha = F.softplus(self.alpha_raw) + 1e-6
        return torch.tanh(alpha * (taps @ self.a.T + self.c))

    def gate_and_multiplier(self, taps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.rails(taps)
        z = torch.sigmoid(r @ self.w.T + self.beta)
        return z, 2.0 * (1.0 - z)

    def forward(self, taps: torch.Tensor) -> torch.Tensor:
        return self.gate_and_multiplier(taps)[1]


def tap_rails_param_count(n_taps: int = 8, n_nodes: int = 25, B: int = 2) -> int:
    """Analytic count: rails ``B*(taps+2)`` plus node mixes ``N*(B+1)``."""
    return int(B) * (int(n_taps) + 2) + int(n_nodes) * (int(B) + 1)
