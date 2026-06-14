"""StageTransfer: width-changing layer between DifferentialStages.

Truncation or zero-padding. No learnable parameters. New nodes (when
out_nodes > in_nodes) are initialized to zero differential state.
"""

from __future__ import annotations

import torch
import torch.nn as nn


__all__ = ["StageTransfer"]


class StageTransfer(nn.Module):
    """Truncation/zero-padding between stages of differing width."""

    def __init__(self, in_nodes: int, out_nodes: int) -> None:
        super().__init__()
        if in_nodes < 0 or out_nodes < 0:
            raise ValueError(f"Negative node count: in={in_nodes}, out={out_nodes}")
        self.in_nodes = int(in_nodes)
        self.out_nodes = int(out_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.in_nodes:
            raise ValueError(
                f"Expected x shape (batch, {self.in_nodes}), got {tuple(x.shape)}"
            )
        if self.out_nodes == self.in_nodes:
            return x
        if self.out_nodes < self.in_nodes:
            return x[:, : self.out_nodes]
        pad = x.new_zeros(x.size(0), self.out_nodes - self.in_nodes)
        return torch.cat([x, pad], dim=1)

    def extra_repr(self) -> str:
        return f"in_nodes={self.in_nodes}, out_nodes={self.out_nodes}"
