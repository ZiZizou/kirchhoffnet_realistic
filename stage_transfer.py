"""StageTransfer: width-changing layer between DifferentialStages.

Truncation or zero-padding. No learnable parameters. New nodes (when
out_nodes > in_nodes) are initialized to zero differential state. An
optional pointwise non-linearty (``activation="relu"``) is applied after
the width adjustment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["StageTransfer", "STAGE_TRANSFER_VALID_ACTIVATIONS"]

STAGE_TRANSFER_VALID_ACTIVATIONS = ("none", "relu")


class StageTransfer(nn.Module):
    """Truncation/zero-padding between stages of differing width.

    Args:
        in_nodes: Number of input nodes (state width of the upstream stage).
        out_nodes: Number of output nodes (state width of the downstream
            stage).
        activation: Optional pointwise non-linearity applied after the
            width adjustment. ``"none"`` (default) keeps the transfer as a
            pure identity/truncate/pad. ``"relu"`` applies ``F.relu`` to
            the resulting state vector.
    """

    def __init__(self, in_nodes: int, out_nodes: int, activation: str = "none") -> None:
        super().__init__()
        if in_nodes < 0 or out_nodes < 0:
            raise ValueError(f"Negative node count: in={in_nodes}, out={out_nodes}")
        if activation not in STAGE_TRANSFER_VALID_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {STAGE_TRANSFER_VALID_ACTIVATIONS}, "
                f"got {activation!r}"
            )
        self.in_nodes = int(in_nodes)
        self.out_nodes = int(out_nodes)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.in_nodes:
            raise ValueError(
                f"Expected x shape (batch, {self.in_nodes}), got {tuple(x.shape)}"
            )
        if self.out_nodes == self.in_nodes:
            out = x
        elif self.out_nodes < self.in_nodes:
            out = x[:, : self.out_nodes]
        else:
            pad = x.new_zeros(x.size(0), self.out_nodes - self.in_nodes)
            out = torch.cat([x, pad], dim=1)

        if self.activation == "relu":
            out = F.relu(out)
        return out

    def extra_repr(self) -> str:
        return f"in_nodes={self.in_nodes}, out_nodes={self.out_nodes}, activation={self.activation}"
