"""StageTransfer: width-changing layer between DifferentialStages.

Truncation or zero-padding. New nodes (when out_nodes > in_nodes) are
initialized to zero differential state. An optional pointwise transform
applied after the width adjustment:

* ``activation="none"`` (default): pure identity / truncate / pad.
* ``activation="relu"``: ``F.relu`` on the state vector.
* ``activation="residual"``: per-node learnable residual
  ``W1 * x + W2 * tanh(x)`` on non-driven nodes. Driven nodes (those
  receiving persistent drive current) pass through as identity. W1
  is initialized to 1 and W2 to 0, so the network starts as identity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["StageTransfer", "STAGE_TRANSFER_VALID_ACTIVATIONS"]

STAGE_TRANSFER_VALID_ACTIVATIONS = ("none", "relu", "residual")


class StageTransfer(nn.Module):
    """Truncation/zero-padding between stages of differing width.

    Args:
        in_nodes: Number of input nodes (state width of the upstream stage).
        out_nodes: Number of output nodes (state width of the downstream
            stage).
        activation: Optional transform applied after the width adjustment.
            ``"none"`` (default): identity / truncate / pad.
            ``"relu"``: ``F.relu`` on the state vector.
            ``"residual"``: per-node learnable
            ``residual_w1 * x + residual_w2 * tanh(x)`` on non-driven
            nodes; driven nodes pass through as identity. Adds
            ``out_nodes`` learnable parameters per side (W1 init=1, W2
            init=0).
        drive_mask: Optional list of node indices that should pass
            through unchanged (identity) when ``activation="residual"``.
            For other activations the mask is ignored. ``None`` (default)
            means no nodes are treated as driven, so the residual
            transform is applied to all nodes.
    """

    def __init__(
        self,
        in_nodes: int,
        out_nodes: int,
        activation: str = "none",
        drive_mask: list[int] | None = None,
    ) -> None:
        super().__init__()
        if in_nodes < 0 or out_nodes < 0:
            raise ValueError(f"Negative node count: in={in_nodes}, out={out_nodes}")
        if activation not in STAGE_TRANSFER_VALID_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {STAGE_TRANSFER_VALID_ACTIVATIONS}, "
                f"got {activation!r}"
            )
        if drive_mask is not None:
            if any(i < 0 or i >= out_nodes for i in drive_mask):
                raise ValueError(
                    f"drive_mask entries must be in [0, {out_nodes}), got {drive_mask}"
                )
        self.in_nodes = int(in_nodes)
        self.out_nodes = int(out_nodes)
        self.activation = activation

        if activation == "residual":
            self.residual_w1 = nn.Parameter(torch.full((self.out_nodes,), 1.0))
            self.residual_w2 = nn.Parameter(torch.full((self.out_nodes,), 0.0))
        else:
            self.register_parameter("residual_w1", None)
            self.register_parameter("residual_w2", None)

        if drive_mask is not None:
            self.register_buffer(
                "_drive_mask", torch.tensor(list(drive_mask), dtype=torch.long)
            )
        else:
            self.register_buffer("_drive_mask", torch.empty(0, dtype=torch.long))

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
        elif self.activation == "residual":
            transformed = self.residual_w1 * out + self.residual_w2 * torch.tanh(out)
            if self._drive_mask.numel() > 0:
                transformed[:, self._drive_mask] = out[:, self._drive_mask]
            out = transformed
        return out

    def extra_repr(self) -> str:
        parts = [
            f"in_nodes={self.in_nodes}",
            f"out_nodes={self.out_nodes}",
            f"activation={self.activation}",
        ]
        if self._drive_mask.numel() > 0:
            parts.append(f"drive_nodes={self._drive_mask.numel()}")
        return ", ".join(parts)
