"""Variation context for the analog KirchhoffNet.

A SimContext carries the PVT (process, voltage, temperature) corner plus
per-edge mismatch tensors for a single forward pass. Sample a new context
every training iteration to inject variation; for validation set
edge_mismatch=None.

Default variation magnitudes come from config.VARIATION.

Note (RR-C): ``temp_c`` is DEPRECATED. The dataclass field is retained for
backward compatibility but ``sample_random_context`` no longer randomizes it
(it always returns ``config.VARIATION["temp_c_default"]``). External code
that explicitly passes a custom ``temp_c`` value will receive a
:class:`DeprecationWarning` so existing notebooks continue to work while
encouraging migration.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
import random

import torch

from config import VARIATION


@dataclass
class SimContext:
    """PVT + mismatch context for one forward pass.

    Attributes:
        temp_c: Junction temperature in Celsius. **Deprecated**: the value
            is no longer used by the analog model and ``sample_random_context``
            always returns ``VARIATION["temp_c_default"]``. Kept for API
            compatibility; passing a non-default value emits a
            :class:`DeprecationWarning`.
        global_gain_shift: Scalar log-multiplicative shift on cell gm
            (e.g. +0.05 ≈ +5% global gm drift). Applied as gm *= exp(shift).
        global_isat_shift: Scalar log-multiplicative shift on cell isat.
            Applied as isat *= exp(shift).
        edge_mismatch: Optional [num_edges, num_cells] log-multiplicative
            mismatch tensor applied to gm. Pass None to disable.
        edge_isat_mismatch: Optional [num_edges, num_cells]
            log-multiplicative mismatch tensor applied to isat.
            Pass None to disable.
    """

    temp_c: float = VARIATION["temp_c_default"]
    global_gain_shift: float = 0.0
    global_isat_shift: float = 0.0
    edge_mismatch: torch.Tensor | None = None
    edge_isat_mismatch: torch.Tensor | None = None

    def __post_init__(self):
        if self.temp_c != VARIATION["temp_c_default"]:
            warnings.warn(
                "SimContext.temp_c is deprecated and ignored by the analog "
                "model. Remove explicit temp_c=... arguments.",
                DeprecationWarning,
                stacklevel=2,
            )

    def to(self, device, dtype=None):
        """Move mismatch tensors to device/dtype; return self for chaining."""
        kw = {"device": device}
        if dtype is not None:
            kw["dtype"] = dtype
        if self.edge_mismatch is not None:
            self.edge_mismatch = self.edge_mismatch.to(**kw)
        if self.edge_isat_mismatch is not None:
            self.edge_isat_mismatch = self.edge_isat_mismatch.to(**kw)
        return self

    def slice_edges(self, start: int, end: int) -> "SimContext":
        """Return a per-stage view of this context for edges [start:end).

        All scalar fields are propagated unchanged; both ``edge_mismatch``
        and ``edge_isat_mismatch`` tensors are sliced along dim 0 if present.
        """
        return SimContext(
            temp_c=self.temp_c,
            global_gain_shift=self.global_gain_shift,
            global_isat_shift=self.global_isat_shift,
            edge_mismatch=(
                None
                if self.edge_mismatch is None
                else self.edge_mismatch[start:end]
            ),
            edge_isat_mismatch=(
                None
                if self.edge_isat_mismatch is None
                else self.edge_isat_mismatch[start:end]
            ),
        )


def sample_random_context(
    num_edges: int,
    num_cells: int,
    *,
    temp_choices: list[float] | None = None,
    gain_shift_std: float | None = None,
    mismatch_std: float | None = None,
    global_isat_shift_std: float | None = None,
    isat_mismatch_std: float | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
    seed: int | None = None,
    legacy_temp: bool = False,
) -> SimContext:
    """Sample a random SimContext for a training iteration.

    If `gain_shift_std`, `mismatch_std`, `global_isat_shift_std`, or
    `isat_mismatch_std` are not provided, defaults from config.VARIATION
    are used. If `seed` is provided (and `generator` is not), a new
    generator is created from that seed.

    Note (RR-C): ``temp_c`` sampling is deprecated. By default
    ``sample_random_context`` returns ``VARIATION["temp_c_default"]`` (27.0).
    To preserve the old behaviour for legacy callers, pass
    ``legacy_temp=True`` together with ``temp_choices``; a one-off
    :class:`DeprecationWarning` is emitted to flag the path.
    """
    if gain_shift_std is None:
        gain_shift_std = float(VARIATION["global_gain_shift_std"])
    if mismatch_std is None:
        mismatch_std = float(VARIATION["edge_mismatch_std"])
    if global_isat_shift_std is None:
        global_isat_shift_std = float(
            VARIATION.get("global_isat_shift_std", 0.0)
        )
    if isat_mismatch_std is None:
        isat_mismatch_std = float(
            VARIATION.get("edge_isat_mismatch_std", 0.0)
        )
    if generator is None:
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(int(seed))
    gain_shift = (
        torch.randn(1, generator=generator, dtype=dtype, device=device).item()
        * gain_shift_std
    )
    isat_shift = (
        torch.randn(1, generator=generator, dtype=dtype, device=device).item()
        * global_isat_shift_std
    )
    if legacy_temp:
        warnings.warn(
            "legacy_temp=True samples temp_c from temp_choices. This path is "
            "deprecated; remove temp_c handling from your code.",
            DeprecationWarning,
            stacklevel=2,
        )
        choices = temp_choices if temp_choices is not None else list(VARIATION["temp_c_choices"])
        temp = random.choice(choices)
    else:
        temp = float(VARIATION["temp_c_default"])
    mismatch = (
        torch.randn(
            num_edges,
            num_cells,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        * mismatch_std
    )
    isat_mismatch = (
        torch.randn(
            num_edges,
            num_cells,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        * isat_mismatch_std
        if isat_mismatch_std > 0.0
        else None
    )
    return SimContext(
        temp_c=temp,
        global_gain_shift=gain_shift,
        global_isat_shift=isat_shift,
        edge_mismatch=mismatch,
        edge_isat_mismatch=isat_mismatch,
    )
