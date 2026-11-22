"""DifferentialStage: sparse COO graph + KCL scatter-add + Heun integration.

Per-node dynamics:
    C_eff * dx_j/dt = sum_in I_edge - sum_out I_edge - leak_j * x_j - clip_j(x_j)

Edge currents are computed by a per-edge device library.
Heun integration (predictor-corrector, 2nd order) is used for fixed-step
BPTT. The stage returns both the final state and the full trajectory so
that regularizers can be evaluated along the path.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    INIT,
    PHYS,
    REF,
    SOLVER,
    VCA,
)
from cell_library import (
    FreeTanhLibrary,
)
from gating.gln import gln_gm_override
from gating.vca import isat_eff_scale, vca_edge_gain

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from hw import HWConfig


__all__ = ["DifferentialStage"]


class DifferentialStage(nn.Module):
    """A single stage of the reduced differential KirchhoffNet.

    Args:
        num_nodes: Number of differential nodes in this stage's internal state.
        src: List of source node ids (length E).
        dst: List of destination node ids (length E).
        cell_lib: Edge device library used to compute edge currents.
        c_eff: Effective node capacitance (default from config).
        x_max: Differential rail limit (default from config).
        clip_current: Soft rail clip current magnitude (default from config).
        clip_softness: Soft rail clip transition width (default from config).
        learnable_clip_sharpness: When ``True``, the soft-rail clip
            denominator ``s`` becomes a per-stage learnable scalar
            (``clip_sharpness_raw``) mapped into
            ``[clip_sharpness_min, clip_sharpness_max]`` via sigmoid and
            logit-initialized so the mapped value equals
            ``clip_sharpness_init`` at startup (epoch-0 forward identical to
            the fixed path when ``clip_sharpness_init == clip_softness``).
            When ``False`` (default) ``s`` is the fixed ``clip_softness``
            float and the forward is bit-identical to the pre-F1 baseline.
        clip_sharpness_init: Initial mapped sharpness for the learnable
            path (default from config ``PHYS['clip_sharpness_init']``).
        clip_sharpness_min: Lower bound of the mapped learnable sharpness
            (default from config ``PHYS['clip_sharpness_min']``).
        clip_sharpness_max: Upper bound of the mapped learnable sharpness
            (default from config ``PHYS['clip_sharpness_max']``).
        leak_mode: ``"programmable"`` (default) creates a learnable per-node
            ``raw_leak`` parameter. ``"non-programmable"`` uses a fixed scalar
            ``leak_constant`` (see ``leak_constant``) for all nodes, saving
            parameters and eliminating leak gradients.
        leak_constant: Fixed leak value used when ``leak_mode="non-programmable"``.
            When ``None``, defaults to ``config.INIT["leak_constant"]``
            (0.0486, matching ``softplus(raw_leak_init)``). Ignored when
            ``leak_mode="programmable"``.
        freeze_read: When ``True``, edge currents (cell_lib output, edge gate,
            and KCL scatter-add) are computed **once** at the
            start of the stage from the initial state ``x0`` and held constant
            throughout all Heun sub-iterations. Leak, clip, and drive
            current still read the evolving ``x`` at each step. Default
            ``False`` (standard behavior: read and write at the same time).
        freeze_boundary: When ``True``, boundary fan-out edge currents (the
            tanh cell contribution with edge gate + VCA gate folded in) are
            computed **once** from ``(u, x0)`` and held constant across all
            Heun sub-iterations, while the family's resistive shunt
            (when present) remains dynamic per-step. Mirrors the
            ``freeze_read`` semantics for the boundary edge family.
            Independent of ``freeze_read`` (can be combined). No-op when the
            stage has no boundary edges. Default ``False``.
        freeze_temporal_read: When ``True``, temporal-readout edge currents
            (the tanh cell contribution with edge gate + VCA gate folded
            in) are computed once from ``x0`` and held constant across all
            Heun iterations. The family's resistive shunt (when
            present) stays dynamic per-step, mirroring the
            ``freeze_read`` behavior for the core graph. Independent of
            ``freeze_read`` and ``freeze_boundary`` (can be combined). No-op
            when the stage has no temporal-readout edges.
            In ``readout_mode="shared_sense"`` the same flag dispatches
            ``_compute_frozen_sense`` (returning ``[B, n_sense]`` per-sense
            currents) and the shared branch's frozen path uses those as the
            tanh contribution while the resistive shunt stays dynamic.
            Default ``False``.
        boundary_src: List of input-terminal indices for sparse OTA edges
            from fixed-voltage boundary terminals into the dynamic fabric.
            Length must equal ``len(boundary_dst)``. ``None`` (default)
            disables boundary edges for this stage.
        boundary_dst: List of target dynamic-node indices for the boundary
            OTA edges (same length as ``boundary_src``). Indices are in
            the compact 0..num_nodes-1 coordinate space, matching
            ``write_idx``.
        boundary_cell_lib: Cell library instance used to compute boundary
            edge currents ``I_OTA(u_i, x_j)``. Must match the cell type
            of the core ``cell_lib`` and be sized for ``len(boundary_src)``
            edges. Must be provided when boundary edges are configured.
        output_ode_src: List of source node indices (compact 0..num_nodes-1
            coordinate space) for the temporal-readout OTA edges. These edges
            inject current from a hidden (or projection) node into an output
            ODE accumulator node. The source is read-only (its voltage drives
            the OTA current but no current is drained from the source); the
            output ODE node is the writable destination. Length must equal
            ``len(output_ode_dst)`` when provided. ``None`` (default)
            disables temporal-readout edges for this stage.
        output_ode_dst: List of destination node indices (compact 0..num_nodes-1
            coordinate space, same length as ``output_ode_src``) for the
            temporal-readout OTA edges. Indices typically lie in the
            output-ode accumulator region (e.g., ``[core_count, num_nodes)``),
            but any valid node index is permitted so a hidden→hidden
            temporal-readout edge is also expressible.
        output_ode_cell_lib: Cell library instance used to compute temporal
            readout edge currents ``I_OTA(x_src, x_dst)``. Must match the
            cell type of the core ``cell_lib`` and be sized for
            ``len(output_ode_src)`` edges. Required when
            ``output_ode_src``/``output_ode_dst`` are provided.
        readout_mode: ``"ota_mesh"`` (default) keeps today's h*d_out
            temporal-readout OTA mesh (each edge has its own cell, gate, and
            VCA tap). ``"shared_sense"`` replaces the mesh with one sense
            OTA per hidden node (optionally two at different operating
            points, see ``readout_senses_per_node``) plus a plain dense
            crossbar ``readout_crossbar_W`` that mixes the sense currents
            into the output-ODE accumulator tail. Exclusive with
            ``output_ode_src``/``output_ode_dst`` (pass ``None`` for those in
            shared mode).
        readout_senses_per_node: ``1`` or ``2``; only meaningful when
            ``readout_mode="shared_sense"``. ``2`` builds two sense OTAs per
            hidden node (the second copy is init-jittered by the topology
            builder so the operating points differ).
        readout_sense_src: For ``shared_sense`` only: list of source node
            indices (compact ``0..num_nodes-1``), one entry per sense cell
            (``s*h`` entries, hidden node ``h`` repeated ``s`` times).
        readout_sense_cell_lib: For ``shared_sense`` only: per-stage-owned
            cell library sized ``len(readout_sense_src)``, used to compute
            sense currents ``I_OTA(x_src_j, Vref)`` against the stage's
            private learnable ``Vref`` rail.
        readout_crossbar_shape: For ``shared_sense`` only: ``(d_out,
            n_sense)`` where ``d_out`` is the number of output-ODE
            accumulator nodes at the state tail and ``n_sense ==
            len(readout_sense_src)``. The caller passes the shape explicitly
            because the stage does not know ``d_out`` directly.
    """

    def __init__(
        self,
        num_nodes: int,
        src: list[int],
        dst: list[int],
        cell_lib: FreeTanhLibrary,
        c_eff: float | None = None,
        x_max: float | None = None,
        clip_current: float | None = None,
        clip_softness: float | None = None,
        learnable_clip_sharpness: bool = False,
        clip_sharpness_init: float | None = None,
        clip_sharpness_min: float | None = None,
        clip_sharpness_max: float | None = None,
        gln_rails: "GLNRails | None" = None,
        leak_mode: str = "programmable",
        leak_constant: float | None = None,
        read_only_source: bool = False,
        freeze_read: bool = False,
        freeze_boundary: bool = False,
        freeze_temporal_read: bool = False,
        boundary_src: list[int] | None = None,
        boundary_dst: list[int] | None = None,
        boundary_cell_lib: FreeTanhLibrary | None = None,
output_ode_src: list[int] | None = None,
        output_ode_dst: list[int] | None = None,
        output_ode_cell_lib: FreeTanhLibrary | None = None,
        readout_mode: str = "ota_mesh",
        readout_senses_per_node: int = 1,
        readout_sense_src: list[int] | None = None,
        readout_sense_cell_lib: FreeTanhLibrary | None = None,
        readout_crossbar_shape: tuple[int, int] | None = None,
        vca_enabled: bool = False,
        vca_rank: int = 2,
        vca_in_dim: int = 0,
        vca_core_enabled: bool = False,
        vca_gate_shunt: bool = False,
        vca_separate_core_bus: bool = False,
        vca_bias: bool | None = None,
        hw: "HWConfig | None" = None,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.cell_lib = cell_lib
        # Hardware configuration (``hw.HWConfig``) or ``None``.  ``None`` keeps
        # the pre-constraint behaviour bit-for-bit.  Plain attribute (not a
        # parameter/buffer), so the state_dict is unchanged.
        self.hw = hw
        self.read_only_source = read_only_source
        self.freeze_read = bool(freeze_read)
        self.freeze_boundary = bool(freeze_boundary)
        self.freeze_temporal_read = bool(freeze_temporal_read)
        if leak_mode not in ("programmable", "non-programmable"):
            raise ValueError(f"leak_mode must be 'programmable' or 'non-programmable', got {leak_mode!r}")
        self.leak_mode = leak_mode
        self.vca_enabled = bool(vca_enabled)
        self.vca_rank = int(vca_rank) if vca_enabled else int(vca_rank)
        self._vca_in_dim = int(vca_in_dim)
        self.vca_bias_enabled = bool(VCA.get("bias", False) if vca_bias is None else vca_bias)
        if self.vca_enabled and self.vca_rank < VCA["min_rank"]:
            raise ValueError(
                f"vca_rank must be >= {VCA['min_rank']}, got {self.vca_rank}"
            )

        self.c_eff = float(c_eff if c_eff is not None else PHYS["C_eff"])
        self.x_max = float(x_max if x_max is not None else PHYS["x_max"])
        self.clip_current = float(clip_current if clip_current is not None else PHYS["clip_current"])
        self.clip_softness = float(clip_softness if clip_softness is not None else PHYS["clip_softness"])
        # Learnable soft-rail sharpness (F1): one scalar per stage. Off-path
        # (default) keeps the fixed ``clip_softness`` float so the rhs is
        # bit-identical to the pre-F1 baseline. On-path maps
        # ``clip_sharpness_raw`` via sigmoid into
        # ``[clip_sharpness_min, clip_sharpness_max]``, logit-initialized so
        # the mapped value equals ``clip_sharpness_init`` at startup.
        self._learnable_clip_sharpness = bool(learnable_clip_sharpness)
        self.clip_sharpness_init = float(
            clip_sharpness_init if clip_sharpness_init is not None
            else PHYS["clip_sharpness_init"]
        )
        self.clip_sharpness_min = float(
            clip_sharpness_min if clip_sharpness_min is not None
            else PHYS["clip_sharpness_min"]
        )
        self.clip_sharpness_max = float(
            clip_sharpness_max if clip_sharpness_max is not None
            else PHYS["clip_sharpness_max"]
        )
        if self.clip_sharpness_max <= self.clip_sharpness_min:
            raise ValueError(
                f"clip_sharpness_max must be > clip_sharpness_min, got "
                f"[{self.clip_sharpness_min}, {self.clip_sharpness_max}]"
            )
        if self._learnable_clip_sharpness:
            t = (self.clip_sharpness_init - self.clip_sharpness_min) / (
                self.clip_sharpness_max - self.clip_sharpness_min
            )
            t = float(min(max(t, 1e-4), 1.0 - 1e-4))
            raw0 = math.log(t / (1.0 - t))
            self.clip_sharpness_raw = nn.Parameter(
                torch.tensor(raw0, dtype=torch.float32)
            )
        else:
            # Fixed path: no parameter, no buffer — the float stays in
            # ``self.clip_softness`` (kept for backward compat with the
            # pre-F1 rhs and state_dicts).
            self.clip_sharpness_raw = None
        # GLN rails (F2): one shared GLNRails module for the whole net,
        # owned/registered by the enclosing KirchhoffNetWithIO (never here).
        # Stored as a plain attribute (bypassing nn.Module auto-registration)
        # so stage.parameters()/state_dict() do not duplicate the shared
        # tensors across stages. ``None`` disables GLN (default).
        object.__setattr__(self, "gln_rails", gln_rails)

        if len(src) != len(dst):
            raise ValueError(f"src/dst length mismatch: {len(src)} vs {len(dst)}")
        if any(s == d for s, d in zip(src, dst)):
            raise ValueError("Self-loops are not allowed in DifferentialStage edges.")
        if any(s < 0 or s >= num_nodes or d < 0 or d >= num_nodes for s, d in zip(src, dst)):
            raise ValueError("Edge endpoint out of range for num_nodes.")

        self.register_buffer("src", torch.tensor(src, dtype=torch.long))
        self.register_buffer("dst", torch.tensor(dst, dtype=torch.long))

        if self.leak_mode == "programmable":
            self.raw_leak = nn.Parameter(torch.full((num_nodes,), float(INIT["raw_leak_init"])))
        else:
            self.leak_constant = float(leak_constant if leak_constant is not None else INIT["leak_constant"])

        # Edge gate: z_e = sigmoid(z_logits) multiplies the edge current.
        z_init = float(INIT.get("z_logit_init", 5.0))
        self.z_logits = nn.Parameter(torch.full((len(src),), z_init))

        # Parallel resistive shunt (FreeTanhLibrary): if the cell library
        # exposes a ``resistive_current(x_src, x_dst)`` method, route an
        # additional current ``G * (Vsrc - Vdest)`` per edge from evolving
        # voltages, bypassing ``freeze_read``. Cached here so the dispatch
        # in ``rhs`` is a fast attribute check instead of a hasattr scan.
        self._has_resistive = hasattr(cell_lib, "resistive_current")

        # Boundary-terminal OTA edges (boundary-fan-out plan).
        # Optional sparse programmable edges from fixed-voltage input
        # terminals (carried in ``u``) into dynamic-node targets. The
        # boundary terminals are ideal voltage sources: current flows
        # only into the destination node, the terminal voltage is never
        # drained. Gated by a separate ``boundary_z_logits`` parameter
        # so they can be pruned/trained independently of the core edges.
        if boundary_src is not None or boundary_dst is not None:
            if boundary_src is None or boundary_dst is None:
                raise ValueError(
                    "DifferentialStage: boundary_src and boundary_dst must "
                    "be provided together (got one without the other)"
                )
            if len(boundary_src) != len(boundary_dst):
                raise ValueError(
                    f"DifferentialStage: boundary_src/dst length mismatch: "
                    f"{len(boundary_src)} vs {len(boundary_dst)}"
                )
            if any(s < 0 or s >= self.num_nodes for s in boundary_dst):
                raise ValueError(
                    f"DifferentialStage: boundary_dst entries must be in "
                    f"[0, {self.num_nodes}), got {boundary_dst}"
                )
            if any(s < 0 for s in boundary_src):
                raise ValueError(
                    f"DifferentialStage: boundary_src entries must be "
                    f"non-negative, got {boundary_src}"
                )
            if boundary_cell_lib is None:
                raise ValueError(
                    "DifferentialStage: boundary_cell_lib is required when "
                    "boundary_src/boundary_dst are provided"
                )
            self.register_buffer(
                "boundary_src", torch.tensor(boundary_src, dtype=torch.long),
            )
            self.register_buffer(
                "boundary_dst", torch.tensor(boundary_dst, dtype=torch.long),
            )
            self.boundary_cell_lib = boundary_cell_lib
            self.boundary_z_logits = nn.Parameter(
                torch.full((len(boundary_src),), z_init),
            )
            self._has_boundary = True
        else:
            self.register_buffer(
                "boundary_src", torch.empty(0, dtype=torch.long),
            )
            self.register_buffer(
                "boundary_dst", torch.empty(0, dtype=torch.long),
            )
            self.boundary_cell_lib = None
            self.boundary_z_logits = None
            self._has_boundary = False

        # Temporal-readout OTA edges (temporal-readout plan).
        # Sparse programmable edges from hidden/projection nodes (read-only
        # source) into the output ODE accumulator nodes (writable destination).
        # The output ODE nodes are the last ``output_ode_count`` entries of the
        # state vector and are part of the ODE dynamics (they receive leak,
        # clip, and the OTA current injected here). The source node is never
        # drained — only the destination receives current — matching the
        # boundary-fan-out pattern.
        if output_ode_src is not None or output_ode_dst is not None:
            if output_ode_src is None or output_ode_dst is None:
                raise ValueError(
                    "DifferentialStage: output_ode_src and output_ode_dst "
                    "must be provided together (got one without the other)"
                )
            if len(output_ode_src) != len(output_ode_dst):
                raise ValueError(
                    f"DifferentialStage: output_ode_src/dst length mismatch: "
                    f"{len(output_ode_src)} vs {len(output_ode_dst)}"
                )
            if any(
                s < 0 or s >= self.num_nodes or d < 0 or d >= self.num_nodes
                for s, d in zip(output_ode_src, output_ode_dst)
            ):
                raise ValueError(
                    f"DifferentialStage: output_ode_src/dst entries must be "
                    f"in [0, {self.num_nodes}), got src={output_ode_src} "
                    f"dst={output_ode_dst}"
                )
            if any(s == d for s, d in zip(output_ode_src, output_ode_dst)):
                raise ValueError(
                    "DifferentialStage: self-loops are not allowed in "
                    "output_ode edges."
                )
            if output_ode_cell_lib is None:
                raise ValueError(
                    "DifferentialStage: output_ode_cell_lib is required when "
                    "output_ode_src/output_ode_dst are provided"
                )
            self.register_buffer(
                "output_ode_src", torch.tensor(output_ode_src, dtype=torch.long),
            )
            self.register_buffer(
                "output_ode_dst", torch.tensor(output_ode_dst, dtype=torch.long),
            )
            self.output_ode_cell_lib = output_ode_cell_lib
            self.output_ode_z_logits = nn.Parameter(
                torch.full((len(output_ode_src),), z_init),
            )
            self._has_output_ode = True
        else:
            self.register_buffer(
                "output_ode_src", torch.empty(0, dtype=torch.long),
            )
            self.register_buffer(
                "output_ode_dst", torch.empty(0, dtype=torch.long),
            )
            self.output_ode_cell_lib = None
            self.output_ode_z_logits = None
            self._has_output_ode = False

        # Shared-sense + crossbar readout (shared-sense-crossbar plan).
        # Replaces the h*d_out temporal OTA mesh with one sense OTA per
        # hidden node (optionally two at different operating points) driven
        # against a private learnable Vref rail, plus a plain dense crossbar
        # W [d_out x s*h] that mixes the sense currents into the last d_out
        # output-ODE accumulator nodes (the readout tail). The crossbar is a
        # plain nn.Parameter — no z_logits, no VCA on the taps. VCA
        # (``vca_v_readout``) gates the sense currents only.
        self.readout_mode = str(readout_mode)
        self.readout_senses_per_node = int(readout_senses_per_node)
        self._has_shared_readout = False
        self._sense_has_resistive = False
        self._readout_dst_start = 0
        if self.readout_mode not in ("ota_mesh", "shared_sense"):
            raise ValueError(
                f"DifferentialStage: readout_mode must be 'ota_mesh' or "
                f"'shared_sense', got {readout_mode!r}"
            )
        if self.readout_senses_per_node not in (1, 2):
            raise ValueError(
                f"DifferentialStage: readout_senses_per_node must be 1 or 2, "
                f"got {self.readout_senses_per_node}"
            )
        if self.readout_mode == "shared_sense":
            if (
                readout_sense_src is None
                or readout_sense_cell_lib is None
                or readout_crossbar_shape is None
            ):
                raise ValueError(
                    "DifferentialStage: shared_sense readout requires "
                    "readout_sense_src, readout_sense_cell_lib, and "
                    "readout_crossbar_shape"
                )
            if output_ode_src is not None or output_ode_dst is not None:
                raise ValueError(
                    "DifferentialStage: shared_sense readout is incompatible "
                    "with output_ode_src/output_ode_dst (the legacy temporal "
                    "mesh); pass None for those in shared mode"
                )
            s = self.readout_senses_per_node
            n_sense = len(readout_sense_src)
            if n_sense <= 0:
                raise ValueError(
                    "DifferentialStage: shared_sense readout needs at least "
                    "one sense cell"
                )
            if n_sense % s != 0:
                raise ValueError(
                    f"DifferentialStage: len(readout_sense_src)={n_sense} must "
                    f"be divisible by readout_senses_per_node={s}"
                )
            if any(
                i < 0 or i >= self.num_nodes for i in readout_sense_src
            ):
                raise ValueError(
                    f"DifferentialStage: readout_sense_src entries must be in "
                    f"[0, {self.num_nodes}), got {readout_sense_src}"
                )
            d_out, crossbar_sense = readout_crossbar_shape
            if int(crossbar_sense) != n_sense:
                raise ValueError(
                    f"DifferentialStage: readout_crossbar_shape[1]="
                    f"{crossbar_sense} must equal len(readout_sense_src)="
                    f"{n_sense}"
                )
            if int(d_out) <= 0 or int(d_out) >= self.num_nodes:
                raise ValueError(
                    f"DifferentialStage: readout_crossbar_shape[0]={d_out} must "
                    f"be in (0, num_nodes={self.num_nodes}) so the readout tail "
                    f"is a proper slice of the state"
                )
            self.register_buffer(
                "readout_sense_src",
                torch.tensor(readout_sense_src, dtype=torch.long),
            )
            self.readout_sense_cell_lib = readout_sense_cell_lib
            self.readout_sense_z_logits = nn.Parameter(
                torch.full((n_sense,), z_init),
            )
            self.readout_crossbar_W = nn.Parameter(
                torch.randn(int(d_out), n_sense) * (1.0 / math.sqrt(n_sense))
            )
            self.raw_vref_sense = nn.Parameter(
                torch.tensor([float(REF["raw_vref_init"])], dtype=torch.float32)
            )
            self._readout_dst_start = self.num_nodes - int(d_out)
            self._has_shared_readout = True
            self._sense_has_resistive = hasattr(
                readout_sense_cell_lib, "resistive_current"
            )
        else:
            # ota_mesh mode: no shared-readout state is created at all (the
            # empty ``output_ode_*`` buffers above already exist on legacy
            # stages) so legacy state_dicts stay byte-identical and old
            # temporal checkpoints keep loading with strict=True.
            self.readout_sense_cell_lib = None
            self.readout_sense_z_logits = None
            self.readout_crossbar_W = None
            self.raw_vref_sense = None
            self._has_shared_readout = False
            self._sense_has_resistive = False
            self._readout_dst_start = 0

        # Low-rank input-driven VCA (Voltage-Controlled Amplifier) gating.
        # When enabled, builds per-edge embeddings for boundary,
        # temporal-readout, and optionally core edges plus a shared input
        # projection. The VCA gate per gated edge is
        #   gate_e = 2 * sigma( (u @ W) @ v_e.T )
        # where ``W`` (in_dim x rank) is the shared projection and
        # ``v_e`` (rank) is the per-edge embedding. The 2-sigma gain
        # range (vs 1-sigma) lets the network amplify AND attenuate.
        #
        # At init, ``vca_W`` is set to ALL-ZERO so ``u @ W = 0`` for every
        # input, hence ``z_e = 0`` for every edge regardless of ``v_e``.
        # Therefore ``gate_e = 2 * sigma(0) = 1.0`` at init and the VCA
        # multiplies edge currents by 1.0 — bit-identical to VCA-off
        # baseline at epoch 0 (the correct null hypothesis). Per-edge
        # embeddings are random at init so ``dL/dW`` receives signal at
        # step 0 via the ``u ⊗ v_e`` path; ``v_e`` itself receives signal
        # once ``W`` moves (zero at step 0 by construction).
        #
        # Requires rank >= VCA['min_rank']. Requires at least one of
        # boundary / temporal-readout / core edge families.
        if self.vca_enabled:
            n_b = int(self.boundary_src.numel())
            # In shared_sense mode the "readout" family is the sense bank
            # (s*h cells); in ota_mesh mode it is the legacy h*d_out mesh.
            # The VCA readout embeddings size to the active readout family.
            n_r = (
                int(self.readout_sense_src.numel())
                if self._has_shared_readout
                else int(self.output_ode_src.numel())
            )
            n_c = int(self.src.numel())
            # Core gating is auto-enabled when VCA is on but neither
            # boundary nor readout edges exist (run C ablation). The
            # shared sense bank counts as a readout family (so shared mode
            # never spuriously auto-enables core gating).
            self._vca_core_enabled = bool(
                vca_core_enabled or (n_b == 0 and n_r == 0)
            )
            if not self._vca_core_enabled and n_b == 0 and n_r == 0:
                raise ValueError(
                    "DifferentialStage: --vca requires at least one gated "
                    "edge family (boundary, temporal-readout, or core)."
                )
            if self._vca_in_dim <= 0:
                raise ValueError(
                    "DifferentialStage: vca_in_dim must be > 0 when "
                    "vca_enabled=True"
                )
            self.vca_gate_shunt = bool(vca_gate_shunt)
            self.vca_separate_core_bus = bool(vca_separate_core_bus)
            init_scale = float(VCA["scale_init"])
            # Shared input projection: ALL ZERO at init ensures
            # gate_e = 1.0 exactly at epoch 0.
            self.vca_W = nn.Parameter(
                torch.zeros(self._vca_in_dim, self.vca_rank)
            )
            # Optional separate bus for core edge family
            # (--vca-separate-core-bus ablation). Same zero-init rule.
            if self.vca_separate_core_bus and self._vca_core_enabled:
                self.vca_W_core = nn.Parameter(
                    torch.zeros(self._vca_in_dim, self.vca_rank)
                )
            else:
                self.vca_W_core = None
            if n_b > 0:
                self.vca_v_boundary = nn.Parameter(
                    torch.empty(n_b, self.vca_rank).normal_(std=init_scale)
                )
            else:
                self.vca_v_boundary = None
            self.vca_b_boundary = nn.Parameter(torch.zeros(n_b)) if (n_b > 0 and self.vca_bias_enabled) else None
            if n_r > 0:
                self.vca_v_readout = nn.Parameter(
                    torch.empty(n_r, self.vca_rank).normal_(std=init_scale)
                )
            else:
                self.vca_v_readout = None
            self.vca_b_readout = nn.Parameter(torch.zeros(n_r)) if (n_r > 0 and self.vca_bias_enabled) else None
            if self._vca_core_enabled:
                self.vca_v_core = nn.Parameter(
                    torch.empty(n_c, self.vca_rank).normal_(std=init_scale)
                )
            else:
                self.vca_v_core = None
            self.vca_b_core = nn.Parameter(torch.zeros(n_c)) if (self._vca_core_enabled and self.vca_bias_enabled) else None
        else:
            self.vca_W = None
            self.vca_W_core = None
            self.vca_v_boundary = None
            self.vca_v_readout = None
            self.vca_v_core = None
            self.vca_b_boundary = None
            self.vca_b_readout = None
            self.vca_b_core = None
            self._vca_core_enabled = False
            self.vca_gate_shunt = False
            self.vca_separate_core_bus = False

    def num_edges(self) -> int:
        return int(self.src.numel())

    def _effective_leak(self, num_nodes: int | None = None) -> torch.Tensor:
        """Return the per-node effective leak.

        Programmable: ``softplus(raw_leak)`` (per-node).
        Non-programmable: ``leak_constant`` (scalar, same for all nodes).
        """
        if num_nodes is None:
            num_nodes = self.num_nodes
        if self.leak_mode == "programmable":
            return F.softplus(self.raw_leak)
        return torch.full((num_nodes,), self.leak_constant, dtype=torch.float32)

    # ------------------------------------------------------------------
    # hardware helpers (Phase 3a 3.10)
    # ------------------------------------------------------------------
    @property
    def hw_constraints(self):
        """The ``HWConstraints`` of ``self.hw`` or ``None`` when the hardware layer is off."""
        hw = getattr(self, "hw", None)
        return None if hw is None else hw.constraints

    def _hw_vca_bits(self) -> int | None:
        """3.10: the width of the VCA logit DAC (``None`` = no quantisation)."""
        c = self.hw_constraints
        return None if c is None else c.hw_vca_bits

    def _clamp_isat_eff(self, i_edge: torch.Tensor, cell_lib, mask: torch.Tensor,
                        gain: torch.Tensor | None) -> torch.Tensor:
        """3.10: keep ``Isat_eff = Isat * sigma(z) * g_vca`` under the hardware ceiling.

        No-op when ``hw_isat_eff_clamp`` is off.
        """
        c = self.hw_constraints
        if c is None or not c.hw_isat_eff_clamp:
            return i_edge
        return i_edge * isat_eff_scale(cell_lib.isat_values(), mask, gain)

    def edge_gain(self, family: str, u: torch.Tensor | None) -> torch.Tensor | None:
        """VCA interface: per-edge gain ``2*sigmoid((u @ W) @ v_e.T [+ b])`` ``[B, E]``.

        ``family`` is ``"core"`` (uses ``vca_W_core`` when the separate core bus is on),
        ``"boundary"`` or ``"readout"`` (temporal-readout mesh or shared-sense OTAs).
        Returns ``None`` when VCA does not gate the family (or ``u`` is absent), so
        callers do ``if g is not None: i = i * g``. The gain is exactly 1 at init
        (``vca_W`` is zero-initialised).
        """
        if not self.vca_enabled or u is None:
            return None
        if family == "core":
            if not self._vca_core_enabled or self.vca_v_core is None:
                return None
            W = self.vca_W_core if self.vca_separate_core_bus else self.vca_W
            return vca_edge_gain(u, W, self.vca_v_core, self.vca_b_core,
                                  bits=self._hw_vca_bits())
        if family == "boundary":
            v_e, bias = self.vca_v_boundary, self.vca_b_boundary
        elif family == "readout":
            v_e, bias = self.vca_v_readout, self.vca_b_readout
        else:
            raise ValueError(f"edge_gain: unknown family {family!r}")
        if v_e is None:
            return None
        return vca_edge_gain(u, self.vca_W, v_e, bias, bits=self._hw_vca_bits())

    def _gln_rails_z(self, u: torch.Tensor | None) -> torch.Tensor | None:
        """GLN rail activations ``[B, n_rails]`` for input ``u`` (``None`` when GLN is off)."""
        rails = getattr(self, "gln_rails", None)
        if rails is None or u is None:
            return None
        return rails.rails(u)

    def edge_gm_scale(self, family: str, cell_lib, gln_z: torch.Tensor | None) -> torch.Tensor | None:
        """GLN interface: input-conditioned gm ``[B, E]`` for ``family`` (``"boundary"`` or
        ``"readout"``), or ``None`` when GLN does not gate it (use the cell's static gm).

        With ``hw_gln_mode == "tail"`` (3.10) the override is an
        ``hw.constraints.IsatScale`` marker: a multiplier on Isat instead of on the slope.
        """
        c = self.hw_constraints
        mode = None if c is None else c.hw_gln_mode
        delta_max = None if c is None else c.hw_gln_delta_max
        return gln_gm_override(getattr(self, "gln_rails", None), family, cell_lib, gln_z,
                               mode=mode, delta_max=delta_max)

    def _ota_current(self, cell_lib, x_src, x_dst, gm) -> torch.Tensor:
        """Full OTA current (tanh + resistive shunt); GLN gm enters inside the tanh only."""
        if gm is None:
            return cell_lib(x_src=x_src, x_dst=x_dst, x_max=self.x_max)
        return cell_lib.forward_tanh(
            x_src=x_src, x_dst=x_dst, x_max=self.x_max, gm_override=gm,
        ) + cell_lib.resistive_current(x_src, x_dst)

    def _compute_frozen_boundary(self, u: torch.Tensor, x0: torch.Tensor) -> torch.Tensor | None:
        """Precompute the frozen boundary-fan-out tanh KCL contribution.

        Computed once from ``(u, x0)`` at stage entry (or per sample in
        the sequence path): the tanh cell evaluation via ``forward_tanh``
        when the boundary cell library exposes it (else full ``forward``),
        multiplied by the per-edge gate ``sigmoid(boundary_z_logits)`` and
        — when VCA is enabled — the per-edge VCA gate. The result is
        scattered into a ``[batch, num_nodes]`` per-node accumulator in
        ``x0``'s dtype, suitable for adding into ``acc`` each rhs call.

        The family's resistive shunt is intentionally NOT folded into this
        tensor; it is recomputed from evolving voltages each rhs call,
        mirroring the Heun-path ``freeze_read`` behavior for the core
        family. Returns ``None`` when boundary edges are absent (so
        ``freeze_boundary`` becomes a no-op as expected).
        """
        if not self._has_boundary or u is None or self.boundary_src.numel() == 0:
            return None
        u_src0 = u[:, self.boundary_src]
        x_dst0 = x0[:, self.boundary_dst]
        cell_lib = self.boundary_cell_lib
        # GLN (F2): fold the input-conditioned gm modulation into the frozen
        # tensor exactly like the dynamic path (tanh with modulated gm;
        # resistive shunt stays dynamic in rhs).
        gm_b = self.edge_gm_scale("boundary", cell_lib, self._gln_rails_z(u))
        if gm_b is not None:
            i_edge = cell_lib.forward_tanh(
                x_src=u_src0, x_dst=x_dst0, x_max=self.x_max,
                gm_override=gm_b,
            )
        elif hasattr(cell_lib, "forward_tanh"):
            i_edge = cell_lib.forward_tanh(
                x_src=u_src0, x_dst=x_dst0, x_max=self.x_max,
            )
        else:
            i_edge = cell_lib(
                x_src=u_src0, x_dst=x_dst0, x_max=self.x_max,
            )
        b_mask = torch.sigmoid(self.boundary_z_logits)  # [Eb]
        i_edge = i_edge * b_mask.unsqueeze(0)
        _g = self.edge_gain("boundary", u)
        if _g is not None:
            i_edge = i_edge * _g
        i_edge = self._clamp_isat_eff(i_edge, cell_lib, b_mask, _g)
        acc_b = torch.zeros_like(x0, dtype=torch.float32)
        acc_b.index_add_(1, self.boundary_dst, i_edge.float())
        return acc_b.to(dtype=x0.dtype)

    def _compute_frozen_readout(self, u: torch.Tensor | None, x0: torch.Tensor) -> torch.Tensor | None:
        """Precompute the frozen temporal-readout tanh KCL contribution.

        Computed once from ``(u, x0)`` at stage entry (or per sample in
        the sequence path): the tanh cell evaluation via ``forward_tanh``
        when the readout cell library exposes it (else full ``forward``),
        multiplied by the per-edge gate ``sigmoid(output_ode_z_logits)``
        and — when VCA is enabled — the per-edge VCA gate (which requires
        ``u``). The result is scattered into a ``[batch, num_nodes]``
        per-node accumulator in ``x0``'s dtype.

        The family's resistive shunt is intentionally NOT folded into this
        tensor; it is recomputed from evolving voltages each rhs call,
        mirroring the Heun-path ``freeze_read`` behavior for the core
        family. Returns ``None`` when readout edges are absent (so
        ``freeze_temporal_read`` becomes a no-op as expected).

        NOTE: the return shape is mode-dependent. In ``ota_mesh`` mode this
        is the legacy ``[batch, num_nodes]`` accumulator tensor. In
        ``shared_sense`` mode this dispatches to :meth:`_compute_frozen_sense`
        and returns ``[batch, n_sense]`` per-sense currents (the crossbar
        matmul happens inside ``rhs``).
        """
        if self._has_shared_readout and self.readout_sense_src.numel() > 0:
            return self._compute_frozen_sense(u, x0)
        if not self._has_output_ode or self.output_ode_src.numel() == 0:
            return None
        x_src0 = x0[:, self.output_ode_src]
        x_dst0 = x0[:, self.output_ode_dst]
        cell_lib = self.output_ode_cell_lib
        if hasattr(cell_lib, "forward_tanh"):
            i_edge = cell_lib.forward_tanh(
                x_src=x_src0, x_dst=x_dst0, x_max=self.x_max,
            )
        else:
            i_edge = cell_lib(
                x_src=x_src0, x_dst=x_dst0, x_max=self.x_max,
            )
        o_mask = torch.sigmoid(self.output_ode_z_logits)  # [Eo]
        i_edge = i_edge * o_mask.unsqueeze(0)
        _g = self.edge_gain("readout", u)
        if _g is not None:
            i_edge = i_edge * _g
        i_edge = self._clamp_isat_eff(i_edge, cell_lib, o_mask, _g)
        acc_o = torch.zeros_like(x0, dtype=torch.float32)
        acc_o.index_add_(1, self.output_ode_dst, i_edge.float())
        return acc_o.to(dtype=x0.dtype)

    def _compute_frozen_sense(
        self, u: torch.Tensor | None, x0: torch.Tensor,
    ) -> torch.Tensor | None:
        """Precompute the frozen shared-sense tanh KCL contribution.

        Mirrors :meth:`_compute_frozen_readout` for the shared_sense family:
        the sense currents ``I_OTA(x0[:, readout_sense_src], Vref)`` are
        computed once from ``(u, x0)`` at stage entry, multiplied by the
        per-sense gate ``sigmoid(readout_sense_z_logits)`` and — when VCA is
        enabled — the per-edge VCA gate (which requires ``u``).

        Unlike the legacy mesh path this returns **per-sense** currents of
        shape ``[batch, n_sense]`` (NOT a scattered ``[batch, num_nodes]``
        accumulator): the dense crossbar matmul happens inside ``rhs`` so the
        frozen/dynamic split stays clean (frozen tanh senses x dynamic
        crossbar ``W``). The family's resistive shunt is NOT folded in; it is
        recomputed from evolving voltages each rhs call. Returns ``None``
        when shared readout is absent.
        """
        if not self._has_shared_readout or self.readout_sense_src.numel() == 0:
            return None
        x_src0 = x0[:, self.readout_sense_src]            # [B, n_sense]
        cell_lib = self.readout_sense_cell_lib
        # Sense destination is the private Vref rail (an ideal voltage
        # source held constant during integration). Pass it as the
        # FreeTanh ``x_dst`` argument so the cell is reused exactly.
        vref = torch.sigmoid(self.raw_vref_sense) * self.x_max
        x_dst0 = vref.to(x0.dtype).view(1, 1).expand_as(x_src0)
        # GLN (F2): fold the input-conditioned sense-gm modulation into the
        # frozen per-sense currents (crossbar W stays untouched).
        gm_s = self.edge_gm_scale("readout", cell_lib, self._gln_rails_z(u))
        if gm_s is not None:
            i_edge = cell_lib.forward_tanh(
                x_src=x_src0, x_dst=x_dst0, x_max=self.x_max,
                gm_override=gm_s,
            )
        elif hasattr(cell_lib, "forward_tanh"):
            i_edge = cell_lib.forward_tanh(
                x_src=x_src0, x_dst=x_dst0, x_max=self.x_max,
            )
        else:
            i_edge = cell_lib(
                x_src=x_src0, x_dst=x_dst0, x_max=self.x_max,
            )
        sense_mask = torch.sigmoid(self.readout_sense_z_logits)  # [n_sense]
        i_edge = i_edge * sense_mask.unsqueeze(0)
        _g = self.edge_gain("readout", u)
        if _g is not None:
            i_edge = i_edge * _g
        i_edge = self._clamp_isat_eff(i_edge, cell_lib, sense_mask, _g)
        return i_edge.to(dtype=x0.dtype)  # [B, n_sense]

    def _compute_i_edge_const(
        self,
        x_src_state: torch.Tensor,
        gate_core_cached: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the core nonlinear KCL contribution that ``freeze_read`` holds.

        Recomputes the cell library (or ``forward_tanh`` when a parallel resistive shunt
        exists — so the resistive term stays dynamic in ``rhs``), folds in the
        per-edge gate, and core VCA gate, and returns
        the ``[batch, num_nodes]`` KCL accumulator in ``x_src_state``'s dtype.

        Args:
            x_src_state: Source state tensor ``[B, num_nodes]`` to read
                ``x[src]`` and ``x[dst]`` from (the stage-entry state).
            gate_core_cached: Precomputed ``[B, E_core]`` VCA gate (when VCA
                is enabled on the core family), or ``None``.

        Returns:
            ``[batch, num_nodes]`` tensor in ``x_src_state``'s dtype.
        """
        x_src0 = x_src_state[:, self.src]
        x_dst0 = x_src_state[:, self.dst]
        if self._has_resistive and hasattr(self.cell_lib, "forward_tanh"):
            i_edge = self.cell_lib.forward_tanh(
                x_src=x_src0, x_dst=x_dst0, x_max=self.x_max,
            )
        else:
            i_edge = self.cell_lib(
                x_src=x_src0, x_dst=x_dst0, x_max=self.x_max,
            )
        edge_mask = torch.sigmoid(self.z_logits)
        i_edge = i_edge * edge_mask.unsqueeze(0)
        if gate_core_cached is not None:
            i_edge = i_edge * gate_core_cached
        i_edge = self._clamp_isat_eff(i_edge, self.cell_lib, edge_mask, gate_core_cached)
        i_edge_f32 = i_edge.float()
        acc_const = torch.zeros_like(x_src_state, dtype=torch.float32)
        acc_const.index_add_(1, self.dst, i_edge_f32)
        if not self.read_only_source:
            acc_const.index_add_(1, self.src, -i_edge_f32)
        return acc_const.to(dtype=x_src_state.dtype)

    def clip_sharpness(self) -> float | torch.Tensor:
        """Live soft-rail clip sharpness ``s`` (denominator of the sigmoid rails).

        Returns the fixed ``clip_softness`` float on the off-path (default,
        bit-identical to the pre-F1 baseline) or the sigmoid-mapped
        ``clip_sharpness_raw`` scalar in ``[clip_sharpness_min,
        clip_sharpness_max]`` on the learnable path.
        """
        if self._learnable_clip_sharpness and self.clip_sharpness_raw is not None:
            t = torch.sigmoid(self.clip_sharpness_raw)
            return self.clip_sharpness_min + (
                self.clip_sharpness_max - self.clip_sharpness_min
            ) * t
        return self.clip_softness

    def soft_clip(self, x: torch.Tensor) -> torch.Tensor:
        """Soft-rail clip current ``clip_current * (sigma - sigma)`` with live sharpness.

        ``clip(x) = clip_current * (sigma((x - x_max)/s) - sigma((-x - x_max)/s))``
        with ``s = self.clip_sharpness()`` (fixed ``clip_softness`` float on
        the default path, learnable per-stage scalar on the F1 path).
        """
        s = self.clip_sharpness()
        clip = torch.sigmoid((x - self.x_max) / s)
        clip = clip - torch.sigmoid((-x - self.x_max) / s)
        return self.clip_current * clip

    def rhs(self, x: torch.Tensor,
            u: torch.Tensor | None = None,
            i_edge_const: torch.Tensor | None = None,
            i_boundary_const: torch.Tensor | None = None,
            i_readout_const: torch.Tensor | None = None,
            vca_gate_core: torch.Tensor | None = None) -> torch.Tensor:
        """Compute dx/dt at state x. x: [batch, num_nodes].

        Gate application:
        - Edge gate: i_edge *= sigmoid(z_logits) — multiplies the edge current
          after cell-library evaluation. When z_e -> 0 the edge contributes
          zero current.

        When ``i_edge_const`` is provided (the ``freeze_read=True`` path), the
        cell_lib evaluation, edge gate, and KCL scatter-add are
        skipped — the provided ``[batch, num_nodes]`` tensor is used directly
        as the KCL contribution. Leak, clip, and drive current are still
        computed from the current ``x``.

        Boundary-terminal OTA edges (boundary-fan-out plan):
        - When ``u`` is provided and ``self._has_boundary``, the boundary
          edges inject ``I_OTA(u[:, boundary_src], x[:, boundary_dst])``
          into the destination nodes only (no source drain — terminals
          are fixed voltages). They are NOT frozen by ``freeze_read``:
          the destination voltage evolves, so the OTA current is
          recomputed every step. When ``i_boundary_const`` is provided
          (the ``freeze_boundary=True`` path) the tanh cell contribution
          + edge gate + VCA gate are precomputed once from ``(u, x0)``
          and the per-step block only recomputes the family's resistive
          shunt (when present).

        Temporal-readout OTA edges (temporal-readout plan):
        - When ``self._has_output_ode``, ``I_OTA(x[output_ode_src[e]],
          x[output_ode_dst[e]])`` is injected into the destination (output
          ODE accumulator) only. The source (hidden/projection) is a
          read-only voltage — no current is drained from it. Like boundary
          and reference edges, the temporal-readout current is NOT frozen
          by ``freeze_read`` (the destination voltage evolves). When
          ``i_readout_const`` is provided (the ``freeze_temporal_read=True``
          path) the tanh cell contribution + edge gate + VCA gate are
          precomputed once from ``x0`` and the per-step block only
          recomputes the family's resistive shunt (when present).

        Shared-sense + crossbar readout (shared-sense-crossbar plan):
        - When ``self._has_shared_readout`` (mutually exclusive with the
          legacy mesh), one sense OTA per hidden node drives against a
          private learnable ``Vref`` rail; the per-sense currents are mixed
          by a plain dense crossbar ``readout_crossbar_W`` and injected into
          the last ``d_out`` output-ODE accumulator nodes only (no source
          drain). The family is NOT frozen by ``freeze_read`` (the sense
          sources evolve). When ``i_readout_const`` is provided
          (``freeze_temporal_read=True``) its shape is ``[B, n_sense]`` in
          shared mode (mode-dependent: ``[B, num_nodes]`` in the legacy mesh
          path) and the per-step block only recomputes the family's
          resistive shunt (when present).
        """
        x_src = x[:, self.src]
        x_dst = x[:, self.dst]

        # GLN rails (F2): compute the rail activations once per stage entry
        # (u is constant per sample during the ODE integration) and reuse
        # them for every gated family below. GLN modulates gm *inside* the
        # tanh argument; VCA (when on) multiplies the resulting current
        # *outside*, so both compose without double-applying.
        gln_z = self._gln_rails_z(u)  # [B, n_rails] or None

        # Edge gate: multiply each edge's current by its gate. Computed once
        # here and reused for both the tanh current and the resistive shunt
        # (when applicable), avoiding a redundant sigmoid call.
        edge_mask = torch.sigmoid(self.z_logits)  # [E]
        if i_edge_const is None:
            i_edge = self.cell_lib(
                x_src=x_src,
                x_dst=x_dst,
                x_max=self.x_max,
            )
            i_edge = i_edge * edge_mask.unsqueeze(0)  # [B, E]
            # Core VCA gate (freeze_read OFF): multiplied per-substep
            # from the cached gate (constant per sample). Semantically,
            # scales the entire nonlinear I-V curve per sample, like a
            # real VCA. No-op when vca_gate_core is None.
            if vca_gate_core is not None:
                i_edge = i_edge * vca_gate_core                # [B, E]
            i_edge = self._clamp_isat_eff(i_edge, self.cell_lib, edge_mask, vca_gate_core)

            # KCL scatter-add: accumulate in float32 for AMP robustness.
            # Under torch.autocast the node/edge gate multiplications promote
            # i_edge to fp32 even when x is fp16, so x.new_zeros() creates a
            # Half accumulator while the source is Float → index_add_ error.
            # Accumulating in float32 then casting back to x.dtype is safe and
            # numerically preferable for scatter operations.
            i_edge_f32 = i_edge.float()
            acc = torch.zeros_like(x, dtype=torch.float32)
            acc.index_add_(1, self.dst, i_edge_f32)
            if not self.read_only_source:
                acc.index_add_(1, self.src, -i_edge_f32)
            acc = acc.to(dtype=x.dtype)
        else:
            # Frozen path: i_edge_const is the precomputed KCL contribution
            # [batch, num_nodes] in x's dtype. The tanh contribution (with
            # the core VCA gate already folded in at precompute time) was
            # computed from x0 and held constant. The resistive shunt (if
            # any) is added below from evolving voltages.
            acc = i_edge_const

        # Parallel resistive shunt (FreeTanhLibrary): always uses evolving
        # voltages, bypassing ``freeze_read``. Gated by the same edge_mask /
        # edge_mask as the tanh current.
        if self._has_resistive:
            i_res = self.cell_lib.resistive_current(x_src, x_dst)
            i_res = i_res * edge_mask.unsqueeze(0)              # [B, E]
            # Core VCA gate on the resistive shunt (--vca-gate-shunt).
            # Makes M(u) input-dependent: input-dependent routing of
            # current flow, not just scaling. Passivity preserved as long
            # as gate is in (0, 2) (it is). Disabled by default.
            if vca_gate_core is not None and self.vca_gate_shunt:
                i_res = i_res * vca_gate_core                  # [B, E]
            i_res_f32 = i_res.float()
            # Clone when freeze_read is active so we don't mutate the
            # shared ``i_edge_const`` tensor across rhs calls.
            if i_edge_const is not None:
                acc = acc.clone()
            acc_res = torch.zeros_like(x, dtype=torch.float32)
            acc_res.index_add_(1, self.dst, i_res_f32)
            if not self.read_only_source:
                acc_res.index_add_(1, self.src, -i_res_f32)
            acc = (acc.float() + acc_res).to(dtype=x.dtype)

        # Boundary-terminal OTA edges: I_OTA(u_i, x_j) injected into dst only.
        # Boundary terminals are ideal voltage sources, never drained.
        if self._has_boundary and u is not None and self.boundary_src.numel() > 0:
            u_src = u[:, self.boundary_src]
            x_dst_b = x[:, self.boundary_dst]
            boundary_mask = torch.sigmoid(self.boundary_z_logits)  # [Eb]
            if i_boundary_const is None:
                # Dynamic path: full cell forward (tanh + resistive shunt).
                # GLN (F2): modulate the family's gm with the shared rails
                # (gm inside tanh arg); the resistive shunt is never gated.
                i_boundary = self._ota_current(
                    self.boundary_cell_lib, u_src, x_dst_b,
                    self.edge_gm_scale("boundary", self.boundary_cell_lib, gln_z),
                )
                i_boundary = i_boundary * boundary_mask.unsqueeze(0)   # [B, Eb]
                _g = self.edge_gain("boundary", u)
                if _g is not None:
                    i_boundary = i_boundary * _g
                i_boundary = self._clamp_isat_eff(
                    i_boundary, self.boundary_cell_lib, boundary_mask, _g)
                i_boundary_f32 = i_boundary.float()
                # Clone when freeze_read is active so we don't mutate the shared
                # ``acc`` (= i_edge_const) tensor.
                if i_edge_const is not None:
                    acc = acc.clone()
                acc_b = torch.zeros_like(x, dtype=torch.float32)
                acc_b.index_add_(1, self.boundary_dst, i_boundary_f32)
                # NOTE: no `acc_b.index_add_(1, boundary_src, -i_boundary_f32)` —
                # boundary terminals are fixed voltages, never drained.
                acc = (acc.float() + acc_b).to(dtype=x.dtype)
            else:
                # Frozen-tanh path (freeze_boundary=True): the tanh cell
                # contribution (edge gate + VCA gate already folded in at
                # precompute time) is added directly. The family's resistive
                # shunt, if present, is recomputed per-step from evolving
                # voltages so it remains dynamic — mirrors the Heun-path
                # freeze_read behavior for the core family.
                acc = (acc.float() + i_boundary_const.float()).to(dtype=x.dtype)
                if hasattr(self.boundary_cell_lib, "resistive_current"):
                    i_res_b = self.boundary_cell_lib.resistive_current(
                        u_src, x_dst_b,
                    )  # [B, Eb]
                    i_res_b = i_res_b * boundary_mask.unsqueeze(0)
                    _g = self.edge_gain("boundary", u)
                    if _g is not None:
                        i_res_b = i_res_b * _g
                    acc_b_res = torch.zeros_like(x, dtype=torch.float32)
                    acc_b_res.index_add_(1, self.boundary_dst, i_res_b.float())
                    # NOTE: no source drain — boundary terminals are fixed.
                    acc = (acc.float() + acc_b_res).to(dtype=x.dtype)

        # Temporal-readout OTA edges (temporal-readout plan).
        # For each edge e: I_out = I_OTA(x[output_ode_src[e]], x[output_ode_dst[e]]).
        # Current is injected into the destination (output ODE accumulator)
        # only; the source (hidden/projection) is read-only and is not
        # drained. The destination voltage evolves through the ODE so the
        # OTA current is recomputed every step (NOT frozen by freeze_read).
        if self._has_output_ode and self.output_ode_src.numel() > 0:
            x_src_o = x[:, self.output_ode_src]  # hidden (read-only)
            x_dst_o = x[:, self.output_ode_dst]  # output ODE (writable)
            out_mask = torch.sigmoid(self.output_ode_z_logits)  # [Eo]
            if i_readout_const is None:
                # Dynamic path: full cell forward (tanh + resistive shunt).
                i_out = self.output_ode_cell_lib(
                    x_src=x_src_o, x_dst=x_dst_o, x_max=self.x_max,
                )
                i_out = i_out * out_mask.unsqueeze(0)  # [B, Eo]
                _g = self.edge_gain("readout", u)
                if _g is not None:
                    i_out = i_out * _g
                i_out = self._clamp_isat_eff(
                    i_out, self.output_ode_cell_lib, out_mask, _g)
                i_out_f32 = i_out.float()
                if i_edge_const is not None:
                    acc = acc.clone()
                acc_out = torch.zeros_like(x, dtype=torch.float32)
                acc_out.index_add_(1, self.output_ode_dst, i_out_f32)
                # NOTE: no source drain on output_ode_src — the hidden/projection
                # grid is untouched, only the output accumulator receives current.
                acc = (acc.float() + acc_out).to(dtype=x.dtype)
            else:
                # Frozen-tanh path (freeze_temporal_read=True): the tanh cell
                # contribution (edge gate + VCA gate already folded in at
                # precompute time) is added directly. The family's resistive
                # shunt, if present, is recomputed per-step from evolving
                # voltages — mirrors the Heun-path freeze_read behavior for
                # the core family.
                acc = (acc.float() + i_readout_const.float()).to(dtype=x.dtype)
                if hasattr(self.output_ode_cell_lib, "resistive_current"):
                    i_res_o = self.output_ode_cell_lib.resistive_current(
                        x_src_o, x_dst_o,
                    )  # [B, Eo]
                    i_res_o = i_res_o * out_mask.unsqueeze(0)
                    _g = self.edge_gain("readout", u)
                    if _g is not None:
                        i_res_o = i_res_o * _g
                    acc_out_res = torch.zeros_like(x, dtype=torch.float32)
                    acc_out_res.index_add_(1, self.output_ode_dst, i_res_o.float())
                    # NOTE: no source drain on output_ode_src.
                    acc = (acc.float() + acc_out_res).to(dtype=x.dtype)

        # Shared-sense + crossbar readout (shared-sense-crossbar plan).
        # Mutually exclusive with the legacy temporal mesh above (in shared
        # mode ``self._has_output_ode`` is False). Dynamics:
        #   I_j = Isat_j * tanh(gm_j * (A_j * x_src_j - B_j * Vref + theta_j))
        #        (+ leaky-ALPHA term + resistive shunt, unchanged)
        #   a_dot_i = sum_j W[i, j] * I_j - leak_i * a_i - clip_i(a_i)
        # where ``a`` is the last ``d_out`` output-ODE accumulator slice and
        # the crossbar ``W`` is a plain dense weight (no gate, no VCA). VCA
        # gates the sense currents only.
        if self._has_shared_readout and self.readout_sense_src.numel() > 0:
            W = self.readout_crossbar_W                    # [d_out, n_sense]
            sense_mask = torch.sigmoid(self.readout_sense_z_logits)  # [n_sense]
            x_j = x[:, self.readout_sense_src]             # [B, n_sense]
            vref = (torch.sigmoid(self.raw_vref_sense) * self.x_max).to(x.dtype)
            x_dst_s = vref.view(1, 1).expand_as(x_j)
            if i_readout_const is None:
                # Dynamic path: full cell forward (tanh + resistive shunt).
                # GLN (F2): modulate the sense OTA gm with the shared rails
                # (one gm per sense; the crossbar W is never touched by GLN).
                i_s = self._ota_current(
                    self.readout_sense_cell_lib, x_j, x_dst_s,
                    self.edge_gm_scale("readout", self.readout_sense_cell_lib, gln_z),
                )
                i_s = i_s * sense_mask.unsqueeze(0)        # [B, n_sense]
                _g = self.edge_gain("readout", u)
                if _g is not None:
                    i_s = i_s * _g
                i_s = self._clamp_isat_eff(
                    i_s, self.readout_sense_cell_lib, sense_mask, _g)
            else:
                # Frozen-tanh path (freeze_temporal_read=True): the tanh cell
                # contribution (sense gate + VCA gate already folded in at
                # precompute time) arrives as ``[B, n_sense]``; the family's
                # resistive shunt is recomputed below from evolving voltages.
                i_s = i_readout_const
            i_acc = (i_s.float() @ W.float().T)            # [B, d_out]
            if i_edge_const is not None:
                acc = acc.clone()
            acc_f32 = acc.float()
            dst_slice = slice(self._readout_dst_start, self.num_nodes)
            acc_f32[:, dst_slice] = (
                acc_f32[:, dst_slice] + i_acc.to(dtype=acc_f32.dtype)
            )
            acc = acc_f32.to(dtype=x.dtype)
            # Resistive shunt stays dynamic per-step like every other family
            # (kept for exact FreeTanh reuse even though Vref is an ideal
            # source). Gate matches the tanh path so it can be pruned away.
            if i_readout_const is not None and self._sense_has_resistive:
                i_res_s = self.readout_sense_cell_lib.resistive_current(x_j, x_dst_s)
                i_res_s = i_res_s * sense_mask.unsqueeze(0)
                _g = self.edge_gain("readout", u)
                if _g is not None:
                    i_res_s = i_res_s * _g
                i_res_acc = (i_res_s.float() @ W.float().T)  # [B, d_out]
                acc_f32 = acc.float()
                acc_f32[:, dst_slice] = (
                    acc_f32[:, dst_slice] + i_res_acc.to(dtype=acc_f32.dtype)
                )
                acc = acc_f32.to(dtype=x.dtype)

        leak = self._effective_leak().unsqueeze(0).to(x.device)  # [1, N]
        leak_term = leak * x

        clip_term = self.soft_clip(x)

        return (acc - leak_term - clip_term) / self.c_eff

    def compile_rhs(self, backend: str = "inductor"):
        """Compile `rhs` with `torch.compile` for kernel fusion.

        Note: `index_add_` on `acc` is a scatter op that may force a graph
        break in some PyTorch versions. The scatter itself remains in eager
        mode; the `cell_lib` call and post-scatter math are fused.
        """
        self.rhs = torch.compile(self.rhs, backend=backend)

    def forward(
        self,
        x0: torch.Tensor,
        t_span: float | None = None,
        num_steps: int | None = None,
        store_trajectory: bool = True,
        u: torch.Tensor | None = None,
        x_hold: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Integrate the stage with fixed-step Heun (2nd-order predictor-corrector).

        Returns
        -------
        x_final : torch.Tensor
            Stage output state.
        traj : torch.Tensor or None
            ``[batch, num_nodes, num_steps+1]`` when ``store_trajectory``.
        """
        # ``x_hold`` is forwarded only when set, so the legacy call signature
        # of ``_forward_heun`` (and test hooks that wrap it) is unchanged.
        extra = {} if x_hold is None else {"x_hold": x_hold}
        return self._forward_heun(
            x0=x0, t_span=t_span, num_steps=num_steps,
            store_trajectory=store_trajectory,
            u=u,
            **extra,
        )

    def _forward_heun(
        self,
        x0: torch.Tensor,
        t_span: float | None,
        num_steps: int | None,
        store_trajectory: bool,
        u: torch.Tensor | None = None,
        x_hold: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        t_span = float(t_span if t_span is not None else SOLVER["t_span"])
        num_steps = int(num_steps if num_steps is not None else SOLVER["num_steps"])
        dt = t_span / float(num_steps)

        # Compute the core VCA gate cache once per forward (u is constant
        # per sample). Hoisted out of the Heun loop to avoid recomputing
        # the matmul per substep (zero extra cost in the freeze_read path
        # — gate folds into i_edge_const).
        gate_core_cached = self.edge_gain("core", u)
        # Diagnostic attribute (not read inside compiled rhs — rhs
        # receives the gate via the explicit kwarg below).
        self._gate_core_cached = gate_core_cached

        # freeze_read: precompute edge currents (cell_lib + edge gate +
        # core VCA gate + KCL scatter-add) once from x0 and hold them
        # constant across all sub-steps. Leak, clip, drive, and boundary-edge
        # currents still read the current x (boundary terminals are fixed but
        # the dynamic target node voltage evolves, so the OTA current is
        # per-step). For cell libraries with a parallel resistive shunt, use
        # ``forward_tanh`` here so the resistive term stays dynamic in ``rhs``
        # (the resistive current is added per-step from evolving voltages).
        i_edge_const = None
        if self.freeze_read:
            # Time-multiplexed fabric (Phase 3b 2.3): the frozen core read
            # may see the value the hold capacitor presents (``x_hold``).
            # Every other current (shunt, sense, boundary, leak) still
            # reads the live ``x``.  ``x_hold=None`` is the legacy path.
            x_core = x0 if x_hold is None else x_hold
            i_edge_const = self._compute_i_edge_const(x_core, gate_core_cached)

        # freeze_boundary: precompute the boundary tanh KCL contribution once
        # from (u, x0). Mirrors the freeze_read pattern for this edge family.
        # The family's resistive shunt is NOT folded in — rhs recomputes it
        # per-step. Independent of freeze_read (core edges may still be
        # dynamic). No-op when boundary edges are absent.
        i_boundary_const = None
        if self.freeze_boundary:
            i_boundary_const = self._compute_frozen_boundary(u, x0)

        # freeze_temporal_read: same pattern for the temporal-readout family.
        i_readout_const = None
        if self.freeze_temporal_read:
            i_readout_const = self._compute_frozen_readout(u, x0)

        x = x0
        traj_chunks = [x] if store_trajectory else None

        for step in range(num_steps):
            k1 = self.rhs(x, u=u,
                          i_edge_const=i_edge_const,
                          i_boundary_const=i_boundary_const,
                          i_readout_const=i_readout_const,
                          vca_gate_core=gate_core_cached)
            x_pred = x + dt * k1
            k2 = self.rhs(x_pred, u=u,
                          i_edge_const=i_edge_const,
                          i_boundary_const=i_boundary_const,
                          i_readout_const=i_readout_const,
                          vca_gate_core=gate_core_cached)
            x = x + 0.5 * dt * (k1 + k2)
            if store_trajectory:
                traj_chunks.append(x)

        traj = torch.stack(traj_chunks, dim=2) if store_trajectory else None
        return x, traj

    def edge_gates(self) -> torch.Tensor:
        """Return edge gate values z_e = σ(z_logits), shape [E]."""
        return torch.sigmoid(self.z_logits)

    def active_edge_mask(self, threshold: float = 0.01) -> torch.Tensor:
        """Boolean mask of edges that survive pruning at the given threshold."""
        return self.edge_gates() > threshold

    def parameter_breakdown(self) -> dict:
        """Return parameter counts including gate parameters (for diagnostics)."""
        if isinstance(self.cell_lib, FreeTanhLibrary):
            device_n = (
                int(self.cell_lib.a_raw.numel())
                + int(self.cell_lib.b_raw.numel())
                + int(self.cell_lib.s_raw.numel())
                + int(self.cell_lib.gm_raw.numel())
                + int(self.cell_lib.isat_raw.numel())
            )
            if hasattr(self.cell_lib, "theta_raw"):
                device_n += int(self.cell_lib.theta_raw.numel())
        else:
            device_n = 0
        raw_leak_n = int(self.raw_leak.numel()) if hasattr(self, "raw_leak") else 0
        bz = int(self.boundary_z_logits.numel()) if self.boundary_z_logits is not None else 0
        bdev = 0
        if self.boundary_cell_lib is not None:
            if isinstance(self.boundary_cell_lib, FreeTanhLibrary):
                bdev = (
                    int(self.boundary_cell_lib.a_raw.numel())
                    + int(self.boundary_cell_lib.b_raw.numel())
                    + int(self.boundary_cell_lib.s_raw.numel())
                    + int(self.boundary_cell_lib.gm_raw.numel())
                    + int(self.boundary_cell_lib.isat_raw.numel())
                )
                if hasattr(self.boundary_cell_lib, "theta_raw"):
                    bdev += int(self.boundary_cell_lib.theta_raw.numel())
        # Temporal-readout OTA edge stats
        out_z = int(self.output_ode_z_logits.numel()) if self.output_ode_z_logits is not None else 0
        out_dev = 0
        if self.output_ode_cell_lib is not None:
            if hasattr(self.output_ode_cell_lib, "param"):
                out_dev = int(self.output_ode_cell_lib.param.numel())
            elif hasattr(self.output_ode_cell_lib, "alpha_raw"):
                out_dev = int(self.output_ode_cell_lib.alpha_raw.numel())
                if hasattr(self.output_ode_cell_lib, "bias_raw"):
                    out_dev += int(self.output_ode_cell_lib.bias_raw.numel())
            elif hasattr(self.output_ode_cell_lib, "gm_raw"):
                out_dev = int(self.output_ode_cell_lib.gm_raw.numel())
                if hasattr(self.output_ode_cell_lib, "isat_raw"):
                    out_dev += int(self.output_ode_cell_lib.isat_raw.numel())
                if hasattr(self.output_ode_cell_lib, "a_raw"):
                    out_dev += int(self.output_ode_cell_lib.a_raw.numel())
                if hasattr(self.output_ode_cell_lib, "b_raw"):
                    out_dev += int(self.output_ode_cell_lib.b_raw.numel())
                if hasattr(self.output_ode_cell_lib, "s_raw"):
                    out_dev += int(self.output_ode_cell_lib.s_raw.numel())
                if hasattr(self.output_ode_cell_lib, "theta_raw"):
                    out_dev += int(self.output_ode_cell_lib.theta_raw.numel())
                if hasattr(self.output_ode_cell_lib, "kappa_raw"):
                    out_dev += int(self.output_ode_cell_lib.kappa_raw.numel())
        # Shared-sense readout stats (shared-sense-crossbar plan). Zero in
        # the legacy ota_mesh mode where the output_ode_* keys above are
        # populated instead.
        sense_z = (
            int(self.readout_sense_z_logits.numel())
            if getattr(self, "readout_sense_z_logits", None) is not None else 0
        )
        sense_dev = 0
        if self._has_shared_readout and self.readout_sense_cell_lib is not None:
            # Same per-type counting ladder as the output_ode block above,
            # plus the FreeTanh resistive shunt (F6: legacy ladders
            # propagate the same omission, but the sense bank must include
            # ``g_resistive_raw`` since ``--no-resistive-shunt`` zeroes it).
            if hasattr(self.readout_sense_cell_lib, "param"):
                sense_dev = int(self.readout_sense_cell_lib.param.numel())
            elif hasattr(self.readout_sense_cell_lib, "alpha_raw"):
                sense_dev = int(self.readout_sense_cell_lib.alpha_raw.numel())
                if hasattr(self.readout_sense_cell_lib, "bias_raw"):
                    sense_dev += int(self.readout_sense_cell_lib.bias_raw.numel())
            elif hasattr(self.readout_sense_cell_lib, "gm_raw"):
                sense_dev = int(self.readout_sense_cell_lib.gm_raw.numel())
                if hasattr(self.readout_sense_cell_lib, "isat_raw"):
                    sense_dev += int(self.readout_sense_cell_lib.isat_raw.numel())
                if hasattr(self.readout_sense_cell_lib, "a_raw"):
                    sense_dev += int(self.readout_sense_cell_lib.a_raw.numel())
                if hasattr(self.readout_sense_cell_lib, "b_raw"):
                    sense_dev += int(self.readout_sense_cell_lib.b_raw.numel())
                if hasattr(self.readout_sense_cell_lib, "s_raw"):
                    sense_dev += int(self.readout_sense_cell_lib.s_raw.numel())
                if hasattr(self.readout_sense_cell_lib, "theta_raw"):
                    sense_dev += int(self.readout_sense_cell_lib.theta_raw.numel())
                if hasattr(self.readout_sense_cell_lib, "kappa_raw"):
                    sense_dev += int(self.readout_sense_cell_lib.kappa_raw.numel())
                if hasattr(self.readout_sense_cell_lib, "g_resistive_raw"):
                    sense_dev += int(self.readout_sense_cell_lib.g_resistive_raw.numel())
        sense_crossbar = (
            int(self.readout_crossbar_W.numel())
            if getattr(self, "readout_crossbar_W", None) is not None else 0
        )
        sense_vref = (
            int(self.raw_vref_sense.numel())
            if getattr(self, "raw_vref_sense", None) is not None else 0
        )
        vca_proj_n = int(self.vca_W.numel()) if self.vca_W is not None else 0
        if getattr(self, "vca_W_core", None) is not None:
            vca_proj_n += int(self.vca_W_core.numel())
        if getattr(self, "vca_W_hidden", None) is not None:
            vca_proj_n += int(self.vca_W_hidden.numel())
        if getattr(self, "vca_W_core_hidden", None) is not None:
            vca_proj_n += int(self.vca_W_core_hidden.numel())
        vca_embed_n = 0
        if self.vca_v_boundary is not None:
            vca_embed_n += int(self.vca_v_boundary.numel())
        if self.vca_v_readout is not None:
            vca_embed_n += int(self.vca_v_readout.numel())
        if getattr(self, "vca_v_core", None) is not None:
            vca_embed_n += int(self.vca_v_core.numel())
        vca_bias_n = sum(
            int(getattr(self, name).numel())
            for name in ("vca_b_boundary", "vca_b_readout", "vca_b_core")
            if getattr(self, name, None) is not None
        )
        return {
            "raw_leak": raw_leak_n,
            "z_logits": int(self.z_logits.numel()),
            "device_param": device_n,
            "boundary_z_logits": bz,
            "boundary_device_param": bdev,
            "output_ode_z_logits": out_z,
            "output_ode_device_param": out_dev,
            "readout_sense_z_logits": sense_z,
            "readout_sense_device_param": sense_dev,
            "readout_crossbar": sense_crossbar,
            "raw_vref_sense": sense_vref,
            "vca_proj": vca_proj_n,
            "vca_embed": vca_embed_n,
            "vca_bias": vca_bias_n,
            "total": (
                raw_leak_n
                + int(self.z_logits.numel())
                + device_n
                + bz
                + bdev
                + out_z
                + out_dev
                + sense_z
                + sense_dev
                + sense_crossbar
                + sense_vref
                + vca_proj_n
                + vca_embed_n
                + vca_bias_n
            ),
        }
