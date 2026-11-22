"""All tunable constants for the reduced differential KirchhoffNet.

Every other module imports from this file. To tune the system, edit this file.

Only the ``tanh_free`` cell library (``cell_library.FreeTanhLibrary``) is
supported; the legacy L/S/P/Z, v1.5 and v2 cell tables were removed in
surgery 5/6 (see the ``pre-surgery`` tag).

Units (R7): All conductance, current, and capacitance values are NORMALIZED
to lie in physically plausible ranges. They are NOT calibrated to a
consistent timescale: ``t_span`` and ``num_steps`` in SOLVER set the
integration window, and ``C_eff`` is a pure scaling. No pretence is made
that the resulting ω = g/C is a real analog time constant.
"""

# Named cell-library configs (only ``tanh_free`` is supported).
CELL_LIBRARIES = {
    "tanh_free": {"device": "tanh_free", "BIAS_ENABLED": True},
}

# tanh_free defaults: per-edge gm and Isat use bounded sigmoid
# parameterization gm = gm_min + (gm_max - gm_min) * sigmoid(gm_raw),
# isat = isat_min + (isat_max - isat_min) * sigmoid(isat_raw). Overridable
# via the FreeTanhLibrary constructor.
TANH_REALISTIC_GM_MIN = 0.01
TANH_REALISTIC_GM_MAX = 10.0
TANH_REALISTIC_ISAT_MIN = 0.01
TANH_REALISTIC_ISAT_MAX = 10.0

# Normalized physical limits (R7: not calibrated to real SI units).
PHYS = {
    "x_max": 3.0,
    "C_eff": 1.0,
    "beta_softness": 0.02,
    "clip_current": 0.05,
    "clip_softness": 0.02,
    # Learnable soft-rail sharpness (F1): the clip denominator ``s`` in
    #   clip(x) = clip_current * (sigma((x - x_max)/s) - sigma((-x - x_max)/s))
    # Smaller s = sharper rails. ``clip_sharpness_init`` equals the fixed
    # ``clip_softness`` so the learnable path is identity-equal at init.
    # The mapped sharpness lives in [clip_sharpness_min, clip_sharpness_max]
    # via sigmoid(clip_sharpness_raw) (logit-init so mapped == init).
    "clip_sharpness_init": 0.02,
    "clip_sharpness_min": 1e-3,
    "clip_sharpness_max": 0.2,
}

# BO search windows for the bounded-sigmoid gm/isat upper rails (F1). These
# are *search* ranges for kn_bayes_opt only — the build-time defaults stay
# at ``TANH_REALISTIC_GM_MAX=10.0`` / ``TANH_REALISTIC_ISAT_MAX=10.0`` so
# non-BO runs and existing checkpoints do not silently change. Param-count
# neutral: changing gm_max/isat_max adds no tensors.
BO_GM_MAX_RANGE = (1.0, 50.0)
BO_ISAT_MAX_RANGE = (1.0, 50.0)

# Shared-sense readout reference rail (``raw_vref_sense``): raw 0 maps to
# Vref = x_max/2 (mid-rail init).
REF = {
    "raw_vref_init": 0.0,
}

# Training hyperparameters
_BASE_BATCH_SIZE = 1024
_BASE_LR = 3e-4
_FINAL_BATCH_SIZE = 4096
OPTIM = {
    # LR auto-scales linearly with batch size (Goyal et al., 2017):
    # lr_new = base_lr * (batch_size / base_batch_size)
    # Compensates for fewer optimizer steps per epoch at larger batch sizes.
    "weight_decay": 1e-4,
    "grad_clip_norm": 5.0,
    "epochs": 800,
    "batch_size": _FINAL_BATCH_SIZE,
    "lr": _BASE_LR * (_FINAL_BATCH_SIZE / _BASE_BATCH_SIZE),
    "reg_warmup_epochs": 100,
    "reg_anneal_epochs": 50,
    "scheduler_T_0": 50,
    "scheduler_T_mult": 1,
    "scheduler_eta_min": 1e-5,
}

# Temperature annealing for soft library selection (R6.1: monotonic,
# R5-prune-retrain-fixes: smooth linear hardening over the last 20% of
# training instead of a step at 90%).
TAU = {
    "init": 1.0,
    "final": 0.1,
    "min": 0.15,
    "T_0": 80,
    "hardening_epoch_frac": 0.1,
    # Two-phase tau for pruning (R2-phase-tau):
    # Pre-prune annealing stops at final_pretrain (gentle specialization,
    # prevents z-death). Retrain continues from final_pretrain down to
    # final (aggressive hardening, safe because surviving edges are
    # proven non-Z). Only active when --prune is passed.
    "final_pretrain": 0.8,
}

# Regularizer weights
# (CP: complexity-regularized pruning decomposition. Replaces the single
# merged "complexity" proxy with four per-component terms:
#   edge_gate     : λ_E · Σ_e σ(z_logits)             (active edge count)
#   node_gate     : λ_N · Σ_j σ(u_logits)             (DEPRECATED: always 0)
#   power         : λ_P · Σ_e z_e · m_e · Σ_q w_q·gm_q (static power proxy)
#   capacitance   : λ_C · C_eff · Σ_j u_j             (DEPRECATED: always 0)
# RR-D: per-preset overrides live on each PRESET entry as a "lambdas" dict
# and are merged over this global LAMBDAS by train_script.
# deprecate-node-gates: node_gate and capacitance lambdas are hardcoded to
# 0.0 in every config — nodes are pruned by connectivity only, and the
# C_eff·Σu proxy would be a constant per stage if node_mask were 1.0.
LAMBDAS = {
    "sparsity": 0,
    "rail": 0.1,
    "edge_gate": 0,
    "node_gate": 0.0,       # DEPRECATED (deprecate-node-gates)
    "power": 1e-6,
    "capacitance": 0.0,     # DEPRECATED (deprecate-node-gates)
    "entropy": 1e-4,
    # tanh saturation penalty for FreeTanhLibrary edges: mean(tanh(u)^2) over edges,
    # penalizes edges operating in the saturated region of tanh.
    "tanh_sat": 0.0,
}

# Pruning thresholds for the overprovision-then-prune pipeline (CP-5).
# (three-phase-schedule: updated to higher defaults — 0.01 was too forgiving
# for the new gate-trained regime. 0.1 for edges, 0.05 for nodes gives a
# usable Pareto frontier when gates have been pushed by Phase B regularizers.)
# deprecate-node-gates: prune_nodes_by_gate is now hardcoded to False in
# every config (PRUNE, SCHEDULE_*, presets). The flag is preserved on the
# function signature of prune_stage / prune_network for backward compat
# but has no effect; CLI flags --prune-nodes-by-gate / --no-prune-nodes-by-gate
# are kept as deprecated no-ops.
PRUNE = {
    "edge_threshold": 0.001,
    "node_threshold": 0.05,
    "prune_nodes_by_gate": False,  # DEPRECATED (deprecate-node-gates): always False
}

# Three-phase training schedule (three-phase-schedule plan, refined by four-phase-redesign).
# Generic schedule that any preset can opt into via ``preset["schedule"] = "three_phase"``.
# Splits the total epoch budget into three phases with independently configured
# tau annealing and structural regularizer magnitudes. Calibrated for the
# post-x_max=3.0 operating regime where task loss is ~0.03 (not ~1.0) — see
# spec/three-phase-schedule.md for the full scale discipline rationale.
#
# four-phase-redesign/Phase 1a updates:
#   - prune_edge_threshold 0.1 -> 0.05 (less violent pruning at B->C)
#   - prune_nodes_by_gate True -> False (nodes only die via connectivity backstop)
#   - node_gate in lambdas_b 1e-5 -> 0.0 (node_gate is too destructive when
#     used in concert with edge_gate; remove the regularizer from B entirely)
SCHEDULE_THREE_PHASE = {
    # Fraction of total epochs allocated to each phase.
    "frac_a": 0.3,            # Phase A: fit (no structure pressure)
    "frac_b": 0.7,            # Phase B: compress (Strategy 2 gate penalties)
    "frac_c": 0.0,            # Phase C: retrain after prune
    # Tau targets per phase.
    "tau_a": 0.8,              # Fixed tau during fit
    "tau_b_init": 0.8,         # Tau at start of compress
    "tau_b_final": 0.05,        # Tau at end of compress
    "tau_c_init": 0.4,         # Tau at start of retrain
    "tau_c_final": 0.1,        # Tau at end of retrain
    # Lambda warmup within Phase B: ramp from 0 to full over this fraction.
    "warmup_frac_b": 1.0 / 2.0,
    # Phase B target lambdas (Strategy 2: gate pruning first, tiny Z pressure).
    # deprecate-node-gates: node_gate=0, capacitance=0 (both regularizers
    # are no-ops; node pruning is connectivity-only).
    "lambdas_b": {
        "sparsity": 0,
        "edge_gate": 0,
        "node_gate": 0.0,        # DEPRECATED (deprecate-node-gates)
        "power": 1e-5,
        "capacitance": 0.0,      # DEPRECATED (deprecate-node-gates)
        "tanh_sat": 0.0,
    },
    # Phase C retrain lambdas: gate penalties off (irrelevant post-prune),
    # tiny sparsity for crisp cell family, rail unchanged.
    "lambdas_c": {
        "sparsity": 0,
        "edge_gate": 0.0,
        "node_gate": 0.0,        # DEPRECATED (deprecate-node-gates)
        "power": 1e-7,
        "capacitance": 0.0,      # DEPRECATED (deprecate-node-gates)
        "tanh_sat": 0.0,
    },
    # Prune thresholds used at the Phase B->C boundary.
    # four-phase-redesign/1a: edge 0.1 -> 0.05 (gentler cut, retains more
    # edges so the retrain has material to work with).
    "prune_edge_threshold": 0.001,
    "prune_node_threshold": 0.001,
    # deprecate-node-gates: always False (node gates are bypassed; nodes
    # only die via the connectivity backstop).
    "prune_nodes_by_gate": False,  # DEPRECATED (deprecate-node-gates)
}

# Four-phase training schedule (four-phase-redesign/Phase 3a).
# Splits Phase B from the three-phase schedule into B1 (cell commitment,
# no pruning) and B2 (edge pruning, readiness-gated). Adds teacher
# distillation (lambda_kd) and STE cell mode. See spec/four-phase-schedule.md.
SCHEDULE_FOUR_PHASE = {
    # Fraction of total epochs allocated to each phase. Must sum to 1.0.
    "frac_a": 0.3,            # Phase A: free fit (soft teacher)
    "frac_b1": 0.2,           # Phase B1: cell commitment (no pruning)
    "frac_b2": 0.2,           # Phase B2: edge pruning (readiness-gated)
    "frac_c": 0.3,            # Phase C: retrain compact model
    # Tau targets per phase.
    "tau_a": 0.6,              # Fixed tau during free fit
    "tau_b1_init": 0.6,        # Tau at start of B1
    "tau_b1_final": 0.5,       # Tau at end of B1
    "tau_b2_init": 0.5,        # Tau at start of B2
    "tau_b2_final": 0.4,       # Tau at end of B2 (readiness check here)
    "tau_c_init": 0.4,         # Tau at start of retrain
    "tau_c_final": 0.1,        # Tau at end of retrain
    # Phase B1 lambdas: cell commitment only, NO edge gate.
    # deprecate-node-gates: node_gate/capacitance default to 0.0 via .get.
    "lambdas_b1": {
        "sparsity": 5e-5,
        "power": 1e-4,
        "tanh_sat": 0.0,
    },
    # Phase B2 lambdas: sparsity + edge_gate. No node_gate (DEPRECATED).
    "lambdas_b2": {
        "sparsity": 5e-5,
        "edge_gate": 0,
        "power": 1e-5,
        "tanh_sat": 0.0,
    },
    # Phase C retrain lambdas: tiny sparsity for crisp cell family.
    "lambdas_c": {
        "sparsity": 1e-5,
        "tanh_sat": 0.0,
    },
    # Warmup within each compress phase: ramp from 0 to full over this fraction.
    "warmup_frac_b1": 0.25,
    "warmup_frac_b2": 0.25,
    # Teacher distillation weight (active in B1 and B2 only).
    "lambda_kd": 1.0,
    # Readiness-based prune trigger (Phase B2 -> Phase C boundary).
    # ALL conditions must be true to fire (AND-logic).
    "readiness_ratio_max": 1.2,        # val_argmax / val_soft must be below this
    "readiness_prob_min": 0.85,        # mean_max_cell_prob must be above this
    "readiness_stability_max": 0.02,   # std(frac_sigma_z_below_0.1) over window
    "readiness_improvement_min": 1e-4, # val_argmax improvement rate per epoch
    "readiness_window": 10,            # number of recent readings for stability checks
    # Prune thresholds (used at the B2->C boundary, whether readiness-gated
    # or fallback).
    "prune_edge_threshold": 0.01,
    "prune_node_threshold": 0.01,
    "prune_nodes_by_gate": False,  # DEPRECATED (deprecate-node-gates)
}

# Fixed-step Heun solver
SOLVER = {
    "method": "heun",
    "t_span": 7.0,
    "num_steps": 70,
}

# Low-rank input-driven VCA (Voltage-Controlled Amplifier) gating.
# Applied to boundary, temporal-readout, and (with --vca-core) core
# edges. VCA gate per edge is
#   gate_e = 2 * sigma( b_e + (u^T W) v_e )
# where ``W`` (in_dim x rank) is a shared input projection and ``v_e``
# (rank) is a per-edge embedding. Physically: ``rank`` global control
# buses broadcast from the input terminals, each gated edge taps into
# them with a programmable weight.
#
# Identity-at-init: ``vca_W`` is initialized to all-zero so ``u^T W = 0``
# and ``gate_e = 2 * sigma(0) = 1.0`` at epoch 0 regardless of v_e.
# Bit-identical to VCA-off baseline at step 0 — the correct null
# hypothesis. Per-edge embeddings are random at init so ``dL/dW`` receives
# signal at step 0 via the u ⊗ v_e path; v_e itself receives signal once
# W moves.
#
# The 2-sigma gain range (vs 1-sigma) lets the network amplify AND
# attenuate edge currents symmetrically.
#
# Default rank=2 keeps the parameter cost small. Biases are opt-in via
# ``--vca-bias`` (or ``VCA['bias'] = True``). Compatible with
# --freeze-read because ``u`` is constant per sample across all Heun
# sub-iterations, so the VCA gate is computed once per sample at stage
# entry and either folded into the frozen KCL contribution (freeze_read
# ON) or passed as a kwarg to rhs (freeze_read OFF).
VCA = {
    "rank": 2,                # projection rank r; must be >= VCA['min_rank']
    "scale_init": 0.01,       # std for v_e init (W is init to all-zero)
    "min_rank": 1,
    "bias": False,             # enable learned per-edge affine VCA offsets
}

# Initialization biases
# (fix-z-death: logits_z_bias=0.0 (was 1.0) gives all four cells equal
# P=0.25 probability at init. Previously Z started at ~42% and, combined
# with tau annealing 1.0→0.1, was amplified to ~99.99% by the end of
# training — stage0_logits gradient (~6e-6) was 5M× too small to overcome
# the +1.0 logit bias within 100 epochs.
# Gates: z_logit_init=0.0, u_logit_init=0.0 → σ=0.5, dσ/dz=0.25. This
# sits at the maximum-gradient point of the sigmoid, giving gates 2.4×
# more sensitivity to regularization than z=2.0 (σ≈0.88, dσ/dz≈0.10).
# Prior value of 2.0 caused 0 edges to ever prune (frac_sigma_z_below_0.1
# stuck at 0 for entire training) because the regularizer gradient λ*σ*(1-σ)
# ≈ 1e-6 was negligible. With z=0 the same regularizer gives 2.5e-6 and
# gates can actually respond to structural pressure.
# Node gates (u_logit) are changed for symmetry, though node gate pruning
# is not currently active — nodes only die via edge disconnection.
# Trade-off: 50% open gates reduce raw signal flow vs 88% open. This is
# accepted because the prior 88% open was effectively permanent — gates
# never closed anyway. A 50% start that CAN close is more useful than
# 88% that cannot.)
INIT = {
    "logits_z_bias": 0.0,
    "raw_mult_init": 0.0,
    "raw_leak_init": -3.0,
    "leak_constant": 0.0486,  # softplus(-3.0), fixed leak when --leak non-programmable
    "gain_scale": 1.0,
    "z_logit_init": 0.0,
    "u_logit_init": 0.0,
}


# =============================================================================
# Preset topology configs
# =============================================================================
# Each stage config in `stages` is consumed by StageTopologyBuilder:
#   num_inputs   : placeholder input nodes (filtered out before ODE)
#   num_hidden   : hidden nodes that participate in the ODE dynamics
#   num_proj     : projection nodes (read via OutputMapper)
#   num_outputs  : placeholder output nodes (see note below)
#   hidden_family: 'line' | 'ring' | 'grid' | 'small_world' |
#                  'torus' | 'empty'
#   *_pattern    : 'all_to_all' | 'one_to_one' | 'none' for input/output/proj
#                  bipartite connections
#   t_span, num_steps: per-stage Heun integration horizon and step count
#
# num_outputs:
#   Adds output placeholder nodes to the SparseTopology graph, connected to
#   hidden+proj nodes via `output_pattern`. These are FILTERED OUT by
#   topology_to_stage() and never enter the ODE core. The actual output
#   dimension is controlled by the preset-level `out_dim`, mapped by
#   OutputMapper(node_dim=read_dim, out_dim=out_dim) reading from the
#   final stage's projection nodes (or hidden if no projections). See
#   topology.py:build_net_from_config.
#   Setting num_outputs=0 is correct and safe for all stages; non-zero values
#   are vestigial and create no-op placeholder nodes in the topology.
#
# Paper v1 active presets (R4):
#   - sinx:    1D regression, ring hidden + projection nodes
#   - housing: 8D regression, ring hidden + projection nodes (kept as a
#              appendix-level sanity check, not central to the paper story)
#
# Deprecated presets (removed from PRESETS, code preserved in modules):
#   - xor:    2-bit logic gate, weak analog motivation (R4.2)
#   - solver: sparse linear system, scope creep for paper v1 (R4.3, 2.10)
# =============================================================================
PRESET_SINX = {
    "stages": [
        {
            "num_inputs": 1,
            "num_hidden": 8,
            "num_proj": 2,
            "num_outputs": 0,
            "hidden_family": "line",
            "hidden_kwargs": {"radius": 2},
            "input_pattern": "all_to_all",
            "output_pattern": "all_to_all",
            "proj_pattern": "all_to_all",
            "t_span": SOLVER["t_span"],
            "num_steps": SOLVER["num_steps"],
        },
    ],
    "use_robust_input": False,
    "loss": "mse",
    "out_dim": 1,
    "write_idx": [0],
    "read_idx": [7],
    "lambdas": {
        "rail": 1.0,
    },
}

PRESET_HOUSING = {
    "stages": [
        {
            "num_inputs": 8,
            "num_hidden": 16,
            "num_proj": 4,
            "num_outputs": 0,
            "hidden_family": "line",
            "hidden_kwargs": {"radius": 2},
            "input_pattern": "all_to_all",
            "output_pattern": "all_to_all",
            "proj_pattern": "all_to_all",
            "t_span": SOLVER["t_span"],
            "num_steps": SOLVER["num_steps"],
        },
    ],
    "use_robust_input": True,
    "loss": "huber",
    "out_dim": 1,
    "write_idx": [0, 1, 2, 3, 4, 5, 6, 7],
    "read_idx": [15],
}

PRESET_SMOOTH2D = {
    "stages": [
        {
            "num_inputs": 2,
            "num_hidden": 10,
            "num_proj": 2,
            "num_outputs": 0,
            "hidden_family": "line",
            "hidden_kwargs": {"radius": 2},
            "input_pattern": "all_to_all",
            "output_pattern": "all_to_all",
            "proj_pattern": "all_to_all",
            "t_span": SOLVER["t_span"],
            "num_steps": SOLVER["num_steps"],
        },
    ],
    "use_robust_input": False,
    "loss": "huber",
    "out_dim": 1,
    "write_idx": [0, 1],
    "read_idx": [9],
}

# Grid-topology variant of the smooth2d preset (smooth2d-grid-preset spec).
# Grid size is configurable via ``make_smooth2d_grid_preset(grid_size)``,
# which also accepts `--grid-size N` from train_script.py.
# Caller formula for exact topology layout is documented in
# spec/grid-size-cli.md.
#
# Fit-first: structural regularizers (edge_gate, node_gate, power,
# capacitance) are zeroed in the preset's 'lambdas' override. The
# --prune flag can still be passed for retrain experiments, but
# gates do not receive any gradient pressure during pre-prune fit.


def make_smooth2d_grid_preset(
    grid_size: int,
    num_stages: int = 3,
    num_proj: int = 3,
    bidirectional: bool = False,
    edge_repeats: int = 2,
    leak_mode: str = "programmable",
    leak_constant: float | None = None,
) -> dict:
    """Dynamically build the smooth2d_grid preset dict for any square grid size.

    ``grid_size`` is the height/width of the hidden grid (N×N). Each stage
    has ``grid_size ** 2`` hidden nodes, ``num_proj`` projection nodes,
    and ``num_inputs=2`` input nodes.

    The write fan-out and read indices are computed to match the patterns
    validated in the original git history (5×5 and 4×4 variants). See
    spec/grid-size-cli.md for the exact formulas.

    When ``bidirectional=True``, each grid_graph edge is realized as two
    directed edges (i->j and j->i), giving asymmetric cell types (P/rectifier)
    true bidirectional capability. Edge count is exactly 2× the single-direction
    count.

    ``edge_repeats`` (default 2) controls the number of parallel edges per
    hidden node pair. Composes multiplicatively with ``bidirectional``; each
    repeated edge gets independent per-edge parameters in DifferentialStage.
    """
    if edge_repeats < 1 or edge_repeats > 8:
        raise ValueError(f"edge_repeats must be in [1, 8], got {edge_repeats}")
    num_hidden = grid_size * grid_size
    n_stages = max(1, num_stages)
    _stage_cfg = {
        "num_inputs": 2,
        "num_hidden": num_hidden,
        "num_proj": num_proj,
        "num_outputs": 0,
        "hidden_family": "grid",
        "hidden_kwargs": {"height": grid_size, "width": grid_size, "kernel_size": 3, "bidirectional": bidirectional},
        "edge_repeats": edge_repeats,
        "input_pattern": "all_to_all",
        "output_pattern": "all_to_all",
        "proj_pattern": "all_to_all",
        "t_span": SOLVER["t_span"] / n_stages,
        "num_steps": round(SOLVER["num_steps"] / n_stages),
    }

    # Fan-out write: left column for input 0, right column for input 1.
    # For grid_size >= 5 use every other row (matches original 5×5 pattern);
    # for smaller grids use consecutive rows 0..height-2 (matches 4×4).
    if grid_size >= 5:
        rows = list(range(0, grid_size, 2))
    else:
        rows = list(range(grid_size - 1))
    fan_out = {
        0: [r * grid_size for r in rows],
        1: [r * grid_size + (grid_size - 1) for r in rows],
    }

    # Read indices: for grid_size >= 5 the center column is >1 hop from
    # the write columns, so include center-column hidden nodes + proj.
    # For smaller grids, only proj nodes are valid (every hidden node is
    # within 1 hop of a write target).
    if grid_size >= 5:
        center_col = grid_size // 2
        center_nodes = [r * grid_size + center_col for r in range(grid_size)]
        read_idx = center_nodes + list(range(num_hidden, num_hidden + num_proj))
    else:
        read_idx = list(range(num_hidden, num_hidden + num_proj))

    preset = {
        "stages": [_stage_cfg] * n_stages,
        "use_robust_input": False,
        "loss": "huber",
        "out_dim": 1,
        "write_mode": "fan_out",
        "write_fan_out": fan_out,
        "read_idx": read_idx,
        "schedule": "three_phase",
        "lambdas": {
            "sparsity": 1e-6,
            "edge_gate": 0,
            "node_gate": 0.0,        # DEPRECATED (deprecate-node-gates)
            "power": 1e-6,
            "capacitance": 0.0,      # DEPRECATED (deprecate-node-gates)
            "rail": 0.1,
        },
        "tau_anneal": True,
    }
    if leak_mode != "programmable" or leak_constant is not None:
        preset["leak_mode"] = leak_mode
        if leak_constant is not None:
            preset["leak_constant"] = leak_constant
    return preset


# Static default: 7×7 grid (49 hidden nodes), 3 stages, 3 proj nodes.
# Increased from 5×5 to give the network more capacity to specialize —
# at 5×5 the network showed zero train/val gap (underfitting) and used
# all edges uniformly (0 pruned), indicating the capacity was fully
# consumed by the blended-behavior solution.
PRESET_SMOOTH2D_GRID = make_smooth2d_grid_preset(grid_size=7)


def make_housing_grid_preset(
    grid_size: int,
    num_stages: int = 3,
    num_proj: int = 3,
    bidirectional: bool = False,
    edge_repeats: int = 2,
    leak_mode: str = "programmable",
    leak_constant: float | None = None,
) -> dict:
    """Build the housing_grid preset dict for any square grid size.

    Reuses the 5x5 grid + 3-stage topology from ``smooth2d_grid`` but
    swaps in the 8 California-housing features and uses dense write
    (all-to-all) since the features have no spatial structure.

    ``grid_size`` is the height/width of the hidden grid (N×N). Each
    stage has ``grid_size ** 2`` hidden nodes, ``num_proj`` projection
    nodes, and ``num_inputs=8`` (the eight California-housing features).

    Loss is declared as ``huber`` for preset metadata consistency; the
    actual training loss is ``F.huber_loss(delta=1.0)`` in
    ``train_script.make_data_housing_grid``. Validation logs are also
    reported in original housing-price units (USD × 100k) via inverse
    normalization of the standardized targets.

    When ``bidirectional=True``, each grid_graph edge is realized as two
    directed edges (i->j and j->i), giving asymmetric cell types (P/rectifier)
    true bidirectional capability. Edge count is exactly 2× the single-direction
    count.

    ``edge_repeats`` (default 2) controls the number of parallel edges per
    hidden node pair. Composes multiplicatively with ``bidirectional``; each
    repeated edge gets independent per-edge parameters in DifferentialStage.
    """
    if edge_repeats < 1 or edge_repeats > 8:
        raise ValueError(f"edge_repeats must be in [1, 8], got {edge_repeats}")
    num_hidden = grid_size * grid_size
    n_stages = max(1, num_stages)
    _stage_cfg = {
        "num_inputs": 8,
        "num_hidden": num_hidden,
        "num_proj": num_proj,
        "num_outputs": 0,
        "hidden_family": "grid",
        "hidden_kwargs": {"height": grid_size, "width": grid_size, "kernel_size": 3, "bidirectional": bidirectional},
        "edge_repeats": edge_repeats,
        "input_pattern": "all_to_all",
        "output_pattern": "all_to_all",
        "proj_pattern": "all_to_all",
        "t_span": SOLVER["t_span"] / n_stages,
        "num_steps": round(SOLVER["num_steps"] / n_stages),
    }

    # Read indices: same convention as smooth2d_grid (center column + proj)
    # so the model can read hidden activations that are 2+ hops from any
    # input source. This is a heuristic; for housing_grid there is no
    # spatial meaning to the grid, so it primarily acts as a structural
    # regularizer (limits the OutputMapper to a small subset).
    if grid_size >= 3:
        center_col = grid_size // 2
        center_nodes = [r * grid_size + center_col for r in range(grid_size)]
        read_idx = center_nodes + list(range(num_hidden, num_hidden + num_proj))
    else:
        read_idx = list(range(num_hidden, num_hidden + num_proj))

    preset = {
        "stages": [_stage_cfg] * n_stages,
        "use_robust_input": False,
        "loss": "huber",
        "out_dim": 1,
        "write_mode": "dense",
        "read_idx": read_idx,
        "schedule": "three_phase",
        "lambdas": {
            "sparsity": 1e-5,
            "edge_gate": 0,
            "node_gate": 0.0,        # DEPRECATED (deprecate-node-gates)
            "power": 1e-5,
            "capacitance": 0.0,      # DEPRECATED (deprecate-node-gates)
            "rail": 0.1,
        },
        "tau_anneal": True,
    }
    if leak_mode != "programmable" or leak_constant is not None:
        preset["leak_mode"] = leak_mode
        if leak_constant is not None:
            preset["leak_constant"] = leak_constant
    return preset


# Static default: 5x5 grid, 3 stages, 3 proj nodes (mirrors smooth2d_grid).
PRESET_HOUSING_GRID = make_housing_grid_preset(grid_size=5)


# Friedman synthetic regression tasks (friedman-problems/REQ).
# Three canonical Friedman problems (Friedman 1991 MARS paper) on a torus
# hidden topology with three_phase schedule and Huber loss. All three use
# sparse write (strided write_idx across the torus grid) so that each input
# feature lands on a distinct column/region of the grid; this gives the
# model locality prior rather than broadcasting every input everywhere.
# The CLI may override any field; this is just the starting point.

def make_friedman1_preset() -> dict:
    """Friedman #1 preset: 10-dim U(0,1) inputs (5 noisy), 5x5 torus.

    Writes are strided at row 0 (cols 0..4) + row 2 (cols 0..4) so 10 unique
    cells in the torus. Reads use ``read_mode='dense'`` (full state) because
    the per-column coverage in the 5x5 torus leaves no hidden node >1 hop
    from every write node under the kernel=3 neighbor rule.
    """
    num_hidden = 25
    return {
        "stages": [
            {
                "num_inputs": 10,
                "num_hidden": num_hidden,
                "num_proj": 0,
                "num_outputs": 0,
                "hidden_family": "torus",
                "hidden_kwargs": {"height": 5, "width": 5, "kernel_size": 3, "bidirectional": False},
                "input_pattern": "all_to_all",
                "output_pattern": "all_to_all",
                "proj_pattern": "all_to_all",
                "t_span": SOLVER["t_span"],
                "num_steps": SOLVER["num_steps"],
            },
        ],
        "use_robust_input": False,
        "loss": "huber",
        "out_dim": 1,
        "write_mode": "sparse_proj",
        "read_mode": "dense",
        "write_idx": [0, 1, 2, 3, 4, 10, 11, 12, 13, 14],
        "schedule": "three_phase",
        "tau_anneal": True,
    }


def make_friedman2_preset(
    hidden_family: str = "torus",
    small_world_k: int = 4,
    small_world_p: float = 0.3,
    small_world_seed: int = 0,
) -> dict:
    """Friedman #2 preset: 4-dim inputs with custom ranges, 4x4 torus.

    Each input lands on a distinct column (stride=4: nodes 0,4,8,12). With
    4 writes covering all 4 columns, reads use ``read_mode='dense'`` (full
    state) for the same reason as Friedman #1.

    With ``hidden_family='small_world'`` the hidden topology is a Watts-Strogatz
    small-world graph with parameters ``k``, ``p``, and ``seed``.

    Args:
        hidden_family: Topology family: ``"torus"`` or ``"small_world"``.
        small_world_k: For ``hidden_family='small_world'`` only; even ring-lattice
            degree (must be even and < 16).
        small_world_p: For ``hidden_family='small_world'`` only; rewiring
            probability in ``[0, 1]``.
        small_world_seed: For ``hidden_family='small_world'`` only; RNG seed for
            rewiring (deterministic graph).
    """
    if hidden_family not in ("torus", "small_world"):
        raise ValueError(f"hidden_family must be 'torus' or 'small_world', got {hidden_family!r}")
    if hidden_family == "small_world":
        if small_world_k < 2 or small_world_k % 2 != 0 or small_world_k >= 16:
            raise ValueError(
                f"small_world_k must be even, >=2, and <16 (num_hidden=16), got {small_world_k}"
            )
        if not (0.0 <= small_world_p <= 1.0):
            raise ValueError(f"small_world_p must be in [0, 1], got {small_world_p}")

    num_hidden = 16
    if hidden_family == "torus":
        hidden_kwargs = {"height": 4, "width": 4, "kernel_size": 3, "bidirectional": False}
    else:
        hidden_kwargs = {
            "k": small_world_k,
            "p": small_world_p,
            "seed": small_world_seed,
            "bidirectional": False,
        }

    return {
        "stages": [
            {
                "num_inputs": 4,
                "num_hidden": num_hidden,
                "num_proj": 0,
                "num_outputs": 0,
                "hidden_family": hidden_family,
                "hidden_kwargs": hidden_kwargs,
                "input_pattern": "all_to_all",
                "output_pattern": "all_to_all",
                "proj_pattern": "all_to_all",
                "t_span": SOLVER["t_span"],
                "num_steps": SOLVER["num_steps"],
            },
        ],
        "use_robust_input": False,
        "loss": "huber",
        "out_dim": 1,
        "write_mode": "sparse_proj",
        "read_mode": "dense",
        "write_idx": [0, 4, 8, 12],
        "schedule": "three_phase",
        "tau_anneal": True,
    }


def make_friedman3_preset() -> dict:
    """Friedman #3 preset: same shape as #2, different target.

    For consistency with ``make_friedman2_preset()``, expose the same topology
    options as defaults (torus with 4x4 hidden shape). Explicit overrides
    are not available for friedman3 (kept simple).
    """
    return make_friedman2_preset()


PRESET_FRIEDMAN1 = make_friedman1_preset()
PRESET_FRIEDMAN2 = make_friedman2_preset()
PRESET_FRIEDMAN3 = make_friedman3_preset()


PRESETS = {
    "sinx": PRESET_SINX,
    "housing": PRESET_HOUSING,
    "smooth2d": PRESET_SMOOTH2D,
    "smooth2d_grid": PRESET_SMOOTH2D_GRID,
    "housing_grid": PRESET_HOUSING_GRID,
    "friedman1": PRESET_FRIEDMAN1,
    "friedman2": PRESET_FRIEDMAN2,
    "friedman3": PRESET_FRIEDMAN3,
}
