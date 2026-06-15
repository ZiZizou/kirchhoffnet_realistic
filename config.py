"""All tunable constants for the reduced differential KirchhoffNet.

Every other module imports from this file. To tune the system, edit this file.

Cell library ordering: L (weak linear), S (saturating tanh), P (smooth bounded
rectifier), Z (disabled). Z is at index 3 so that logits[:, 3] = +1.0 yields
softmax ≈ [0.155, 0.155, 0.155, 0.421] — about 58% active cells at init.
(RR-A: lowered from +2.0 to start with more active edges so the network can
learn freely before staged regularizers are annealed in).

Units (R7): All conductance, current, and capacitance values are NORMALIZED
to lie in physically plausible ranges. They are NOT calibrated to a
consistent timescale: ``t_span`` and ``num_steps`` in SOLVER set the
integration window, and ``C_eff`` is a pure scaling. No pretence is made
that the resulting ω = g/C is a real analog time constant.
"""

# Cell library: gm (normalized transconductance, μS-scale), isat (sat current,
# μA-scale), rho (destination feedback coefficient, dimensionless), gleak
# (residual linear leakage, μS-scale), bias (per-family fixed offset, μA-scale).
CELL_L = {
    "gm": 0.2,
    "isat": 10.0,
    "rho": 1.0,
    "gleak": 0.01,
    "bias": 0.0,
    "theta": 0.0,
    "beta": 1.0,
}

CELL_S = {
    "gm": 1.0,
    "isat": 0.5,
    "rho": 1.0,
    "gleak": 0.01,
    "bias": 0.0,
    "theta": 0.0,
    "beta": 1.0,
}

CELL_P = {
    "gm": 1.0,
    "isat": 1.0,
    "rho": 1.0,
    "gleak": 0.0,
    "bias": 0.0,
    "theta": 0.0,
    "beta": 0.1,
}

CELL_Z = {
    "gm": 0.0,
    "isat": 0.0,
    "rho": 0.0,
    "gleak": 0.0,
    "bias": 0.0,
    "theta": 0.0,
    "beta": 1.0,
}

CELL_LIBRARY = {
    "L": CELL_L,
    "S": CELL_S,
    "P": CELL_P,
    "Z": CELL_Z,
}

CELL_ORDER = ["L", "S", "P", "Z"]
NUM_CELLS = len(CELL_ORDER)
Z_INDEX = CELL_ORDER.index("Z")

# Normalized physical limits (R7: not calibrated to real SI units).
PHYS = {
    "x_max": 3.0,
    "C_eff": 1.0,
    "beta_softness": 0.02,
    "clip_current": 0.05,
    "clip_softness": 0.02,
}

# Training hyperparameters
_BASE_BATCH_SIZE = 1024
_BASE_LR = 3e-4
_FINAL_BATCH_SIZE = 2048
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
#   node_gate     : λ_N · Σ_j σ(u_logits)             (active hidden node count)
#   power         : λ_P · Σ_e z_e · m_e · Σ_q w_q·gm_q (static power proxy)
#   capacitance   : λ_C · C_eff · Σ_j u_j             (capacitance area proxy)
# RR-D: per-preset overrides live on each PRESET entry as a "lambdas" dict
# and are merged over this global LAMBDAS by train_script.)
LAMBDAS = {
    "sparsity": 1e-3,
    "rail": 1.0,
    "edge_gate": 5e-4,
    "node_gate": 1e-4,
    "power": 1e-4,
    "capacitance": 1e-5,
    "entropy": 0.0,
}

# Pruning thresholds for the overprovision-then-prune pipeline (CP-5).
# (three-phase-schedule: updated to higher defaults — 0.01 was too forgiving
# for the new gate-trained regime. 0.1 for edges, 0.05 for nodes gives a
# usable Pareto frontier when gates have been pushed by Phase B regularizers.)
PRUNE = {
    "edge_threshold": 0.1,
    "node_threshold": 0.05,
    # When True (legacy), nodes with σ(u_logits) <= node_threshold are pruned
    # independently, which causes collateral edge removal. When False, no
    # independent node pruning — nodes are only removed by the connectivity
    # backstop (dead island purge). See spec/edge-only-prune.md.
    "prune_nodes_by_gate": True,
}

# Three-phase training schedule (three-phase-schedule plan).
# Generic schedule that any preset can opt into via ``preset["schedule"] = "three_phase"``.
# Splits the total epoch budget into three phases with independently configured
# tau annealing and structural regularizer magnitudes. Calibrated for the
# post-x_max=3.0 operating regime where task loss is ~0.03 (not ~1.0) — see
# spec/three-phase-schedule.md for the full scale discipline rationale.
SCHEDULE_THREE_PHASE = {
    # Fraction of total epochs allocated to each phase.
    "frac_a": 0.30,            # Phase A: fit (no structure pressure)
    "frac_b": 0.40,            # Phase B: compress (Strategy 2 gate penalties)
    "frac_c": 0.30,            # Phase C: retrain after prune
    # Tau targets per phase.
    "tau_a": 1.0,              # Fixed tau during fit
    "tau_b_init": 1.0,         # Tau at start of compress
    "tau_b_final": 0.6,        # Tau at end of compress
    "tau_c_init": 0.6,         # Tau at start of retrain
    "tau_c_final": 0.1,        # Tau at end of retrain
    # Lambda warmup within Phase B: ramp from 0 to full over this fraction.
    "warmup_frac_b": 1.0 / 6.0,
    # Phase B target lambdas (Strategy 2: gate pruning first, tiny Z pressure).
    "lambdas_b": {
        "sparsity": 1e-4,
        "edge_gate": 5e-5,
        "node_gate": 1e-5,
        "power": 1e-5,
        "capacitance": 1e-6,
    },
    # Phase C retrain lambdas: gate penalties off (irrelevant post-prune),
    # tiny sparsity for crisp cell family, rail unchanged.
    "lambdas_c": {
        "sparsity": 1e-5,
        "edge_gate": 0.0,
        "node_gate": 0.0,
        "power": 0.0,
        "capacitance": 0.0,
    },
    # Prune thresholds used at the Phase B→C boundary.
    "prune_edge_threshold": 0.1,
    "prune_node_threshold": 0.05,
    # See PRUNE['prune_nodes_by_gate']. Default (True) is the legacy
    # behavior; set to False to keep low-u nodes alive (edge-only prune).
    "prune_nodes_by_gate": True,
}

# Fixed-step Heun solver
SOLVER = {
    "method": "heun",
    "t_span": 5,
    "num_steps": 50,
}

# Initialization biases
# (fix-z-death: logits_z_bias=0.0 (was 1.0) gives all four cells equal
# P=0.25 probability at init. Previously Z started at ~42% and, combined
# with tau annealing 1.0→0.1, was amplified to ~99.99% by the end of
# training — stage0_logits gradient (~6e-6) was 5M× too small to overcome
# the +1.0 logit bias within 100 epochs.
# Gates: z_logit_init=2.0, u_logit_init=2.0 → σ≈0.88, dσ/dz≈0.10. This
# gives 88% open gates (strong signal flow through deep ODE stacks) while
# keeping gate gradients 14× healthier than z=5.0 (dσ/dz=0.007) and only
# 2× worse than z=0.0. Balanced between expressivity and learnability.)
INIT = {
    "logits_z_bias": 0.0,
    "raw_mult_init": 0.0,
    "raw_leak_init": -3.0,
    "gain_scale": 1.0,
    "z_logit_init": 2.0,
    "u_logit_init": 2.0,
}

# Variation context defaults
# (RR-C: temp_c sampling is deprecated. The dataclass field is kept for
# backward compatibility but no longer randomized; ``temp_c_choices`` is
# only read by external code that explicitly opts in via the
# ``sample_random_context(..., legacy_temp=True)`` path.)
VARIATION = {
    "temp_c_default": 27.0,
    "temp_c_choices": [0.0, 27.0, 75.0],
    "global_gain_shift_std": 0.05,
    "edge_mismatch_std": 0.05,
}


def cells_to_tensor_dict():
    """Stack CELL_LIBRARY into a dict of 1-D tensors ordered by CELL_ORDER.

    Returns dict of tensors with shape [NUM_CELLS] for: gm, isat, rho, gleak,
    bias, theta, beta. theta/beta are only used by rectifier cells (P); other
    cells carry neutral dummy values (theta=0, beta=1) for buffer alignment.
    """
    import torch
    keys = ["gm", "isat", "rho", "gleak", "bias", "theta", "beta"]
    return {k: torch.tensor([CELL_LIBRARY[c][k] for c in CELL_ORDER], dtype=torch.float32) for k in keys}


# =============================================================================
# Preset topology configs
# =============================================================================
# Each stage config in `stages` is consumed by StageTopologyBuilder:
#   num_inputs   : placeholder input nodes (filtered out before ODE)
#   num_hidden   : hidden nodes that participate in the ODE dynamics
#   num_proj     : projection nodes (read via OutputMapper)
#   num_outputs  : placeholder output nodes (see note below)
#   hidden_family: 'line' | 'ring' | 'grid' | 'cluster' | 'empty'
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
    "loss": "mae",
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
    "loss": "mse",
    "out_dim": 1,
    "write_idx": [0, 1],
    "read_idx": [9],
}

# Grid-topology variant of the smooth2d preset (smooth2d-grid-preset spec).
# 4x4 = 16 hidden grid nodes with 8-neighbor local connectivity
# (grid_graph kernel_size=3). 3 projection nodes are connected
# bidirectionally to all 16 hidden nodes (proj_pattern="all_to_all",
# yielding 16*3 = 48 projection edges; 42 hidden edges).
#
# Multi-stage (multistage-smooth2d-grid spec): 3 independent stages with
# untied weights. Each stage has the same 4x4 grid + 3 proj topology (19
# active nodes, 90 edges) but its own learnable parameters. Total t_span=5
# is split equally (5/3 ≈ 1.667 per stage) and num_steps=50 proportionally
# (17 per stage), keeping dt ≈ 0.098 constant. StageTransfer between stages
# is identity (19→19). This triples the edge parameters (~1767 total) while
# making each per-stage backward pass 3x shallower, directly addressing
# the vanishing gradient issue. Gradient norms per stage are logged
# separately in grad_norms.txt.
#
# Fan-out write (smooth2d-sanity-pass spec): each input writes to 3
# designated hidden nodes (left/right columns of the 4x4 grid) via
# FanOutInputMapper. With 4 rows, write targets are left column rows
# 0,1,2 = [0, 4, 8] and right column rows 0,1,2 = [3, 7, 11]. This
# gives 6 total write targets.
#
# Read (proj-only): 3 projection nodes only. With 4-wide grid and
# 8-neighbor connectivity, every hidden node is within 1 hop of the
# left/right write columns, so the >1-hop topology check would fail
# for any hidden read. Topology check is skipped when all reads are
# proj (see topology.py:validate_topology_degrees). OutputMapper is a
# Linear(3, 1) over these 3 read positions. read_idx targets the LAST
# stage's state vector.
#
# Fit-first: structural regularizers (edge_gate, node_gate, power,
# capacitance) are zeroed in the preset's 'lambdas' override. The
# --prune flag can still be passed for retrain experiments, but
# gates do not receive any gradient pressure during pre-prune fit.
_STAGE_CFG_16G_3P = {
    "num_inputs": 2,
    "num_hidden": 16,
    "num_proj": 3,
    "num_outputs": 0,
    "hidden_family": "grid",
    "hidden_kwargs": {"height": 4, "width": 4, "kernel_size": 3},
    "input_pattern": "all_to_all",
    "output_pattern": "all_to_all",
    "proj_pattern": "all_to_all",
    "t_span": SOLVER["t_span"] / 3,
    "num_steps": round(SOLVER["num_steps"] / 3),
}
PRESET_SMOOTH2D_GRID = {
    "stages": [
        _STAGE_CFG_16G_3P,
        _STAGE_CFG_16G_3P,
        _STAGE_CFG_16G_3P,
    ],
    "use_robust_input": False,
    "loss": "mse",
    "out_dim": 1,
    "write_mode": "fan_out",
    "write_fan_out": {0: [0, 4, 8], 1: [3, 7, 11]},
    "read_idx": [16, 17, 18],
    "schedule": "three_phase",
    # Legacy per-preset lambda overrides kept for backward compatibility with
    # the --schedule legacy code path. When three_phase is active, the schedule
    # functions in train.py replace these with the phase-aware values.
    "lambdas": {
        "sparsity": 1e-4,
        "edge_gate": 5e-5,
        "node_gate": 1e-5,
        "power": 1e-5,
        "capacitance": 1e-6,
        "rail": 0.1,
    },
    "tau_anneal": True,
}

PRESETS = {
    "sinx": PRESET_SINX,
    "housing": PRESET_HOUSING,
    "smooth2d": PRESET_SMOOTH2D,
    "smooth2d_grid": PRESET_SMOOTH2D_GRID,
}
