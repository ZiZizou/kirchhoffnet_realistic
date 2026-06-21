"""Train a reduced differential KirchhoffNet on a chosen task.

CLI:
    train_script.py --problem {sinx,housing} [--output OUTPUT]
                    [--epochs EPOCHS] [--lr LR]
                    [--ablation {none,mapper-only,empty-graph}]
                    [--variation]

Outputs to --output:
  - loss_history.txt
  - loss_curve.png
  - model.pt
  - config_snapshot.txt
  - stage_*_graph_init.png  (one per stage)
  - stage_*_graph_trained.png
  - cell_selection_trained.png
  - trajectories.png
  - output_fit.png
  - pipeline.png

Regularizer warm-up (RR-A): ``[0, W)`` no penalty, ``[W, W+A)`` linear
anneal, ``[W+A, ∞)`` full penalty (defaults: W=50, A=50, see
``config.OPTIM``). Per-preset ``lambdas`` overrides (RR-D) are merged on
top of the global ``LAMBDAS`` dict before each epoch.
"""

import argparse
import os
import sys
import copy
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)

from config import (
    OPTIM,
    PRESETS,
    LAMBDAS,
    SCHEDULE_FOUR_PHASE,
    SOLVER,
    TAU,
    VARIATION,
    make_smooth2d_grid_preset,
    make_housing_grid_preset,
)
from cell_library import IdealizedCellLibrary, make_cell_library, SimpleEdgeLibrary
from topology import build_net_from_preset
from sim_context import SimContext, sample_random_context
from train import (
    compute_loss,
    compute_solver_loss,
    make_optimizer,
    tau_for_epoch,
    reg_schedule,
    phase_boundaries,
    phase_for_epoch,
    three_phase_tau,
    three_phase_lambdas,
    four_phase_boundaries,
    phase_for_epoch_four,
    four_phase_tau,
    four_phase_lambdas,
    four_phase_kd_active,
    prune_readiness_check,
    compute_solidification_metrics,
    validate_argmax,
)
from io_mapper import (
    FanOutInputMapper,
    InputMapper,
    OutputMapper,
    RobustInputMapper,
    SparseInputMapper,
)

# Monkey-patch _compute_regularizers to handle DataParallel wrapping
# (train.py is read-only on Kaggle; DataParallel strips module internals)
import train as _train_module
_orig_compute_regularizers = _train_module._compute_regularizers

def _dp_safe_compute_regularizers(net, trajs, tau, lambdas):
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    return _orig_compute_regularizers(net, trajs, tau, lambdas)

_train_module._compute_regularizers = _dp_safe_compute_regularizers
from visualize import (
    plot_stage_graph,
    plot_cell_selection,
    plot_trajectories,
    plot_output_fit,
    plot_network,
)


def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _ensure_dir(path: Path) -> Path:
    if path.exists():
        suffix = time.strftime("%Y%m%d_%H%M%S")
        path = path.with_name(f"{path.name}_{suffix}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_lambdas(problem: str) -> dict:
    """Build the active lambdas dict for ``problem`` (RR-D).

    Per-preset ``lambdas`` overrides are merged on top of the global
    ``LAMBDAS`` so each preset can tune a single knob (e.g. lower the
    rail penalty on sinx) without redefining the entire dict.
    """
    merged = dict(LAMBDAS)
    preset_lambdas = PRESETS[problem].get("lambdas", {})
    merged.update(preset_lambdas)
    return merged


def _resolve_schedule(problem: str, cli_value: str | None) -> str:
    """Resolve the active schedule mode (three-phase-schedule, four-phase-redesign).

    Precedence: explicit CLI flag > preset['schedule'] > 'legacy'.
    """
    if cli_value is not None:
        return cli_value
    preset_val = PRESETS[problem].get("schedule")
    if preset_val in ("legacy", "three_phase", "four_phase"):
        return preset_val
    return "legacy"


def _resolve_cell_mode(cli_value: str, phase: str, schedule_mode: str) -> str:
    """Resolve the cell selection mode for the current epoch
    (four-phase-redesign/Phase 2b).

    Behavior:
    - ``cli_value == 'soft'`` or ``'ste'``: honor the explicit override.
    - ``cli_value == 'auto'`` (default): use ``'soft'`` for Phase A and
      ``'ste'`` for Phase B/C. Outside of a phased schedule, always
      ``'soft'``.
    """
    if cli_value in ("soft", "ste"):
        return cli_value
    # 'auto'
    if schedule_mode in ("three_phase", "four_phase") and phase in ("B", "C", "B1", "B2"):
        return "ste"
    return "soft"


def _apply_ablation_set(args, schedule_mode: str) -> None:
    """Mutate ``args`` to apply a diagnostic ablation preset in place
    (four-phase-redesign/Phase 1c).

    The three meaningful presets are:

    - ``reg-only`` (A): keep tau=1.0 throughout, turn on gate
      regularizers, disable pruning. Tests whether regularization alone
      commits cells.
    - ``tau-only`` (B): anneal tau normally, all structural regularizers
      off, disable pruning. Tests whether tau annealing alone is enough.
    - ``edge-only`` (C): normal B path but disable node-gate pruning and
      use a lower edge threshold. Tests whether node-gate damage is the
      real culprit.

    Only mutates flags that the user did NOT explicitly set on the CLI
    (we read from ``sys.argv`` to detect explicit overrides). This lets a
    user still override the preset for individual knobs.
    """
    if args.ablation_set in (None, "none"):
        return

    import sys as _sys
    argv_tokens = set(_sys.argv[1:])

    def _set_if_unset(attr: str, value) -> None:
        flag_long = f"--{attr.replace('_', '-')}"
        if flag_long not in argv_tokens and attr not in argv_tokens:
            setattr(args, attr, value)

    if args.ablation_set == "reg-only":
        # A: regularization-only test. Force tau_b_final == tau_b_init == 1.0,
        # leave gate regularizers on (default three_phase behavior), and
        # skip pruning entirely.
        _set_if_unset("prune", False)
        print(
            "[ablation-set=reg-only] tau fixed at 1.0 through B, "
            "gate regularizers active, --prune disabled"
        )
    elif args.ablation_set == "tau-only":
        # B: tau-only test. Disable structural regularizers and skip pruning.
        # The current three_phase path zeros out structural regularizers in
        # Phase A, so the only moving piece here is pruning. We disable it
        # and zero lambdas_b to be explicit.
        _set_if_unset("prune", False)
        # We zero structural lambdas in Phase B by overriding the schedule
        # via a sentinel. The train loop reads the active schedule_cfg so
        # we mutate the schedule dict in place for the run duration
        # (a non-invasive in-memory override).
        from config import SCHEDULE_THREE_PHASE
        for k in ("sparsity", "edge_gate", "node_gate", "power", "capacitance"):
            if k in SCHEDULE_THREE_PHASE.get("lambdas_b", {}):
                SCHEDULE_THREE_PHASE["lambdas_b"][k] = 0.0
        from config import SCHEDULE_FOUR_PHASE
        for phase_key in ("lambdas_b1", "lambdas_b2"):
            if phase_key in SCHEDULE_FOUR_PHASE:
                for k in ("sparsity", "edge_gate", "power", "capacitance"):
                    if k in SCHEDULE_FOUR_PHASE[phase_key]:
                        SCHEDULE_FOUR_PHASE[phase_key][k] = 0.0
        print(
            "[ablation-set=tau-only] structural regularizers zeroed in B/B1/B2, "
            "tau anneals normally, --prune disabled"
        )
    elif args.ablation_set == "edge-only":
        # C: edge-only pruning test (DEPRECATED: node-gate pruning is
        # permanently disabled; this ablation set is a no-op).
        _set_if_unset("prune_edge_threshold", 0.05)
        _set_if_unset("stage_lr_scale", 1.0)
        _set_if_unset("retrain_stage_lr_scale", 1.0)
        _set_if_unset("mapper_lr_scale", 1.0)
        _set_if_unset("retrain_mapper_lr_scale", 1.0)
        _set_if_unset("freeze_mappers", False)
        print(
            "[ablation-set=edge-only] node-gate pruning permanently disabled "
            "(deprecate-node-gates), edge threshold 0.05, "
            "stage_lr_scale=1.0, mapper_lr_scale=1.0"
        )


def _validate_hidden_family_args(args) -> None:
    """Validate --hidden-family / --num-hidden / --num-stages / --edge-repeats
    combinations before any expensive setup. Raises ``ValueError`` on bad
    combos so argparse-style early failure is obvious to the user.

    Rules:
      - --hidden-family=cluster requires --num-hidden N (N >= 2)
      - --hidden-family=cluster + --grid-size is an error
        (cluster ignores grid; mixing is misleading)
      - --edge-repeats must be in [1, 8]
      - --num-stages must be >= 1
      - --num-hidden without --hidden-family emits a warning (ignored)
    """
    hf = args.hidden_family
    nh = args.num_hidden
    gs = args.grid_size
    er = args.edge_repeats
    ns = args.num_stages

    if er is not None and (er < 1 or er > 8):
        raise ValueError(
            f"--edge-repeats must be in [1, 8], got {er}"
        )
    if ns is not None and ns < 1:
        raise ValueError(
            f"--num-stages must be >= 1, got {ns}"
        )

    if hf is None:
        if nh is not None:
            print(
                f"[train] WARNING: --num-hidden={nh} is ignored because "
                f"--hidden-family was not specified"
            )
        return

    if hf == "cluster":
        if nh is None:
            raise ValueError(
                "--hidden-family=cluster requires --num-hidden N to be set. "
                "Pass an integer (e.g. --num-hidden 25) to specify the "
                "number of hidden nodes."
            )
        if nh < 2:
            raise ValueError(
                f"--num-hidden must be >= 2 for cluster family, got {nh}"
            )
        if gs is not None:
            raise ValueError(
                "--grid-size is not compatible with --hidden-family=cluster "
                "(cluster has no spatial grid; --num-hidden controls size)."
            )
    elif hf == "grid":
        if nh is not None:
            print(
                f"[train] note: --num-hidden={nh} is ignored for grid family; "
                f"use --grid-size N (default per problem)"
            )


def _make_dynamic_preset(
    problem: str,
    hidden_family: str,
    num_hidden: int,
    num_stages: int = 1,
    edge_repeats: int = 2,
    grid_size: int | None = None,
    bidirectional: bool = False,
    write_mode_override: str | None = None,
    read_mode_override: str | None = None,
) -> dict:
    """Build a fresh preset dict that overrides the topology of the named
    problem. Preserves problem-specific fields (num_inputs, loss, out_dim,
    use_robust_input, schedule, lambdas, tau_anneal, write_idx default).

    Args:
        problem: Base problem name (e.g. 'housing', 'sinx').
        hidden_family: 'grid' or 'cluster'.
        num_hidden: Number of hidden nodes. For 'grid', must equal
            grid_size**2 (we recompute grid_size from num_hidden if not
            given). For 'cluster', used directly.
        num_stages: Number of ODE stages (default 1).
        edge_repeats: Parallel edges per hidden pair (default 2).
        grid_size: For 'grid' family; height/width. If None and family='grid',
            defaults to 5 (matches housing_grid default).
        bidirectional: Emit two directed edges per hidden node pair (i->j
            AND j->i). Doubles the hidden edge count and gives asymmetric
            cells (P/rectifier) true bidirectional capability. Composes
            multiplicatively with ``edge_repeats``.
        write_mode_override: If set, use this write_mode instead of
            family default. CLI --write-mode takes precedence.
        read_mode_override: If set, use this read_mode instead of family
            default. CLI --read-mode takes precedence.

    Returns:
        Fresh dict ready to assign to ``PRESETS[problem]`` before calling
        ``build_net_from_preset(problem)``.
    """
    from config import SOLVER
    base = dict(PRESETS[problem])
    # Use the FIRST stage's num_inputs as the canonical input dimension
    # (multi-stage presets all share num_inputs; this is just defensive).
    num_inputs = int(base["stages"][0]["num_inputs"])
    n_stages = max(1, int(num_stages))
    er = int(edge_repeats)

    if hidden_family == "cluster":
        num_proj = 0
        eff_num_hidden = int(num_hidden)
        hidden_kwargs = {
            "edge_prob": 1.0,
            "seed": 0,
            "bidirectional": bidirectional,
        }
        eff_write_mode = (
            write_mode_override if write_mode_override is not None else "dense"
        )
        read_idx = list(range(eff_num_hidden))
        eff_read_mode = "sparse" if read_mode_override is None else read_mode_override
        # Remove preset write_idx (incompatible with dense write).
        base.pop("write_idx", None)
    elif hidden_family == "grid":
        if grid_size is None:
            grid_size = max(2, round(int(num_hidden) ** 0.5))
        eff_num_hidden = grid_size * grid_size
        if num_hidden is not None and eff_num_hidden != int(num_hidden):
            print(
                f"[train] note: --num-hidden={num_hidden} rounded to "
                f"grid_size={grid_size} (eff_num_hidden={eff_num_hidden})"
            )
        num_proj = 3
        hidden_kwargs = {
            "height": grid_size,
            "width": grid_size,
            "kernel_size": 3,
            "bidirectional": bidirectional,
        }
        eff_write_mode = (
            write_mode_override if write_mode_override is not None
            else base.get("write_mode", "fan_out")
        )
        if grid_size >= 3:
            center_col = grid_size // 2
            center_nodes = [r * grid_size + center_col for r in range(grid_size)]
            read_idx = center_nodes + list(range(eff_num_hidden, eff_num_hidden + num_proj))
        else:
            read_idx = list(range(eff_num_hidden, eff_num_hidden + num_proj))
        eff_read_mode = "sparse" if read_mode_override is None else read_mode_override
    else:
        raise ValueError(f"Unknown hidden_family: {hidden_family!r}")

    stage_cfg = {
        "num_inputs": num_inputs,
        "num_hidden": eff_num_hidden,
        "num_proj": num_proj,
        "num_outputs": 0,
        "hidden_family": hidden_family,
        "hidden_kwargs": hidden_kwargs,
        "edge_repeats": er,
        "input_pattern": "all_to_all",
        "output_pattern": "all_to_all",
        "proj_pattern": "all_to_all",
        "t_span": SOLVER["t_span"] / n_stages,
        "num_steps": round(SOLVER["num_steps"] / n_stages),
    }

    new_preset = dict(base)
    new_preset["stages"] = [stage_cfg] * n_stages
    new_preset["write_mode"] = eff_write_mode
    new_preset["read_idx"] = read_idx
    new_preset["read_mode"] = eff_read_mode
    # Remove schedule/lambdas overrides if present so dynamic topology uses
    # the global defaults from LAMBDAS. Per-problem schedules (e.g. housing's
    # default 'three_phase') are still preserved by the explicit
    # base.get('schedule') check below.
    return new_preset


def _log_solidification(log_path, epoch: int, metrics: dict) -> None:
    """Append one row of solidification metrics to ``log_path``.

    Writes header on first call (or if the file is empty). Value ordering
    is deterministic (sorted keys).
    """
    if not isinstance(log_path, Path):
        log_path = Path(log_path)
    sorted_keys = sorted(k for k in metrics.keys())
    if not log_path.exists() or log_path.stat().st_size == 0:
        with open(log_path, "w") as f:
            f.write("epoch\t" + "\t".join(sorted_keys) + "\n")
    with open(log_path, "a") as f:
        row = [str(epoch)]
        for k in sorted_keys:
            v = metrics[k]
            row.append(f"{v:.6e}" if isinstance(v, float) else str(v))
        f.write("\t".join(row) + "\n")


def _save_config_snapshot(out_dir: Path, problem: str, args, lambdas: dict,
                          net=None) -> None:
    snap_path = out_dir / "config_snapshot.txt"
    with open(snap_path, "w") as f:
        f.write(f"problem: {problem}\n")
        f.write(f"preset: {PRESETS[problem]}\n\n")
        f.write("LAMBDAS (global):\n")
        for k, v in LAMBDAS.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nLAMBDAS (effective for {problem}, preset overrides applied):\n")
        for k, v in lambdas.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nOPTIM:\n")
        for k, v in OPTIM.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nVARIATION:\n")
        for k, v in VARIATION.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nI/O MAPPING:\n")
        if net is not None:
            from io_mapper import FanOutInputMapper, SparseInputMapper, InputMapper
            if isinstance(net.input_mapper, FanOutInputMapper):
                effective_write = "fan_out"
            elif isinstance(net.input_mapper, SparseInputMapper):
                effective_write = "one_to_one"
            elif isinstance(net.input_mapper, InputMapper):
                effective_write = "dense"
            else:
                effective_write = type(net.input_mapper).__name__
            f.write(f"  write_mode: {effective_write}\n")
            f.write(f"  input_mapper: {type(net.input_mapper).__name__}\n")
            if effective_write == "fan_out":
                f.write(f"  fan_out_map: {net.input_mapper.fan_out_map}\n")
            f.write(f"  write_idx: {list(net.write_idx) if net.write_idx is not None else None}\n")
            f.write(f"  output_mapper: {type(net.output_mapper).__name__}\n")
            f.write(f"  read_idx: {list(net.read_idx) if net.read_idx is not None else None}\n")
            f.write(f"  hid_count: {net.hid_count}\n")
            f.write(f"  proj_count: {net.proj_count}\n")
            f.write(f"  stage_lr_scale: {args.stage_lr_scale}\n")
            f.write(f"  mapper_lr_scale: {args.mapper_lr_scale}\n")
            f.write(f"  retrain_stage_lr_scale: {args.retrain_stage_lr_scale}\n")
            f.write(f"  retrain_mapper_lr_scale: {args.retrain_mapper_lr_scale}\n")
            f.write(f"  freeze_mappers: {args.freeze_mappers}\n")
        else:
            f.write(f"  write_mode: {args.write_mode}\n")
            f.write(f"  read_mode: {args.read_mode}\n")
            f.write(f"  write_idx: {args.write_idx}\n")
            f.write(f"  read_idx: {args.read_idx}\n")


def make_data_sinx(batch_size: int, val_size: int = 1024):
    train_size = 8192
    u_train = (torch.rand(train_size, 1) * 2 * torch.pi) - torch.pi
    y_train = torch.sin(u_train)
    u_val = (torch.rand(val_size, 1) * 2 * torch.pi) - torch.pi
    y_val = torch.sin(u_val)

    train_loader = DataLoader(
        TensorDataset(u_train, y_train), batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        TensorDataset(u_val, y_val), batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, val_loader, F.mse_loss


def _load_california_housing_data() -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Load raw California housing: (X, y_normalized, y_mean, y_std).

    Targets are standardized (subtract mean, divide by std). Features
    are returned *unnormalized* so that each caller can apply its own
    normalization scheme.
    """
    try:
        from sklearn.datasets import fetch_california_housing
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for the housing preset. "
            "Install with: uv pip install scikit-learn"
        ) from e

    data = fetch_california_housing()
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.float32).unsqueeze(1)
    y_mean = float(y.mean().item())
    y_std = float(y.std().clamp(min=1e-6).item())
    y_norm = (y - y_mean) / y_std
    return X, y_norm, y_mean, y_std


def _make_data_split(
    X: torch.Tensor, y: torch.Tensor, batch_size: int, seed: int = 42
) -> tuple[DataLoader, DataLoader]:
    """Deterministic 80/20 train/val split + DataLoaders."""
    n = X.shape[0]
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng)
    n_train = int(0.8 * n)
    train_ds = TensorDataset(X[perm[:n_train]], y[perm[:n_train]])
    val_ds = TensorDataset(X[perm[n_train:]], y[perm[n_train:]])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def make_data_housing(batch_size: int):
    """California-housing regression on the line topology.

    Features are min-max scaled to [0, 1] per column. This handles both
    non-negative features (Population) and signed features (Longitude ~-124
    to -114): dividing by column max alone would clamp the negative
    Longitude max to 1e-6 and produce values of ~-1e8 that overflow float16
    under AMP (zeroing all gradients via 0*inf=NaN in the backward pass).
    Targets are standardized. Training loss is L1 (MAE).

    Returns ``(train_loader, val_loader, task_fn, inverse_stats)`` where
    ``inverse_stats`` is ``{"y_mean": ..., "y_std": ...}`` for
    denormalizing validation predictions back to original units.
    """
    X, y_norm, y_mean, y_std = _load_california_housing_data()

    x_min = X.min(dim=0, keepdim=True).values
    x_max = X.max(dim=0, keepdim=True).values
    x_range = (x_max - x_min).clamp(min=1e-6)
    X = (X - x_min) / x_range

    train_loader, val_loader = _make_data_split(X, y_norm, batch_size)

    def task_fn(y_pred, y_target):
        return F.l1_loss(y_pred, y_target)

    inverse_stats = {"y_mean": y_mean, "y_std": y_std}
    return train_loader, val_loader, task_fn, inverse_stats


def make_data_housing_grid(batch_size: int, huber_delta: float = 1.0):
    """California-housing regression on the 5x5 grid topology.

    Features are min-max scaled to [0, 1] per column (handles both
    non-negative features like ``Population`` and signed features like
    ``Longitude``). Targets are standardized. Training loss is Huber
    (delta=1.0), blending MSE's smooth gradients for small errors with
    L1's robustness to heavy tails in housing prices.

    Returns ``(train_loader, val_loader, task_fn, inverse_stats)`` where
    ``inverse_stats`` is ``{"y_mean": ..., "y_std": ...}`` for
    denormalizing validation predictions back to original units.
    """
    X, y_norm, y_mean, y_std = _load_california_housing_data()

    x_min = X.min(dim=0, keepdim=True).values
    x_max = X.max(dim=0, keepdim=True).values
    x_range = (x_max - x_min).clamp(min=1e-6)
    X = (X - x_min) / x_range

    train_loader, val_loader = _make_data_split(X, y_norm, batch_size)

    def task_fn(y_pred, y_target):
        return F.huber_loss(y_pred, y_target, delta=huber_delta)

    inverse_stats = {"y_mean": y_mean, "y_std": y_std}
    return train_loader, val_loader, task_fn, inverse_stats


def denormalize_targets(y_norm: torch.Tensor, inverse_stats: dict) -> torch.Tensor:
    """Map standardized targets back to original California-housing units."""
    return y_norm * inverse_stats["y_std"] + inverse_stats["y_mean"]


def _franke(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    t1 = -((9 * x1 - 2) ** 2) / 4 - ((9 * x2 - 2) ** 2) / 4
    t2 = -(9 * x1 + 1) ** 2 / 49 - (9 * x2 + 1) / 10
    t3 = -((9 * x1 - 7) ** 2) / 4 - ((9 * x2 - 3) ** 2) / 4
    t4 = -(9 * x1 - 4) ** 2 - (9 * x2 - 7) ** 2
    return (
        0.75 * t1.exp()
        + 0.75 * t2.exp()
        + 0.5 * t3.exp()
        - 0.2 * t4.exp()
    )


def _lhs_samples(n: int, d: int, seed: int = 42) -> torch.Tensor:
    """Latin Hypercube samples in [0, 1]^d with guaranteed stratification."""
    from scipy.stats.qmc import LatinHypercube
    sampler = LatinHypercube(d=d, seed=seed)
    return torch.from_numpy(sampler.random(n=n)).float()


def make_data_smooth2d(batch_size: int, val_size: int = 4000):
    # Fixed seed for reproducible train/val splits and noise across runs.
    n_train = 20000
    u_train = _lhs_samples(n_train, 2, seed=42)
    y_train = _franke(u_train[:, 0], u_train[:, 1]).unsqueeze(1)
    torch.manual_seed(42)
    u_val = torch.rand(val_size, 2)
    y_val = _franke(u_val[:, 0], u_val[:, 1]).unsqueeze(1)

    noise_std = 0.01
    y_train = y_train + noise_std * torch.randn_like(y_train)

    y_mean = y_train.mean()
    y_std = y_train.std().clamp(min=1e-6)
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std

    train_loader = DataLoader(
        TensorDataset(u_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(u_val, y_val), batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader, F.mse_loss


def make_data(problem: str, batch_size: int):
    if problem == "sinx":
        return make_data_sinx(batch_size)
    if problem == "housing":
        return make_data_housing(batch_size)
    if problem == "smooth2d":
        return make_data_smooth2d(batch_size)
    if problem == "smooth2d_grid":
        return make_data_smooth2d(batch_size)
    if problem == "housing_grid":
        return make_data_housing_grid(batch_size)
    raise ValueError(f"Unknown problem: {problem}")


def validate(net, val_loader, task_fn, ctx_factory, device, cell_mode: str = "soft") -> float:
    net.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for u, target in val_loader:
            u = u.to(device)
            target = target.to(device)
            ctx = ctx_factory(u.size(0), device=device)
            out, _ = net(u, ctx=ctx, store_trajectory=False, cell_mode=cell_mode)
            loss = task_fn(out, target)
            total += float(loss.item()) * u.size(0)
            n += u.size(0)
    net.train()
    return total / max(1, n)


def validate_with_inverse(
    net,
    val_loader,
    task_fn,
    ctx_factory,
    device,
    inverse_stats: dict | None,
    cell_mode: str = "soft",
) -> dict:
    """Validation that also reports metrics in the original (denormalized) target units.

    Returns a dict with:
      - ``val``: training-space loss (mean of ``task_fn(out, target)``)
      - ``mae_orig``: MAE in original target units (USD x 100k for California housing)
      - ``rmse_orig``: RMSE in original target units
    If ``inverse_stats`` is None, ``mae_orig`` and ``rmse_orig`` are
    reported as NaN (no denormalization available).
    """
    net.eval()
    total = 0.0
    n = 0
    se_sum = 0.0
    ae_sum = 0.0
    with torch.no_grad():
        for u, target in val_loader:
            u = u.to(device)
            target = target.to(device)
            ctx = ctx_factory(u.size(0), device=device)
            out, _ = net(u, ctx=ctx, store_trajectory=False, cell_mode=cell_mode)
            loss = task_fn(out, target)
            total += float(loss.item()) * u.size(0)
            if inverse_stats is not None:
                pred_orig = denormalize_targets(out, inverse_stats)
                targ_orig = denormalize_targets(target, inverse_stats)
                ae_sum += float((pred_orig - targ_orig).abs().sum().item())
                se_sum += float(((pred_orig - targ_orig) ** 2).sum().item())
            n += u.size(0)
    net.train()
    out_dict = {"val": total / max(1, n)}
    if inverse_stats is not None and n > 0:
        out_dict["mae_orig"] = ae_sum / n
        out_dict["rmse_orig"] = (se_sum / n) ** 0.5
    else:
        out_dict["mae_orig"] = float("nan")
        out_dict["rmse_orig"] = float("nan")
    return out_dict


def collect_predictions(net, inputs, ctx_factory, device) -> torch.Tensor:
    net.eval()
    with torch.no_grad():
        ctx = ctx_factory(inputs.size(0), device=device)
        out, _ = net(inputs, ctx=ctx, store_trajectory=True)
    net.train()
    return out


def apply_ablation(net, ablation: str) -> None:
    """Wrapper around train.apply_ablation with a [ablation=X] log line."""
    from train import apply_ablation as _apply_ablation
    _apply_ablation(net, ablation)
    print(f"[ablation={ablation}] applied to net")


def _parse_int_list(spec: str | None) -> list[int] | None:
    """Parse '0,3,5' -> [0,3,5]. None passes through."""
    if spec is None:
        return None
    spec = spec.strip()
    if not spec:
        return None
    return [int(x) for x in spec.split(",") if x.strip()]


def collect_gradient_norms(raw_net):
    """Walk ``raw_net`` (assumed unwrapped from DataParallel) and collect
    per-group L2 gradient norms.

    Returns: dict with keys
      stage{i}_logits, stage{i}_raw_mult, stage{i}_raw_leak,
      stage{i}_z_logits, stage{i}_u_logits  (one entry per stage)
      stage_transfer
      in_mapper
      out_mapper
    Each value is either a float (the L2 norm) or None if the group has
    no parameter or no gradient.
    """
    stage_sq = {}
    stage_components = ("logits", "raw_mult", "raw_leak", "z_logits", "u_logits")
    transfer_sq = 0.0
    transfer_found = False
    in_sq = 0.0
    in_found = False
    out_sq = 0.0
    out_found = False

    # First pass: register all expected keys (even when grads are None),
    # so callers always find them in the dict.
    for name, p in raw_net.named_parameters():
        if ".stages." in name:
            for comp in stage_components:
                if name.endswith("." + comp):
                    stage_idx = int(name.split(".stages.")[1].split(".")[0])
                    stage_sq.setdefault(f"stage{stage_idx}_{comp}", 0.0)
            if name.endswith(".cell_lib.param"):
                stage_idx = int(name.split(".stages.")[1].split(".")[0])
                stage_sq.setdefault(f"stage{stage_idx}_device_param", 0.0)

    # Second pass: accumulate squared norms where gradients exist.
    for name, p in raw_net.named_parameters():
        if p.grad is None:
            continue
        gnorm_sq = float(p.grad.norm().item()) ** 2
        if ".stages." in name:
            for comp in stage_components:
                if name.endswith("." + comp):
                    stage_idx = int(name.split(".stages.")[1].split(".")[0])
                    key = f"stage{stage_idx}_{comp}"
                    stage_sq[key] = stage_sq.get(key, 0.0) + gnorm_sq
            if name.endswith(".cell_lib.param"):
                stage_idx = int(name.split(".stages.")[1].split(".")[0])
                key = f"stage{stage_idx}_device_param"
                stage_sq[key] = stage_sq.get(key, 0.0) + gnorm_sq
        if "transfers" in name or "stage_transfer" in name:
            transfer_sq += gnorm_sq
            transfer_found = True
        if "input_mapper" in name:
            in_sq += gnorm_sq
            in_found = True
        elif "output_mapper" in name:
            out_sq += gnorm_sq
            out_found = True

    out = {}
    for k, v in stage_sq.items():
        out[k] = v ** 0.5 if v > 0 else None
    out["stage_transfer"] = transfer_sq ** 0.5 if transfer_found else None
    out["in_mapper"] = in_sq ** 0.5 if in_found else None
    out["out_mapper"] = out_sq ** 0.5 if out_found else None
    return out


def _grad_norm_keys(norms):
    """Deterministic key ordering for gradient norm output (shared by header and data rows)."""
    return sorted(
        [k for k in norms.keys() if k.startswith("stage")]
        + [k for k in ("stage_transfer", "in_mapper", "out_mapper") if k in norms]
    )


def log_gradient_norms(grad_log_path, epoch, raw_net, *, retrain=False, optimizer=None):
    """Append one row of per-group L2 gradient norms to ``grad_log_path``.

    On the first call, also writes the header row.

    If an ``optimizer`` with multiple param groups is provided, appends the
    per-group learning rates as additional columns (``lr0``, ``lr1``, ...).
    """
    raw = raw_net.module if isinstance(raw_net, torch.nn.DataParallel) else raw_net
    norms = collect_gradient_norms(raw)
    ordered_keys = _grad_norm_keys(norms)

    lr_keys = []
    if optimizer is not None and len(optimizer.param_groups) > 1:
        lr_keys = [f"lr{i}" for i in range(len(optimizer.param_groups))]

    if not grad_log_path.exists():
        cols = ordered_keys + lr_keys
        with open(grad_log_path, "w") as f:
            f.write("epoch\t" + "\t".join(cols) + "\n")

    prefix = "retrain_" if retrain else ""
    row_parts = [f"{prefix}{epoch}"]
    for k in ordered_keys:
        v = norms.get(k)
        row_parts.append(f"{v:.6e}" if v is not None else "-")
    if lr_keys:
        for g in optimizer.param_groups:
            row_parts.append(f"{g['lr']:.6e}")
    with open(grad_log_path, "a") as f:
        f.write("\t".join(row_parts) + "\n")


def make_static_ctx_factory():
    """Build a ctx_factory that always returns a default (variation-off) context."""
    def _factory(batch_size_: int, device: torch.device = "cpu", **_):
        return SimContext()
    return _factory


def _add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Populate ``parser`` with the train_script CLI flags.  Split out of
    ``main()`` so smoke tests can introspect the flag surface (PP-5)."""
    parser.add_argument(
        "--problem", choices=["sinx", "housing", "smooth2d", "smooth2d_grid", "housing_grid"], default="sinx",
        help="Task to train (default: sinx)",
    )
    parser.add_argument(
        "--grid-size", type=int, default=None, dest="grid_size",
        help="Hidden grid height/width for smooth2d_grid and housing_grid. "
             "Per-problem default: smooth2d_grid=7, housing_grid=5. "
             "Explicit --grid-size N overrides either. "
             "Only applies when --problem smooth2d_grid or --problem housing_grid.",
    )
    parser.add_argument(
        "--cell-library", type=str, default=None, dest="cell_library",
        choices=["legacy", "v15", "v2", "relu", "tanh"],
        help="Cell library: 'legacy' (L,S,P,Z, default), 'v15' (O_weak,O_hard,P0,N0,D1,Z), "
             "'v2' (mix-code/bias-code bounded library), "
             "'relu' (I=ReLU(p0*Vsrc+p1*Vdest+p2)), 'tanh' (I=tanh(p0*Vsrc+p1*Vdest+p2)). "
             "Overrides the preset's cell_library key if present.",
    )
    parser.add_argument(
        "--hidden-family", type=str, default=None, dest="hidden_family",
        choices=["grid", "cluster"],
        help="Hidden-node topology family (default: from preset). "
             "'grid' uses a 2D grid graph (requires --grid-size). "
             "'cluster' uses a fully connected Erdos-Renyi graph "
             "(requires --num-hidden). When set, dynamically rebuilds the "
             "preset's stages config to use the specified family instead "
             "of the hardcoded preset topology. Mutually exclusive with "
             "the preset's default family when combined with --grid-size.",
    )
    parser.add_argument(
        "--num-hidden", type=int, default=None, dest="num_hidden",
        help="Number of hidden nodes (default: from preset). "
             "Required when --hidden-family=cluster (must be >= 2). "
             "Ignored for grid family (uses --grid-size).",
    )
    parser.add_argument(
        "--num-stages", type=int, default=None, dest="num_stages",
        help="Number of ODE stages (default: from preset, or 1 if not "
             "specified). Each stage gets an identical topology. "
             "t_span and num_steps are divided evenly across stages.",
    )
    parser.add_argument(
        "--edge-repeats", type=int, default=None, dest="edge_repeats",
        help="Parallel edges per hidden node pair (default: 2, range 1-8). "
             "Each repeated edge gets independent logits/gate/multiplier. "
             "I/O and projection edges are NOT repeated. Composes "
             "multiplicatively with --bidirectional. Set to 1 for the "
             "previous single-edge behavior.",
    )
    parser.add_argument(
        "--bidirectional", dest="bidirectional", action="store_true", default=False,
        help="Emit two directed edges per unique node pair in the hidden graph "
             "(i->j AND j->i). Doubles the hidden edge count and gives "
             "asymmetric cells (P/rectifier) true bidirectional capability. "
             "Default: off (single edge per pair).",
    )
    parser.add_argument(
        "--no-bidirectional", dest="bidirectional", action="store_false",
        help="Disable dual edges per node pair (default).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("./output"),
        help="Output directory for artifacts (default: ./output)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=f"Number of epochs (default: {OPTIM['epochs']})",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help=f"Learning rate (default: {OPTIM['lr']})",
    )
    parser.add_argument(
        "--stage-lr-scale", type=float, default=1.0,
        help="Per-stage geometric LR multiplier (stage-lr-scaling). "
             "When 1.0 (default), all parameters share the same LR. "
             "When >1.0, stage i gets lr * scale^(S-1-i) where S is the "
             "number of stages. Compensates for vanishing gradients in "
             "deep ODE stacks (e.g. scale=10 with 3 stages: stage0=lr*100, "
             "stage1=lr*10, stage2=lr).",
    )
    parser.add_argument(
        "--retrain-stage-lr-scale", type=float, default=1.0,
        help="Per-stage LR scaling for retrain (default: 1.0). "
             "Warm-started pruned networks need gentle fine-tuning, so "
             "this defaults to uniform LR. Set to match --stage-lr-scale "
             "if you want geometric scaling during retrain.",
    )
    parser.add_argument(
        "--mapper-lr-scale", type=float, default=1.0,
        help="LR multiplier for I/O mapper params (input_mapper + output_mapper). "
             "Default 1.0 (no change). Use <1.0 (e.g. 0.1) to slow mapper learning "
             "when mapper gradient norms dominate core by ~300x. Composes with "
             "--stage-lr-scale.",
    )
    parser.add_argument(
        "--retrain-mapper-lr-scale", type=float, default=1.0,
        help="Mapper LR scale for the pruned-network retrain optimizer "
             "(default: 1.0). Mirrors --mapper-lr-scale if you want to slow "
             "mapper learning during retrain.",
    )
    parser.add_argument(
        "--freeze-mappers", dest="freeze_mappers", action="store_true", default=False,
        help="Freeze mapper requires_grad during the first half of the combined "
             "B1+B2 duration (four_phase schedule only). Mappers train normally "
             "in Phase A, freeze at B1 start, unfreeze at the midpoint. After "
             "unfreeze mappers resume at the --mapper-lr-scale rate. "
             "Ignored for three_phase and legacy schedules.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device 'cpu' or 'cuda' (default: auto-detect)",
    )
    parser.add_argument(
        "--amp", dest="amp", action="store_true", default=None,
        help="Enable mixed precision (AMP) via torch.cuda.amp (default: on when CUDA)",
    )
    parser.add_argument(
        "--no-amp", dest="amp", action="store_false",
        help="Disable mixed precision",
    )
    parser.add_argument(
        "--compile", dest="compile", action="store_true", default=None,
        help="Enable torch.compile on hot paths (default: on when CUDA)",
    )
    parser.add_argument(
        "--no-compile", dest="compile", action="store_false",
        help="Disable torch.compile",
    )
    parser.add_argument(
        "--parallel", dest="parallel", action="store_true", default=None,
        help="Enable DataParallel across multiple GPUs (default: on when ≥2 GPUs)",
    )
    parser.add_argument(
        "--no-parallel", dest="parallel", action="store_false",
        help="Disable DataParallel",
    )
    parser.add_argument(
        "--validate-every", type=int, default=5,
        help="Validate every N epochs (default: 5). Use 1 for every epoch.",
    )
    parser.add_argument(
        "--early-stop", dest="early_stop", action="store_true", default=True,
        help="Enable early stopping (default: on)",
    )
    parser.add_argument(
        "--no-early-stop", dest="early_stop", action="store_false",
        help="Disable early stopping",
    )
    parser.add_argument(
        "--patience", type=int, default=500,
        help="Early stopping patience in epochs (default: 500)",
    )
    parser.add_argument(
        "--min-delta", type=float, default=1e-4,
        help="Early stopping min improvement in val loss (default: 1e-4)",
    )
    parser.add_argument(
        "--amp-dtype", choices=["float16", "bfloat16"], default="float16",
        help="Autocast dtype (default: float16; bfloat16 needs Ampere+)",
    )
    parser.add_argument(
        "--ablation", choices=["none", "mapper-only", "empty-graph"], default="none",
        help="Structural ablation to apply (default: none). R2.",
    )
    parser.add_argument(
        "--variation", dest="variation", action="store_true", default=False,
        help="Enable PVT/mismatch injection during training (default: off, R6.3).",
    )
    parser.add_argument(
        "--write-mode", choices=["one_to_one", "dense", "fan_out"], default=None,
        help="Input write mapping (default: from preset). 'one_to_one' uses "
             "SparseInputMapper, 'dense' uses InputMapper (nn.Linear), "
             "'fan_out' uses FanOutInputMapper with preset-defined targets.",
    )
    parser.add_argument(
        "--read-mode", choices=["sparse", "dense"], default=None,
        help="Output read mapping (default: from preset, typically 'sparse'). "
             "'sparse' uses OutputMapper with preset-defined read_idx; "
             "'dense' uses full-projection readout.",
    )
    parser.add_argument(
        "--write-idx", type=str, default=None,
        help="Comma-separated hidden node indices for sparse input write "
             "(overrides preset write_idx). E.g. '0,2,4'.",
    )
    parser.add_argument(
        "--read-idx", type=str, default=None,
        help="Comma-separated full-state indices for sparse output read "
             "(overrides preset read_idx). E.g. '7'.",
    )
    parser.add_argument(
        "--prune", dest="prune", action="store_true", default=False,
        help="Run gate-based pruning after training (CP). Prunes edges and "
             "nodes below the configured thresholds, then either retrains "
             "the compact network or saves it as-is (see --no-retrain).",
    )
    parser.add_argument(
        "--retrain", dest="retrain", action="store_true", default=True,
        help="After pruning, retrain the compact network warm-started from "
             "the surviving pre-prune parameters (default: on). Use "
             "--no-retrain to skip retraining, or --fresh-init to retrain "
             "from random init instead.",
    )
    parser.add_argument(
        "--no-retrain", dest="retrain", action="store_false",
        help="Skip retraining after pruning; only transfer the surviving "
             "parameters from the overcomplete network into the compact one.",
    )
    parser.add_argument(
        "--prune-edge-threshold", type=float, default=None,
        help="Override config.PRUNE['edge_threshold'] for pruning.",
    )
    parser.add_argument(
        "--prune-node-threshold", type=float, default=None,
        help="Override config.PRUNE['node_threshold'] for pruning.",
    )
    parser.add_argument(
        "--prune-nodes-by-gate", dest="prune_nodes_by_gate",
        action="store_true", default=None,
        help="DEPRECATED (deprecate-node-gates): no-op, kept for backward "
             "compat. Node pruning is now connectivity-only regardless of "
             "this flag. Use --no-prune-nodes-by-gate to silence the warning.",
    )
    parser.add_argument(
        "--no-prune-nodes-by-gate", dest="prune_nodes_by_gate",
        action="store_false",
        help="DEPRECATED (deprecate-node-gates): no-op, kept for backward "
             "compat. Nodes are pruned by connectivity only.",
    )
    parser.add_argument(
        "--retrain-epochs", type=int, default=None,
        help="Number of epochs to retrain the compact network (default: "
             "the same value as --epochs, capped at half).",
    )
    parser.add_argument(
        "--retrain-lr", type=float, default=None,
        help="Learning rate for the retrain phase (default: same as --lr).",
    )
    parser.add_argument(
        "--fresh-init", dest="fresh_init", action="store_true", default=False,
        help="Re-initialize the pruned network from scratch (skip warm "
             "start from pre-prune parameters). Default: warm-start.",
    )
    parser.add_argument(
        "--no-scheduler", dest="use_scheduler", action="store_false", default=True,
        help="Disable LR scheduler (default: on).",
    )
    parser.add_argument(
        "--scheduler-type", choices=["cosine", "warm_restarts"], default="cosine",
        help="LR scheduler type when --scheduler is enabled (default: 'cosine' — "
             "plain cosine decay over total epochs, no restarts). 'warm_restarts' "
             "uses CosineAnnealingWarmRestarts (legacy behavior).",
    )
    parser.add_argument(
        "--grad-log", dest="grad_log", action="store_true", default=False,
        help="Periodically log per-parameter-group gradient L2 norms to "
             "grad_norms.txt (default: off).",
    )
    parser.add_argument(
        "--grad-log-every", type=int, default=10,
        help="Log gradient norms every N epochs (default: 10). Only used "
             "when --grad-log is enabled.",
    )
    parser.add_argument(
        "--schedule", choices=["legacy", "three_phase", "four_phase"], default=None,
        help="Training schedule mode (default: from preset['schedule'], "
             "fallback 'legacy'). 'three_phase' implements the phased "
             "fit-compress-prune pipeline (Phase A: fit with no structure "
             "pressure, Phase B: compress via gate penalties, Phase C: "
             "auto-prune + retrain). 'four_phase' adds a cell-commitment "
             "Phase B1 (no pruning), readiness-gated Phase B2 (edge "
             "pruning), and a KD-anchored retrain Phase C. See "
             "spec/four-phase-schedule.md.",
    )
    parser.add_argument(
        "--no-argmax-val", dest="argmax_val", action="store_false", default=True,
        help="Disable argmax-vs-soft validation diagnostic (default: on "
             "when --schedule three_phase is active).",
    )
    parser.add_argument(
        "--ablation-set",
        choices=["none", "reg-only", "tau-only", "edge-only"], default="none",
        help="Diagnostic ablation preset (four-phase-redesign/Phase 1c). "
             "Each preset overrides the relevant combination of "
             "tau/regularizer/pruning flags. 'reg-only' keeps tau=1.0 "
             "through B and turns on gate regularizers without pruning. "
             "'tau-only' anneals tau with all structural regularizers off "
             "and no pruning. 'edge-only' is the normal B path but with "
             "node-gate pruning disabled and a lower edge threshold "
             "(matches the four-phase-redesign defaults). 'none' is the "
             "standard schedule behavior with no overrides.",
    )
    parser.add_argument(
        "--cell-mode", choices=["soft", "ste", "auto"], default="auto",
        help="Cell selection mode (four-phase-redesign/Phase 2b). 'soft' "
             "uses a softmax-weighted mixture of cells per edge. 'ste' "
             "uses one cell per edge in the forward pass (argmax) with "
             "straight-through soft gradients in the backward pass. "
             "'auto' uses 'soft' for Phase A and 'ste' for B/C (only "
             "meaningful with --schedule three_phase / four_phase).",
    )


# ----------------------------------------------------------------
# Pruning helpers (PIT): transferable I/O mapper reconstruction.
# ----------------------------------------------------------------

def _remap_indices(idx_list, remap):
    """Remap a list of compact node ids through a node_remap dict.

    Pruned indices (not present in ``remap``) are silently dropped. This
    is safe because:
      - write targets are protected from pruning (see ``protected_nodes``
        in ``prune_stage``), so write_idx entries always survive.
      - read targets are allowed to be pruned (elastic readout), and the
        surviving entries are remapped; pruned ones are dropped from the
        read_idx list, which the caller handles by rebuilding the
        OutputMapper with fewer input features.

    Returns a list of remapped indices (may be shorter than ``idx_list``).
    """
    out = []
    for i in idx_list:
        if i not in remap:
            continue
        out.append(remap[i])
    return out


def _transfer_input_mapper(raw_mapper, raw_write_idx, stage0_remap,
                            pruned_first_n, in_dim):
    """Return a new InputMapper for the pruned stage with weights
    transferred from ``raw_mapper``.

    For DenseMapper/RobustInputMapper, rows are selectively copied
    for surviving nodes when stage 0's node count changed.

    For SparseInputMapper, ``raw_write_idx`` is remapped through
    ``stage0_remap`` and the per-feature (gain, bias) parameters
    are copied directly (they are indexed by input, not by node).

    For FanOutInputMapper, each target list in ``fan_out_map`` is
    remapped through ``stage0_remap``; per-(input, target)
    (gain, bias) parameters are copied directly.
    """
    if isinstance(raw_mapper, SparseInputMapper):
        if raw_write_idx is None:
            new_write_idx = list(range(min(in_dim, pruned_first_n)))
        else:
            new_write_idx = _remap_indices(raw_write_idx, stage0_remap)
        new_mapper = SparseInputMapper(
            in_dim=in_dim, out_dim=pruned_first_n, write_idx=new_write_idx,
        )
        with torch.no_grad():
            new_mapper.gain.data.copy_(raw_mapper.gain.data)
            new_mapper.bias.data.copy_(raw_mapper.bias.data)
        return new_mapper, new_write_idx

    if isinstance(raw_mapper, FanOutInputMapper):
        new_fan_out = {}
        for inp, targets in raw_mapper.fan_out_map.items():
            new_targets = _remap_indices(targets, stage0_remap)
            new_fan_out[inp] = new_targets
        new_mapper = FanOutInputMapper(
            in_dim=raw_mapper.in_dim,
            out_dim=pruned_first_n,
            fan_out_map=new_fan_out,
            x_max=raw_mapper.x_max,
        )
        with torch.no_grad():
            new_mapper.gain.data.copy_(raw_mapper.gain.data)
            new_mapper.bias.data.copy_(raw_mapper.bias.data)
        return new_mapper, None

    if isinstance(raw_mapper, (InputMapper, RobustInputMapper)):
        old_out = raw_mapper.gain.out_features
        surviving_old = sorted(stage0_remap.keys())
        surviving_new = [stage0_remap[o] for o in surviving_old]
        is_identity = (old_out == pruned_first_n and
                       surviving_new == list(range(pruned_first_n)))
        if is_identity:
            return copy.deepcopy(raw_mapper), raw_write_idx
        if isinstance(raw_mapper, RobustInputMapper):
            new_mapper = RobustInputMapper(
                in_dim=in_dim, out_dim=pruned_first_n, x_max=raw_mapper.x_max,
            )
        else:
            new_mapper = InputMapper(
                in_dim=in_dim, out_dim=pruned_first_n, x_max=raw_mapper.x_max,
            )
        with torch.no_grad():
            for old_id, new_id in zip(surviving_old, surviving_new):
                if old_id < old_out:
                    new_mapper.gain.weight.data[new_id].copy_(
                        raw_mapper.gain.weight.data[old_id]
                    )
                    new_mapper.gain.bias.data[new_id].copy_(
                        raw_mapper.gain.bias.data[old_id]
                    )
                else:
                    new_mapper.gain.weight.data[new_id].zero_()
                    new_mapper.gain.bias.data[new_id].zero_()
            if isinstance(raw_mapper, RobustInputMapper):
                new_mapper.log_scale.data.copy_(raw_mapper.log_scale.data)
        return new_mapper, raw_write_idx

    raise TypeError(
        f"transfer_input_mapper: unsupported mapper type {type(raw_mapper).__name__}"
    )


def _transfer_output_mapper(raw_mapper, raw_read_idx, last_remap,
                            pruned_last_n, out_dim):
    """Return a new OutputMapper for the pruned stage with weights
    transferred from ``raw_mapper``.

    For sparse read mode, ``raw_read_idx`` is remapped through
    ``last_remap``. If all read nodes survive, position ``i`` in the new
    proj corresponds to old position ``i`` and the proj weight columns
    can be copied directly. If some read nodes were pruned (elastic
    readout), the new mapper has fewer input features; we copy only the
    columns that correspond to surviving read positions, preserving the
    learned readout for those nodes.

    For dense read mode (no read_idx), if last-stage node count
    unchanged, deepcopy. Otherwise, copy ``proj.weight`` columns
    for surviving nodes.
    """
    if raw_read_idx is not None:
        new_read_idx = _remap_indices(raw_read_idx, last_remap)
        new_mapper = OutputMapper(
            node_dim=pruned_last_n, out_dim=out_dim, read_idx=new_read_idx,
        )
        # Determine which columns of the old weight matrix correspond to
        # surviving read positions (those entries not pruned away).
        surviving_old_positions = [
            i for i, idx in enumerate(raw_read_idx) if idx in last_remap
        ]
        with torch.no_grad():
            new_mapper.proj.weight.data.copy_(
                raw_mapper.proj.weight.data[:, surviving_old_positions]
            )
            new_mapper.proj.bias.data.copy_(raw_mapper.proj.bias.data)
        return new_mapper, new_read_idx

    old_dim = raw_mapper.proj.in_features
    if old_dim == pruned_last_n:
        return copy.deepcopy(raw_mapper), None
    surviving_old = sorted(last_remap.keys())
    surviving_new = [last_remap[o] for o in surviving_old]
    new_mapper = OutputMapper(node_dim=pruned_last_n, out_dim=out_dim)
    with torch.no_grad():
        new_mapper.proj.weight.data[:, surviving_new].copy_(
            raw_mapper.proj.weight.data[:, surviving_old]
        )
        new_mapper.proj.bias.data.copy_(raw_mapper.proj.bias.data)
    return new_mapper, None


def main():
    parser = argparse.ArgumentParser(
        description="Train a reduced differential KirchhoffNet."
    )
    _add_argparse_args(parser)
    args = parser.parse_args()

    amp_enabled = args.amp if args.amp is not None else torch.cuda.is_available()
    compile_enabled = args.compile if args.compile is not None else torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    parallel_enabled = (
        args.parallel if args.parallel is not None else (n_gpus >= 2)
    )

    epochs = args.epochs if args.epochs is not None else OPTIM["epochs"]
    lr = args.lr if args.lr is not None else OPTIM["lr"]
    batch_size = int(OPTIM["batch_size"])

    out_dir = _ensure_dir(args.output.resolve())
    lambdas = _resolve_lambdas(args.problem)
    schedule_mode = _resolve_schedule(args.problem, args.schedule)
    # four-phase-redesign/Phase 1c: apply diagnostic ablation preset
    # overrides BEFORE any other flag is consumed (e.g. before the
    # pruning-threshold resolution below).
    _apply_ablation_set(args, schedule_mode)
    # cluster-topology CLI: validate new --hidden-family/--num-hidden/
    # --edge-repeats/--num-stages combinations early (cheap, before any
    # expensive setup like data loading or model construction).
    _validate_hidden_family_args(args)
    if schedule_mode == "three_phase":
        a_end, b_end, c_end = phase_boundaries(epochs)
        print(
            f"[train] three_phase schedule: A=[0,{a_end}) B=[{a_end},{b_end}) "
            f"C=[{b_end},{c_end}) (total={epochs})"
        )
    elif schedule_mode == "four_phase":
        fp_a_end, fp_b1_end, fp_b2_end, fp_c_end = four_phase_boundaries(epochs)
        print(
            f"[train] four_phase schedule: A=[0,{fp_a_end}) "
            f"B1=[{fp_a_end},{fp_b1_end}) "
            f"B2=[{fp_b1_end},{fp_b2_end}) "
            f"C=[{fp_b2_end},{fp_c_end}) (total={epochs})"
        )
        if args.freeze_mappers:
            mapper_unfreeze_epoch = fp_a_end + (fp_b2_end - fp_a_end) // 2
            print(
                f"[train] four_phase freeze_mappers: freeze at epoch {fp_a_end}, "
                f"unfreeze at epoch {mapper_unfreeze_epoch}"
            )
        else:
            mapper_unfreeze_epoch = -1
    else:
        mapper_unfreeze_epoch = -1
    if args.freeze_mappers and schedule_mode != "four_phase":
        print(
            f"[train] WARNING: --freeze-mappers ignored for schedule_mode={schedule_mode} "
            f"(only supported for four_phase)"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device is not None:
        device = args.device
    write_idx_arg = _parse_int_list(args.write_idx)
    read_idx_arg = _parse_int_list(args.read_idx)

    # Resolve cell library: CLI overrides preset, preset overrides default.
    lib_name = "legacy"
    if args.problem in PRESETS:
        lib_name = PRESETS[args.problem].get("cell_library", "legacy")
    if args.cell_library is not None:
        lib_name = args.cell_library
    cell_lib = make_cell_library(lib_name)
    # Per-problem grid-size default (grid7-gate0: smooth2d_grid=7, housing_grid=5).
    # Explicit --grid-size N overrides either.
    if args.problem == "smooth2d_grid":
        resolved_grid_size = args.grid_size if args.grid_size is not None else 7
        PRESETS["smooth2d_grid"] = make_smooth2d_grid_preset(
            grid_size=resolved_grid_size,
            bidirectional=args.bidirectional,
        )
    elif args.problem == "housing_grid":
        resolved_grid_size = args.grid_size if args.grid_size is not None else 5
        PRESETS["housing_grid"] = make_housing_grid_preset(
            grid_size=resolved_grid_size,
            bidirectional=args.bidirectional,
        )
    else:
        resolved_grid_size = args.grid_size
    args.grid_size = resolved_grid_size
    # cluster-topology CLI: when --hidden-family is specified, dynamically
    # rebuild the problem's preset to use the requested family. Done AFTER
    # grid-size resolution so the cluster-family branch can read
    # args.grid_size (for error detection).
    if args.hidden_family is not None:
        eff_num_hidden = (
            args.num_hidden if args.num_hidden is not None
            else (resolved_grid_size ** 2 if resolved_grid_size else 5)
        )
        eff_num_stages = args.num_stages if args.num_stages is not None else 1
        eff_edge_repeats = args.edge_repeats if args.edge_repeats is not None else 2
        new_preset = _make_dynamic_preset(
            problem=args.problem,
            hidden_family=args.hidden_family,
            num_hidden=eff_num_hidden,
            num_stages=eff_num_stages,
            edge_repeats=eff_edge_repeats,
            grid_size=resolved_grid_size,
            bidirectional=args.bidirectional,
            write_mode_override=args.write_mode,
            read_mode_override=args.read_mode,
        )
        PRESETS[args.problem] = new_preset
        print(
            f"[train] dynamic preset (hidden_family={args.hidden_family}): "
            f"num_hidden={eff_num_hidden} num_stages={eff_num_stages} "
            f"edge_repeats={eff_edge_repeats} "
            f"bidirectional={args.bidirectional} "
            f"write_mode={new_preset['write_mode']} read_mode={new_preset['read_mode']}"
        )
    net = build_net_from_preset(
        args.problem,
        cell_lib=cell_lib,
        write_mode=args.write_mode,
        read_mode=args.read_mode,
        write_idx=write_idx_arg,
        read_idx=read_idx_arg,
    )
    net.to(device)
    grid_label = (
        f" {args.grid_size}×{args.grid_size} grid,"
        if args.problem in ("smooth2d_grid", "housing_grid")
        else ""
    )
    in_mapper_name = type(net.input_mapper).__name__
    out_mapper_name = type(net.output_mapper).__name__
    print(
        f"[train] problem={args.problem}{grid_label} epochs={epochs} lr={lr} device={device} "
        f"output={out_dir} amp={amp_enabled} compile={compile_enabled} "
        f"parallel={parallel_enabled} ({n_gpus} GPUs) "
        f"validate_every={args.validate_every} early_stop={args.early_stop} "
        f"ablation={args.ablation} variation={args.variation} "
        f"input_mapper={in_mapper_name} output_mapper={out_mapper_name} "
        f"hid_count={net.hid_count} proj_count={net.proj_count} "
        f"write_idx={list(net.write_idx) if net.write_idx is not None else None} "
        f"read_idx={list(net.read_idx) if net.read_idx is not None else None}"
    )
    if args.bidirectional or (args.edge_repeats is not None and args.edge_repeats > 1):
        eff_er = args.edge_repeats if args.edge_repeats is not None else 2
        mult = 2 if args.bidirectional else 1
        mult *= eff_er
        if args.bidirectional:
            edges_per_stage = [s.num_edges() for s in net.core.stages]
            print(
                f"[train] bidirectional={args.bidirectional} "
                f"edge_repeats={eff_er}: {edges_per_stage} edges per stage "
                f"({mult}× single-edge baseline)"
            )
        else:
            print(
                f"[train] edge_repeats={eff_er}: ({mult}× single-edge baseline)"
            )
    _save_config_snapshot(out_dir, args.problem, args, lambdas, net=net)

    effective_write_mode = (
        "fan_out" if isinstance(net.input_mapper, FanOutInputMapper)
        else "one_to_one" if isinstance(net.input_mapper, SparseInputMapper)
        else "dense"
    )
    effective_read_mode = "sparse" if net.read_idx is not None else "dense"

    data_out = make_data(args.problem, batch_size)
    if len(data_out) == 4:
        train_loader, val_loader, task_fn, inverse_stats = data_out
    else:
        train_loader, val_loader, task_fn = data_out
        inverse_stats = None

    if args.ablation != "none":
        apply_ablation(net, args.ablation)

    if parallel_enabled and n_gpus >= 2:
        compile_enabled = False
        print("[train] disabling compile for DataParallel compatibility (use single GPU for compile)")

    if compile_enabled and isinstance(device, str) and device.startswith("cuda"):
        try:
            cell_lib.compile_forward()
            for stage in net.core.stages:
                if stage.num_edges() > 0:
                    stage.compile_rhs()
            print("[train] torch.compile enabled on cell_lib.forward and stage.rhs")
        except Exception as e:
            print(f"[train] torch.compile setup failed: {e}; continuing without compile")
            compile_enabled = False

    if parallel_enabled and n_gpus >= 2:
        try:
            net = torch.nn.DataParallel(net, device_ids=list(range(n_gpus)))
            print(f"[train] DataParallel enabled on {n_gpus} GPUs")
        except Exception as e:
            print(f"[train] DataParallel setup failed: {e}; continuing single-GPU")

    def _unwrap(m):
        return m.module if isinstance(m, torch.nn.DataParallel) else m

    raw_net = _unwrap(net)

    if args.variation:
        def ctx_factory(batch_size_: int, device: torch.device = device, **_):
            total_edges = sum(s.num_edges() for s in raw_net.core.stages)
            return sample_random_context(
                num_edges=total_edges,
                num_cells=raw_net.core.stages[0].cell_lib.num_cells,
                device=device,
                gain_shift_std=VARIATION["global_gain_shift_std"],
                mismatch_std=VARIATION["edge_mismatch_std"],
            )
    else:
        ctx_factory = make_static_ctx_factory()

    optimizer = make_optimizer(
        net, lr=lr,
        stage_lr_scale=args.stage_lr_scale,
        mapper_lr_scale=args.mapper_lr_scale,
    )
    if args.stage_lr_scale != 1.0 or args.mapper_lr_scale != 1.0:
        lr_strs = [f"{g['lr']:.1e}" for g in optimizer.param_groups]
        print(
            f"[train] stage_lr_scale={args.stage_lr_scale}, "
            f"mapper_lr_scale={args.mapper_lr_scale}: per-group LRs = {lr_strs}"
        )
    if args.use_scheduler:
        if schedule_mode == "three_phase":
            _, _ab_end, _ = phase_boundaries(epochs)
            sched_tmax = max(1, _ab_end)
        else:
            sched_tmax = max(1, epochs)
        if args.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=sched_tmax,
                eta_min=OPTIM["scheduler_eta_min"],
            )
        else:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=OPTIM["scheduler_T_0"],
                T_mult=OPTIM["scheduler_T_mult"],
                eta_min=OPTIM["scheduler_eta_min"],
            )
    else:
        scheduler = None

    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    scaler = (
        torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    )

    history = []
    val_history = []

    for i, stage in enumerate(raw_net.core.stages):
        plot_stage_graph(
            stage, save_path=str(out_dir / f"stage{i + 1}_graph_init.png"),
            title=f"{args.problem} — Stage {i + 1} (init)",
        )

    print("[train] starting training loop")
    try:
        from tqdm import tqdm
    except ImportError:
        warnings.warn("tqdm not installed, falling back to plain prints", stacklevel=2)
        tqdm = None

    grad_log_path = out_dir / "grad_norms.txt" if args.grad_log else None
    solid_log_path = out_dir / "solidification_metrics.txt" if schedule_mode in ("three_phase", "four_phase") else None

    # ---------- Determine effective training scope ----------
    # Three-phase: Phase A+B use the overcomplete network (epochs 0..b_end).
    # Four-phase: Phase A+B1+B2 use the overcomplete network (epochs 0..b2_end).
    # Legacy: single phase over all epochs.
    if schedule_mode == "three_phase":
        a_end, b_end, _ = phase_boundaries(epochs)
        ab_total = b_end  # Phase A+B epoch count
        c_total = max(1, epochs - b_end)
        needs_prune = True  # three_phase always prunes at the B→C boundary
    elif schedule_mode == "four_phase":
        fp_a_end, fp_b1_end, fp_b2_end, _ = four_phase_boundaries(epochs)
        a_end = fp_a_end
        b1_end = fp_b1_end
        b2_end = fp_b2_end
        ab_total = b2_end  # Phase A+B1+B2 epoch count
        c_total = max(1, epochs - b2_end)
        needs_prune = False  # four_phase handles pruning inline during B2
    else:
        ab_total = epochs
        c_total = 0
        needs_prune = args.prune

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    best_metric_name = "val"  # four-phase-redesign/Phase 1b: track which metric
    epochs_without_improve = 0
    stop_training = False

    # four-phase-redesign: teacher distillation state
    teacher_net = None
    teacher_frozen = False
    # Readiness tracking for four_phase B2
    readiness_histories: dict = {}
    readiness_prune_fired = False
    readiness_prune_epoch = -1

    # ---- Phase A + B: overcomplete training ----
    tau_anneal_enabled = PRESETS[args.problem].get("tau_anneal", True)
    if schedule_mode == "legacy" and tau_anneal_enabled:
        tau_kwargs = {}
        if args.prune:
            tau_kwargs["tau_final"] = float(TAU.get("final_pretrain", TAU["final"]))
    if schedule_mode == "three_phase":
        ab_desc = f"train {args.problem} [A+B]"
    elif schedule_mode == "four_phase":
        ab_desc = f"train {args.problem} [A+B1+B2]"
    else:
        ab_desc = f"train {args.problem} [ablation={args.ablation}]"
    ab_iter = (
        tqdm(range(ab_total), desc=ab_desc, unit="epoch")
        if tqdm is not None else range(ab_total)
    )
    # Track argmax validation alongside soft (when enabled).
    argmax_val_enabled = (
        (schedule_mode in ("three_phase", "four_phase")) and args.argmax_val
    )
    val_argmax_history = [] if argmax_val_enabled else None
    # four-phase-redesign: solidification metrics stored for readiness check
    solid_metrics_history: list[dict] = []
    # Validate-only histories (matching cadence of solid_metrics_history)
    # Avoids duplicate entries on non-validate epochs for the readiness check.
    val_v_history: list[float] = []
    val_argmax_v_history: list[float] = [] if argmax_val_enabled else None
    # housing_grid: per-epoch dicts of {val, mae_orig, rmse_orig} for
    # logging in original housing-price units (USD x 100k).
    val_orig_history: list[dict] = [] if inverse_stats is not None else None
    # mapper-lr-control: freeze state for --freeze-mappers (four_phase only)
    mappers_frozen = False

    for epoch in ab_iter:
        if stop_training:
            break
        net.train()

        if schedule_mode == "four_phase":
            tau = four_phase_tau(epoch, epochs)
            phase = phase_for_epoch_four(epoch, epochs)
            effective_lambdas = four_phase_lambdas(epoch, epochs, lambdas)
            reg_scale = 1.0  # four_phase_lambdas already includes warmup

            # Teacher cloning at phase transition A -> B1.
            if phase in ("B1", "B2") and not teacher_frozen and best_state is not None:
                # Deep-copy the architecture and load the best Phase A
                # checkpoint into the frozen teacher.
                import copy as _copy
                raw = net.module if isinstance(net, torch.nn.DataParallel) else net
                teacher_net = _copy.deepcopy(raw)
                teacher_net.load_state_dict(best_state)
                teacher_net.requires_grad_(False)
                teacher_net.eval()
                teacher_net.to(device)
                teacher_frozen = True
                print(
                    f"[teacher] cloned best Phase A state (epoch {best_epoch}, "
                    f"{best_metric_name}={best_val:.4f}) as frozen teacher"
                )

            # mapper-lr-control: freeze mappers at A->B1 boundary, unfreeze
            # at the midpoint of the combined B1+B2 duration. Only when
            # --freeze-mappers is set and schedule_mode is four_phase.
            if args.freeze_mappers:
                raw = net.module if isinstance(net, torch.nn.DataParallel) else net
                if epoch == fp_a_end and not mappers_frozen:
                    raw.input_mapper.requires_grad_(False)
                    raw.output_mapper.requires_grad_(False)
                    mappers_frozen = True
                    print(
                        f"[mapper] frozen mappers at epoch {epoch} "
                        f"(unfreeze at {mapper_unfreeze_epoch})"
                    )
                elif epoch == mapper_unfreeze_epoch and mappers_frozen:
                    raw.input_mapper.requires_grad_(True)
                    raw.output_mapper.requires_grad_(True)
                    mappers_frozen = False
                    print(
                        f"[mapper] unfrozen mappers at epoch {epoch} "
                        f"(lr_scale={args.mapper_lr_scale})"
                    )

            # If readiness triggered on a previous validate step, stop here.
            if phase == "B2" and readiness_prune_fired:
                print(
                    f"[four_phase] readiness pruning triggered at epoch {epoch} "
                    f"(B2 cut short, was scheduled for {b2_end})"
                )
                break

        elif schedule_mode == "three_phase":
            tau = three_phase_tau(epoch, epochs)
            phase = phase_for_epoch(epoch, epochs)
            effective_lambdas = three_phase_lambdas(epoch, epochs, lambdas)
            reg_scale = 1.0  # three_phase_lambdas already includes warmup
        else:
            phase = ""
            if tau_anneal_enabled:
                tau = tau_for_epoch(epoch, total_epochs=epochs, **tau_kwargs)
            else:
                tau = float(TAU["init"])
            reg_scale = reg_schedule(epoch)
            effective_lambdas = lambdas

        # four-phase-redesign/Phase 2b: per-epoch cell selection mode.
        # 'auto' uses 'ste' for Phase B/C of phased schedules.
        cell_mode = _resolve_cell_mode(args.cell_mode, phase, schedule_mode)

        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            ctx = ctx_factory(batch[0].size(0), device=device)
            optimizer.zero_grad()
            u, target = batch
            u = u.to(device)
            target = target.to(device)
            # four-phase-redesign/Phase 3c: pass teacher for KD when active.
            kd_teacher = None
            kd_lambda = 0.0
            if schedule_mode == "four_phase" and teacher_frozen and four_phase_kd_active(epoch, epochs):
                kd_teacher = teacher_net
                kd_lambda = float(SCHEDULE_FOUR_PHASE.get("lambda_kd", 1.0))
            loss_task, loss_structural, _ = compute_loss(
                net, u, target, ctx, task_fn,
                lambdas=effective_lambdas, tau=tau, return_parts=True,
                amp=amp_enabled, amp_dtype=amp_dtype, reg_scale=reg_scale,
                cell_mode=cell_mode,
                teacher=kd_teacher, lambda_kd=kd_lambda, teacher_tau=1.0,
                teacher_cell_mode="soft",
            )
            if scaler is not None and scaler._enabled:
                ( scaler.scale(loss_task) + scaler.scale(loss_structural) ).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=OPTIM["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                (loss_task + loss_structural).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=OPTIM["grad_clip_norm"])
                optimizer.step()
            total_loss += float((loss_task + loss_structural).item())
            n_batches += 1

        avg_train = total_loss / max(1, n_batches)
        do_validate = (epoch % args.validate_every == 0) or (epoch == ab_total - 1)
        if do_validate:
            if inverse_stats is not None:
                val_metrics = validate_with_inverse(
                    net, val_loader, task_fn, ctx_factory, device,
                    inverse_stats=inverse_stats, cell_mode=cell_mode,
                )
                val_loss = val_metrics["val"]
            else:
                val_loss = validate(net, val_loader, task_fn, ctx_factory, device, cell_mode=cell_mode)
                val_metrics = None
            val_v_history.append(val_loss)
            if inverse_stats is not None and val_metrics is not None:
                val_orig_history.append(val_metrics)
            # Argmax validation for phased schedules
            if val_argmax_history is not None:
                val_arg = validate_argmax(net, val_loader, task_fn, ctx_factory, device)
                val_argmax_history.append(val_arg)
                val_argmax_v_history.append(val_arg)
            if solid_log_path is not None and phase in ("A", "B", "B1", "B2"):
                metrics = compute_solidification_metrics(net, tau=tau)
                _log_solidification(solid_log_path, epoch, metrics)
                # Store for four_phase readiness check.
                if schedule_mode == "four_phase":
                    solid_metrics_history.append(metrics)
            # four-phase-redesign/Phase 3d: readiness check during B2.
            # Uses validate-only histories (val_v_history, val_argmax_v_history)
            # to avoid duplicate entries on non-validate epochs.
            if (
                schedule_mode == "four_phase"
                and phase == "B2"
                and val_argmax_v_history is not None
                and len(val_argmax_v_history) >= 10
                and len(val_v_history) >= 10
                and len(solid_metrics_history) >= 10
            ):
                is_ready, ready_details = prune_readiness_check(
                    val_v_history, val_argmax_v_history, solid_metrics_history,
                )
                # Also log readiness diagnostics.
                if solid_log_path is not None:
                    _log_solidification(
                        solid_log_path, epoch,
                        {"ready_ratio": ready_details.get("ratio", -1.0),
                         "ready_prob": ready_details.get("max_cell_prob", -1.0),
                         "ready_stability": ready_details.get("stability", -1.0),
                         "ready_improvement": ready_details.get("improvement_rate", -1.0),
                         "all_ready": 1.0 if ready_details.get("all_ready", False) else 0.0},
                    )
                if is_ready and not readiness_prune_fired:
                    readiness_prune_fired = True
                    readiness_prune_epoch = epoch
                    print(
                        f"[four_phase] READINESS TRIGGERED at epoch {epoch}: "
                        f"ratio={ready_details['ratio']:.3f}, "
                        f"prob={ready_details['max_cell_prob']:.3f}, "
                        f"stability={ready_details['stability']:.4f}, "
                        f"improvement={ready_details['improvement_rate']:.6f}"
                    )
        else:
            val_loss = val_history[-1] if val_history else avg_train
            if val_argmax_history is not None:
                val_argmax_history.append(val_argmax_history[-1] if val_argmax_history else val_loss)

        history.append(avg_train)
        val_history.append(val_loss)

        if do_validate:
            # four-phase-redesign/Phase 1b: For three_phase mode in Phase B,
            # and for four_phase mode in B1/B2, the deployable model is the
            # hard-cell (argmax) version. Use val_argmax as checkpoint metric
            # instead of soft val, which includes cell mixture that vanishes
            # at deployment. Phase A still uses soft val.
            use_argmax_ckpt = (
                (schedule_mode == "three_phase" and phase == "B")
                or (schedule_mode == "four_phase" and phase in ("B1", "B2"))
            ) and val_argmax_history is not None and len(val_argmax_history) > 0
            if use_argmax_ckpt:
                sel_metric = float(val_argmax_history[-1])
                sel_name = "val_argmax"
            else:
                sel_metric = float(val_loss)
                sel_name = "val"
            if sel_metric < best_val - args.min_delta:
                best_val = sel_metric
                best_epoch = epoch
                best_metric_name = sel_name
                epochs_without_improve = 0
                raw = net.module if isinstance(net, torch.nn.DataParallel) else net
                best_state = {k: v.detach().clone() for k, v in raw.state_dict().items()}
            else:
                epochs_without_improve += args.validate_every
                if args.early_stop and epochs_without_improve >= args.patience:
                    print(
                        f"[train] early stopping at epoch {epoch}: "
                        f"no {best_metric_name} improvement for {epochs_without_improve} epochs "
                        f"(best {best_metric_name}={best_val:.4f} @ epoch {best_epoch})"
                    )
                    stop_training = True

        if scheduler is not None:
            scheduler.step()

        if grad_log_path is not None and epoch % args.grad_log_every == 0:
            raw = net.module if isinstance(net, torch.nn.DataParallel) else net
            log_gradient_norms(grad_log_path, epoch, raw, optimizer=optimizer)

        _lrs = [g["lr"] for g in optimizer.param_groups]
        lr_str = f"{min(_lrs):.1e}..{max(_lrs):.1e}" if len(_lrs) > 1 else f"{_lrs[0]:.2e}"
        phase_tag = f" [{phase}]" if phase else ""
        if tqdm is not None:
            ab_iter.set_description_str(f"{ab_desc}  train={avg_train:.4f}  val={val_loss:.4f}")
            postfix_dict = dict(tau=f"{tau:.3f}", lr=lr_str)
            if schedule_mode == "legacy":
                postfix_dict["reg"] = f"{reg_scale:.2f}"
            ab_iter.set_postfix(**postfix_dict)
        else:
            print(
                f"  epoch {epoch:4d}{phase_tag}  train={avg_train:.4f}  val={val_loss:.4f}  "
                f"tau={tau:.3f}  lr={lr_str}"
            )

    # ---- End of Phase A+B (or A+B1+B2) ----
    if schedule_mode == "three_phase" and not stop_training:
        a_end, b_end, _ = phase_boundaries(epochs)
        print(
            f"[phase] A+B complete (epoch {b_end}), best {best_metric_name}={best_val:.4f} "
            f"@ epoch {best_epoch}"
        )
    elif schedule_mode == "four_phase":
        if stop_training and not teacher_frozen:
            # Early stopped in Phase A — no pruning, just save.
            print(
                f"[four_phase] early stop in Phase A at epoch {epoch}, "
                f"skipping prune+retrain"
            )
        else:
            if readiness_prune_fired:
                print(
                    f"[four_phase] A+B1+B2 complete, prune triggered at epoch "
                    f"{readiness_prune_epoch}, best {best_metric_name}={best_val:.4f} "
                    f"@ epoch {best_epoch}"
                )
            elif stop_training and teacher_frozen:
                # Early stop during B1/B2 — still prune since compression
                # has already begun.
                print(
                    f"[four_phase] early stop at epoch {epoch} during "
                    f"B1/B2, pruning anyway"
                )
            else:
                print(
                    f"[four_phase] A+B1+B2 complete (fallback prune at epoch {b2_end}), "
                    f"best {best_metric_name}={best_val:.4f} @ epoch {best_epoch}"
                )
                readiness_prune_fired = True
            needs_prune = True

    history_path = out_dir / "loss_history.txt"
    with open(history_path, "w") as f:
        has_argmax = val_argmax_history is not None and len(val_argmax_history) == len(val_history)
        has_orig = val_orig_history is not None and len(val_orig_history) == len(val_history)
        if has_argmax and has_orig:
            f.write("epoch\ttrain\tval\tval_argmax\tmae_orig\trmse_orig\tphase\n")
            for i, (t, v, va, m) in enumerate(zip(history, val_history, val_argmax_history, val_orig_history)):
                if schedule_mode == "three_phase":
                    p = phase_for_epoch(i, epochs)
                elif schedule_mode == "four_phase":
                    p = phase_for_epoch_four(i, epochs)
                else:
                    p = "A"
                f.write(f"{i}\t{t}\t{v}\t{va}\t{m['mae_orig']:.6f}\t{m['rmse_orig']:.6f}\t{p}\n")
        elif has_argmax:
            f.write("epoch\ttrain\tval\tval_argmax\tphase\n")
            for i, (t, v, va) in enumerate(zip(history, val_history, val_argmax_history)):
                if schedule_mode == "three_phase":
                    p = phase_for_epoch(i, epochs)
                elif schedule_mode == "four_phase":
                    p = phase_for_epoch_four(i, epochs)
                else:
                    p = "A"
                f.write(f"{i}\t{t}\t{v}\t{va}\t{p}\n")
        elif has_orig:
            f.write("epoch\ttrain\tval\tmae_orig\trmse_orig\tphase\n")
            for i, (t, v, m) in enumerate(zip(history, val_history, val_orig_history)):
                if schedule_mode == "three_phase":
                    p = phase_for_epoch(i, epochs)
                elif schedule_mode == "four_phase":
                    p = phase_for_epoch_four(i, epochs)
                else:
                    p = "A"
                f.write(f"{i}\t{t}\t{v}\t{m['mae_orig']:.6f}\t{m['rmse_orig']:.6f}\t{p}\n")
        else:
            f.write("epoch\ttrain\tval\tphase\n")
            for i, (t, v) in enumerate(zip(history, val_history)):
                if schedule_mode == "three_phase":
                    p = phase_for_epoch(i, epochs)
                elif schedule_mode == "four_phase":
                    p = phase_for_epoch_four(i, epochs)
                else:
                    p = "A"
                f.write(f"{i}\t{t}\t{v}\t{p}\n")

    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history, label="train", color="C0")
    ax.plot(val_history, label="val", color="C3")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(
        f"{args.problem} — training loss [ablation={args.ablation}, variation={args.variation}]"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    if best_state is not None:
        raw = net.module if isinstance(net, torch.nn.DataParallel) else net
        raw.load_state_dict(best_state)
        print(
            f"[train] restored best model from epoch {best_epoch} "
            f"(val={best_val:.4f})"
        )
    torch.save(raw_net.state_dict(), out_dir / "model.pt")
    print(f"[train] saved model to {out_dir / 'model.pt'}")

    for i, stage in enumerate(raw_net.core.stages):
        plot_stage_graph(
            stage, save_path=str(out_dir / f"stage{i + 1}_graph_trained.png"),
            title=f"{args.problem} — Stage {i + 1} (trained)",
        )

    for i, stage in enumerate(raw_net.core.stages):
        plot_cell_selection(
            stage.logits, cell_order=stage.cell_lib._cell_order,
            save_path=str(out_dir / f"stage{i + 1}_cell_selection_trained.png"),
            title=f"{args.problem} — Stage {i + 1} cell selection (trained)",
        )

    val_batch = next(iter(val_loader))
    u_val, y_val = val_batch[0][:64].to(device), val_batch[1][:64].to(device)
    ctx = ctx_factory(u_val.size(0), device=device)
    with torch.no_grad():
        out, trajs = net(u_val, ctx=ctx, store_trajectory=True)
    if isinstance(trajs, list) and trajs:
        plot_trajectories(
            trajs[0], stage_idx=0,
            save_path=str(out_dir / "trajectories.png"),
            title=f"{args.problem} — Stage 1 trajectories (trained)",
        )

    plot_output_fit(
        out, y_val, loss_name=PRESETS[args.problem]["loss"],
        save_path=str(out_dir / "output_fit.png"),
        title=f"{args.problem} — Output fit (trained)",
    )

    plot_network(
        raw_net, cell_order=raw_net.core.stages[0].cell_lib._cell_order,
        save_path=str(out_dir / "pipeline.png"),
    )

    # ----------------------------------------------------------------
    # Complexity-regularized pruning pipeline (CP-6).
    # Three-phase schedule: pruning auto-triggers at the B→C boundary.
    # ----------------------------------------------------------------
    if needs_prune:

        from config import PRUNE
        from topology import prune_network
        from kirchhoff_net import KirchhoffNet, KirchhoffNetWithIO

        # Import SCHEDULE_THREE_PHASE for schedule-specific edge/node thresholds.
        from config import SCHEDULE_THREE_PHASE

        # For phased schedules, use schedule-specific thresholds;
        # CLI overrides take precedence.
        if args.prune_edge_threshold is None and args.prune_node_threshold is None:
            if schedule_mode == "four_phase":
                _scfg = SCHEDULE_FOUR_PHASE
                edge_thresh = float(_scfg.get("prune_edge_threshold", 0.05))
                node_thresh = float(_scfg.get("prune_node_threshold", 0.05))
            elif schedule_mode == "three_phase":
                _scfg = SCHEDULE_THREE_PHASE
                edge_thresh = float(_scfg.get("prune_edge_threshold", 0.1))
                node_thresh = float(_scfg.get("prune_node_threshold", 0.05))
            else:
                edge_thresh = float(PRUNE["edge_threshold"])
                node_thresh = float(PRUNE["node_threshold"])
        else:
            edge_thresh = args.prune_edge_threshold if args.prune_edge_threshold is not None else float(PRUNE["edge_threshold"])
            node_thresh = args.prune_node_threshold if args.prune_node_threshold is not None else float(PRUNE["node_threshold"])

        # DEPRECATED (deprecate-node-gates): prune_nodes_by_gate is always
        # False; the CLI flag is kept as a deprecated no-op.
        if args.prune_nodes_by_gate is not None:
            import warnings
            warnings.warn(
                "--prune-nodes-by-gate / --no-prune-nodes-by-gate is deprecated "
                "(deprecate-node-gates); node pruning is connectivity-only "
                "regardless of this flag.",
                DeprecationWarning,
                stacklevel=2,
            )
        pnbg = False

        pre_edges = sum(s.num_edges() for s in raw_net.core.stages)
        pre_nodes = sum(s.num_nodes for s in raw_net.core.stages)
        print(
            f"[prune] pre-prune: {pre_edges} edges, {pre_nodes} nodes "
            f"(edge_thresh={edge_thresh}, prune_nodes_by_gate={pnbg})"
        )

        pruned_core, stage_remaps = prune_network(
            raw_net.core,
            edge_threshold=edge_thresh,
            node_threshold=node_thresh,
            transfer_params=not args.fresh_init,
            write_idx=list(raw_net.write_idx) if raw_net.write_idx is not None else None,
            read_idx=list(raw_net.read_idx) if raw_net.read_idx is not None else None,
            prune_nodes_by_gate=pnbg,
        )

        post_edges = sum(s.num_edges() for s in pruned_core.stages)
        post_nodes = sum(s.num_nodes for s in pruned_core.stages)
        print(
            f"[prune] post-prune: {post_edges} edges, {post_nodes} nodes "
            f"(removed {pre_edges - post_edges} edges, "
            f"{pre_nodes - post_nodes} nodes)"
        )

        # Rebuild InputMapper/OutputMapper with weights transferred from
        # the pre-prune network.  We treat all surviving nodes as hidden
        # (proj=0) because the hidden/projection split is unknown after
        # remapping.  ``stage_remaps[0]`` maps first-stage compact ids,
        # ``stage_remaps[-1]`` maps last-stage compact ids.
        preset_cfg = PRESETS[args.problem]
        pruned_first_n = pruned_core.stages[0].num_nodes
        pruned_last_n = pruned_core.stages[-1].num_nodes
        in_dim = preset_cfg["stages"][0]["num_inputs"]
        out_dim = preset_cfg.get("out_dim", 1)
        stage0_remap = stage_remaps[0]
        last_remap = stage_remaps[-1]

        raw_write_idx = list(raw_net.write_idx) if raw_net.write_idx is not None else None
        raw_read_idx = list(raw_net.read_idx) if raw_net.read_idx is not None else None
        if effective_read_mode == "sparse" and raw_read_idx is None:
            raw_read_idx = list(read_idx_arg) if read_idx_arg is not None else [0]

        if not args.fresh_init:
            input_mapper_pruned, pruned_write_idx = _transfer_input_mapper(
                raw_net.input_mapper, raw_write_idx, stage0_remap,
                pruned_first_n, in_dim,
            )
            output_mapper_pruned, pruned_read_idx = _transfer_output_mapper(
                raw_net.output_mapper, raw_read_idx, last_remap,
                pruned_last_n, out_dim,
            )
        else:
            if effective_write_mode == "one_to_one":
                if raw_write_idx is None:
                    pruned_write_idx = list(range(min(in_dim, pruned_first_n)))
                else:
                    pruned_write_idx = _remap_indices(raw_write_idx, stage0_remap)
                input_mapper_pruned = SparseInputMapper(
                    in_dim=in_dim, out_dim=pruned_first_n, write_idx=pruned_write_idx,
                )
            elif effective_write_mode == "fan_out":
                fan_out_map = preset_cfg.get("write_fan_out")
                if fan_out_map is None:
                    raise ValueError(
                        "write_mode='fan_out' requires 'write_fan_out' in config"
                    )
                new_fan_out = {
                    inp: _remap_indices(targets, stage0_remap)
                    for inp, targets in fan_out_map.items()
                }
                input_mapper_pruned = FanOutInputMapper(
                    in_dim=in_dim, out_dim=pruned_first_n, fan_out_map=new_fan_out,
                )
                pruned_write_idx = None
            else:
                MapperCls = (RobustInputMapper
                             if preset_cfg.get("use_robust_input", False)
                             else InputMapper)
                input_mapper_pruned = MapperCls(in_dim=in_dim, out_dim=pruned_first_n)
                pruned_write_idx = None

            if effective_read_mode == "sparse":
                if raw_read_idx is None:
                    pruned_read_idx = [0]
                else:
                    pruned_read_idx = _remap_indices(raw_read_idx, last_remap)
                output_mapper_pruned = OutputMapper(
                    node_dim=pruned_last_n, out_dim=out_dim, read_idx=pruned_read_idx,
                )
            else:
                output_mapper_pruned = OutputMapper(node_dim=pruned_last_n, out_dim=out_dim)
                pruned_read_idx = None

        pruned_net = KirchhoffNetWithIO(
            input_mapper_pruned,
            pruned_core,
            output_mapper_pruned,
            hid_count=pruned_first_n,
            proj_count=0,
            final_hid_count=pruned_last_n,
            final_proj_count=0,
            write_idx=pruned_write_idx if effective_write_mode == "one_to_one" else None,
            read_idx=pruned_read_idx if effective_read_mode == "sparse" else None,
        )
        pruned_net.to(device)

        # For three_phase, Phase C retrain is always enabled and uses the remainder
        # of the epoch budget. For legacy, respect the --retrain flag.
        retrain_enabled = args.retrain if schedule_mode == "legacy" else True
        if retrain_enabled:
            if schedule_mode in ("three_phase", "four_phase"):
                c_epochs = c_total
                c_lr = args.retrain_lr if args.retrain_lr is not None else lr
            else:
                c_epochs = args.retrain_epochs if args.retrain_epochs is not None else max(50, epochs // 2)
                c_lr = args.retrain_lr if args.retrain_lr is not None else lr
            print(
                f"[prune] retraining compact network for {c_epochs} epochs "
                f"(lr={c_lr}, warm_start={not args.fresh_init}, "
                f"scheduler={args.use_scheduler})"
            )
            retrain_optimizer = make_optimizer(
                pruned_net, lr=c_lr,
                stage_lr_scale=args.retrain_stage_lr_scale,
                mapper_lr_scale=args.retrain_mapper_lr_scale,
            )
            if args.retrain_stage_lr_scale != 1.0 or args.retrain_mapper_lr_scale != 1.0:
                lr_strs = [f"{g['lr']:.1e}" for g in retrain_optimizer.param_groups]
                print(
                    f"[prune] retrain stage_lr_scale={args.retrain_stage_lr_scale}, "
                    f"retrain_mapper_lr_scale={args.retrain_mapper_lr_scale}: "
                    f"per-group LRs = {lr_strs}"
                )
            if args.use_scheduler:
                if args.scheduler_type == "cosine":
                    retrain_scheduler = CosineAnnealingLR(
                        retrain_optimizer,
                        T_max=max(1, c_epochs),
                        eta_min=OPTIM["scheduler_eta_min"],
                    )
                else:
                    retrain_scheduler = CosineAnnealingWarmRestarts(
                        retrain_optimizer,
                        T_0=OPTIM["scheduler_T_0"],
                        T_mult=OPTIM["scheduler_T_mult"],
                        eta_min=OPTIM["scheduler_eta_min"],
                    )
            else:
                retrain_scheduler = None
            retrain_scaler = (
                torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
            )
            retrain_history = []
            retrain_val_history = []
            retrain_val_argmax = [] if val_argmax_history is not None else None
            retrain_orig_history = [] if inverse_stats is not None else None
            best_val_pruned = float("inf")
            best_epoch_pruned = -1
            best_state_pruned = None
            best_metric_name_c = "val"  # four-phase-redesign/Phase 1b
            ewop = 0
            for repoch in range(c_epochs):
                pruned_net.train()
                if schedule_mode == "three_phase":
                    global_epoch = b_end + repoch
                    tau_r = three_phase_tau(global_epoch, epochs)
                    effective_c_lambdas = three_phase_lambdas(global_epoch, epochs, lambdas)
                    reg_r = 1.0
                    # Solidification metrics during Phase C.
                    if solid_log_path is not None and repoch % args.validate_every == 0:
                        c_metrics = compute_solidification_metrics(pruned_net, tau=tau_r)
                        _log_solidification(solid_log_path, global_epoch, c_metrics)
                elif schedule_mode == "four_phase":
                    global_epoch = b2_end + repoch
                    tau_r = four_phase_tau(global_epoch, epochs)
                    effective_c_lambdas = four_phase_lambdas(global_epoch, epochs, lambdas)
                    reg_r = 1.0
                    if solid_log_path is not None and repoch % args.validate_every == 0:
                        c_metrics = compute_solidification_metrics(pruned_net, tau=tau_r)
                        _log_solidification(solid_log_path, global_epoch, c_metrics)
                else:
                    retrain_warmup = (0 if (not args.fresh_init) else max(1, c_epochs // 2))
                    retrain_tau_init = float(TAU.get("final_pretrain", TAU["init"]))
                    retrain_tau_final = float(TAU["final"])
                    tau_r = tau_for_epoch(
                        repoch, total_epochs=c_epochs,
                        tau_init=retrain_tau_init,
                        tau_final=retrain_tau_final,
                    )
                    reg_r = reg_schedule(
                        repoch,
                        warmup=retrain_warmup,
                        anneal=max(25, c_epochs // 4),
                    )
                    effective_c_lambdas = lambdas
                # four-phase-redesign/Phase 2b: cell_mode in Phase C.
                # Phase C is post-prune and STE is the deployable form.
                cell_mode_c = _resolve_cell_mode(args.cell_mode, "C", schedule_mode)
                tot = 0.0
                nb = 0
                for batch in train_loader:
                    ctx = ctx_factory(batch[0].size(0), device=device)
                    retrain_optimizer.zero_grad()
                    u_b, tgt_b = batch
                    u_b = u_b.to(device)
                    tgt_b = tgt_b.to(device)
                    loss_task, loss_structural, _ = compute_loss(
                        pruned_net, u_b, tgt_b, ctx, task_fn,
                        lambdas=effective_c_lambdas, tau=tau_r, return_parts=True,
                        amp=amp_enabled, amp_dtype=amp_dtype, reg_scale=reg_r,
                        cell_mode=cell_mode_c,
                    )
                    if retrain_scaler is not None and retrain_scaler._enabled:
                        ( retrain_scaler.scale(loss_task) + retrain_scaler.scale(loss_structural) ).backward()
                        retrain_scaler.unscale_(retrain_optimizer)
                        torch.nn.utils.clip_grad_norm_(pruned_net.parameters(), max_norm=OPTIM["grad_clip_norm"])
                        retrain_scaler.step(retrain_optimizer)
                        retrain_scaler.update()
                    else:
                        (loss_task + loss_structural).backward()
                        torch.nn.utils.clip_grad_norm_(pruned_net.parameters(), max_norm=OPTIM["grad_clip_norm"])
                        retrain_optimizer.step()
                    tot += float((loss_task + loss_structural).item())
                    nb += 1
                if retrain_scheduler is not None:
                    retrain_scheduler.step()
                if grad_log_path is not None and repoch % args.grad_log_every == 0:
                    log_gradient_norms(
                        grad_log_path, repoch, pruned_net, retrain=True,
                        optimizer=retrain_optimizer,
                    )
                avg = tot / max(1, nb)
                retrain_history.append(avg)
                if repoch % args.validate_every == 0 or repoch == c_epochs - 1:
                    if inverse_stats is not None:
                        val_metrics_c = validate_with_inverse(
                            pruned_net, val_loader, task_fn, ctx_factory, device,
                            inverse_stats=inverse_stats, cell_mode=cell_mode_c,
                        )
                        val = val_metrics_c["val"]
                    else:
                        val_metrics_c = None
                        val = validate(pruned_net, val_loader, task_fn, ctx_factory, device, cell_mode=cell_mode_c)
                    if retrain_val_argmax is not None:
                        val_arg = validate_argmax(pruned_net, val_loader, task_fn, ctx_factory, device)
                        retrain_val_argmax.append(val_arg)
                    retrain_val_history.append(val)
                    if retrain_orig_history is not None and val_metrics_c is not None:
                        retrain_orig_history.append(val_metrics_c)
                    # four-phase-redesign/Phase 1b: Phase C is post-prune, the
                    # deployable model IS the hard-cell (argmax) version, so
                    # use val_argmax for checkpoint selection.
                    if (
                        schedule_mode in ("three_phase", "four_phase")
                        and retrain_val_argmax is not None
                        and len(retrain_val_argmax) > 0
                    ):
                        sel_metric_c = float(retrain_val_argmax[-1])
                        sel_name_c = "val_argmax"
                    else:
                        sel_metric_c = float(val)
                        sel_name_c = "val"
                    if sel_metric_c < best_val_pruned - args.min_delta:
                        best_val_pruned = sel_metric_c
                        best_epoch_pruned = repoch
                        best_metric_name_c = sel_name_c
                        ewop = 0
                        best_state_pruned = {k: v.detach().clone() for k, v in pruned_net.state_dict().items()}
                    else:
                        ewop += args.validate_every
                        if args.early_stop and ewop >= args.patience:
                            print(
                                f"[prune] retrain early stop at epoch {repoch} "
                                f"(best {best_metric_name_c}={best_val_pruned:.4f})"
                            )
                            break
                else:
                    retrain_val_history.append(retrain_val_history[-1] if retrain_val_history else avg)
                    if retrain_val_argmax is not None:
                        retrain_val_argmax.append(retrain_val_argmax[-1] if retrain_val_argmax else avg)
                    if retrain_orig_history is not None:
                        retrain_orig_history.append(
                            retrain_orig_history[-1] if retrain_orig_history
                            else {"val": avg, "mae_orig": float("nan"), "rmse_orig": float("nan")}
                        )
                phase_tag = " [C]" if schedule_mode in ("three_phase", "four_phase") else ""
                print(
                    f"  {'retrain' if schedule_mode == 'legacy' else 'phase-C'} epoch {repoch:4d}{phase_tag}  "
                    f"train={avg:.4f}  "
                    f"val={retrain_val_history[-1]:.4f}  tau={tau_r:.3f}  "
                    f"lr={retrain_optimizer.param_groups[0]['lr']:.2e}"
                )

            if best_state_pruned is not None:
                pruned_net.load_state_dict(best_state_pruned)
                pruned_net.to(device)
                print(
                    f"[prune] restored best retrain state "
                    f"(epoch {best_epoch_pruned}, {best_metric_name_c}={best_val_pruned:.4f})"
                )

            if schedule_mode in ("three_phase", "four_phase"):
                phase_c_start = b_end if schedule_mode == "three_phase" else b2_end
                with open(history_path, "r") as f:
                    header = f.readline().strip()
                has_argmax = "val_argmax" in header
                has_orig = "mae_orig" in header
                with open(history_path, "a") as f:
                    f.write(f"\n# Phase C (prune + retrain): {post_edges}/{pre_edges} edges, {post_nodes}/{pre_nodes} nodes survived\n")
                    if has_argmax and has_orig and retrain_orig_history is not None:
                        for i, (t, v, va, m) in enumerate(
                            zip(retrain_history, retrain_val_history, retrain_val_argmax, retrain_orig_history)
                        ):
                            global_ep = phase_c_start + i
                            f.write(f"{global_ep}\t{t}\t{v}\t{va}\t{m['mae_orig']:.6f}\t{m['rmse_orig']:.6f}\tC\n")
                    elif has_argmax:
                        for i, (t, v, va) in enumerate(zip(retrain_history, retrain_val_history, retrain_val_argmax)):
                            global_ep = phase_c_start + i
                            f.write(f"{global_ep}\t{t}\t{v}\t{va}\tC\n")
                    elif has_orig and retrain_orig_history is not None:
                        for i, (t, v, m) in enumerate(zip(retrain_history, retrain_val_history, retrain_orig_history)):
                            global_ep = phase_c_start + i
                            f.write(f"{global_ep}\t{t}\t{v}\t{m['mae_orig']:.6f}\t{m['rmse_orig']:.6f}\tC\n")
                    else:
                        for i, (t, v) in enumerate(zip(retrain_history, retrain_val_history)):
                            global_ep = phase_c_start + i
                            f.write(f"{global_ep}\t{t}\t{v}\tC\n")
            else:
                with open(history_path, "a") as f:
                    f.write(
                        f"\n[prune] pre-prune: {pre_edges} edges, {pre_nodes} nodes "
                        f"(edge_thresh={edge_thresh}, node_thresh={node_thresh})\n"
                    )
                    f.write(
                        f"[prune] post-prune: {post_edges} edges, {post_nodes} nodes "
                        f"(removed {pre_edges - post_edges} edges, "
                        f"{pre_nodes - post_nodes} nodes)\n"
                    )
                    f.write(f"retrain_epoch\ttrain\tval\n")
                    for i, (t, v) in enumerate(zip(retrain_history, retrain_val_history)):
                        f.write(f"{i}\t{t}\t{v}\n")

            # Save compact-network plots.
            for i, stage in enumerate(pruned_core.stages):
                plot_stage_graph(
                    stage, save_path=str(out_dir / f"stage{i + 1}_graph_pruned.png"),
                    title=f"{args.problem} — Stage {i + 1} (pruned, {stage.num_edges()} edges, {stage.num_nodes} nodes)",
                )
                plot_cell_selection(
                    stage.logits, cell_order=stage.cell_lib._cell_order,
                    save_path=str(out_dir / f"stage{i + 1}_cell_selection_pruned.png"),
                    title=f"{args.problem} — Stage {i + 1} cell selection (pruned)",
                )

            # Pruned output fit.
            with torch.no_grad():
                out_pruned, _ = pruned_net(u_val, ctx=ctx_factory(u_val.size(0), device=device), store_trajectory=False)
            plot_output_fit(
                out_pruned, y_val, loss_name=PRESETS[args.problem]["loss"],
                save_path=str(out_dir / "output_fit_pruned.png"),
                title=f"{args.problem} — Output fit (pruned, retrained)",
            )

            # Save pruned model.
            torch.save(pruned_net.state_dict(), out_dir / "model_pruned.pt")
            print(f"[prune] saved pruned model to {out_dir / 'model_pruned.pt'}")
        else:
            # No-retrain mode: pruned model with transferred parameters.
            torch.save(pruned_net.state_dict(), out_dir / "model_pruned.pt")
            print(
                f"[prune] --no-retrain: saved pruned model (with transferred params) "
                f"to {out_dir / 'model_pruned.pt'}"
            )

        # Write a pruning summary file.
        with open(out_dir / "prune_summary.txt", "w") as f:
            f.write(f"edge_threshold: {edge_thresh}\n")
            f.write(f"node_threshold: {node_thresh}\n")
            f.write(f"pre_edges: {pre_edges}\n")
            f.write(f"post_edges: {post_edges}\n")
            f.write(f"pre_nodes: {pre_nodes}\n")
            f.write(f"post_nodes: {post_nodes}\n")
            f.write(f"edges_removed: {pre_edges - post_edges}\n")
            f.write(f"nodes_removed: {pre_nodes - post_nodes}\n")
            f.write(f"retrain: {retrain_enabled}\n")
            f.write(f"fresh_init: {args.fresh_init}\n")
            if retrain_enabled:
                f.write(f"retrain_epochs: {c_epochs}\n")
                f.write(f"retrain_lr: {c_lr}\n")
                f.write(f"best_val_pruned: {best_val_pruned:.6f}\n")
                f.write(f"best_epoch_pruned: {best_epoch_pruned}\n")
                f.write(f"scheduler: {args.use_scheduler}\n")

    print(f"[train] done — artifacts in {out_dir}")


if __name__ == "__main__":
    main()