"""Original KirchhoffNet (CircuitNet) baseline for the Friedman #2 synthetic regression task.

Purpose: provide a parameter-matched benchmark for the reduced differential
KirchhoffNet on the classic Friedman (1991) MARS test problem #2, using the
original ODE-based CircuitNet model from ``../kirchhoffnet_original``.

Friedman #2 is a 4-dimensional regression problem. The target is:

    y = sqrt(x1^2 + (x2 * x3 - 1 / (x2 * x4))^2) + noise

with variable-specific ranges:
    x1 ~ U(0, 100)
    x2 ~ U(40*pi, 560*pi)
    x3 ~ U(0, 1)
    x4 ~ U(1, 11)

The canonical noise standard deviation is sigma = 1.0.

CLI:
    kn_benchmark_friedman2.py [--num-nodes 8] [--num-stages 1]
                              [--device-model ShiftTanh2]
                              [--topology {fully_connect,torus}]
                              [--solver-method dopri5] [--t-end 1.0]
                              [--solver-tol 1e-4]
                              [--epochs 800] [--lr 1.2e-3] [--target-noise-std 1.0]
                              [--output OUTPUT] [--device DEVICE]

Outputs to --output:
  - loss_history.txt
  - loss_curve.png
  - model.pt
  - config_snapshot.txt
  - output_fit.png
  - final_metrics.txt

Defaults match the KirchhoffNet training script (train_script.py):
AdamW (lr=OPTIM.lr, weight_decay=OPTIM.weight_decay), grad_clip_norm=5.0,
batch_size=OPTIM.batch_size, early stopping with patience=50, min_delta=1e-4.
"""

import argparse
import math
import random
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_THIS_FILE = Path(__file__).resolve()
sys.path.insert(0, str(_THIS_FILE.parent))

from config import OPTIM
from train_script import _lhs_samples

_KN_ORIGINAL_SRC = _THIS_FILE.parent.parent.parent / "kirchhoffnet_original" / "kirchhoffnet" / "src"
if not _KN_ORIGINAL_SRC.is_dir():
    _KN_ORIGINAL_SRC = _THIS_FILE.parent.parent / "kirchhoffnet_original" / "kirchhoffnet" / "src"
if not _KN_ORIGINAL_SRC.is_dir():
    raise FileNotFoundError(
        f"Could not locate kirchhoffnet_original 'src' directory at {_KN_ORIGINAL_SRC}. "
        "Pass --kirchhoffnet-original-src to override the search path."
    )
sys.path.insert(0, str(_KN_ORIGINAL_SRC))

from utils.model import CircuitNet
from utils.topology import generate_topology

__all__ = ["CircuitNetRegressor", "count_parameters", "_friedman2",
           "make_data_friedman2", "_DEVICE_MODELS", "_TOPOLOGY_MODES"]


_FRIEDMAN2_IN_DIM = 4
_FRIEDMAN2_PI = math.pi
_FRIEDMAN2_RANGES = [
    (0.0, 100.0),
    (40.0 * _FRIEDMAN2_PI, 560.0 * _FRIEDMAN2_PI),
    (0.0, 1.0),
    (1.0, 11.0),
]


_DEVICE_MODELS = {
    "ShiftTanh2",
    "ShiftTanh1",
    "ShiftRelu2",
    "ShiftRelu1",
    "ShiftLeakyRelu2",
    "ShiftLeakyRelu1",
    "Conductance",
}

_TOPOLOGY_MODES = {"fully_connect", "torus"}

_SOLVER_METHODS = {"dopri5", "dopri8", "adams", "tsit5", "euler", "midpoint", "rk4"}


def _friedman2(x: torch.Tensor) -> torch.Tensor:
    """Friedman #2 target.

    Args:
        x: Tensor of shape (..., 4) with x[..., 0] ~ U(0,100),
           x[..., 1] ~ U(40π, 560π), x[..., 2] ~ U(0,1), x[..., 3] ~ U(1,11).

    Returns:
        Tensor of shape (...) with the deterministic (noise-free) target.
    """
    if x.shape[-1] != _FRIEDMAN2_IN_DIM:
        raise ValueError(
            f"_friedman2 requires exactly {_FRIEDMAN2_IN_DIM} input columns, "
            f"got {x.shape[-1]}"
        )
    x1 = x[..., 0]
    x2 = x[..., 1]
    x3 = x[..., 2]
    x4 = x[..., 3]
    inner = (x2 * x3) - (1.0 / (x2 * x4))
    return torch.sqrt(x1 ** 2 + inner ** 2)


def _scale_lhs_to_ranges(u_unit: torch.Tensor, ranges: list[tuple[float, float]]) -> torch.Tensor:
    """Linearly scale [0,1] uniform/LHS samples to per-dim (lo, hi) ranges."""
    out = u_unit.clone()
    for i, (lo, hi) in enumerate(ranges):
        out[..., i] = lo + (hi - lo) * u_unit[..., i]
    return out


def _minmax_normalize_inputs(
    u_train: torch.Tensor, u_val: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-dim min-max normalize features to [0, 1] using train-set stats."""
    u_min = u_train.amin(dim=0, keepdim=True)
    u_max = u_train.amax(dim=0, keepdim=True)
    u_range = (u_max - u_min).clamp(min=1e-8)
    u_train_n = (u_train - u_min) / u_range
    u_val_n = (u_val - u_min) / u_range
    return u_train_n, u_val_n, u_min, u_range


def make_data_friedman2(batch_size: int, noise_std: float = 1.0, val_size: int = 4000,
                        normalize_inputs: bool = True):
    """Build Friedman #2 train/val loaders with target standardization.

    Mirrors the MLP benchmark's data generation exactly so that the MLP and
    KirchhoffNet benchmarks can be compared on identical train/val splits
    and target normalization.
    """
    n_train = 20000
    u_train_unit = _lhs_samples(n_train, _FRIEDMAN2_IN_DIM, seed=42)
    u_train = _scale_lhs_to_ranges(u_train_unit, _FRIEDMAN2_RANGES)
    y_train = _friedman2(u_train).unsqueeze(1)

    torch.manual_seed(42)
    u_val_unit = torch.rand(val_size, _FRIEDMAN2_IN_DIM)
    u_val = _scale_lhs_to_ranges(u_val_unit, _FRIEDMAN2_RANGES)
    y_val = _friedman2(u_val).unsqueeze(1)

    if noise_std > 0:
        y_train = y_train + noise_std * torch.randn_like(y_train)

    if normalize_inputs:
        u_train, u_val, u_min, u_range = _minmax_normalize_inputs(u_train, u_val)

    y_mean = float(y_train.mean().item())
    y_std = float(y_train.std().clamp(min=1e-6).item())
    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    train_loader = DataLoader(
        TensorDataset(u_train, y_train_n), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(u_val, y_val_n), batch_size=batch_size, shuffle=False
    )
    inverse_stats = {"y_mean": y_mean, "y_std": y_std}
    if normalize_inputs:
        inverse_stats["u_min"] = u_min.squeeze(0).tolist()
        inverse_stats["u_range"] = u_range.squeeze(0).tolist()
    return train_loader, val_loader, F.mse_loss, inverse_stats


def count_parameters(net: nn.Module) -> int:
    """Total learnable parameters (weights + biases)."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def _build_torus_topology(num_nodes: int, kernel_size: int = 3) -> torch.Tensor:
    """Build a 2-D torus (grid with wrap-around) edge list for the circuit.

    Generates a ``root_height * root_width = num_nodes`` square grid with
    per-node ``kernel_size x kernel_size`` neighborhood (including the
    self-wrap on the borders). Each non-self (i, j) pair in the neighborhood
    produces a directed edge (i -> j). Returns a long tensor of shape
    (E, 3) with columns [src, des, edge_id]; node 0 is reserved for ground
    and excluded from the hidden nodes (so hidden nodes are 1..num_nodes).

    Args:
        num_nodes: number of hidden circuit nodes (excludes ground). Must be
            a positive perfect square (e.g. 4, 9, 16, 25).
        kernel_size: odd neighborhood side length (default 3, meaning the
            8 surrounding cells of each hidden node on the torus).

    Returns:
        Long tensor of shape (E, 3) ready to be consumed by CircuitBlock.
    """
    side = int(round(math.sqrt(num_nodes)))
    if side * side != num_nodes or num_nodes < 4:
        raise ValueError(
            f"_build_torus_topology requires num_nodes to be a perfect "
            f"square >= 4, got {num_nodes}"
        )
    if kernel_size < 1 or kernel_size % 2 != 0 and kernel_size != 1:
        # accept any odd kernel_size >= 1 (1 -> self-loops only, skipped)
        if kernel_size < 3:
            kernel_size = 3

    hidden_ids = torch.arange(1, num_nodes + 1).view(side, side)

    edges = []
    cnt = 0
    half = (kernel_size - 1) // 2
    for r in range(side):
        for c in range(side):
            src = int(hidden_ids[r, c].item())
            for dr in range(-half, half + 1):
                for dc in range(-half, half + 1):
                    nr = (r + dr) % side
                    nc = (c + dc) % side
                    if nr == r and nc == c:
                        continue
                    des = int(hidden_ids[nr, nc].item())
                    edges.append([src, des, cnt])
                    cnt += 1
    if not edges:
        raise RuntimeError(
            "_build_torus_topology produced 0 edges; check num_nodes and kernel_size"
        )
    return torch.tensor(edges, dtype=torch.long)


class CircuitNetRegressor(nn.Module):
    """ODE-based KirchhoffNet (CircuitNet) regressor for 4-dim inputs.

    Architecture: Linear(4 -> num_nodes) [encoder] -> ODE-integrated
    CircuitNet -> Linear(num_nodes -> 1) [projector].

    Input: (B, in_dim=4)
    Output: (B, out_dim=1)

    The CircuitNet uses ``use_augment=False`` (regression, not generative),
    allowing direct regression-head training without log-probability
    tracking. The numerical solver is torchdiffeq's adaptive odeint.
    """

    def __init__(
        self,
        in_dim: int = _FRIEDMAN2_IN_DIM,
        num_nodes: int = 8,
        out_dim: int = 1,
        num_stages: int = 1,
        topology_mode: str = "fully_connect",
        device_model_name: str = "ShiftTanh2",
        initialization: str = "kaiming",
        solver_method: str = "dopri5",
        t_end: float = 1.0,
        solver_tol: float = 1e-4,
        first_step: float = 1e-3,
        min_step: float = 1e-6,
        step_size: float = 1e-6,
        adjoint: bool = True,
    ) -> None:
        super().__init__()
        if topology_mode not in _TOPOLOGY_MODES:
            raise ValueError(
                f"topology_mode must be one of {sorted(_TOPOLOGY_MODES)}, "
                f"got {topology_mode!r}"
            )
        if device_model_name not in _DEVICE_MODELS:
            raise ValueError(
                f"device_model_name must be one of {sorted(_DEVICE_MODELS)}, "
                f"got {device_model_name!r}"
            )
        if solver_method not in _SOLVER_METHODS:
            raise ValueError(
                f"solver_method must be one of {sorted(_SOLVER_METHODS)}, "
                f"got {solver_method!r}"
            )
        if num_stages < 1:
            raise ValueError(f"num_stages must be >= 1, got {num_stages}")
        if num_nodes < 2:
            raise ValueError(f"num_nodes must be >= 2, got {num_nodes}")
        if topology_mode == "torus" and int(round(math.sqrt(num_nodes))) ** 2 != num_nodes:
            raise ValueError(
                f"topology='torus' requires num_nodes to be a perfect square, "
                f"got {num_nodes}"
            )

        self.in_dim = int(in_dim)
        self.num_nodes = int(num_nodes)
        self.out_dim = int(out_dim)
        self.num_stages = int(num_stages)
        self.topology_mode = topology_mode
        self.device_model_name = device_model_name
        self.initialization = initialization
        self.solver_method = solver_method
        self.t_end = float(t_end)
        self.solver_tol = float(solver_tol)
        self.adjoint = bool(adjoint)

        self.encoder = nn.Linear(int(in_dim), int(num_nodes))
        self.projector = nn.Linear(int(num_nodes), int(out_dim))

        circuit_topology = self._build_topologies()
        sim_dict = {
            "t_end": self._t_end_anchors(),
            "tol": float(solver_tol),
            "min_step": float(min_step),
            "first_step": float(first_step),
            "step_size": float(step_size),
            "method": solver_method,
        }
        circuit_dict = {
            "model": {"name": device_model_name, "args": {}},
            "initialization": initialization,
            "residual": None,
            "fill": None,
        }
        self.circuit = CircuitNet(
            circuit_topology=circuit_topology,
            sim_dict=sim_dict,
            circuit_dict=circuit_dict,
            encoder=None,
            projector=None,
            use_augment=False,
            adjoint=self.adjoint,
        )

    def _build_topologies(self) -> list[torch.Tensor]:
        topos = []
        for _ in range(self.num_stages):
            if self.topology_mode == "fully_connect":
                edges, _ = self._fully_connect_local(
                    num_node=self.num_nodes + 1, include_gnd=True, repeat=[1, 1]
                )
            elif self.topology_mode == "torus":
                edges = _build_torus_topology(self.num_nodes, kernel_size=3)
            else:
                raise ValueError(f"Unsupported topology_mode: {self.topology_mode}")
            topos.append(edges)
        return topos

    @staticmethod
    def _fully_connect_local(num_node: int, include_gnd: bool = True,
                              repeat: tuple[int, int] = (1, 1)) -> tuple[list, int]:
        edge_matrix = []
        node_start = 0 if include_gnd else 1
        node_end = num_node if include_gnd else num_node + 1
        cnt = 0
        for i in range(node_start, node_end):
            for j in range(node_start, node_end):
                if i != j:
                    for _k1 in range(repeat[0]):
                        edge_matrix.append([i, j, cnt])
                        cnt += 1
                    for _k2 in range(repeat[1]):
                        edge_matrix.append([j, i, cnt])
                        cnt += 1
        return edge_matrix, cnt

    def _t_end_anchors(self) -> list[float]:
        if self.num_stages == 1:
            return [0.0, self.t_end]
        anchors = [0.0]
        for i in range(1, self.num_stages):
            anchors.append(self.t_end * i / self.num_stages)
        anchors.append(self.t_end)
        return anchors

    def prepare(self, device: list[int] | int) -> None:
        if isinstance(device, int):
            device = [device] if device >= 0 else [-1]
        self.circuit.prepare(device)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        h = self.encoder(u)
        out, _ = self.circuit(h, reverse=False, return_middle=False)
        return self.projector(out)


def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        warnings.warn("matplotlib not installed; skipping plots", stacklevel=2)
        return None


def validate(net, val_loader, task_fn, device) -> float:
    net.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for u, target in val_loader:
            u = u.to(device)
            target = target.to(device)
            out = net(u)
            loss = task_fn(out, target)
            total += float(loss.item()) * u.size(0)
            n += u.size(0)
    net.train()
    return total / max(1, n)


def compute_orig_metrics(net, val_loader, inverse_stats, device):
    """Compute MSE/RMSE in original (un-standardized) target units."""
    net.eval()
    preds, targets = [], []
    with torch.no_grad():
        for u, t in val_loader:
            u = u.to(device)
            t = t.to(device)
            preds.append(net(u))
            targets.append(t)
    y_pred_std = torch.cat(preds, dim=0)
    y_true_std = torch.cat(targets, dim=0)
    y_mean = inverse_stats["y_mean"]
    y_std = inverse_stats["y_std"]
    y_pred = y_pred_std * y_std + y_mean
    y_true = y_true_std * y_std + y_mean
    mse = float(F.mse_loss(y_pred, y_true).item())
    rmse = float(torch.sqrt(F.mse_loss(y_pred, y_true)).item())
    mae = float(F.l1_loss(y_pred, y_true).item())
    net.train()
    return mse, rmse, mae


def plot_output_fit(out, target, save_path, title):
    plt = _import_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(target.cpu().numpy().ravel(), out.detach().cpu().numpy().ravel(),
               s=4, alpha=0.4, color="C0")
    lo = float(min(target.min().item(), out.min().item()))
    hi = float(max(target.max().item(), out.max().item()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y = x")
    ax.set_xlabel("target (normalized)")
    ax.set_ylabel("prediction (normalized)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(history, val_history, save_path, title, loss_label: str = "MSE"):
    plt = _import_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history, label="train", color="C0")
    ax.plot(val_history, label="val", color="C3")
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{loss_label} loss")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Train an ODE-based KirchhoffNet (CircuitNet) on the "
                    "Friedman #2 synthetic regression task."
    )
    parser.add_argument("--num-nodes", type=int, default=8,
                        help="Number of circuit nodes (excluding ground). "
                             "For --topology torus this must be a perfect square (default: 8).")
    parser.add_argument("--num-stages", type=int, default=1,
                        help="Number of ODE integration stages (each adds "
                             "another CircuitLayer; default: 1).")
    parser.add_argument("--topology", choices=sorted(_TOPOLOGY_MODES),
                        default="fully_connect",
                        help="Circuit topology (default: fully_connect). "
                             "'torus' places nodes on a sqrt(N)x sqrt(N) wrap-around grid.")
    parser.add_argument("--device-model", choices=sorted(_DEVICE_MODELS),
                        default="ShiftTanh2",
                        help="Per-edge I-V device model (default: ShiftTanh2).")
    parser.add_argument("--initialization", default="kaiming",
                        choices=["uniform", "zeros", "ones", "xavier", "gauss", "kaiming"],
                        help="Initialization scheme for the device-model "
                             "parameters (default: kaiming).")
    parser.add_argument("--solver-method", choices=sorted(_SOLVER_METHODS),
                        default="dopri5",
                        help="torchdiffeq ODE solver (default: dopri5).")
    parser.add_argument("--t-end", type=float, default=1.0,
                        help="Total integration horizon (default: 1.0).")
    parser.add_argument("--solver-tol", type=float, default=1e-4,
                        help="ODE solver atol/rtol (default: 1e-4).")
    parser.add_argument("--no-adjoint", action="store_true",
                        help="Disable torchdiffeq adjoint (memory-efficient) "
                             "and use plain odeint instead.")
    parser.add_argument("--kirchhoffnet-original-src", default=None,
                        help="Override path to kirchhoffnet_original/.../src "
                             "(default: auto-detect).")

    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Number of training epochs (default: {OPTIM['epochs']})")
    parser.add_argument("--lr", type=float, default=None,
                        help=f"Learning rate (default: {OPTIM['lr']})")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help=f"Weight decay (default: {OPTIM['weight_decay']})")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Batch size (default: {OPTIM['batch_size']})")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience in epochs (default: 50)")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="Early stopping min improvement in val loss (default: 1e-4)")
    parser.add_argument("--validate-every", type=int, default=5,
                        help="Validate every N epochs (default: 5)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--target-noise-std", type=float, default=1.0,
                        help="Standard deviation of additive Gaussian target "
                             "noise (default: 1.0, the Friedman #2 canonical value).")
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse",
                        help="Loss function (default: mse). Huber uses delta=1.0.")
    parser.add_argument("--output", type=Path, default=Path("./output/kn_friedman2"),
                        help="Output directory (default: ./output/kn_friedman2)")
    parser.add_argument("--device", default=None,
                        help="Device 'cpu' or 'cuda' (default: auto-detect)")
    parser.add_argument("--normalize-inputs", dest="normalize_inputs",
                        action="store_true", default=True,
                        help="Per-dim min-max normalize inputs to [0, 1] "
                             "(default: on). Use --no-normalize-inputs to disable for ablation.")
    parser.add_argument("--no-normalize-inputs", dest="normalize_inputs",
                        action="store_false",
                        help="Disable per-dim input normalization.")
    args = parser.parse_args()

    global _KN_ORIGINAL_SRC
    if args.kirchhoffnet_original_src is not None:
        _KN_ORIGINAL_SRC = Path(args.kirchhoffnet_original_src).resolve()
        if str(_KN_ORIGINAL_SRC) in sys.path:
            sys.path.remove(str(_KN_ORIGINAL_SRC))
        sys.path.insert(0, str(_KN_ORIGINAL_SRC))
        import importlib
        import utils.model as _utils_model
        import utils.topology as _utils_topology
        importlib.reload(_utils_model)
        importlib.reload(_utils_topology)
        globals()["CircuitNet"] = _utils_model.CircuitNet
        globals()["generate_topology"] = _utils_topology.generate_topology

    epochs = args.epochs if args.epochs is not None else int(OPTIM["epochs"])
    lr = args.lr if args.lr is not None else float(OPTIM["lr"])
    weight_decay = args.weight_decay if args.weight_decay is not None else float(OPTIM["weight_decay"])
    batch_size = args.batch_size if args.batch_size is not None else int(OPTIM["batch_size"])
    grad_clip_norm = float(OPTIM["grad_clip_norm"])
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    net = CircuitNetRegressor(
        in_dim=_FRIEDMAN2_IN_DIM,
        num_nodes=args.num_nodes,
        out_dim=1,
        num_stages=args.num_stages,
        topology_mode=args.topology,
        device_model_name=args.device_model,
        initialization=args.initialization,
        solver_method=args.solver_method,
        t_end=args.t_end,
        solver_tol=args.solver_tol,
        adjoint=not args.no_adjoint,
    )
    n_params = count_parameters(net)
    net.to(device)
    if device == "cpu":
        net.prepare([-1])
    else:
        net.prepare([0])

    train_loader, val_loader, task_fn, inverse_stats = make_data_friedman2(
        batch_size=batch_size, noise_std=args.target_noise_std,
        normalize_inputs=args.normalize_inputs,
    )
    if args.loss == "huber":
        task_fn = lambda o, t: F.huber_loss(o, t, delta=1.0)
    else:
        task_fn = F.mse_loss
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    print(
        f"[kn_friedman2] num_nodes={args.num_nodes} num_stages={args.num_stages} "
        f"topology={args.topology} device_model={args.device_model} "
        f"init={args.initialization} solver={args.solver_method} t_end={args.t_end} "
        f"tol={args.solver_tol} adjoint={not args.no_adjoint} "
        f"params={n_params} loss={args.loss} epochs={epochs} lr={lr} "
        f"weight_decay={weight_decay} batch_size={batch_size} "
        f"grad_clip_norm={grad_clip_norm} target_noise_std={args.target_noise_std} "
        f"device={device} output={out_dir}"
    )

    with open(out_dir / "config_snapshot.txt", "w") as f:
        f.write(f"model: CircuitNetRegressor(4 -> {args.num_nodes} -> 1, "
                f"stages={args.num_stages}, topology={args.topology}, "
                f"device={args.device_model}, init={args.initialization}, "
                f"solver={args.solver_method}, t_end={args.t_end}, "
                f"tol={args.solver_tol}, adjoint={not args.no_adjoint})\n")
        f.write(f"kirchhoffnet_original_src: {_KN_ORIGINAL_SRC}\n")
        f.write(f"param_count: {n_params}\n")
        f.write(f"num_nodes: {args.num_nodes}\n")
        f.write(f"num_stages: {args.num_stages}\n")
        f.write(f"topology: {args.topology}\n")
        f.write(f"device_model: {args.device_model}\n")
        f.write(f"initialization: {args.initialization}\n")
        f.write(f"solver_method: {args.solver_method}\n")
        f.write(f"t_end: {args.t_end}\n")
        f.write(f"solver_tol: {args.solver_tol}\n")
        f.write(f"adjoint: {not args.no_adjoint}\n")
        f.write(f"loss: {args.loss}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"lr: {lr}\n")
        f.write(f"weight_decay: {weight_decay}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"grad_clip_norm: {grad_clip_norm}\n")
        f.write(f"patience: {args.patience}\n")
        f.write(f"min_delta: {args.min_delta}\n")
        f.write(f"validate_every: {args.validate_every}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"target_noise_std: {args.target_noise_std}\n")
        f.write(f"device: {device}\n")
        f.write(f"dataset: Friedman #2 (4-dim, ranges x1=(0,100) x2=(40pi,560pi) "
                f"x3=(0,1) x4=(1,11), 20k LHS train / 4k uniform val, "
                f"sigma={args.target_noise_std}, normalized targets)\n")

    loss_label = args.loss.upper()
    loss_metric_key = f"val_{args.loss}"

    history = []
    val_history = []
    orig_rmse_history = []
    orig_mae_history = []
    best_val = float("inf")
    best_epoch = -1
    best_state = None
    epochs_without_improve = 0
    stop_training = False

    start = time.time()
    for epoch in range(epochs):
        if stop_training:
            break
        net.train()
        total_loss = 0.0
        n_batches = 0
        for u, target in train_loader:
            u = u.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            out = net(u)
            loss = task_fn(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        avg_train = total_loss / max(1, n_batches)
        do_validate = (epoch % args.validate_every == 0) or (epoch == epochs - 1)
        if do_validate:
            val_loss = validate(net, val_loader, task_fn, device)
            _, rmse, mae = compute_orig_metrics(net, val_loader, inverse_stats, device)
        else:
            val_loss = val_history[-1] if val_history else avg_train
            rmse = orig_rmse_history[-1] if orig_rmse_history else 0.0
            mae = orig_mae_history[-1] if orig_mae_history else 0.0

        history.append(avg_train)
        val_history.append(val_loss)
        orig_rmse_history.append(rmse)
        orig_mae_history.append(mae)

        if do_validate:
            if val_loss < best_val - args.min_delta:
                best_val = float(val_loss)
                best_epoch = epoch
                epochs_without_improve = 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                epochs_without_improve += args.validate_every
                if epochs_without_improve >= args.patience:
                    print(
                        f"[kn_friedman2] early stopping at epoch {epoch}: "
                        f"no val improvement for {epochs_without_improve} epochs "
                        f"(best val={best_val:.6f} @ epoch {best_epoch})"
                    )
                    stop_training = True

        if epoch % args.validate_every == 0 or epoch == epochs - 1:
            print(
                f"  epoch {epoch:4d}  train={avg_train:.6f}  val={val_loss:.6f}  "
                f"RMSE_orig={rmse:.4f}  MAE_orig={mae:.4f}  "
                f"nfe={net.circuit.nfe[0] if hasattr(net.circuit, 'nfe') else 'n/a'}"
            )

    elapsed = time.time() - start
    print(f"[kn_friedman2] training done in {elapsed:.1f}s ({len(history)} epochs)")

    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"[kn_friedman2] restored best model from epoch {best_epoch} (val={best_val:.6f})")

    torch.save(net.state_dict(), out_dir / "model.pt")

    with open(out_dir / "loss_history.txt", "w") as f:
        f.write("epoch\ttrain\tval\trmse_orig\tmae_orig\n")
        for i, (t, v, r, m) in enumerate(zip(history, val_history, orig_rmse_history, orig_mae_history)):
            f.write(f"{i}\t{t}\t{v}\t{r}\t{m}\n")

    plot_loss_curve(
        history, val_history,
        save_path=str(out_dir / "loss_curve.png"),
        title=f"CircuitNet (nodes={args.num_nodes}, stages={args.num_stages}, {args.device_model}, {n_params} params) - Friedman #2",
        loss_label=loss_label,
    )

    val_batch = next(iter(val_loader))
    u_val, y_val = val_batch[0][:64], val_batch[1][:64]
    u_val = u_val.to(device)
    y_val = y_val.to(device)
    with torch.no_grad():
        out = net(u_val)
    plot_output_fit(
        out, y_val,
        save_path=str(out_dir / "output_fit.png"),
        title=f"CircuitNet (nodes={args.num_nodes}, {args.device_model}) - Friedman #2 output fit",
    )

    full_val_loss = validate(net, val_loader, task_fn, device)
    final_mse, final_rmse, final_mae = compute_orig_metrics(net, val_loader, inverse_stats, device)
    print(
        f"[kn_friedman2] final val {loss_label} = {full_val_loss:.6f} "
        f"(best val = {best_val:.6f} @ epoch {best_epoch})  "
        f"MSE_orig={final_mse:.4f}  RMSE_orig={final_rmse:.4f}  MAE_orig={final_mae:.4f}"
    )
    with open(out_dir / "final_metrics.txt", "w") as f:
        f.write(f"param_count: {n_params}\n")
        f.write(f"best_{loss_metric_key}: {best_val:.6f}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"final_{loss_metric_key}: {full_val_loss:.6f}\n")
        f.write(f"final_mse_orig: {final_mse:.6f}\n")
        f.write(f"final_rmse_orig: {final_rmse:.6f}\n")
        f.write(f"final_mae_orig: {final_mae:.6f}\n")
        f.write(f"epochs_run: {len(history)}\n")
        f.write(f"elapsed_seconds: {elapsed:.2f}\n")

    print(f"[kn_friedman2] artifacts in {out_dir}")


if __name__ == "__main__":
    main()
