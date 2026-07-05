"""Network visualization utilities for the reduced differential KirchhoffNet.

All functions lazily import matplotlib and networkx so that this module is
importable even if those packages are not installed. The first call to any
plot_* function will raise a clear ImportError if matplotlib or networkx is
missing; install them with:

    uv pip install matplotlib networkx

Conventions:
  - matplotlib backend is forced to 'Agg' (headless / no display required)
  - figures are closed via plt.close() after each save_* call
  - color scheme matches SparseTopology: tab:green=input, tab:blue=hidden,
    tab:orange=proj, tab:red=output; edges inherit their endpoint color
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from differential_stage import DifferentialStage
    from topology import MultiStageTopology, SparseTopology
    from kirchhoff_net import KirchhoffNetWithIO


_matplotlib = None
_networkx = None


_NODE_COLORS = {
    "input": "tab:green",
    "hidden": "tab:blue",
    "proj": "tab:orange",
    "output": "tab:red",
}




def _ensure_matplotlib():
    """Lazy-import matplotlib with Agg backend. Idempotent."""
    global _matplotlib
    if _matplotlib is None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            _matplotlib = _plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install with: uv pip install matplotlib"
            ) from e
    return _matplotlib


def _ensure_networkx():
    """Lazy-import networkx. Idempotent."""
    global _networkx
    if _networkx is None:
        try:
            import networkx as _nx
            _networkx = _nx
        except ImportError as e:
            raise ImportError(
                "networkx is required for graph layout. "
                "Install with: uv pip install networkx"
            ) from e
    return _networkx


def _layout(G, layout: str, seed: int):
    """Compute node positions using a networkx layout algorithm."""
    nx = _ensure_networkx()
    if layout == "spring":
        return nx.spring_layout(G, seed=seed, k=0.8, iterations=50)
    if layout == "circular":
        return nx.circular_layout(G)
    if layout == "kamada_kawai":
        try:
            return nx.kamada_kawai_layout(G)
        except Exception:
            return nx.spring_layout(G, seed=seed)
    if layout == "shell":
        return nx.shell_layout(G)
    raise ValueError(f"Unknown layout: {layout!r}")


def _save(fig, save_path: str | None) -> None:
    """Save figure to disk if path given, then close."""
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    _ensure_matplotlib().close(fig)


# ---------- topology graph ----------

def plot_sparse_topology(
    topo: "SparseTopology",
    ax=None,
    title: str | None = None,
    layout: str = "spring",
    seed: int = 42,
    save_path: str | None = None,
    show_edge_types: bool = True,
    stage: "DifferentialStage | None" = None,
):
    """Plot a SparseTopology as a directed graph.

    Nodes are colored by kind (input/hidden/proj/output) and edges by type
    (input/hidden/proj/output) when show_edge_types=True.

    Input/output edges remain dashed.

    Returns (fig, ax).
    """
    plt = _ensure_matplotlib()
    nx = _ensure_networkx()

    G = nx.DiGraph()
    G.add_nodes_from(range(topo.num_nodes))

    edge_colors = []
    edge_styles = []
    edge_widths = []
    for s, d, t in zip(topo.src, topo.dst, topo.edge_type):
        G.add_edge(s, d)
        c = _NODE_COLORS.get(t, "gray")
        ew = 1.0
        edge_colors.append(c)
        edge_styles.append("-" if t in ("hidden", "proj") else "--")
        edge_widths.append(ew)

    pos = _layout(G, layout, seed)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    for kind, color in _NODE_COLORS.items():
        ids = [i for i, k in enumerate(topo.node_kind) if k == kind]
        if not ids:
            continue
        nx.draw_networkx_nodes(
            G, pos, nodelist=ids, node_color=[color] * len(ids),
            node_size=350, edgecolors="black", linewidths=0.6, ax=ax,
            label=f"{kind} ({len(ids)})",
        )

    for etype in ("hidden", "proj", "input", "output"):
        idx = [i for i, t in enumerate(topo.edge_type) if t == etype]
        if not idx:
            continue
        elist = [(topo.src[i], topo.dst[i]) for i in idx]
        ecolors = [edge_colors[i] for i in idx]
        ewidths = [edge_widths[i] for i in idx]
        estyle = "-" if etype in ("hidden", "proj") else "--"
        nx.draw_networkx_edges(
            G, pos, edgelist=elist, edge_color=ecolors, style=estyle,
            arrows=True, arrowsize=10, width=ewidths, alpha=0.7, ax=ax,
        )

    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

    ax.set_title(title or f"SparseTopology: {topo.num_nodes} nodes, {topo.num_edges()} edges")
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    leg = ax.get_legend()

    _save(fig, save_path)
    return fig, ax


def plot_stage_graph(
    stage: "DifferentialStage",
    ax=None,
    title: str | None = None,
    layout: str = "spring",
    seed: int = 42,
    save_path: str | None = None,
):
    """Plot a DifferentialStage's COO graph.

    Note: by the time a SparseTopology is converted to a DifferentialStage,
    input/output nodes and edges have been filtered out. So all nodes shown
    are active (hidden+proj in the original topology) and all edges are
    hidden or proj type.
    """
    plt = _ensure_matplotlib()
    nx = _ensure_networkx()

    src_list = stage.src.tolist() if hasattr(stage.src, "tolist") else list(stage.src)
    dst_list = stage.dst.tolist() if hasattr(stage.dst, "tolist") else list(stage.dst)

    G = nx.DiGraph()
    G.add_nodes_from(range(stage.num_nodes))
    for s, d in zip(src_list, dst_list):
        G.add_edge(s, d)

    pos = _layout(G, layout, seed)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    nx.draw_networkx_nodes(
        G, pos, node_color="tab:blue", node_size=350,
        edgecolors="black", linewidths=0.6, ax=ax,
        label=f"active ({stage.num_nodes})",
    )
    nx.draw_networkx_edges(
        G, pos, edge_color="tab:gray", arrows=True, arrowsize=10,
        width=1.0, alpha=0.7, ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

    n_edges = len(src_list)
    ax.set_title(title or f"Stage: {stage.num_nodes} active nodes, {n_edges} edges")
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    _save(fig, save_path)
    return fig, ax


def plot_multi_stage_topology(
    multi: "MultiStageTopology",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    layout: str = "spring",
    seed: int = 42,
    save_path: str | None = None,
):
    """Plot all stages in a MultiStageTopology as side-by-side subplots."""
    plt = _ensure_matplotlib()
    n = len(multi)
    if figsize is None:
        figsize = (5.5 * n, 5.5)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for i, topo in enumerate(multi.stages):
        plot_sparse_topology(
            topo, ax=axes[i], title=f"Stage {i + 1}",
            layout=layout, seed=seed + i,
        )

    if title:
        fig.suptitle(title, fontsize=12)
    _save(fig, save_path)
    return fig, axes


# ---------- trajectories ----------

def plot_trajectories(
    trajs: torch.Tensor,
    stage_idx: int = 0,
    ax=None,
    title: str | None = None,
    save_path: str | None = None,
    downsample: int = 1,
    max_nodes: int = 16,
):
    """Plot node voltage trajectories over the integration horizon.

    Args:
        trajs: Tensor of shape [batch, num_nodes, num_steps + 1] (from
            DifferentialStage.forward with store_trajectory=True).
        stage_idx: Stage index, used only in default title.
        downsample: Plot every k-th time step.
        max_nodes: If num_nodes > max_nodes, only plot the first max_nodes
            to keep the figure legible.
    """
    plt = _ensure_matplotlib()

    if trajs.dim() == 2:
        trajs = trajs.unsqueeze(0)
    if trajs.dim() != 3:
        raise ValueError(
            f"trajs must be [batch, nodes, time] or [nodes, time], got shape {tuple(traj.shape)}"
        )

    x = trajs.detach().mean(dim=0)
    n_nodes, n_steps = x.shape
    n_nodes = min(n_nodes, max_nodes)
    t = torch.arange(n_steps)[::downsample].float()
    x = x[:n_nodes, ::downsample].cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure

    cmap = plt.cm.viridis
    for i in range(n_nodes):
        c = cmap(i / max(n_nodes - 1, 1))
        ax.plot(t.numpy(), x[i], color=c, linewidth=1.0, alpha=0.85,
                label=f"node {i}" if n_nodes <= 8 else None)

    ax.set_xlabel("integration step")
    ax.set_ylabel("node voltage")
    ax.set_title(title or f"Stage {stage_idx} trajectories ({n_nodes} nodes, {n_steps} steps)")
    ax.grid(True, alpha=0.3)
    if n_nodes <= 8:
        ax.legend(fontsize=8, loc="best")

    _save(fig, save_path)
    return fig, ax


# ---------- cell library selection ----------

# ---------- output fit ----------

def plot_output_fit(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    loss_name: str = "loss",
    ax=None,
    title: str | None = None,
    save_path: str | None = None,
):
    """Scatter plot of predicted vs target with diagonal + residual subplot.

    Returns (fig, (ax_fit, ax_resid)).
    """
    plt = _ensure_matplotlib()

    yp = y_pred.detach().cpu().flatten().numpy()
    yt = y_true.detach().cpu().flatten().numpy()
    resid = yp - yt

    fig, (ax_fit, ax_resid) = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    lo = float(min(yp.min(), yt.min()))
    hi = float(max(yp.max(), yt.max()))
    ax_fit.scatter(yt, yp, s=18, alpha=0.65, color="tab:blue", edgecolors="none")
    ax_fit.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    ax_fit.set_xlabel("target")
    ax_fit.set_ylabel("prediction")
    ax_fit.set_title(title or f"Output fit ({loss_name})")
    ax_fit.set_xlim(lo, hi)
    ax_fit.set_ylim(lo, hi)
    ax_fit.set_aspect("equal", adjustable="box")
    ax_fit.grid(True, alpha=0.3)
    ax_fit.legend(fontsize=8)

    ax_resid.scatter(yt, resid, s=18, alpha=0.65, color="tab:red", edgecolors="none")
    ax_resid.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax_resid.set_xlabel("target")
    ax_resid.set_ylabel("prediction - target")
    ax_resid.set_title("Residuals")
    ax_resid.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, save_path)
    return fig, (ax_fit, ax_resid)


# ---------- convenience ----------

def plot_network(
    net: "KirchhoffNetWithIO",
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
):
    """Convenience: plot the full multi-stage pipeline of a KirchhoffNetWithIO.

    Includes annotation boxes describing the input/output mapper dimensions.
    """
    from topology import MultiStageTopology

    plt = _ensure_matplotlib()
    stages = net.core.stages
    n = len(stages)
    if figsize is None:
        figsize = (5.5 * n + 2, 6.0)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, n + 2, width_ratios=[1.2] + [1.0] * n + [1.2], height_ratios=[3, 0.4])
    axes = [fig.add_subplot(gs[0, i + 1]) for i in range(n)]

    nx = _ensure_networkx()
    multi = MultiStageTopology([
        _stage_to_sparse_topology(s, n_idx=i, n_stages=n) for i, s in enumerate(stages)
    ])

    for i, topo in enumerate(multi.stages):
        plot_sparse_topology(topo, ax=axes[i], title=f"Stage {i + 1}", seed=42 + i, stage=stages[i])

    in_ax = fig.add_subplot(gs[0, 0]); in_ax.axis("off")
    out_ax = fig.add_subplot(gs[0, n + 1]); out_ax.axis("off")
    in_ax.text(
        0.5, 0.5, f"InputMapper\nin → x0", ha="center", va="center",
        fontsize=10, bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
    )
    out_ax.text(
        0.5, 0.5, f"OutputMapper\ny → ŷ", ha="center", va="center",
        fontsize=10, bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5),
    )

    for i in range(n - 1):
        arrow_ax = fig.add_subplot(gs[1, i + 1]); arrow_ax.axis("off")
        arrow_ax.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5),
                           arrowprops=dict(arrowstyle="->", lw=1.5))
    fig.suptitle("KirchhoffNetWithIO pipeline", fontsize=12)
    _save(fig, save_path)
    return fig, axes


def _stage_to_sparse_topology(stage: "DifferentialStage", n_idx: int, n_stages: int):
    """Reconstruct a minimal SparseTopology from a DifferentialStage for plotting.

    Marks all nodes as 'hidden' (since input/output edges are filtered out by
    topology_to_stage). Use plot_sparse_topology directly on the original
    SparseTopology to see input/output nodes.
    """
    from topology import (
        SparseTopology, EDGE_TYPE_HIDDEN, NODE_KIND_HIDDEN,
    )

    src_list = stage.src.tolist() if hasattr(stage.src, "tolist") else list(stage.src)
    dst_list = stage.dst.tolist() if hasattr(stage.dst, "tolist") else list(stage.dst)
    return SparseTopology(
        num_nodes=stage.num_nodes,
        src=src_list,
        dst=dst_list,
        edge_type=[EDGE_TYPE_HIDDEN] * len(src_list),
        node_kind=[NODE_KIND_HIDDEN] * stage.num_nodes,
        hidden_node_ids=list(range(stage.num_nodes)),
    )
