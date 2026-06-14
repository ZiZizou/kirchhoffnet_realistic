"""Union-graph topology builder for the sparse linear solver.

The sparsity pattern of the dataset's A matrices is accumulated into a single
fixed supergraph. Edges appearing in at least `min_freq` fraction of samples
become hidden graph edges in the KirchhoffNet stage. The resulting topology
is fed through StageTopologyBuilder to add input/output/projection edges
matching the existing pipeline.
"""

from __future__ import annotations

import torch

from topology import (
    SparseTopology,
    StageTopologyBuilder,
    EDGE_TYPE_HIDDEN,
    NODE_KIND_HIDDEN,
)


__all__ = ["build_union_topology"]


def build_union_topology(
    dataset,
    n: int,
    num_proj: int = 4,
    min_freq: float = 0.1,
    edge_threshold: float = 1e-3,
) -> SparseTopology:
    """Build a fixed supergraph covering all likely edges in `dataset`.

    Args:
        dataset: iterable yielding (b, x_star, A) tuples.
        n: matrix dimension (== num_hidden nodes).
        num_proj: number of projection nodes.
        min_freq: keep edges appearing in at least this fraction of samples.
        edge_threshold: |A[i,j]| > threshold counts as an edge.

    Returns:
        SparseTopology ready for topology_to_stage().
    """
    mask = torch.zeros(n, n)
    count = 0
    for _, _, A in dataset:
        mask += (A.abs() > edge_threshold).float()
        count += 1

    threshold = max(1, int(count * min_freq))
    active = mask >= threshold

    src: list[int] = []
    dst: list[int] = []
    for i in range(n):
        for j in range(i + 1, n):
            if active[i, j]:
                src.extend([i, j])
                dst.extend([j, i])

    hid_topo = SparseTopology(
        num_nodes=n,
        src=src,
        dst=dst,
        edge_type=[EDGE_TYPE_HIDDEN] * len(src),
        node_kind=[NODE_KIND_HIDDEN] * n,
        input_node_ids=[],
        output_node_ids=[],
        hidden_node_ids=list(range(n)),
        proj_node_ids=[],
    )

    builder = StageTopologyBuilder(
        num_inputs=n,
        num_outputs=0,
        num_hidden=n,
        num_proj=num_proj,
    )
    return builder.build(
        hid_topo,
        input_pattern="all_to_all",
        output_pattern="all_to_all",
        proj_pattern="all_to_all",
    )
