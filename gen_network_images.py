"""Generate network visualization images for all 3 task presets.

Saves to /home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/network_visualization/
"""
import sys
sys.path.insert(0, '/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal')

import os
import torch
from cell_library import make_cell_library
from topology import build_net_from_preset, MultiStageTopology
from sim_context import SimContext
from visualize import (
    plot_sparse_topology,
    plot_stage_graph,
    plot_multi_stage_topology,
    plot_trajectories,
    plot_output_fit,
    plot_network,
)
from config import PRESETS

OUT_DIR = '/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/network_visualization'
os.makedirs(OUT_DIR, exist_ok=True)


def gen_for_preset(name: str) -> None:
    print(f'\n{"=" * 60}\n  Preset: {name}\n{"=" * 60}')
    cell_lib = make_cell_library('tanh')
    net = build_net_from_preset(name, cell_lib=cell_lib)

    # 1. SparseTopology of each stage (raw graph with I/O nodes)
    multi_topo = MultiStageTopology.from_config(PRESETS[name]['stages'])
    for i, topo in enumerate(multi_topo.stages):
        path = os.path.join(OUT_DIR, f'{name}_stage{i + 1}_sparse_topology.png')
        stage = net.core.stages[i] if i < len(net.core.stages) else None
        plot_sparse_topology(topo, save_path=path, title=f'{name} — Stage {i + 1} SparseTopology',
                             stage=stage)
        print(f'  saved {path}')

    # 2. Stage graph (post-filter, ODE-active nodes only)
    for i, stage in enumerate(net.core.stages):
        path = os.path.join(OUT_DIR, f'{name}_stage{i + 1}_stage_graph.png')
        plot_stage_graph(stage, save_path=path, title=f'{name} — Stage {i + 1} ODE core')
        print(f'  saved {path}')

    # 3. Multi-stage topology
    path = os.path.join(OUT_DIR, f'{name}_multi_stage_topology.png')
    plot_multi_stage_topology(multi_topo, save_path=path, title=f'{name} — Multi-stage topology')
    print(f'  saved {path}')

    # 4. Forward pass + trajectory + output fit
    n_in = PRESETS[name]['stages'][0]['num_inputs']
    batch_size = 64
    u = torch.randn(batch_size, n_in)
    ctx = SimContext()
    with torch.no_grad():
        y, trajs = net(u, ctx=ctx, store_trajectory=True)

    # Trajectory per stage
    if isinstance(trajs, list):
        for i, t in enumerate(trajs):
            path = os.path.join(OUT_DIR, f'{name}_stage{i + 1}_trajectories.png')
            plot_trajectories(t, stage_idx=i, save_path=path,
                              title=f'{name} — Stage {i + 1} trajectories')
            print(f'  saved {path}')

    # 5. Fit plot
    if name == 'sinx':
        y_true = torch.sin(u)
    elif name == 'xor':
        u_xor = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        ctx2 = SimContext()
        with torch.no_grad():
            y, _ = net(u_xor, ctx=ctx2, store_trajectory=True)
        y_true = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
        batch_size = 4
    else:  # housing
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        X = torch.tensor(data.data[:64], dtype=torch.float32)
        y_true = torch.tensor(data.target[:64], dtype=torch.float32).unsqueeze(1)
        X = X / X.max(dim=0, keepdim=True).values.clamp(min=1e-6)
        with torch.no_grad():
            y, _ = net(X, ctx=ctx, store_trajectory=True)

    path = os.path.join(OUT_DIR, f'{name}_output_fit.png')
    plot_output_fit(y, y_true, loss_name=PRESETS[name]['loss'], save_path=path,
                    title=f'{name} — Output fit ({PRESETS[name]["loss"]})')
    print(f'  saved {path}')

    # 6. Full pipeline
    path = os.path.join(OUT_DIR, f'{name}_pipeline.png')
    plot_network(net, save_path=path)
    print(f'  saved {path}')

    print(f'  done: {name}  ({net!r})')


if __name__ == '__main__':
    for name in sorted(PRESETS):
        gen_for_preset(name)
    print(f'\nAll images saved to: {OUT_DIR}')
