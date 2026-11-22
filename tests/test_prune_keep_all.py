"""Keep-all pruning regression/reproduction for the Friedman-2 preset."""
import torch

from cell_library import make_cell_library
from config import PRESETS
from topology import build_net_from_preset, prune_network


def test_friedman2_prune_keep_all_preserves_core_params_and_forward():
    assert PRESETS["friedman2"]["schedule"] == "three_phase"
    net = build_net_from_preset(
        "friedman2", make_cell_library("tanh_free"), freeze_read=True)
    core = net.core
    x = net.input_mapper(torch.randn(8, 4))
    core.eval()
    before = core(x, store_trajectory=False)[0]
    pruned, _ = prune_network(
        core, edge_threshold=-1.0, write_idx=net.write_idx,
        read_idx=net.read_idx)
    after = pruned(x, store_trajectory=False)[0]
    assert sum(p.numel() for p in core.parameters()) == sum(
        p.numel() for p in pruned.parameters())
    torch.testing.assert_close(after, before, rtol=0, atol=0)
