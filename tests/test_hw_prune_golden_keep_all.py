"""Golden-builder keep-all pruning coverage for stage-owned hardware cells."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "golden"))
import make_golden as mg
from topology import prune_network


@pytest.mark.parametrize("spec", [mg.FRIEDMAN2_NET, mg.TOY_NET], ids=["friedman2-ota-mesh", "toy-shared-sense"])
def test_golden_builder_keep_all_pruning_preserves_params_and_forward(spec):
    torch.set_num_threads(2)
    with mg.f64():
        net = mg._build_net(spec).double().eval()
        x = mg._friedman2_xy(16, 23)[0] if spec is mg.FRIEDMAN2_NET else mg._toy_xy(16, 33)[0]
        with torch.no_grad():
            before = mg.model_forward(net, x)
        old_core = net.core
        old_count = sum(p.numel() for p in old_core.parameters())
        old_families = {
            family: sum(p.numel() for name, p in old_core.named_parameters() if family in name)
            for family in ("boundary", "output_ode", "readout_sense", "vca", "gln")
        }

        pruned, _ = prune_network(old_core, edge_threshold=-1.0)
        new_count = sum(p.numel() for p in pruned.parameters())
        new_families = {
            family: sum(p.numel() for name, p in pruned.named_parameters() if family in name)
            for family in old_families
        }
        net.core = pruned
        with torch.no_grad():
            after = mg.model_forward(net, x)

    assert new_count == old_count
    assert new_families == old_families
    torch.testing.assert_close(after, before, rtol=0, atol=0)
