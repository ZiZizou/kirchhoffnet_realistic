"""hw_ratio_ab freezes the dead a_raw / b_raw (Phase 3b W5 step 2).

Under ``hw_ratio_ab`` the edge gain uses ``r_raw`` only.  ``a_raw``/``b_raw``
stay in the state_dict (strict loading of ratio checkpoints) but must not be
trainable, so they do not inflate the trainable count (BO param budget) nor
reach the optimizer.
"""

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "golden"))
sys.path.insert(0, os.path.dirname(HERE))

import make_golden as mg  # noqa: E402
from cell_library import make_cell_library  # noqa: E402
from hw import HWConfig  # noqa: E402
from hw.constraints import HWConstraints  # noqa: E402
from topology import build_net_from_config  # noqa: E402

RATIO = HWConfig(constraints=HWConstraints(hw_ratio_ab=True))


def _build(spec, hw=None):
    spec = dict(spec)
    cfg = mg._build_cfg(spec)
    with mg.f64():
        mg.seed_all(int(spec["seed"]))
        cl = {k: v for k, v in spec["cell_lib"].items() if k != "name"}
        cell_lib = make_cell_library(spec["cell_lib"]["name"], **cl, hw=hw)
        kw = dict(spec["build"])
        kw["boundary_fan_out"] = mg._int_keys(kw["boundary_fan_out"])
        net = build_net_from_config(cfg, cell_lib=cell_lib, hw=hw, **kw)
    return net.double()


def _trainable(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def _numel(net, suffix):
    return sum(p.numel() for n, p in net.named_parameters() if n.endswith(suffix))


@pytest.mark.parametrize("spec_name", ["TOY_NET", "FRIEDMAN2_NET"])
def test_trainable_count_is_legacy_minus_a_b_plus_r(spec_name):
    spec = getattr(mg, spec_name)
    legacy = _build(spec)
    ratio = _build(spec, RATIO)
    n_a, n_b, n_r = _numel(ratio, ".a_raw"), _numel(ratio, ".b_raw"), _numel(ratio, ".r_raw")
    assert n_a > 0 and n_a == n_b == n_r
    assert _trainable(ratio) == _trainable(legacy) - n_a - n_b + n_r
    for n, p in ratio.named_parameters():
        if n.endswith(".a_raw") or n.endswith(".b_raw"):
            assert not p.requires_grad, n
        if n.endswith(".r_raw"):
            assert p.requires_grad, n
    # legacy is untouched
    assert all(p.requires_grad for n, p in legacy.named_parameters()
               if n.endswith(".a_raw") or n.endswith(".b_raw"))


def test_frozen_a_b_stay_in_the_state_dict_and_strict_load_works():
    src = _build(mg.TOY_NET, RATIO)
    sd = src.state_dict()
    assert any(k.endswith(".a_raw") for k in sd) and any(k.endswith(".b_raw") for k in sd)
    dst = _build(mg.TOY_NET, RATIO)
    dst.load_state_dict(sd, strict=True)
    # requires_grad survives the load
    assert not any(p.requires_grad for n, p in dst.named_parameters() if n.endswith(".a_raw"))


def test_optimizer_step_leaves_a_b_untouched_and_moves_r():
    net = _build(mg.TOY_NET, RATIO)
    before = {n: p.detach().clone() for n, p in net.named_parameters()}
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    x = mg.make_inputs(dict(mg.TOY_NET))
    net.train()
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        mg.model_forward(net, x).pow(2).mean().backward()
        opt.step()
    for n, p in net.named_parameters():
        if n.endswith(".a_raw") or n.endswith(".b_raw"):
            assert torch.equal(p, before[n]), n
    assert any(not torch.equal(p, before[n]) for n, p in net.named_parameters()
               if n.endswith(".r_raw"))


def test_bo_param_counter_excludes_the_frozen_a_b():
    import bo_param_sampling as bps
    kw = dict(in_dim=4, out_dim=1, fanout_count=2, use_robust_input=False,
              dagger=False, readout_mode="shared_sense", readout_senses=1)
    n_off = bps.knet_param_count(10, 4, 4, 2, **kw)
    assert n_off == 2401                                    # pinned in test_f1_f2
    assert bps.knet_param_count(10, 4, 4, 2, hw=HWConfig(), **kw) == n_off  # off == legacy
    n_ratio = bps.knet_param_count(10, 4, 4, 2, hw=RATIO, **kw)
    # same as a JSON string (the cache key form)
    assert bps.knet_param_count(10, 4, 4, 2, hw=RATIO.to_json(), **kw) == n_ratio
    # one cell-library slot per edge loses a_raw + b_raw and gains r_raw
    assert n_ratio < n_off
    # cross-check against a direct build of the same net
    import cell_library, topology  # noqa: F401
    feas = bps.knet_feasible_arches(soft_limit=n_ratio, in_dim=4, out_dim=1,
                                    hidden_range=(10, 10), stages_range=(4, 4),
                                    k_choices=(4,), rank_range=(2, 2),
                                    readout_mode="shared_sense", hw=RATIO)
    assert (10, 4, 4, 2, 2) in feas
    feas_legacy = bps.knet_feasible_arches(soft_limit=n_ratio, in_dim=4, out_dim=1,
                                           hidden_range=(10, 10), stages_range=(4, 4),
                                           k_choices=(4,), rank_range=(2, 2),
                                           readout_mode="shared_sense")
    assert (10, 4, 4, 2, 2) not in feas_legacy


def test_bo_param_counter_is_legacy_minus_a_b_plus_r(monkeypatch):
    """The BO count obeys the same identity on the net it builds internally."""
    import bo_param_sampling as bps
    import topology
    kw = dict(in_dim=4, out_dim=1, fanout_count=2, use_robust_input=False,
              dagger=False, readout_mode="shared_sense", readout_senses=1)
    n_off = bps.knet_param_count(10, 4, 4, 2, **kw)
    built = {}
    real_build = topology.build_net_from_config

    def spy(*a, **k):
        built["net"] = real_build(*a, **k)
        return built["net"]
    # the counter imports build_net_from_config at call time; bypass its cache
    monkeypatch.setattr(topology, "build_net_from_config", spy)
    n_ratio = bps._knet_param_count_cached.__wrapped__(
        10, 4, 4, 2, 4, 1, 2, False, 0, 0, False, "shared_sense", 1, False,
        False, 4, 2, "boundary,readout", RATIO.to_json(indent=None))
    net = built["net"]
    n_a, n_b, n_r = _numel(net, ".a_raw"), _numel(net, ".b_raw"), _numel(net, ".r_raw")
    assert n_r > 0 and n_ratio == n_off - n_a - n_b + n_r
    assert n_ratio == bps.knet_param_count(10, 4, 4, 2, hw=RATIO, **kw)


def test_pruning_keeps_r_raw():
    from topology import prune_network
    net = _build(mg.FRIEDMAN2_NET, RATIO)      # ota_mesh readout: prunable
    with torch.no_grad():
        for n, p in net.named_parameters():
            if n.endswith(".r_raw"):
                p.copy_(torch.linspace(-1.0, 1.0, p.numel(), dtype=p.dtype))
    r_before = {n: p.detach().clone() for n, p in net.core.named_parameters()
                if n.endswith("cell_lib.r_raw")}
    pruned, _ = prune_network(net.core, edge_threshold=-1.0)   # keep every edge
    # the pruner rebuilds the per-stage core cell library (stages.<i>.cell_lib)
    r_after = {n: p.detach() for n, p in pruned.named_parameters()
               if n.endswith(".cell_lib.r_raw")}
    assert len(r_after) == 2 and set(r_after) <= set(r_before)
    for n, b in r_after.items():
        assert torch.allclose(r_before[n].to(b.dtype), b), n
        assert not torch.all(b == 0), n        # not re-initialised to zero
    assert not any(p.requires_grad for n, p in pruned.named_parameters()
                   if n.endswith(".a_raw") or n.endswith(".b_raw"))
