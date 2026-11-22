import pytest
import torch

from hw import HWConfig, HWConstraints
from hw.constraints import cap_divider_gain, effective_ab_values
from cell_library import FreeTanhLibrary


def test_cap_divider_levels_range_and_ste_gradient():
    probs = torch.tensor([.01, .15, .30, .45, .60, .75, .90, .99])
    raw = torch.logit(probs).requires_grad_()
    gain = cap_divider_gain(raw, bits=3, r_max=2)
    assert bool(((gain >= .5) & (gain <= 1)).all())
    assert torch.unique(gain).numel() <= 8
    assert torch.allclose(torch.sort(torch.unique(gain)).values,
                          torch.tensor([.5, .5714286, .6428571, .7142857, .7857143, .8571429, .9285714, 1.]))
    gain.sum().backward()
    assert raw.grad is not None and bool((raw.grad > 0).all())


def test_differential_exact_unity_and_dead_legacy_params():
    hw = HWConfig(constraints=HWConstraints(hw_ab_mode="differential", hw_slope_bound=True))
    lib = FreeTanhLibrary(4, hw=hw)
    a, b = effective_ab_values(lib)
    assert torch.equal(a, torch.ones(4)) and torch.equal(b, torch.ones(4))
    assert not lib.a_raw.requires_grad and not lib.b_raw.requires_grad


def test_modes_require_slope_and_legacy_alias_conflict():
    with pytest.raises(ValueError, match="requires hw_slope_bound"):
        HWConstraints(hw_ab_mode="cap_divider")
    with pytest.raises(ValueError, match="conflicts"):
        HWConstraints(hw_ratio_ab=True, hw_ab_mode="differential", hw_slope_bound=True)
    assert HWConstraints(hw_ratio_ab=True).hw_ratio_ab


def test_bo_param_counter_builds_each_ab_mode():
    from bo_param_sampling import knet_param_count
    kwargs = dict(in_dim=4, out_dim=1, fanout_count=1, use_robust_input=False,
                  dagger=False, readout_mode="shared_sense", readout_senses=1)
    counts = {}
    for mode in ("ratio", "differential", "cap_divider"):
        hw = HWConfig(constraints=HWConstraints(hw_ab_mode=mode, hw_slope_bound=True))
        counts[mode] = knet_param_count(8, 1, 2, 1, hw=hw, **kwargs)
    assert counts["cap_divider"] > counts["ratio"] > counts["differential"]


def test_slope_bound_for_cap_divider():
    hw = HWConfig(constraints=HWConstraints(hw_ab_mode="cap_divider", hw_slope_bound=True))
    lib = FreeTanhLibrary(3, hw=hw)
    a, b = lib.ab_values()
    slope = lib.gm_eff_values() * torch.maximum(a, b)
    assert bool((slope <= hw.units.slope_max).all())
