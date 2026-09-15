"""Fast invariants for the Phase-8 tied causal tap-gate mechanism."""
from __future__ import annotations

import copy
import torch

from narma_phase7_leak_bands import (
    BAND_GROUPS, BAND_LEAKS, HIDDEN_DIM, N_STEPS, N_TAPS, T_SPAN,
    build_base, set_leaks,
)
from tap_rails import TapRails, tap_rails_param_count


def _band_values() -> torch.Tensor:
    values = torch.empty(HIDDEN_DIM)
    for band, nodes in BAND_GROUPS.items():
        values[nodes] = BAND_LEAKS[band]
    return values


def _stage_with_banded_leaks(seed: int = 0):
    net = build_base(seed).cpu().eval()
    set_leaks(net.core.stages[0], _band_values())
    return net.core.stages[0]


def main() -> None:
    torch.manual_seed(31)
    static = _stage_with_banded_leaks()
    gated = copy.deepcopy(static)
    gated.tap_rails = TapRails(N_TAPS, HIDDEN_DIM, B=2)
    assert tap_rails_param_count(B=2) == 95
    assert sum(p.numel() for p in gated.tap_rails.parameters()) == 95

    x0, taps = torch.randn(3, HIDDEN_DIM), torch.randn(3, N_TAPS)
    with torch.no_grad():
        y_static, _ = static(x0, t_span=T_SPAN, num_steps=N_STEPS, u=taps)
        y_gated, _ = gated(x0, t_span=T_SPAN, num_steps=N_STEPS, u=taps)
    assert torch.equal(y_static, y_gated), "identity gate changed the Phase-8 forward"
    z, m = gated.tap_rails.gate_and_multiplier(taps)
    assert torch.equal(z, torch.full_like(z, 0.5))
    assert torch.equal(m, torch.ones_like(m))
    print("IDENTITY_FORWARD_EXACT_OK")

    # The nonzero rail weights make w receive a gradient at the identity init.
    y, _ = gated(x0, t_span=T_SPAN, num_steps=N_STEPS, u=taps)
    y.square().mean().backward()
    assert gated.tap_rails.w.grad is not None
    assert gated.tap_rails.w.grad.abs().max().item() > 0
    print("IDENTITY_W_GRADIENT_OK")

    with torch.no_grad():
        gated.tap_rails.w.normal_(std=0.1)
    _, random_m = gated.tap_rails.gate_and_multiplier(taps)
    assert not torch.equal(random_m, torch.ones_like(random_m))
    print("RANDOM_GATE_CHANGES_MULTIPLIER_OK")
    print("ALL_PHASE8_TIED_GATE_AUDITS_OK")


if __name__ == "__main__":
    main()
