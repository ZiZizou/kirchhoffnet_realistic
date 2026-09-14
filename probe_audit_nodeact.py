"""Node-activation + spectral-norm + isolation audit (narma-node-activation).

Lightweight CPU checks (seconds to ~2 min total):

[N1] Epoch-0 identity: ``node_activation='tanh'`` matches ``'none'`` to
     1e-5 abs at small signal (tanh(x) ~= x); ``'identity'`` matches
     exactly (0.0); ``state_dict`` key sets identical (no new params).
[N2] Builder guard: ``node_activation='tanh'`` without
     ``allow_experimental_cells`` raises ValueError; with the guard the
     NARMA build succeeds and the stage carries the flag; default builds
     carry ``'none'``; bogus values raise in both stage and builder.
[N3] Factory closure (narma-cell-isolation): ``make_cell_library`` rejects
     unknown/future cell names (documents the closed registry).
[N4] Spectral hook: report keys present; multi-pass converges from the
     gm=-1 operating point to within 0.05 of 0.95; no-op when already at
     target (scale 1.0, gm_raw untouched); device field resolves per
     stage (``cpu`` locally; ``auto`` follows parameters on CUDA).
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import torch

import narma_experiment as ne  # noqa: E402
import topology as topo  # noqa: E402
from cell_library import make_cell_library  # noqa: E402
from differential_stage import DifferentialStage  # noqa: E402
from cell_library import FreeTanhLibrary  # noqa: E402
from narma_spectral_norm import (  # noqa: E402
    normalize_spectral_radius,
    _stage_device,
)


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    raise SystemExit(2)


def _tiny_stage(act: str) -> DifferentialStage:
    torch.manual_seed(0)
    lib = FreeTanhLibrary(num_edges=6)
    return DifferentialStage(
        num_nodes=4, src=[0, 1, 2, 0, 1, 2], dst=[1, 2, 3, 2, 3, 3],
        cell_lib=lib, x_max=3.0, node_activation=act,
    )


def audit_n1_epoch0_identity() -> None:
    a, b, c = _tiny_stage("none"), _tiny_stage("tanh"), _tiny_stage("identity")
    b.load_state_dict(a.state_dict())
    c.load_state_dict(a.state_dict())
    x = 0.05 * torch.randn(2, 4)
    ra, rb, rc = a.rhs(x), b.rhs(x), c.rhs(x)
    d_tanh = float((ra - rb).detach().abs().max())
    d_ident = float((ra - rc).detach().abs().max())
    if d_tanh < 1e-5:
        _ok(f"[N1a] tanh small-signal identity: maxdiff {d_tanh:.2e} < 1e-5")
    else:
        _fail(f"[N1a] tanh small-signal drift too large: {d_tanh:.2e}")
    if d_ident == 0.0:
        _ok("[N1b] identity is an exact no-op (diff 0.0)")
    else:
        _fail(f"[N1b] identity mismatch: {d_ident:.2e}")
    if set(a.state_dict()) == set(b.state_dict()):
        _ok("[N1c] state_dict keys identical (no new parameters)")
    else:
        _fail("[N1c] state_dict key mismatch")


def audit_n2_builder_guard() -> None:
    import config as _C
    cfg = _C.make_narma_preset(order=10, hidden_dim=25)
    lib = make_cell_library("tanh_free", num_edges=8)
    cfg["node_activation"] = "tanh"
    try:
        topo.build_net_from_config(cfg, lib)
        _fail("[N2a] guard did not raise for unguarded experimental key")
    except ValueError as e:
        if "experimental" in str(e).lower():
            _ok("[N2a] unguarded node_activation='tanh' raises isolation guard")
        else:
            _fail(f"[N2a] wrong ValueError (guard bypassed?): {e}")
    # End-to-end through the real NARMA path (sole guard holder).
    net = ne._build_fabric_net(
        order=10, seed=0, freeze_read=False, t_span=1.0, num_steps=8,
        cell_library="tanh_free", core_refresh_interval=0,
        leak_constant=None, compile_sequence=False, hidden_dim=25,
        node_activation="tanh",
    )[0]
    if net.core.stages[0].node_activation == "tanh":
        _ok("[N2b] NARMA path carries node_activation='tanh'")
    else:
        _fail("[N2b] NARMA path lost the flag")
    net0 = ne._build_fabric_net(
        order=10, seed=0, freeze_read=False, t_span=1.0, num_steps=8,
        cell_library="tanh_free", core_refresh_interval=0,
        leak_constant=None, compile_sequence=False, hidden_dim=25,
    )[0]
    if net0.core.stages[0].node_activation == "none":
        _ok("[N2c] default build carries node_activation='none'")
    else:
        _fail("[N2c] default build mutated")
    try:
        _tiny_stage("bogus")
        _fail("[N2d] stage accepted 'bogus'")
    except ValueError:
        _ok("[N2d] stage rejects bad node_activation")


def audit_n3_factory_closure() -> None:
    for name in ("linear_ota", "dead_zone_ota", "asymmetric_ota", "lif"):
        try:
            make_cell_library(name, num_edges=4)
            _fail(f"[N3] factory accepted {name!r} (registry not closed)")
        except ValueError:
            pass
    _ok("[N3] factory rejects future NARMA cell names (registry closed)")


def audit_n4_spectral_hook() -> None:
    net, ts, ns = ne._build_fabric_net(
        order=10, seed=0, freeze_read=False, t_span=1.0, num_steps=8,
        cell_library="tanh_free", core_refresh_interval=0,
        leak_constant=0.1, compile_sequence=False, hidden_dim=25,
        node_activation="tanh", spectral_radius_target=None,
    )
    if _stage_device(net.core.stages[0]) != "cpu":
        _fail("[N4a] _stage_device misresolved on CPU")
    _ok("[N4a] _stage_device resolves per-stage parameter device")
    with torch.no_grad():
        for st in net.core.stages:
            for ln in ("cell_lib", "boundary_cell_lib", "output_ode_cell_lib"):
                lib = getattr(st, ln, None)
                if lib is None:
                    continue
                if hasattr(lib, "gm_raw"):
                    lib.gm_raw.data.fill_(-1.0)
                if hasattr(lib, "isat_raw"):
                    lib.isat_raw.data.fill_(-5.0)
    rep = normalize_spectral_radius(
        net, target_sr=0.95, t_span=ts, num_steps=ns, max_passes=15,
    )["stages"][0]
    for key in ("sr_before", "sr_after", "scale_applied", "gm_shift",
                "libs_touched", "passes", "note", "device"):
        if key not in rep:
            _fail(f"[N4b] report missing key {key!r}")
    _ok("[N4b] report carries all contract keys")
    if abs(rep["sr_after"] - 0.95) <= 0.05:
        _ok(f"[N4c] multi-pass converges: {rep['sr_before']:.3f} -> "
            f"{rep['sr_after']:.3f} in {rep['passes']} passes")
    else:
        _fail(f"[N4c] no convergence: sr_after={rep['sr_after']:.3f}")
    raw_before = float(net.core.stages[0].cell_lib.isat_raw.flatten()[0].item())
    gm_before = float(net.core.stages[0].cell_lib.gm_raw.flatten()[0].item())
    rep2 = normalize_spectral_radius(
        net, target_sr=rep["sr_after"], t_span=ts, num_steps=ns, tol=0.05,
    )["stages"][0]
    raw_after = float(net.core.stages[0].cell_lib.isat_raw.flatten()[0].item())
    gm_after = float(net.core.stages[0].cell_lib.gm_raw.flatten()[0].item())
    if abs(rep2["scale_applied"] - 1.0) < 1e-9 and raw_after == raw_before \
            and gm_after == gm_before:
        _ok("[N4d] at-target call is a no-op (scale 1.0, raws untouched)")
    else:
        _fail("[N4d] at-target call mutated weights")


def main() -> int:
    print("=== node-activation audit ===")
    audit_n1_epoch0_identity()
    audit_n2_builder_guard()
    audit_n3_factory_closure()
    audit_n4_spectral_hook()
    print("\nALL_AUDITS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
