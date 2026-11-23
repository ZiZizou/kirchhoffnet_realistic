"""Structural coverage for hardware-mode copied Spectre netlists."""
import copy
import pytest
import torch

from export import spectre_export
from export.spectre_netlist import render
from hw import HWConfig, load_preset
from tests.golden.make_golden import TOY_NET, CtleStudent, _build_net, seed_all


def _toy_model(preset):
    cfg = HWConfig.from_dict(copy.deepcopy(load_preset(preset)))
    seed_all(20261122)
    net = _build_net(copy.deepcopy(TOY_NET), hw=cfg).double()
    model = CtleStudent(net, 4, 3, 4.0).double().eval()
    gen = torch.Generator().manual_seed(7319)
    with torch.no_grad():
        for _, param in sorted(model.named_parameters()):
            param.add_(0.015 * torch.randn(param.shape, generator=gen, dtype=param.dtype))
    return model, cfg


@pytest.mark.parametrize("preset,expected", [
    ("phase3a_core", "ab_mode=3"),
    ("phase3a_core_diff", "ab_mode=2"),
])
def test_hw_core_netlist_structural_mapping(tmp_path, preset, expected):
    model, cfg = _toy_model(preset)
    params = spectre_export.export_model(model, tmp_path, mode="hw", fabric="copies", specs=1, hw=cfg)
    text = render(params, tmp_path / "hw.scs", inputs=[[0.08, 0.2, 0.7, 1.4]])
    assert "ota_edge_sampled" in text
    assert "mode=1" in text and expected in text
    assert "floor_term=0" in text
    if preset == "phase3a_core":
        assert "cap_code_a=" in text and "cap_code_b=" in text


@pytest.mark.slow
@pytest.mark.spectre
def test_hw_core_spectre_compiles_and_matches_interpreter(tmp_path):
    """Run one core hardware deck when tcsh, kit and Spectre are installed."""
    import os
    import re
    import shutil
    import subprocess
    from pathlib import Path
    import numpy as np
    from export.netlist_interp import simulate
    from scripts.spectre.characterize import psf_waveforms

    tcsh = shutil.which("tcsh")
    spectre = shutil.which("spectre")
    setup = Path.home() / "workarea_GF22_FDX_EXT/kit.gf22fdx_ext.v1.0_4.1_stack19_oa.csh"
    if not tcsh or not setup.exists():
        pytest.skip("tcsh / GF22 Spectre kit unavailable")
    probe = subprocess.run([tcsh, "-c", f"source {setup} >& /dev/null; which spectre"],
                           text=True, capture_output=True)
    if probe.returncode or "spectre" not in probe.stdout.lower():
        pytest.skip("Spectre executable unavailable")

    run = Path("/home/annaik/Documents/kirchhoffnet_refactor/.agent_scratch/hwnet/pytest_hw")
    run.mkdir(parents=True, exist_ok=True)
    model, cfg = _toy_model("phase3a_core")
    params = spectre_export.export_model(model, run, mode="hw", fabric="copies", specs=1, hw=cfg)
    x = torch.tensor([[0.08, 0.2, 0.7, 1.4]], dtype=torch.float64)
    render(params, run / "hrknet.scs", inputs=x)
    command = (f"cd {run}; source {setup} >& /dev/null; "
               "spectre -format psfascii hrknet.scs +log hrknet.log")
    proc = subprocess.run([tcsh, "-c", command], cwd=run, text=True, capture_output=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    log = (run / "hrknet.log").read_text(errors="replace")
    assert re.search(r"spectre completes with 0 errors", log), log[-2000:]
    ref = simulate(params, x, return_trajectory=True, steps_per_stage=1000)
    traces = psf_waveforms(run / "hrknet.raw" / "tran.tran.tran")
    vu, vcm, tu = (params["units"][k] for k in ("Vu_V", "Vcm_V", "Tu_s"))
    offset = 0.0
    max_error = 0.0
    for si, stage in enumerate(params["resolved"]["stages"]):
        span = float(stage["solver"]["t_span"])
        for node in range(int(stage["num_nodes"])):
            vals = traces[f"n0_{si}_{node}"]
            ts = np.asarray([v[0] for v in vals])
            got = (np.asarray([v[1] for v in vals]) - vcm) / vu
            mask = (ts >= offset - 1e-14) & (ts <= offset + span * tu + 1e-14)
            want = np.interp((ts[mask] - offset) / tu,
                             ref["stage_times"][si].cpu().numpy(),
                             ref["stage_trajectories"][si][0, :, node].cpu().numpy())
            if want.size:
                max_error = max(max_error, float(np.max(np.abs(got[mask] - want))))
        offset += span * tu
    assert max_error < 1e-4, f"max hardware core trajectory error {max_error:.6g} x"
