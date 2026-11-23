"""Structural regressions for copied-fabric Spectre deck generation."""
import re
import pytest

from checkpoints import DEFAULT_CKPT
from export.spectre_export import export

def test_paper_copies_netlist_has_expected_instance_counts(tmp_path):
    export(DEFAULT_CKPT, tmp_path, mode="legacy", fabric="copies", specs=1)
    text = (tmp_path / "hrknet.scs").read_text()
    lines = text.splitlines()
    assert sum(bool(re.match(r"q0_s\d+_e\d+ ", line)) for line in lines) == 7 * 22
    assert sum(" node_cap_leak " in line for line in lines) == 7 * 18
    assert sum(" boundary_sense_ota " in line and "_r" in line.split()[0] for line in lines) == 7 * 22
    assert sum(" ota_edge_sampled " in line for line in lines) == 7 * 22
    assert sum(" crossbar_accumulator_22x7 " in line for line in lines) == 7
    assert sum(" rail_clip " in line for line in lines) == 7 * 18
    assert sum(" stage_gate " in line for line in lines) == 6 * 18
    assert "Vsense" not in text and "Rref" not in text
    assert "maxstep=" in text and "simulatorOptions options reltol=1e-8" in text
    sized = tmp_path / "crossbar_accumulator_22x7.va"
    assert sized.exists()


@pytest.mark.slow
@pytest.mark.spectre
def test_spectre_network_single_spec_matches_true_ode():
    """One real Spectre trajectory per stage; skip on hosts without the kit."""
    import os
    import re
    import shutil
    import subprocess
    from pathlib import Path

    import numpy as np
    import torch
    import pytest

    tcsh = shutil.which("tcsh")
    setup = Path.home() / "workarea_GF22_FDX_EXT/kit.gf22fdx_ext.v1.0_4.1_stack19_oa.csh"
    spectre = shutil.which("spectre")
    if not tcsh or not setup.exists():
        pytest.skip("tcsh / GF22 Spectre kit unavailable")
    probe = subprocess.run([tcsh, "-c", f"source {setup} >& /dev/null; which spectre"],
                           text=True, capture_output=True)
    if probe.returncode or "spectre" not in probe.stdout.lower():
        pytest.skip("Spectre executable unavailable")

    from checkpoints import DEFAULT_CKPT
    from export.spectre_export import export
    from export.spectre_netlist import render
    from export.netlist_interp import simulate
    from scripts.spectre.characterize import psf_waveforms

    canonical = Path.home() / "Documents/ASPDAC_2026/kirchhoff_redesign/ideal/models/canonical_ctle_phase_a_v2.npz"
    if not canonical.exists():
        pytest.skip("canonical input dataset unavailable")
    run_root = Path("/home/annaik/Documents/kirchhoffnet_refactor/.agent_scratch/taskD3/runs")
    run_dir = run_root / f"pytest_network_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(canonical)
    x = data["specs"][data["test_idx"][:1]]
    params = export(DEFAULT_CKPT, run_dir, mode="legacy", fabric="copies", specs=1)
    deck = run_dir / "hrknet.scs"
    render(params, deck, inputs=x)
    command = (f"cd {run_dir}; source {setup} >& /dev/null; "
               f"spectre -format psfascii {deck.name} +log hrknet.log")
    proc = subprocess.Popen([tcsh, "-c", command], cwd=run_dir,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    assert proc.wait(timeout=900) == 0
    log = (run_dir / "hrknet.log").read_text(errors="replace")
    assert re.search(r"spectre completes with 0 errors", log), log[-2000:]

    traces = psf_waveforms(run_dir / "hrknet.raw" / "tran.tran.tran")
    ref = simulate(params, torch.as_tensor(x, dtype=torch.float64),
                   return_trajectory=True, steps_per_stage=1000)
    vu, vcm, tu = (params["units"][k] for k in ("Vu_V", "Vcm_V", "Tu_s"))
    offset = 0.0
    max_error = 0.0
    for si, stage in enumerate(params["resolved"]["stages"]):
        span = float(stage["solver"]["t_span"])
        for node in range(int(stage["num_nodes"])):
            values = traces[f"n0_{si}_{node}"]
            ts = np.asarray([v[0] for v in values])
            measured = (np.asarray([v[1] for v in values]) - vcm) / vu
            mask = (ts >= offset - 1e-14) & (ts <= offset + span * tu + 1e-14)
            expected = np.interp((ts[mask] - offset) / tu,
                                 ref["stage_times"][si].cpu().numpy(),
                                 ref["stage_trajectories"][si][0, :, node].cpu().numpy())
            if expected.size:
                max_error = max(max_error, float(np.max(np.abs(measured[mask] - expected))))
        offset += span * tu
    assert max_error < 1e-4, f"max node trajectory error {max_error:.6g} x; run={run_dir}"


def test_paper_timemux_netlist_uses_shared_fabric_and_recurring_windows(tmp_path):
    params = export(DEFAULT_CKPT, tmp_path, mode="legacy", fabric="timemux", specs=1)
    text = (tmp_path / "hrknet.scs").read_text()
    lines = text.splitlines()
    assert "shared 18-node state; seven parameter banks" in text
    assert sum(" ota_edge_sampled_banked " in line for line in lines) == 22
    assert sum(" boundary_sense_ota_banked " in line for line in lines) == 30
    assert sum(" node_cap_leak_banked " in line for line in lines) == 18
    assert sum("window_clamp " in line for line in lines) == 18
    assert sum("interstage_transfer_banked " in line for line in lines) == 18
    assert "V0_win" in text and "V0_xfer_en" in text and "V0_stage" in text
    assert "t_span[k]*Tu" in text
    assert (tmp_path / "crossbar_accumulator_banked_22x7.va").is_file()

    # Parse generated control PWLs: every transition is a true step, and
    # controls stay enabled / unclamped through each stage's evolve window.
    waveforms = {}
    for name in ("stage", "en", "win", "xfer_en"):
        match = re.search(rf"^V0_{name} \({name}0 0\) vsource type=pwl wave=\[([^]]+)\]", text, re.M)
        assert match, name
        values = [float(v) for v in match.group(1).split()]
        pts = list(zip(values[::2], values[1::2]))
        waveforms[name] = pts
        for previous, edge in zip(pts, pts[1:]):
            if previous[1] != edge[1]:
                assert edge[0] - previous[0] <= 1.01e-12, (name, previous, edge)
    def at(pts, time):
        for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
            if t0 <= time <= t1:
                return v0 + (v1 - v0) * (time - t0) / (t1 - t0)
        return pts[-1][1]
    spans = [float(s["solver"]["t_span"]) * params["units"]["Tu_s"]
             for s in params["resolved"]["stages"]]
    evolve_start = 0.0
    for stage, span in enumerate(spans):
        middle = evolve_start + span / 2
        assert at(waveforms["en"], middle) == 1.0
        assert at(waveforms["win"], middle) == 0.0
        assert at(waveforms["stage"], middle) == float(stage)
        assert at(waveforms["xfer_en"], middle) == 0.0
        evolve_start += span + min(2e-9, min(spans) / 100) * 2 + min(0.2e-9, min(spans) / 1000)
