"""Slow Spectre regression for the shared time-multiplexed paper fabric."""
import pytest


@pytest.mark.slow
@pytest.mark.spectre
def test_spectre_timemux_stage01_matches_interp():
    import os
    import shutil
    import subprocess
    import time
    from pathlib import Path

    import numpy as np
    import torch

    tcsh = shutil.which("tcsh")
    setup = Path.home() / "workarea_GF22_FDX_EXT/kit.gf22fdx_ext.v1.0_4.1_stack19_oa.csh"
    if not tcsh or not setup.exists():
        pytest.skip("tcsh / GF22 Spectre kit unavailable")
    probe = subprocess.run([tcsh, "-c", f"source {setup} >& /dev/null; which spectre"],
                           text=True, capture_output=True)
    if probe.returncode or "spectre" not in probe.stdout.lower():
        pytest.skip("Spectre executable unavailable")

    from checkpoints import DEFAULT_CKPT
    from export.spectre_export import export
    from export.spectre_netlist import render_timemux
    from export.netlist_interp import simulate
    from scripts.spectre.characterize import psf_waveforms

    canonical = Path.home() / "Documents/ASPDAC_2026/kirchhoff_redesign/ideal/models/canonical_ctle_phase_a_v2.npz"
    if not canonical.exists():
        pytest.skip("canonical input dataset unavailable")
    run_root = Path("/home/annaik/Documents/kirchhoffnet_refactor/.agent_scratch/tmnet")
    run_dir = run_root / f"pytest_timemux_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(canonical)
    x = data["specs"][data["test_idx"][:1]]
    params = export(DEFAULT_CKPT, run_dir, mode="legacy", fabric="timemux", specs=1)
    deck = run_dir / "hrknet.scs"
    render_timemux(params, deck, inputs=x)
    log = run_dir / "hrknet.log"
    done = run_dir / "compile.done"
    script = run_dir / "compile.csh"
    script.write_text(
        "#!/bin/tcsh\n"
        f"cd {run_dir}\n"
        f"source {setup} >& /dev/null\n"
        f"spectre -format psfascii {deck.name} +log {log.name} >>& {run_dir / 'spectre.out'}\n"
        "set rc = $status\n"
        f"echo $rc > {done}\n"
        "exit $rc\n"
    )
    subprocess.run(["bash", "-c", f"cd {run_dir} && nohup tcsh {script.name} > /dev/null 2>&1 &"],
                   check=True, timeout=30)
    deadline = time.monotonic() + 900
    while not done.exists() and time.monotonic() < deadline:
        time.sleep(2)
    assert done.exists(), f"Spectre timed out; see {run_dir}"
    assert done.read_text().strip() == "0", f"Spectre failed; see {log}"
    assert "spectre completes with 0 errors" in log.read_text(errors="replace"), f"see {log}"

    traces = psf_waveforms(run_dir / "hrknet.raw" / "tran.tran.tran")
    ref = simulate(params, torch.as_tensor(x, dtype=torch.float64),
                   return_trajectory=True, steps_per_stage=1000)
    vu, vcm, tu = (params["units"][k] for k in ("Vu_V", "Vcm_V", "Tu_s"))
    spans = [float(stage["solver"]["t_span"]) * tu for stage in params["resolved"]["stages"]]
    capture = min(2e-9, min(spans) / 100)
    settle = min(0.2e-9, capture / 10)
    overhead = 2 * capture + settle
    offset = 0.0
    errors = []
    for si, stage in enumerate(params["resolved"]["stages"]):
        stage_error = 0.0
        for node in range(int(stage["num_nodes"])):
            values = np.asarray(traces[f"n0_{node}"])
            ts = values[:, 0]
            measured = (values[:, 1] - vcm) / vu
            mask = (ts >= offset - 1e-14) & (ts < offset + spans[si] - 1e-9)
            expected = np.interp((ts[mask] - offset) / tu,
                                 ref["stage_times"][si].cpu().numpy(),
                                 ref["stage_trajectories"][si][0, :, node].cpu().numpy())
            if expected.size:
                stage_error = max(stage_error, float(np.max(np.abs(measured[mask] - expected))))
        errors.append(stage_error)
        offset += spans[si] + overhead
    assert max(errors[:2]) < 1e-4, f"stage 0-1 errors {errors[:2]}; run={run_dir}"
