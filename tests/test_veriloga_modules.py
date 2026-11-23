"""Fast structural checks for the Verilog-A primitive library."""
from pathlib import Path
import re
import json
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
VA = ROOT / "veriloga"
MODULE_PARAMS = {
    "ota_edge": {"mode", "ab_mode", "A", "B", "Ahat", "Bhat", "s", "theta", "gm", "Isat", "z_gate", "vca", "gln", "cap_code_a", "cap_code_b", "cap_bits", "cap_rmax"},
    "shunt": {"G", "z_gate", "vca", "vca_gate_shunt"},
    "node_cap_leak": {"c_eff", "c_phys", "g_leak"},
    "rail_clip": {"x_max", "sharp", "Iclip"},
    "sample_hold": {"C_hold", "R_on", "R_off", "inject_v", "inject_gain_err", "droop_v", "droop_time", "droop_tau"},
    "boundary_sense_ota": {"gm", "Isat", "A", "B", "theta", "s", "z_gate"},
    "crossbar_accumulator": {"W", "hw_mode", "hw_bits", "hw_bound", "acc_compliance_en", "Wmax", "lambda_v", "x_hard", "Vcm", "Vu", "Ut", "k_soft"},
    "interstage_transfer": {"w1", "w2", "w3", "vth", "x_hard"},
    "vca_bias": {"w0", "w1", "w2", "w3", "bias"},
    "gln_bias": {"w0", "w1", "w2", "w3", "bias", "delta_max"},
    "phase_ctrl": {"stages", "t_hold", "t_transfer", "t_switch", "t_evolve", "Vlogic"},
    "stage_gate": {"w1", "w2", "w3", "vth", "activation", "x_hard", "release_s", "smooth_s", "g_clamp"},
    "ota_edge_sampled": {"A", "B", "s", "theta", "gm", "Isat", "z_gate", "sigma_z", "vca", "x_max", "gate_width", "floor_term"},
}


def test_all_modules_exist_and_include_shared_constants():
    expected_files = {f"{module}.va" for module in MODULE_PARAMS}
    assert {path.name for path in VA.glob("*.va")} == expected_files
    assert (VA / "hrknet_constants.vams").is_file()
    for path in VA.glob("*.va"):
        text = path.read_text()
        assert '`include "hrknet_constants.vams"' in text, path.name
    for module in MODULE_PARAMS:
        text = (VA / f"{module}.va").read_text()
        assert re.search(rf"\bmodule\s+{module}\s*\(", text), module


def test_exporter_parameter_contract_is_declared():
    for module, expected in MODULE_PARAMS.items():
        text = (VA / f"{module}.va").read_text()
        declared = set()
        for names in re.findall(r"parameter\s+(?:integer|real)\s+([^;]+);", text):
            for item in names.split(","):
                declared.update(re.findall(r"[A-Za-z_]\w*", item.split("=", 1)[0]))
        missing = expected - declared
        assert not missing, f"{module} lacks exporter parameters: {sorted(missing)}"


def test_shared_unit_constants_match_hw_units_contract():
    text = (VA / "hrknet_constants.vams").read_text()
    for macro in ("HRK_VDD", "HRK_VCM", "HRK_VU", "HRK_CU", "HRK_TU", "HRK_IU", "HRK_GU"):
        assert f"`define {macro} " in text
    assert "350e-9/4.660583" in text


@pytest.mark.slow
@pytest.mark.spectre
def test_spectre_ota_and_shunt_match_analytic_reference():
    tcsh = shutil.which("tcsh")
    setup = Path.home() / "workarea_GF22_FDX_EXT" / "kit.gf22fdx_ext.v1.0_4.1_stack19_oa.csh"
    if tcsh is None or not setup.is_file():
        pytest.skip("Cadence setup/tcsh is unavailable")
    probe_dir = Path("/home/annaik/Documents/kirchhoffnet_refactor/.agent_scratch/taskD2/runs/probe")
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run([tcsh, "-c", f"cd {probe_dir}; source {setup} >& /dev/null; which spectre"],
                           text=True, capture_output=True, timeout=30)
    if probe.returncode or "spectre" not in probe.stdout.lower():
        pytest.skip("Spectre is unavailable")
    script = ROOT / "scripts" / "spectre" / "characterize.py"
    result = subprocess.run(["/home/annaik/Documents/kirchhoffnet_refactor/venv/bin/python", str(script)],
                            cwd=ROOT, text=True, capture_output=True, timeout=600, check=True)
    report_path = Path(result.stdout.strip().splitlines()[-1])
    report = json.loads(report_path.read_text())
    assert report["measured_cases"] == 84
    assert {r["module"] for r in report["results"]} == {
        "ota_edge", "shunt", "interstage_transfer", "vca_bias", "gln_bias",
        "node_cap_leak", "rail_clip", "boundary_sense_ota", "crossbar_accumulator",
        "sample_hold", "phase_ctrl",
    }
    crossbar = [r for r in report["results"] if r["module"] == "crossbar_accumulator"]
    assert len(crossbar) == 18
    assert {r["mode"] for r in crossbar} == {"ideal_0", "ideal_1", "ideal_2", "hw_offset_binary_compliance"}
    assert max(r["rel_error"] for r in crossbar) <= 1e-5
    boundary_sense = [r for r in report["results"] if r["module"] == "boundary_sense_ota"]
    assert len(boundary_sense) == 16
    assert {r["mode"] for r in boundary_sense} == {"boundary", "sense"}
    legacy = [r for r in report["results"] if r["module"] == "ota_edge" and r["mode"] == "legacy"]
    assert len(legacy) == 16
    assert max(r["rel_error"] for r in legacy) <= 1e-6
    exact = [r for r in report["results"] if r["module"] in {
        "ota_edge", "shunt", "vca_bias", "gln_bias"
    }]
    assert max(r["abs_error_model_units"] for r in exact) < 1e-5
    transfer = [r for r in report["results"] if r["module"] == "interstage_transfer"]
    assert max(r["abs_error_model_units"] for r in transfer) <= 1e-6
    # Every characterization reported by this deck has an explicit tolerance.
    # node_cap_leak's 8.72e-6 x residual is integration truncation error:
    # 200 max steps per RC time constant with tight Spectre reltol/maxstep.
    tolerances = {
        "ota_edge": 1e-5, "shunt": 1e-5, "interstage_transfer": 1e-6,
        "vca_bias": 1e-5, "gln_bias": 1e-5,
        "node_cap_leak": 1e-5, "rail_clip": 1e-6,
        "boundary_sense_ota": 1e-5, "crossbar_accumulator": 1e-5,
        "sample_hold": 5e-3, "phase_ctrl": 1e-10,
    }
    for module, tolerance in tolerances.items():
        rows = [r for r in report["results"] if r["module"] == module]
        assert rows, f"missing Spectre cases for {module}"
        assert max(r["abs_error_model_units"] for r in rows) <= tolerance
    sample_hold = [r for r in report["results"] if r["module"] == "sample_hold"]
    assert len(sample_hold) == 1 and sample_hold[0]["ktc"] == "excluded"
    phase_ctrl = [r for r in report["results"] if r["module"] == "phase_ctrl"]
    assert len(phase_ctrl) == 1
    assert phase_ctrl[0]["phase_edges"] == 8
    assert phase_ctrl[0]["stage_edges"] == 2
