"""Fast structural checks for the Verilog-A primitive library."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
VA = ROOT / "veriloga"
MODULE_PARAMS = {
    "ota_edge": {"mode", "ab_mode", "A", "B", "Ahat", "Bhat", "s", "theta", "gm", "Isat", "z_gate", "vca", "gln", "cap_code_a", "cap_code_b", "cap_bits", "cap_rmax"},
    "shunt": {"G", "z_gate", "vca", "vca_gate_shunt"},
    "node_cap_leak": {"c_eff", "c_phys", "g_leak"},
    "rail_clip": {"x_max", "sharp", "Iclip"},
    "sample_hold": {"C_hold", "R_on", "R_off", "inject_v", "droop_tau"},
    "boundary_sense_ota": {"gm", "Isat", "A", "B", "theta", "s", "z_gate"},
    "crossbar_accumulator": {"weight", "Vref", "Gunit", "Cacc"},
    "interstage_transfer": {"w1", "w2", "w3", "vth", "x_hard"},
    "vca_bias": {"w0", "w1", "w2", "w3", "bias"},
    "gln_bias": {"w0", "w1", "w2", "w3", "bias", "delta_max"},
    "phase_ctrl": {"stages", "t_hold", "t_transfer", "t_switch", "t_evolve", "Vlogic"},
}


def test_all_modules_exist_and_include_shared_constants():
    for module in MODULE_PARAMS:
        path = VA / f"{module}.va"
        assert path.is_file(), f"missing {path.name}"
        text = path.read_text()
        assert '`include "hrknet_constants.vams"' in text, path.name
        assert re.search(rf"\bmodule\s+{module}\s*\(", text), path.name


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
