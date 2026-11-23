"""Fast structural contracts for the seven-bank Verilog-A primitives."""
from pathlib import Path
import re

VA = Path(__file__).resolve().parents[1] / "veriloga"


def test_banked_primitives_select_nearest_volt_and_gate_currents():
    modules = {
        "ota_edge_banked": ("A", "B", "s", "theta", "gm", "Isat"),
        "ota_edge_sampled_banked": ("A", "B", "s", "theta", "gm", "Isat"),
        "shunt_banked": ("G",),
        "boundary_sense_ota_banked": ("gm", "Isat", "G", "A", "B"),
        "node_cap_leak_banked": ("c_eff", "c_phys", "g_leak"),
        "rail_clip_banked": ("x_max", "sharp", "Iclip"),
    }
    for module, params in modules.items():
        text = (VA / f"{module}.va").read_text()
        assert f"module {module}(" in text
        assert "stage,en" in text
        assert "floor(V(stage)+0.5)" in text and "if(k>6) k=6" in text
        assert "V(en)-0.5" in text
        for name in params:
            assert re.search(rf"\b{name}\[0:6\]", text), (module, name)
    crossbar = (VA / "crossbar_accumulator_banked_template.va").read_text()
    assert "W[0:83]" in crossbar and "W[k*12+" in crossbar
    assert "V(stage)+0.5" in crossbar and "V(en)-0.5" in crossbar


def test_characterization_reference_equations_are_present():
    """Spot-check bank equations against copies-module current expressions."""
    edge = (VA / "ota_edge_banked.va").read_text()
    copy_edge = (VA / "ota_edge.va").read_text()
    assert "Isat[k]*gate*(tanh(u)+floor_term*u)*`HRK_IU" in edge
    assert "Isat*gate*(t+floor_term*u)*`HRK_IU" in copy_edge
    shunt = (VA / "shunt_banked.va").read_text()
    copy_shunt = (VA / "shunt.va").read_text()
    assert "G[k]*mult*(V(src)-V(dst))*`HRK_GU" in shunt
    assert "G*mult*(V(src)-V(dst))*`HRK_GU" in copy_shunt
