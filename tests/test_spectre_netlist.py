"""Structural regressions for copied-fabric Spectre deck generation."""
import re

from checkpoints import DEFAULT_CKPT
from export.spectre_export import export

def test_paper_copies_netlist_has_expected_instance_counts(tmp_path):
    export(DEFAULT_CKPT, tmp_path, mode="legacy", fabric="copies", specs=1)
    text = (tmp_path / "hrknet.scs").read_text()
    lines = text.splitlines()
    assert sum(bool(re.match(r"q0_s\d+_e\d+ ", line)) for line in lines) == 7 * 22
    assert sum(" node_cap_leak " in line for line in lines) == 7 * 18
    assert sum(" boundary_sense_ota " in line and "_r" in line.split()[0] for line in lines) == 7 * 22
    assert sum(" crossbar_accumulator_22x7 " in line for line in lines) == 7
    assert sum(" rail_clip " in line for line in lines) == 7 * 18
    assert "maxstep=" in text and "reltol=1e-6" in text
    sized = tmp_path / "crossbar_accumulator_22x7.va"
    assert sized.exists()
