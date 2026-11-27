"""Exact parent-dir match for Kaggle auto-resume checkpoint search."""
import os
from pathlib import Path

from scripts.kaggle.resume_lookup import find_phase_a_checkpoints


def test_search_matches_exact_rung_dir_and_picks_newest(tmp_path):
    root = tmp_path / "input"
    r1_old = root / "dataset_a" / "R1_seed100"
    r1_new = root / "dataset_b" / "nested" / "R1_seed100"
    r1d = root / "dataset_c" / "R1d_seed100"
    other_seed = root / "R1_seed101"
    for folder in (r1_old, r1_new, r1d, other_seed):
        folder.mkdir(parents=True)
        (folder / "phase_a_last.pt").write_bytes(b"ckpt")
        (folder / "hw_config.json").write_text("{}\n")
    os.utime(r1_old / "phase_a_last.pt", (1_700_000_000, 1_700_000_000))
    os.utime(r1_new / "phase_a_last.pt", (1_800_000_000, 1_800_000_000))

    found = find_phase_a_checkpoints(root, "R1", 100)
    assert [p.parent for p in found] == [r1_new, r1_old]
    assert all(p.parent.name == "R1_seed100" for p in found)
    assert all("R1d" not in p.parent.name for p in found)

    r1d_hits = find_phase_a_checkpoints(root, "R1d", 100)
    assert len(r1d_hits) == 1
    assert r1d_hits[0].parent == r1d

    assert find_phase_a_checkpoints(root, "R2", 100) == []
    assert find_phase_a_checkpoints(tmp_path / "missing", "R1", 100) == []

import argparse

import pytest

from hw import hw_config_from_args
from scripts.kaggle.resume_lookup import compare_resume_hw_config


def _write_preset_sibling(path: Path, preset):
    hw = hw_config_from_args(argparse.Namespace(hw_preset=preset, hw_config=None))
    path.parent.mkdir(parents=True, exist_ok=True)
    hw.to_json(str(path))
    return hw


def test_phase3a_core_sibling_passes_for_r1_and_raises_for_r2(tmp_path):
    """R1 is phase3a_core; R2 is phase3a_full. Hashes match the abb9de7 logs."""
    sibling = tmp_path / "R1_seed100" / "hw_config.json"
    hw = _write_preset_sibling(sibling, "phase3a_core")
    assert hw.hash() == "eda59b5b6e0c"

    msg = compare_resume_hw_config(sibling, "phase3a_core")
    assert "eda59b5b6e0c" in msg
    assert "phase3a_core" in msg

    with pytest.raises(ValueError, match="eda59b5b6e0c") as raised:
        compare_resume_hw_config(sibling, "phase3a_full")
    err = str(raised.value)
    assert "9c9fe3fe9ec3" in err
    assert "phase3a_full" in err


def test_missing_sibling_warns_and_r0_requires_hw_off(tmp_path):
    missing = tmp_path / "nope" / "hw_config.json"
    warn = compare_resume_hw_config(missing, "phase3a_core")
    assert warn.startswith("WARNING")
    assert "hw_config.json" in warn

    off = tmp_path / "R0_seed100" / "hw_config.json"
    _write_preset_sibling(off, None)
    msg = compare_resume_hw_config(off, None)
    assert "hardware off" in msg

    on = tmp_path / "R0_bad" / "hw_config.json"
    _write_preset_sibling(on, "phase3a_full")
    with pytest.raises(ValueError, match="9c9fe3fe9ec3"):
        compare_resume_hw_config(on, None)
