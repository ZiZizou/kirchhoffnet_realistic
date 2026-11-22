"""Friedman #2/#3 preset factory: torus default, small-world option, input validation.

Replaces the script-style test_audit.py / test_friedman2_small_world.py (hardcoded ASPDAC
paths, no ``test_`` functions, so pytest collected nothing from them).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import make_friedman2_preset, make_friedman3_preset  # noqa: E402


def test_friedman2_default_is_torus():
    p = make_friedman2_preset()
    st = p["stages"][0]
    assert st["hidden_family"] == "torus"
    assert set(st["hidden_kwargs"]) == {"height", "width", "kernel_size", "bidirectional"}
    assert p["write_mode"] == "sparse_proj"
    assert p["read_mode"] == "dense"
    assert p["write_idx"] == [0, 4, 8, 12]


def test_friedman2_small_world_kwargs():
    p = make_friedman2_preset("small_world", small_world_k=6, small_world_p=0.1, small_world_seed=99)
    st = p["stages"][0]
    assert st["hidden_family"] == "small_world"
    assert st["hidden_kwargs"] == {"k": 6, "p": 0.1, "seed": 99, "bidirectional": False}
    assert p["write_idx"] == [0, 4, 8, 12]


def test_friedman3_is_torus_like_friedman2_default():
    p3 = make_friedman3_preset()
    assert p3["stages"][0]["hidden_family"] == "torus"
    assert p3["write_mode"] == make_friedman2_preset()["write_mode"]


@pytest.mark.parametrize("kwargs", [
    dict(hidden_family="small_world", small_world_k=5),      # odd k
    dict(hidden_family="small_world", small_world_p=1.5),    # p > 1
    dict(hidden_family="small_world", small_world_p=-0.1),   # p < 0
    dict(hidden_family="grid"),                              # unknown family
])
def test_friedman2_validation_errors(kwargs):
    with pytest.raises(ValueError):
        make_friedman2_preset(**kwargs)
