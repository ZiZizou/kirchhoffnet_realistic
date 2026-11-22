"""The hardware CLI reaches the entry scripts (Phase 3b 2.2, W4 step 3).

Runs ``train_script.py`` for one epoch on the small ``sinx`` problem (about
10 s on CPU) with ``--hw-preset phase3a_core``; the config must be printed with
its hash and written to ``hw_config.json``.  (``--problem friedman2 --epochs 1``
works too but takes ~60 s because its preset prunes and retrains.)
"""

import json
import os
import subprocess
import sys

import pytest

from hw import HWConfig
from hw.cli import parse_known_args_hw_guarded

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(args, tmp_path, timeout=300):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    return subprocess.run([sys.executable, os.path.join(ROOT, "train_script.py")] + args,
                          cwd=str(tmp_path), capture_output=True, text=True,
                          timeout=timeout, env=env)


def test_train_script_runs_with_hw_preset(tmp_path):
    out = tmp_path / "run"
    r = _run(["--problem", "sinx", "--epochs", "1", "--seed", "7",
              "--hw-preset", "phase3a_core", "--device", "cpu", "--output", str(out)], tmp_path)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    want = HWConfig.from_dict(json.load(open(os.path.join(ROOT, "hw", "presets",
                                                          "phase3a_core.json"))))
    assert f"[hw] HWConfig ON hash={want.hash()}" in r.stdout
    # _ensure_dir may timestamp an existing dir; find the hw_config.json
    found = [os.path.join(d, "hw_config.json") for d, _, fs in os.walk(tmp_path)
             if "hw_config.json" in fs]
    assert len(found) == 1
    assert HWConfig.from_json(found[0]) == want


def test_train_script_rejects_unknown_hw_flag(tmp_path):
    # train_script uses parse_args (strict): argparse itself refuses the typo
    r = _run(["--problem", "sinx", "--epochs", "1", "--hw-slope-bnd"], tmp_path, timeout=120)
    assert r.returncode == 2 and "--hw-slope-bnd" in r.stderr


def test_train_script_parser_has_hw_flags():
    import argparse
    sys.path.insert(0, ROOT)
    from train_script import _add_argparse_args
    p = argparse.ArgumentParser()
    _add_argparse_args(p)
    a, unknown = parse_known_args_hw_guarded(p, ["--hw-preset", "phase3b_tm", "--hw-tm"])
    assert unknown == [] and a.hw_preset == "phase3b_tm" and a.hw_tm is True
    with pytest.raises(SystemExit):   # no ZIG in train_script
        p.parse_args(["--zig-degrade-thr", "0.2"])


def test_bo_driver_passes_hw_flags_through():
    """Static check of kn_bayes_opt: CRLF kept, flags added, passthrough + fingerprint."""
    with open(os.path.join(ROOT, "kn_bayes_opt.py"), "rb") as fh:
        raw = fh.read()
    assert raw.count(b"\r\n") == raw.count(b"\n"), "kn_bayes_opt.py must stay CRLF"
    src = raw.decode("utf-8")
    for s in ("parse_hw_args(parser)", "add_degrade_thr_arg(parser)",
              "cmd += hw_argv_ctle", "cmd += hw_argv_train", '"hw_config_hash": hw_cfg.hash()',
              'hw_argv_train = ["--hw-config", str(hw_json)]'):
        assert s in src, s


def test_bo_driver_accepts_every_hw_flag():
    """The slurm HW_EXTRA flags (--hw-tm ...) must parse in the BO driver too."""
    r = subprocess.run([sys.executable, os.path.join(ROOT, "kn_bayes_opt.py"), "--help"],
                       capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 0
    for flag in ("--hw-config", "--hw-preset", "--hw-tm", "--hw-tm-c-hold-ff",
                 "--hw-slope-bound", "--zig-degrade-thr"):
        assert flag in r.stdout, flag
