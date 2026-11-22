"""The CTLE failure rule (>= 2 of 4 specs degrade by >= 10%) has ONE definition.

Every default that decides "failed or not" must equal ctle_metrics_config.DEGRADE_REL_THRESHOLD
(0.10): the surrogate scorer, Phase-A ``ctx["error_threshold"]``, the dagger / fixed-distillation
scripts and ``spice_failure_check.py --degrade-thr``.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ctle_metrics_config import DEGRADE_REL_THRESHOLD, MIN_DEGRADED_DIMS  # noqa: E402


def _src(name: str) -> str:
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return f.read()


def test_constant_is_ten_percent():
    assert DEGRADE_REL_THRESHOLD == 0.10
    assert MIN_DEGRADED_DIMS == 2


def test_ctle_dagger_common_defaults():
    import ctle_dagger_common as cdc
    assert cdc.HPARAMS_DEFAULTS["DEGRADE_REL_THRESHOLD"] == DEGRADE_REL_THRESHOLD
    assert cdc.HPARAMS_DEFAULTS["ERROR_THRESHOLD"] == DEGRADE_REL_THRESHOLD
    assert cdc.HPARAMS_DEFAULTS["MIN_DEGRADED_DIMS"] == MIN_DEGRADED_DIMS


def test_surrogate_scorer_uses_constant():
    """compute_forward_errors with no threshold and an empty context: a 15% error fails, 5% does not."""
    import ctle_dagger_common as cdc
    target = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    pred = np.array([[1.15, 1.15, 1.0, 1.0],     # 2 dims off by 15%  -> fail under 10%, pass under 20%
                     [1.05, 1.05, 1.0, 1.0]])    # 2 dims off by 5%   -> pass
    saved = dict(cdc._active_ctx)
    try:
        cdc._active_ctx.clear()
        out = cdc.compute_forward_errors(pred, target, boundary_tols={k: (0.0, 0.0) for k in cdc.BOUNDARY_ABS_TOLERANCES})
    finally:
        cdc._active_ctx.clear()
        cdc._active_ctx.update(saved)
    mask = np.asarray(out[1] if isinstance(out, tuple) else out["failure_mask"])
    assert bool(mask[0]) and not bool(mask[1]), mask


def test_spice_failure_check_default():
    import spice_failure_check as sfc
    args = sfc.build_parser().parse_args(["--help"][:0] + [])  # all-defaults parse
    assert args.degrade_thr == DEGRADE_REL_THRESHOLD
    assert args.min_degraded_dims == MIN_DEGRADED_DIMS


def test_fixed_mlp_distillation_default():
    spec = importlib.util.spec_from_file_location("fixed_mlp_distill", os.path.join(HERE, "fixed-mlp-distillation-kirchhoffnet.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    old = sys.argv
    try:
        sys.argv = ["x", "--teacher-checkpoint", "t.pt", "--dataset", "d.csv", "--output", "o"]
        try:
            args = mod.parse_args()
        except SystemExit:  # required args differ; fall back to a source check
            args = None
    finally:
        sys.argv = old
    if args is not None:
        assert args.zig_degrade_threshold == DEGRADE_REL_THRESHOLD
        assert args.zig_min_degraded_dims == MIN_DEGRADED_DIMS
    src = _src("fixed-mlp-distillation-kirchhoffnet.py")
    assert "default=DEGRADE_REL_THRESHOLD" in src


def test_script_style_entry_points_import_the_constant():
    """The dagger scripts run at import, so check their source: they import the constant and
    define no literal 0.20 degrade threshold of their own."""
    for name in ("dagger-nuance-distillation-kirchhoffnet.py",
                 "dagger-mlp-distillation-kirchhoffnet.py",
                 "generative-distillation-improved-dagger-nuance-mlp.py"):
        src = _src(name)
        assert "from ctle_metrics_config import DEGRADE_REL_THRESHOLD" in src, name
        assert re.search(r"^ERROR_THRESHOLD\s*=\s*DEGRADE_REL_THRESHOLD", src, re.M), name
        assert not re.search(r"^DEGRADE_REL_THRESHOLD\s*=\s*0\.", src, re.M), name
    for name in ("generative-distillation-plain-mlp.py", "ctle_kirchhoff_bo.py"):
        src = _src(name)
        assert '"error_threshold": DEGRADE_REL_THRESHOLD' in src, name
        assert not re.search(r'"error_threshold":\s*0\.\d', src), name


def test_no_literal_020_threshold_left():
    for name in ("ctle_dagger_common.py", "kn_bayes_opt.py", "fixed-distillation-bayes-opt.py",
                 "spice_failure_check.py", "fixed-mlp-distillation-kirchhoffnet.py"):
        src = _src(name)
        for m in re.finditer(r"(?i)(degrade\w*|error_threshold)\W{0,12}0\.20?\b", src):
            raise AssertionError(f"{name}: literal 0.2 near {m.group(0)!r}")
