"""Single source of truth for the CTLE failure rule.

A design FAILS when at least ``MIN_DEGRADED_DIMS`` of the 4 specs (power, jitter,
eye height, eye width) degrade by ``DEGRADE_REL_THRESHOLD`` (relative) or more
versus the target. The paper reports 8.15% test failure computed at 10%
(ctle_dagger_common at 5c6a4f4: ``ctx["error_threshold"] = 0.10`` reaches
``compute_forward_errors``), although the paper text says 20%.

Every default that decides "failed or not" (surrogate scorer, Phase-A
``ctx["error_threshold"]``, the dagger / fixed-distillation scripts and
``spice_failure_check.py --degrade-thr``) imports these constants. Pure Python on
purpose: no torch/numpy imports, so any script can import it cheaply.
"""

DEGRADE_REL_THRESHOLD = 0.10
MIN_DEGRADED_DIMS = 2

__all__ = ["DEGRADE_REL_THRESHOLD", "MIN_DEGRADED_DIMS"]
