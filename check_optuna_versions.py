"""Quick optuna+numpy version check for the host ENV.

Run on the failing host (where the offending ENV lives):

    python check_optuna_versions.py

Prints the installed optuna + numpy versions and a small BOOT AVOIDANCE note.

Crash summary (root cause families we hit):
- optuna#5260 < 4.0: TPESampler crashes on Pruned trials in multi-obj
  studies; lines `_calculate_weights_below_for_multi_objective` raise
  `TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'`
  (this is the second crash we observed).
- optuna#3676: multi-obj TPE mishandles +/-inf objective values -> NaN
  weights. Always use FINITE penalty (no inf) when returning fallback
  values from objective.
- numpy >= 1.24 + ragged `np.asarray([t.values ...])`: raises the first
  crash we saw (`inhomogeneous shape after 1 dimension`).

The fixed-distillation BO is hardened against all three families by code
guards (enum direction check, pre-optimize homogeneous scan, finite
penalty instead of TrialPruned). This script is informational; the
preferred upgrade target is optuna >= 4.0 if the host ENV allows the bump.
"""

from __future__ import annotations

import sys


def main() -> int:
    print("Package versions:")
    for name in ("optuna", "numpy", "torch"):
        try:
            mod = __import__(name)
        except ImportError:
            print(f"  {name}: not installed")
            continue
        print(f"  {name}: {mod.__version__}")

    try:
        import optuna as _o
    except ImportError:
        print("\noptuna missing; cannot assess.")
        return 2

    parts = [int(p) for p in _o.__version__.split(".")[:2]]
    version_tuple = tuple(parts + [0] * (2 - len(parts)))
    print("\nRecommendation:")
    if version_tuple < (3, 6):
        print("  CRITICAL: optuna < 3.6 — both crash families #5260 and #3676"
              " are fixed in 4.0+. Upgrade before the next BO run.")
    elif version_tuple < (4, 0):
        print("  ADVISORY: optuna < 4.0 — #5260 (pruned multi-obj crash) is not"
              " fixed. The code-level guard (penalty instead of TrialPruned)"
              " works on this version, but consider upgrading.")
    else:
        print("  OK: optuna >= 4.0 — both #5260 and #3676 are patched in code.")

    try:
        import numpy as _n
        numpy_vers = tuple(int(p) for p in _n.__version__.split(".")[:2])
        if numpy_vers < (1, 24):
            print(f"  OK: numpy {_n.__version__} < 1.24 — ragged-list infers"
                  " an object array (no inhomogeneous-shape crash).")
        else:
            print(f"  NOTE: numpy {_n.__version__} >= 1.24 — `np.asarray` on a"
                  " ragged list raises. The BO's pre-optimize scan rejects"
                  " any pre-existing trial whose values is None or short.")
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
