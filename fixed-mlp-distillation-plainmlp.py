"""Run the shared fixed CTLE distillation harness with a PlainMLP student.

All split, teacher-label, objective, ZIG evaluation, and artifact behavior is
implemented by ``fixed-mlp-distillation-kirchhoffnet.py``.  This tiny entry
point fixes ``--student-kind mlp`` so MLP and KNet experiments cannot drift.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    shared = Path(__file__).with_name("fixed-mlp-distillation-kirchhoffnet.py")
    sys.argv = [str(shared), "--student-kind", "mlp", *sys.argv[1:]]
    runpy.run_path(str(shared), run_name="__main__")
