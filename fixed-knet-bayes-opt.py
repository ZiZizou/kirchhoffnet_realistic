"""KNet entry point for the shared fixed-distillation Bayesian optimizer."""
from __future__ import annotations
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).with_name("fixed-distillation-bayes-opt.py")
    sys.argv = [str(target), "--student-kind", "knet", *sys.argv[1:]]
    runpy.run_path(str(target), run_name="__main__")
