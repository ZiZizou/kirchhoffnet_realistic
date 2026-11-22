"""End-to-end guard for three-phase pruning and Phase C retraining."""
import math
import os
from pathlib import Path
import re
import subprocess
import sys


def test_three_phase_friedman2_prune_retrains_with_finite_loss(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    result = subprocess.run(
        [
            sys.executable, "train_script.py", "--problem", "friedman2",
            "--epochs", "6", "--seed", "7", "--device", "cpu",
            "--batch-size", "256", "--num-steps", "2",
            "--schedule", "three_phase", "--prune-edge-threshold", "0.5",
            "--no-amp", "--output", str(tmp_path),
        ],
        cwd=repo, env=env, capture_output=True, text=True, timeout=55,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    log = result.stdout
    assert "[train] done" in log
    assert "[prune] pre-prune:" in log and "[prune] post-prune:" in log
    before_matches = re.findall(
        r"train friedman2 \[A\+B\]\s+train=([0-9.eE+-]+)", log
    )
    after_matches = re.findall(
        r"phase-C epoch\s+\d+ \[C\]\s+train=([0-9.eE+-]+)", log
    )
    assert before_matches and after_matches, log
    before, after = float(before_matches[-1]), float(after_matches[-1])
    assert math.isfinite(before) and math.isfinite(after)
    assert after <= 5 * max(before, 1e-12), (before, after)
