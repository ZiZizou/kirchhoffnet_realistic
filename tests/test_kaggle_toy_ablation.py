"""Fast checks for the toy ablation's statistical aggregation."""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hw_toy_ablation import mean_t_ci


def test_mean_t_ci_centered_and_finite():
    mean, low, high = mean_t_ci([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert low < mean < high
    assert math.isfinite(low) and math.isfinite(high)


def test_mean_t_ci_rejects_empty_and_nonfinite():
    try:
        mean_t_ci([])
    except ValueError:
        pass
    else:
        raise AssertionError("empty input should fail")
    try:
        mean_t_ci([1.0, float("nan")])
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite input should fail")
