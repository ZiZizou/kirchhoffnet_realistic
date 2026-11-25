"""Small pure helpers shared by the CTLE closure prediction and tests."""
import csv
from pathlib import Path
import numpy as np

PARAM_COLS = ("fW", "current", "ind", "Rd", "Cs", "Rs", "VDD")


def bounded_params(logits, model):
    """Match PaperStudent's sigmoid decoder into bounded log10 parameters."""
    import torch
    logits = torch.as_tensor(logits, dtype=model.log_lo.dtype, device=model.log_lo.device)
    return model.log_lo.unsqueeze(0) + (model.log_hi-model.log_lo).unsqueeze(0)*torch.sigmoid(logits)


def write_params(path, rows):
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 7 or not np.isfinite(arr).all():
        raise ValueError(f"expected finite (N,7) log10 knob values, got {arr.shape}")
    with Path(path).open("w", newline="") as f:
        w = csv.writer(f); w.writerow(PARAM_COLS)
        w.writerows([[f"{v:.10g}" for v in row] for row in arr])
