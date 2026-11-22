"""Audit-only: instability scan of the training histories (NaN/Inf/spikes)."""
import math
from pathlib import Path

BASE = Path(r"C:\Users\Atharva\Downloads\trained_knet")

def read_tsv(path):
    lines = [l for l in Path(path).read_text().splitlines() if l.strip()]
    header = lines[0].split("\t")
    rows = [l.split("\t") for l in lines[1:]]
    return header, rows

def report(name, path):
    header, rows = read_tsv(path)
    print(f"\n=== {name} :: {path} ===")
    print(f"rows={len(rows)} cols={len(header)}")
    ncol = len(header)
    for c in range(1, min(ncol, 30)):
        vals = []
        for r in rows:
            if c < len(r):
                try:
                    vals.append(float(r[c]))
                except ValueError:
                    pass
        if not vals:
            continue
        bad = sum(1 for v in vals if math.isnan(v) or math.isinf(v))
        mx = max(vals)
        mn = min(vals)
        med = sorted(vals)[len(vals) // 2]
        print(f"  {header[c]:<26s} n={len(vals):<5} min={mn:+.4e} med={med:+.4e} max={mx:+.4e} "
              f"nonfinite={bad}")

report("smooth2d loss", BASE / "test_smooth2d_w_tanh" / "loss_history.txt")
report("smooth2d grad_norms", BASE / "test_smooth2d_w_tanh" / "grad_norms.txt")
report("smooth2d update_norms", BASE / "test_smooth2d_w_tanh" / "update_norms.txt")
report("deq loss", BASE / "deq" / "loss_history.txt")
report("deq validation", BASE / "deq" / "deq_validation_history.txt")
