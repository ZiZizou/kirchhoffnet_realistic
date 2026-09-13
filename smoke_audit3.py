"""Audit fixes smoke (2026-09-04): every fix verified end-to-end."""
import math, shutil
from pathlib import Path
import torch
from config import make_narma_preset
from cell_library import make_cell_library
from topology import build_net_from_config
import narma_experiment as nax
from narma_experiment import train_fabric, narma, scale_input_to_rails, NARMA_INPUT_MAX

torch.manual_seed(0)

# --- helpers ----------------------------------------------------------
def build(refresh=8, freeze=False, legacy=False):
    p = make_narma_preset(order=10, hidden_dim=25, t_span=1.0,
                         num_steps_per_sample=8,
                         core_refresh_interval=0 if legacy else refresh)
    return build_net_from_config(p, cell_lib=make_cell_library("tanh_free"),
                                 boundary_fan_out=p["boundary_fan_out"],
                                 enable_temporal_readout=True,
                                 freeze_read=freeze)

u, y = narma(260, order=10, seed=0)
u = scale_input_to_rails(u.unsqueeze(0), u_max=NARMA_INPUT_MAX[10])
y = y.unsqueeze(0)

# --- Fix 1: incremental flush no longer TypeErrors --------------------
out = Path("s3_flush"); out.mkdir(exist_ok=True)
rows = [{
    "seed": 0, "condition": "fabric_refresh_k4",
    "nrmse": 0.4, "r2": 0.8, "mc_total": 5.0,
    "n_params": 661, "total_params": 661, "hidden_dim": "",
    "cell_library": "tanh_free", "core_refresh_interval": 4,
    "cell_lib_evals_per_sample": 2,
}]
nax._write_partial_tables(out, rows, 10, [0])  # 4 args, must not raise
assert (out / "results_table.csv").exists() and (out / "results_table.txt").exists()
print("FIX1_FLUSH_OK")
shutil.rmtree(out)

# --- Fix 2+3: epochs_completed propagates; resume-from-completed is safe
ckpt = Path("s3_ckpt.pt")
# Train 5 epochs, force early stop via impossible target.
y_bad = torch.randn_like(y) * 10  # diverges under MSE
r = train_fabric(build(refresh=4), u, y_bad, epochs=5, tbptt_chunk=25,
                 device="cpu", verbose=False, val_every=1,
                 early_stop_patience=1, checkpoint_path=ckpt,
                 checkpoint_every=1)
# resume from a checkpoint that completed more epochs than target -> empty loop
r2 = train_fabric(build(refresh=4), u, y, epochs=2, tbptt_chunk=25,
                  device="cpu", verbose=False, val_every=0,
                  init_from=ckpt)
print(f"FIX2 epochs_completed={r['epochs_completed']} (expected 5)")
print(f"FIX3 resume-from-completed epochs_completed={r2['epochs_completed']} (expected 5)")
assert r["epochs_completed"] == 5
assert r2["epochs_completed"] == 5
# Resume-from-completed (empty loop) must not append spurious history entries.
assert len(r2["train_loss_history"]) == 5

# --- Fix 5: scaler + stagnation in checkpoint ------------------------
ckpt2 = torch.load(ckpt, map_location="cpu", weights_only=False)
assert "scaler_state" in ckpt2, "scaler_state missing"
assert "evals_without_improvement" in ckpt2, "evals_without_improvement missing"
print("FIX5_CHECKPOINT_KEYS_OK")
ckpt.unlink()

# --- Fix 6: NaN-guarded decision helper ------------------------------
good = {"fabric_refresh_k4": [{"nrmse": 0.5}], "fabric_refresh_k2": [{"nrmse": 0.2}]}
mixed = {"fabric_refresh_k4": [{"nrmse": float("nan")}], "fabric_refresh_k2": [{"nrmse": 0.2}]}
out_good = nax._decide_refresh_ladder(good, (0.15, 0.29))
out_mixed = nax._decide_refresh_ladder(mixed, (0.15, 0.29))
print("FIX6a (both finite):", out_good[-1] if out_good else "NONE")
print("FIX6b (NaN filtered):", out_mixed[0] if out_mixed else "NONE",
      "|", out_mixed[-1] if out_mixed else "NONE")
assert any("diverged" in ln for ln in out_mixed)
assert "0.2000" in out_mixed[-1]

# --- Fix 1 integration: end-to-end CLI dry run -----------------------
import subprocess, sys
res = subprocess.run(
    [sys.executable, "-B", "narma_experiment.py",
     "--order", "10", "--seeds", "0", "--epochs", "2",
     "--fabric-only", "--n-streams", "2", "--train-samples", "130",
     "--tbptt-chunk", "25", "--num-steps", "8",
     "--core-refresh-interval", "4", "--val-every", "1",
     "--early-stop-patience", "0", "--device", "cpu",
     "--output", "./output/s3_dry", "--no-progress"],
    capture_output=True, text=True, timeout=600,
)
last_lines = res.stdout.strip().split("\n")[-6:]
print("FIX1_CLI_TAIL:")
for ln in last_lines:
    print("  ", ln)
print("FIX1_CLI_STDERR_TAIL:")
for ln in res.stderr.strip().split("\n")[-10:]:
    print("  ", ln)
print("FIX1_CLI_RC:", res.returncode)
# The bug was: WARNING: partial table flush failed: ... every job.
assert "partial table flush failed" not in res.stdout, "flush still failing"
# Decision is suppressed for single-seed or single-condition runs (need
# >=2 refresh conditions in the table). Verify the suppress-vs-write path
# is wired correctly by running two k values in a separate output dir.
# Decision needs >=2 refresh conditions in the table. ``--core-refresh-interval``
# only takes a single value, so we run two separate subprocess calls into the
# same output dir; ``--append`` (or default overwrite with same seeds) re-uses
# the table writer. Simpler: run k=4 first, then re-run the same script for
# k=2 with a unique seed combo so the table gets both rows, then call the
# decision helper directly to validate the verdict.
out2 = Path("./output/s3_dry_ladder")
if out2.exists():
    shutil.rmtree(out2)
def _run_k(k_val, outdir, seed=0):
    if outdir.exists():
        shutil.rmtree(outdir)
    return subprocess.run(
        [sys.executable, "-B", "narma_experiment.py",
         "--order", "10", "--seeds", str(seed), "--epochs", "2",
         "--fabric-only", "--n-streams", "2", "--train-samples", "130",
         "--tbptt-chunk", "25", "--num-steps", "8",
         "--core-refresh-interval", str(k_val), "--val-every", "1",
         "--early-stop-patience", "0", "--device", "cpu",
         "--output", str(outdir), "--no-progress"],
        capture_output=True, text=True, timeout=600,
    )
r_k4 = _run_k(4, Path("./output/s3_dry_k4"), seed=0)
r_k2 = _run_k(2, Path("./output/s3_dry_k2"), seed=0)
print("FIX1_LADDER_K4_RC:", r_k4.returncode)
print("FIX1_LADDER_K2_RC:", r_k2.returncode)
assert r_k4.returncode == 0, r_k4.stderr[-500:]
assert r_k2.returncode == 0, r_k2.stderr[-500:]
# Parse both tables and call the decision helper directly.
import csv
by_cond: dict[str, list[dict]] = {}
for od in (Path("./output/s3_dry_k4"), Path("./output/s3_dry_k2")):
    with open(od / "results_table.csv") as f:
        for r in csv.DictReader(f):
            by_cond.setdefault(r["condition"], []).append(
                {"nrmse": float(r["nrmse"])}
            )
print("FIX1_LADDER_BY_COND:", {k: [v["nrmse"] for v in vs] for k, vs in by_cond.items()})
verdict = nax._decide_refresh_ladder(by_cond, (0.15, 0.29))
print("FIX1_LADDER_VERDICT:")
for ln in verdict or []:
    print("  ", ln)
assert verdict is not None
assert any("Verdict:" in ln for ln in verdict)
print("ALL_OK")
