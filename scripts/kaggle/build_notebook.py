#!/usr/bin/env python3
"""Build the clean Kaggle HW retraining notebook using only stdlib json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "kaggle_hw_retrain.ipynb"

def cell(kind, text):
    return {"cell_type": kind, "metadata": {}, "source": text.splitlines(True),
            **({"execution_count": None, "outputs": []} if kind == "code" else {})}

cells = [
cell("markdown", "# KirchhoffNet hardware retraining ladder\nOne rung per session; set `KNB_*` environment variables before headless execution.\n"),
cell("code", '''import os
RUNG = os.getenv("KNB_RUNG", "R0")
EPOCHS = int(os.getenv("KNB_EPOCHS", "300"))
SEED = int(os.getenv("KNB_SEED", "100"))
LR = float(os.getenv("KNB_LR", "0.007069359" if RUNG == "R0" else "0.03"))
STEPS_PER_STAGE = int(os.getenv("KNB_STEPS_PER_STAGE", "40"))  # 4x paper's 10 steps/stage on HW rungs
DATA_ROOT = os.getenv("KNB_DATA_ROOT", "/kaggle/input")
OUT_DIR = os.getenv("KNB_OUT_DIR", "/kaggle/working/out")
RESUME = os.getenv("KNB_RESUME", "")
SMOKE = os.getenv("KNB_SMOKE", "0").lower() in ("1", "true", "yes")
RUN_TOY_ABLATION = os.getenv("KNB_RUN_TOY_ABLATION", "0").lower() in ("1", "true", "yes")
if SMOKE:
    EPOCHS, LR = 1, float(os.getenv("KNB_LR", "0.03"))
'''),
cell("code", '''import glob, os, shutil, subprocess, sys, zipfile
from pathlib import Path
import torch
print("torch", torch.__version__, "CUDA", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0))
# Kaggle images may omit these runtime dependencies; torch is normally preinstalled.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torchdiffeq", "optuna", "joblib", "scikit-learn", "scipy", "matplotlib"])
data_root = Path(DATA_ROOT)
repo = Path("/kaggle/working/repo") if Path("/kaggle").exists() else Path(OUT_DIR) / "repo"
repo.parent.mkdir(parents=True, exist_ok=True)
inputs = list(data_root.glob("*"))
archives = [p for p in inputs if p.is_file() and p.suffix.lower() == ".zip"]
extracted = [p for p in inputs if p.is_dir() and ((p / "repo").is_dir() or (p / "MANIFEST.txt").exists())]
if archives:
    with zipfile.ZipFile(archives[0]) as zf: zf.extractall(repo.parent)
    if not repo.exists(): repo.mkdir(parents=True, exist_ok=True)
    bundle_root = repo.parent
elif extracted:
    bundle_root = extracted[0]
    if (bundle_root / "repo").is_dir(): repo = bundle_root / "repo"
else:
    raise FileNotFoundError(f"No bundle archive/directory found under {DATA_ROOT}")
DATA_CANONICAL = str(bundle_root / "data/canonical/canonical_ctle_phase_a_v2.npz")
TEACHER_DIR = str(bundle_root / "data/zig")
CSV_DIR = str(bundle_root / "data/ctle_csv")
assert Path(DATA_CANONICAL).is_file(), DATA_CANONICAL
assert Path(TEACHER_DIR).is_dir() and Path(CSV_DIR).is_dir()
os.chdir(repo)
print("repository", repo, "canonical", DATA_CANONICAL, "teacher/ZIG", TEACHER_DIR, "CSV", CSV_DIR)
'''),
cell("code", '''# Fast plumbing checks and exact paper-checkpoint validation.
subprocess.check_call([sys.executable, "-m", "pytest", "-q", "tests/test_hw_cli.py", "tests/test_kaggle_cli_resume.py"], cwd=repo)
eval_cmd = [sys.executable, "scripts/eval_paper_checkpoint.py", "--canonical-dataset", DATA_CANONICAL, "--zig-dir", TEACHER_DIR, "--threads", "2"]
eval_out = subprocess.check_output(eval_cmd, cwd=repo, text=True)
print(eval_out)
assert "val= 8.05%" in eval_out and "test= 8.15%" in eval_out
'''),
cell("code", '''# Execute one ladder rung. Stream output to console and durable log as it arrives.
from datetime import datetime
rungs = {"R0": None, "R1": "phase3a_core", "R1d": "phase3a_core_diff", "R2": "phase3a_full", "R3": "phase3b_tm"}
if RUNG not in rungs: raise ValueError(f"RUNG must be one of {list(rungs)}")
run_dir = Path(OUT_DIR) / f"{RUNG}_seed{SEED}"
run_dir.mkdir(parents=True, exist_ok=True)
cmd = [sys.executable, "dagger-nuance-distillation-kirchhoffnet.py",
 "--canonical-dataset", DATA_CANONICAL, "--phase-a-epochs", str(EPOCHS),
 "--kn-num-stages", "7", "--kn-num-hidden", "11", "--kn-small-world-k", "2", "--kn-small-world-p", "0.2", "--kn-vca-rank", "3",
 "--kn-x-max", "4", "--kn-gm-max", "4.241970", "--kn-isat-max", "16" if rungs[RUNG] else "28.68855",
 "--kn-mapper-lr-scale", "1", "--kn-struct-lr-scale", "4", "--kn-dyn-lr-scale", "1",
 "--lr", str(LR if RUNG != "R0" else float(os.getenv("KNB_LR", "0.007069359"))), "--weight-decay", "0.006087302", "--batch-size", "512", "--earlystop-eval-every", "5",
 "--phase-a-validity-weight", "0.3", "--phase-a-validity-ramp-start", "10", "--phase-a-validity-ramp-epochs", "30",
 "--phase-a-mlp-weight", "1", "--phase-a-fwd-weight", "1", "--phase-a-power-weight", "0.0005", "--phase-a-power-norm", "train_mean",
 "--kn-readout", "shared-x2", "--kn-learnable-clip-sharpness", "--kn-gln-rails", "--kn-gln-B", "4", "--kn-gln-rank", "2", "--kn-gln-families", "boundary,readout",
 "--t-span", "4.660583", "--boundary-fan-out", '{"0": [0, 4], "1": [1, 5], "2": [2, 6], "3": [3, 7]}',
 "--phase-a-mlp-teacher-ckpt", str(repo / "assets/teachers/trial_0019/dagger_student_plain.pt"),
 "--teacher-dir", TEACHER_DIR, "--data-dir", CSV_DIR, "--device", "cuda" if torch.cuda.is_available() else "cpu",
 "--seed", str(SEED), "--output", str(run_dir)]
if RUNG != "R0": cmd += ["--kn-steps-per-stage", str(STEPS_PER_STAGE)]
if rungs[RUNG]: cmd += ["--hw-preset", rungs[RUNG]]
if SMOKE: cmd += ["--phase-a-sample-limit", "32"]
if RESUME: cmd += ["--resume", RESUME]
log_path = run_dir / "train.log"
with open(log_path, "a", buffering=1) as log:
    proc = subprocess.Popen(cmd, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end=""); log.write(line)
    code = proc.wait()
if code: raise subprocess.CalledProcessError(code, cmd)
print("finished", datetime.now().isoformat(), run_dir)
'''),
cell("code", '''# Evaluate the final best model against the canonical 10% failure rule.
import json, csv, math
history_file = run_dir / "phase_a_history.json"
hist = json.loads(history_file.read_text())
final = hist["final"]
for split in ("train", "val", "test"):
    rate = final[split]["failure_rate"]
    print(split, f"failure rate at DEGRADE_REL_THRESHOLD=10%: {rate*100:.3f}%")
    assert math.isfinite(rate), f"{split} evaluation is non-finite"
best = run_dir / "phase_a_best.pt"
if rungs[RUNG]:
    audit = [sys.executable, "-m", "hw.audit", "--checkpoint", str(best), "--kn-isat-max", "16", "--kn-gm-max", "4.241970", "--canonical", DATA_CANONICAL, "--json", str(run_dir / "hw_audit.json")]
    subprocess.run(audit, cwd=repo, check=False)
try:
    import matplotlib.pyplot as plt
    h = hist["history"]
    for key in ("train_loss", "val_loss", "val_failure_rate"):
        plt.figure(); plt.plot(h["epoch"], h[key]); plt.xlabel("epoch"); plt.ylabel(key); plt.grid(True)
        plt.savefig(run_dir / f"{key}.png", dpi=140, bbox_inches="tight"); plt.close()
except Exception as exc: print("plot warning", exc)
result = {"rung": RUNG, "seed": SEED, "lr": LR, "epochs": EPOCHS, **{f"{s}_failure": final[s]["failure_rate"] for s in ("train", "val", "test")}, "checkpoint": str(best)}
csv_path = Path(OUT_DIR) / "results.csv"
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(result))
    if csv_path.stat().st_size == 0: writer.writeheader()
    writer.writerow(result)
print(result)
'''),
cell("code", '''# Optional toy ablation is deliberately isolated from the CTLE ladder.
if RUN_TOY_ABLATION:
    print("Toy ablation: follow docs/hw/cpu_learning_validation.md harness; run >=10 seeds, report mean and 95% CI.")
    print("The CPU validation harness defines the toy presets; no CTLE weights are changed here.")
'''),
cell("code", '''# Downloadable artifact for Kaggle Output / persistence.
archive = shutil.make_archive(str(Path(OUT_DIR).resolve()), "zip", root_dir=OUT_DIR)
print("results archive:", archive)
''')]
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10.0"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(OUT)
