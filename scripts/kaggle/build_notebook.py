#!/usr/bin/env python3
"""Build the clean Kaggle HW retraining notebook using only stdlib json."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "kaggle_hw_retrain.ipynb"

def cell(kind, text):
    cell_id = hashlib.sha1((kind + "\\0" + text).encode("utf-8")).hexdigest()[:8]
    return {"cell_type": kind, "id": cell_id, "metadata": {}, "source": text.splitlines(True),
            **({"execution_count": None, "outputs": []} if kind == "code" else {})}

cells = [
cell("markdown", "# KirchhoffNet hardware retraining ladder\nOne rung per session; set `KNB_*` environment variables before headless execution.\n"),
cell("code", '''import os
from pathlib import Path
RUNG = os.getenv("KNB_RUNG", "R0")
EPOCHS = int(os.getenv("KNB_EPOCHS", "300"))
SEED = int(os.getenv("KNB_SEED", "100"))
LR = float(os.getenv("KNB_LR", "0.007069359" if RUNG == "R0" else "0.03"))
STEPS_PER_STAGE = int(os.getenv("KNB_STEPS_PER_STAGE", "40"))  # 4x paper's 10 steps/stage on HW rungs
OUT_DIR = os.getenv("KNB_OUT_DIR", "/kaggle/working/out")
RESUME = os.getenv("KNB_RESUME", "")
SMOKE = os.getenv("KNB_SMOKE", "0").lower() in ("1", "true", "yes")
RUN_TOY_ABLATION = os.getenv("KNB_RUN_TOY_ABLATION", "0").lower() in ("1", "true", "yes")
TEACHER_CKPT = Path(os.getenv("KNB_TEACHER_CKPT", "/kaggle/input/models/awekill/improved-zig-nf-spline/pytorch/default/4/dagger_student_plain.pt"))
TEACHER_DIR = Path(os.getenv("KNB_TEACHER_DIR", str(TEACHER_CKPT.parent)))
CSV_DIR = Path(os.getenv("KNB_CSV_DIR", "/kaggle/input/datasets/awekill/augmented-cvae-ctle"))
CANONICAL = Path(os.getenv("KNB_CANONICAL", "/kaggle/working/kirchhoffnet_realistic/assets/canonical/canonical_ctle_phase_a_v2.npz"))
if SMOKE:
    EPOCHS, LR = 1, float(os.getenv("KNB_LR", "0.03"))
'''),
cell("code", '''import os, sys
from kaggle_secrets import UserSecretsClient

# Retrieve the token securely
user_secrets = UserSecretsClient()
token = user_secrets.get_secret("GITHUB_TOKEN")
branch = "hw-refactor"  # <--- Specify your branch here ("main" or "deq")

# Define your GitHub details
username = "ZiZizou"
repo = "kirchhoffnet_realistic"

# Clone the specific branch using the -b flag
os.system(f"git clone -b {branch} https://{username}:{token}@github.com/{username}/{repo}.git")


# The exact path to your cloned repository
repo_path = "/kaggle/working/"+repo

if repo_path not in sys.path:
    sys.path.insert(0, repo_path)
'''),
cell("code", '''import os, shutil, subprocess, sys
from pathlib import Path
import torch
print("torch", torch.__version__, "CUDA", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0))
# Kaggle images may omit these runtime dependencies; torch is normally preinstalled.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torchdiffeq", "optuna", "joblib", "scikit-learn", "scipy", "matplotlib"])
repo = Path(repo_path)
os.chdir(repo)
print("repository", repo, "canonical", CANONICAL, "teacher/ZIG", TEACHER_DIR, "CSV", CSV_DIR)
'''),
cell("code", '''# Required Kaggle inputs: verify files before evaluation or training.
import hashlib
CSV_FILES = [
    "dataset_log_part_1_may3.csv", "dataset_log_part_1_may4.csv",
    "dataset_log_part_2_may3.csv", "dataset_log_part_2_may4.csv",
    "dataset_log_part_3_may3.csv", "dataset_log_part_3_may4.csv",
    "dataset_log_part_4_may3.csv", "dataset_log_part_4_may4.csv",
    "dataset_log_part_5_may3.csv", "dataset_log_part_5_may4.csv",
    "dataset_log_part_6_may3.csv", "dataset_log_part_6_may4.csv",
    "dataset_log_may_7.csv", "dataset_log_march12.csv", "dataset_log_may_18.csv",
]
TEACHER_FILES = [
    "ctle_conditional_flow.pt", "flow_scaler_X.pkl", "flow_scaler_C.pkl",
    "hybrid_hurdle_ctle_model.pt", "hybrid_hurdle_scaler_X.pkl",
    "hybrid_hurdle_scaler_y_power.pkl", "hybrid_hurdle_config.pkl",
]
EXPECTED_INPUT_HASHES = {
    "dagger_student_plain.pt": "fa7391fc70be8818b6b89c2199447d2651c47f77f57a9fe087cf230c82d9c142",
    "ctle_conditional_flow.pt": "1d942852ac96bcfed08a76edee77179fed6f4863c1ca5bf6e8a81608c7c8e726",
    "flow_scaler_X.pkl": "109c6c45b889a7df652be4abe1ee411fd152f94aa6d7982b06e54848f579f685",
    "flow_scaler_C.pkl": "deee8f5125b4196cf3fbcd1a5ffc7bf6e90471bbb2b88a3b4f802b9571a00c7b",
    "hybrid_hurdle_ctle_model.pt": "9ea4d46a1e1b96dbc2ffcb18d9626b539ac52de0aa3f1cd4af331be362c5fda7",
    "hybrid_hurdle_scaler_X.pkl": "109c6c45b889a7df652be4abe1ee411fd152f94aa6d7982b06e54848f579f685",
    "hybrid_hurdle_scaler_y_power.pkl": "898dd523a3d6f66e3b7f9328fccf6e0f296f6a3e9c95dae1ef2aad4a6e21e9c1",
    "hybrid_hurdle_config.pkl": "1456a0d09140f9c36ae5e696e78de34525777e76432bf06bdd0a39792c1e395f",
}
required = [TEACHER_CKPT, CANONICAL] + [TEACHER_DIR / name for name in TEACHER_FILES] + [CSV_DIR / name for name in CSV_FILES]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError("Missing required Kaggle inputs:\\n" + "\\n".join(missing))
for path in [TEACHER_CKPT] + [TEACHER_DIR / name for name in TEACHER_FILES]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = EXPECTED_INPUT_HASHES.get(str(path), EXPECTED_INPUT_HASHES.get(path.name, "not recorded"))
    print(f"sha256 {path}: {digest} (expected reference: {expected})")
    if expected != "not recorded" and digest != expected:
        print(f"WARNING: {path.name} hash differs from ssr0 reference; Kaggle model version 4 may differ.")
DATA_CANONICAL = str(CANONICAL)
TEACHER_CKPT = str(TEACHER_CKPT)
TEACHER_DIR = str(TEACHER_DIR)
CSV_DIR = str(CSV_DIR)
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
 "--phase-a-mlp-teacher-ckpt", str(TEACHER_CKPT),
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
    subprocess.run(audit, cwd=repo, check=True)
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
    ablation_dir = Path(OUT_DIR) / "toy_ablation"
    seeds = int(os.getenv("KNB_TOY_SEEDS", "10"))
    steps = int(os.getenv("KNB_TOY_STEPS", "400"))
    jobs = int(os.getenv("KNB_TOY_JOBS", str(max(1, min(32, (os.cpu_count() or 2) // 2)))))
    subprocess.check_call([sys.executable, str(repo / "scripts/hw_toy_ablation.py"),
        "--seeds", str(seeds), "--steps", str(steps), "--jobs", str(jobs),
        "--threads-per-job", "1", "--out-dir", str(ablation_dir)], cwd=repo)
    print("toy ablation reports:", ablation_dir / "toy_ablation_summary.md")
'''),
cell("code", '''# Downloadable artifact for Kaggle Output / persistence.
archive = shutil.make_archive(str(Path(OUT_DIR).resolve()), "zip", root_dir=OUT_DIR)
print("results archive:", archive)
''')]
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10.0"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(OUT)
