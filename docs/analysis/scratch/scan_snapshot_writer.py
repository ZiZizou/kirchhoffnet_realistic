"""Audit-only: fingerprint the config_snapshot writer across all commits.

Extracts the body of _save_config_snapshot from train_script.py at every commit
and prints the ordered list of snapshot keys it writes, so we can match the
three artifact snapshots (smooth2d / deq / friedman2-mlp) to code versions.
"""
import re
import subprocess

REPO = r"C:\Users\Atharva\Documents\kirchhoffnet_realistic"

def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout

TARGET_SMOOTH = [
    "problem", "preset", "LAMBDAS (global)", "LAMBDAS (effective", "OPTIM",
    "VARIATION", "I/O MAPPING", "write_mode", "input_mapper", "write_idx",
    "persistent_drive", "drive_mode", "output_mapper", "read_idx", "hid_count",
    "proj_count", "stage_lr_scale", "mapper_lr_scale",
    "retrain_stage_lr_scale", "retrain_mapper_lr_scale", "freeze_mappers",
]
TARGET_DEQ = [
    "problem", "preset", "LAMBDAS (global)", "LAMBDAS (effective", "OPTIM",
    "VARIATION", "I/O MAPPING", "write_mode", "input_mapper", "fan_out_map",
    "write_idx", "output_mapper", "read_idx", "hid_count", "proj_count",
    "stage_lr_scale", "mapper_lr_scale",
    "retrain_stage_lr_scale", "retrain_mapper_lr_scale", "freeze_mappers",
]

def snapshot_keys(ts_src):
    m = re.search(r"def _save_config_snapshot\(.*?\n(.*?)(?=\ndef |\nclass |\nif __name__)", ts_src, re.S)
    if not m:
        return None
    body = m.group(1)
    keys = []
    for w in re.finditer(r'f\.write\(f?"?\s*\\n?\s*([A-Za-z ()/{}:._-]+?):', body):
        k = w.group(1).strip()
        if k and k not in keys:
            keys.append(k)
    for w in re.finditer(r'f\.write\(f"\\n([A-Z][A-Za-z ()/{}:._-]+?):', body):
        k = w.group(1).strip()
        if k not in keys:
            keys.append(k)
    return keys

def score(keys, target):
    if keys is None:
        return -1
    s = set(keys)
    t = set(target)
    inter = len(s & t)
    miss = len(t - s)
    extra = len(s - t)
    return inter - miss - 0.25 * extra

commits = git("log", "--all", "--format=%H|%ad|%s", "--date=short", "--", "train_script.py").strip().split("\n")
best = []
for line in commits:
    sha, date, subj = line.split("|", 2)
    ts = git("show", f"{sha}:train_script.py")
    if not ts:
        continue
    keys = snapshot_keys(ts)
    if keys is None:
        continue
    ss = score(keys, TARGET_SMOOTH)
    sd = score(keys, TARGET_DEQ)
    best.append((ss, sd, sha[:8], date, keys, subj[:52]))

best.sort(key=lambda r: -r[0])
print("=== TOP 12 by smooth2d-artifact snapshot similarity ===")
for ss, sd, sha, date, keys, subj in best[:12]:
    print(f"{sha} {date} smooth_score={ss:6.1f} deq_score={sd:6.1f} | {subj}")
    print(f"    keys: {keys}")
best.sort(key=lambda r: -r[1])
print("\n=== TOP 12 by deq-artifact snapshot similarity ===")
for ss, sd, sha, date, keys, subj in best[:12]:
    print(f"{sha} {date} deq_score={sd:6.1f} smooth_score={ss:6.1f} | {subj}")
    print(f"    keys: {keys}")
