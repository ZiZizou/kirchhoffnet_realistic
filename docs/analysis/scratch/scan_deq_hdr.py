"""Audit-only: print the exact deq_validation_history header line per commit."""
import subprocess

REPO = r"C:\Users\Atharva\Documents\kirchhoffnet_realistic"

def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout

commits = git("log", "--all", "--format=%H|%ad|%s", "--date=short", "--", "train_script.py").strip().split("\n")
prev = None
for line in commits:
    sha, date, subj = line.split("|", 2)
    ts = git("show", f"{sha}:train_script.py")
    if not ts:
        continue
    hdr = None
    for ln in ts.split("\n"):
        if "residual_mean" in ln and ("epoch" in ln):
            hdr = ln.strip()
            break
    if hdr and hdr != prev:
        print(f"{sha[:8]} {date} :: {hdr[:150]}")
        print(f"        | {subj[:70]}")
        prev = hdr
