"""Audit-only: find the mlp_benchmark.py variant matching the friedman2_normal artifact."""
import subprocess

REPO = r"C:\Users\Atharva\Documents\kirchhoffnet_realistic"

def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout

commits = git("log", "--all", "--format=%H|%ad|%s", "--date=short", "--", "mlp_benchmark.py").strip().split("\n")
prev = None
for line in commits:
    sha, date, subj = line.split("|", 2)
    src = git("show", f"{sha}:mlp_benchmark.py")
    if not src:
        continue
    block = []
    grab = False
    for ln in src.split("\n"):
        if "final_metrics.txt" in ln:
            grab = True
        if grab:
            block.append(ln.strip())
            if len(block) > 22:
                break
    sig = " | ".join(b for b in block if b.startswith("f.write"))
    if sig and sig != prev:
        print(f"{sha[:8]} {date} :: {sig[:260]}")
        print(f"        | {subj[:70]}")
        prev = sig

print("\n=== dataset line in config_snapshot per commit (first occurrence) ===")
prev = None
for line in commits:
    sha, date, subj = line.split("|", 2)
    src = git("show", f"{sha}:mlp_benchmark.py")
    if not src:
        continue
    dl = [l.strip() for l in src.split("\n") if "dataset:" in l and "f.write" in l]
    if dl and dl[0] != prev:
        print(f"{sha[:8]} {date} :: {dl[0][:120]}")
        prev = dl[0]
