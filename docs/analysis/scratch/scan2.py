"""Audit-only: (a) re-probe DEQ symbols including cell_library.py,
(b) show the exact 'preset:' write line per commit in the Sept window."""
import subprocess

REPO = r"C:\Users\Atharva\Documents\kirchhoffnet_realistic"

def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout

commits = git("log", "--all", "--format=%H|%ad|%s", "--date=short").strip().split("\n")
print("=== DEQ symbol probe (incl. cell_library.py) ===")
for line in commits:
    sha, date, subj = line.split("|", 2)
    blob = ""
    for f in ("differential_stage.py", "io_mapper.py", "cell_library.py"):
        blob += git("show", f"{sha}:{f}") or ""
    need = ["raw_drive_g", "cell_type_code", "_is_rect", "raw_mult", "raw_leak"]
    absent = ["boundary_cell_lib", "output_ode_cell_lib", "g_resistive_raw"]
    miss = [t for t in need if t not in blob]
    bad = [t for t in absent if t in blob]
    if not miss and not bad:
        print(f"  MATCH {sha[:8]} {date} | {subj[:60]}")

print("\n=== 'preset:' write line in train_script.py (Sept window + Jul/Aug) ===")
seen = set()
for line in commits:
    sha, date, subj = line.split("|", 2)
    if date < "2026-07-10":
        continue
    ts = git("show", f"{sha}:train_script.py")
    if not ts:
        continue
    hits = [ln.strip() for ln in ts.split("\n") if 'preset:' in ln and 'f.write' in ln]
    for h in hits:
        if h not in seen:
            seen.add(h)
            print(f"  {sha[:8]} {date} :: {h[:110]} | {subj[:40]}")
