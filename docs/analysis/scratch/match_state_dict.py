"""Audit-only: fingerprint repo commits by state_dict-bearing symbol presence.

For each commit we check the presence/absence of the exact attribute names that
appear in the artifact model.pt state_dicts, to bound the code version.
"""
import subprocess

REPO = r"C:\Users\Atharva\Documents\kirchhoffnet_realistic"

def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout

SMOOTH_DS = ["g_resistive_raw", "boundary_cell_lib", "output_ode_cell_lib",
             "z_logits", "u_logits", "ref_dst", "boundary_z_logits"]
SMOOTH_DS_ABS = ["cell_type_code", "_is_rect"]
SMOOTH_IO = ["OutputAffine", "NullInputMapper", "FanOutInputMapper"]
SMOOTH_ST = ["residual_w1", "residual_vth", "residual_w3"]

DEQ_DS = ["raw_drive_g", "cell_type_code", "_is_rect", "raw_mult", "raw_leak"]
DEQ_DS_ABS = ["boundary_cell_lib", "output_ode_cell_lib", "g_resistive_raw"]

def probe(sha, files, need, absent=()):
    blob = ""
    for f in files:
        blob += git("show", f"{sha}:{f}") or ""
    miss = [t for t in need if t not in blob]
    bad = [t for t in absent if t in blob]
    return miss, bad

commits = git("log", "--all", "--format=%H|%ad|%s", "--date=short").strip().split("\n")
rows = []
for line in commits:
    sha, date, subj = line.split("|", 2)
    smiss, sbad = probe(sha, ["differential_stage.py", "io_mapper.py", "stage_transfer.py"],
                        SMOOTH_DS + SMOOTH_IO + SMOOTH_ST, SMOOTH_DS_ABS)
    dmiss, dbad = probe(sha, ["differential_stage.py", "io_mapper.py"], DEQ_DS, DEQ_DS_ABS)
    rows.append((sha[:8], date, subj[:54], smiss, sbad, dmiss, dbad))

def show(title, idx_miss, idx_bad, filter_fn):
    print(f"\n=== {title} ===")
    for r in rows:
        miss, bad = r[idx_miss], r[idx_bad]
        if filter_fn(miss, bad):
            print(f"{r[0]} {r[1]} missing={miss} present_but_should_be_absent={bad} | {r[2]}")

show("SMOOTH artifact: all smooth2d symbols present, none of the old ones",
     3, 4, lambda miss, bad: not miss and not bad)
show("DEQ artifact: all deq symbols present, no boundary/output_ode/g_resistive",
     5, 6, lambda miss, bad: not miss and not bad)
