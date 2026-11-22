"""Audit-only: fingerprint the DEQ-branch lineage against the DEQ artifact."""
import re
import subprocess

REPO = r"C:\Users\Atharva\Documents\kirchhoffnet_realistic"

def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout

commits = git("log", "--oneline", "--format=%H|%ad|%s", "--date=short", "origin/deq").strip().split("\n")
print(f"{'sha':<9}{'date':<12}{'hdr':<46}{'after_phase_ab':<16}{'mem':<6}{'grad=':<7}{'gridprint':<10} note")
for line in commits:
    sha, date, subj = line.split("|", 2)
    ts = git("show", f"{sha}:train_script.py")
    if not ts:
        continue
    hdr = ""
    m = re.search(r'f\.write\(\s*"([^"]*epoch[^"]*)"', ts)
    if m:
        hdr = m.group(1).replace("\t", "|")
    ap = "yes" if "after_phase_ab" in ts else ("no" if "after_phase" in ts else "-")
    mem = "yes" if "[mem]" in ts else "-"
    gr = "yes" if "grad=" in ts else "-"
    gp = "yes" if "grid" in ts and "×" in ts else "-"
    print(f"{sha[:8]:<9}{date:<12}{hdr[:44]:<46}{ap:<16}{mem:<6}{gr:<7}{gp:<10} {subj[:50]}")
