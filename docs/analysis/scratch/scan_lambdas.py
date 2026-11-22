"""Audit-only: print LAMBDAS/OPTIM per commit + how train_script writes the preset line."""
import subprocess

REPO = r"C:\Users\Atharva\Documents\kirchhoffnet_realistic"

def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout

def load_config(src):
    ns = {"__name__": "cfg_audit"}
    try:
        exec(compile(src, "config.py", "exec"), ns)
        return ns
    except Exception as e:
        return {"__error__": f"{type(e).__name__}: {e}"}

rows = git("log", "--all", "--format=%H|%ad|%s", "--date=short", "--since=2026-07-01").strip().split("\n")
prev = None
print(f"{'sha':<9}{'date':<12}{'LAMBDAS (subset)':<120} note")
for line in rows:
    sha, date, subj = line.split("|", 2)
    cfg = git("show", f"{sha}:config.py")
    if not cfg or "skip_linear_l2" not in cfg:
        continue
    ns = load_config(cfg)
    if "__error__" in ns:
        continue
    lam = ns.get("LAMBDAS", {})
    opt = ns.get("OPTIM", {})
    sub = {k: lam.get(k) for k in ("sparsity", "edge_gate", "power", "entropy", "tanh_sat", "skip_linear_l2")}
    key = (tuple(sorted(sub.items())), opt.get("batch_size"), opt.get("lr"), opt.get("epochs"))
    if key != prev:
        print(f"{sha[:8]:<9}{date:<12}{str(sub):<120} bs={opt.get('batch_size')} lr={opt.get('lr')} ep={opt.get('epochs')} | {subj[:40]}")
        prev = key

print("\n=== how train_script writes the 'preset:' line (history) ===")
for sha, date, subj in [l.split("|", 2) for l in
                         git("log", "--all", "--format=%H|%ad|%s", "--date=short", "--", "train_script.py").strip().split("\n")]:
    ts = git("show", f"{sha}:train_script.py")
    if not ts:
        continue
    for pat in ('f.write(f"preset: {PRESETS[problem]}\\n\\n")',
                'f.write(f"preset: {preset}\\n\\n")',
                'f.write(f"preset: {eff_preset}\\n\\n")',
                '"preset: {"'):
        if pat.replace('\\n', '\n') in ts:
            print(f"{sha[:8]} {date} {pat!r}  <- {subj[:60]}")
            break
