"""Audit-only: timeline of config.py knobs around the smooth2d artifact window."""
import subprocess

REPO = r"C:\Users\Atharva\Documents\kirchhoffnet_realistic"

def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout

def load(src):
    ns = {"__name__": "cfg"}
    try:
        exec(compile(src, "config.py", "exec"), ns)
        return ns
    except Exception as e:
        return {"__error__": f"{type(e).__name__}"}

rows = git("log", "--all", "--format=%H|%ad|%s", "--date=short", "--since=2026-06-15").strip().split("\n")
prev = None
for line in rows:
    sha, date, subj = line.split("|", 2)
    cfg = git("show", f"{sha}:config.py")
    if not cfg:
        continue
    ns = load(cfg)
    if "__error__" in ns:
        continue
    lam = ns.get("LAMBDAS", {})
    opt = ns.get("OPTIM", {})
    solver = ns.get("SOLVER", {})
    pres = ns.get("PRESETS", {}).get("smooth2d", {})
    stages = pres.get("stages", []) if isinstance(pres, dict) else []
    st0 = stages[0] if stages else {}
    key = (opt.get("batch_size"), lam.get("sparsity"), lam.get("power"), lam.get("entropy"),
           lam.get("tanh_sat"), lam.get("skip_linear_l2"), len(stages),
           st0.get("hidden_family"), st0.get("t_span"), st0.get("num_steps"),
           solver.get("t_span"), solver.get("num_steps"),
           ("tanh_free" in cfg), ("VCA" in ns), ("REF" in ns), ("DEGREE_BUDGET" in ns),
           ("leak_constant" in cfg))
    if key != prev:
        print(f"{sha[:8]} {date} bs={key[0]} sp={key[1]} pw={key[2]} ent={key[3]} tsat={key[4]} skl={key[5]} "
              f"nSt={key[6]} fam={key[7]} st_tspan={key[8]} st_steps={key[9]} SOLVER=({key[10]},{key[11]}) "
              f"tanh_free={key[12]} VCA={key[13]} REF={key[14]} BUDGET={key[15]} leakconst={key[16]} | {subj[:45]}")
        prev = key
