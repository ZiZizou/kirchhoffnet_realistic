"""Audit-only: find which repo commits match the artifact config snapshots.

For every commit that touches config.py (plus the full log), we:
  * extract config.py via `git show <c>:config.py`
  * exec it in a fresh namespace (it only imports torch inside functions)
  * compare LAMBDAS / OPTIM / PRESETS['smooth2d'] against the artifact snapshots
  * check train_script.py's _save_config_snapshot for the I/O MAPPING key set
"""
import json
import re
import subprocess
import sys

REPO = r"C:\Users\Atharva\Documents\kirchhoffnet_realistic"

def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout

# --- target values from artifacts -------------------------------------------------
# test_smooth2d_w_tanh/config_snapshot.txt
TGT_SMOOTH2D_LAMBDAS = {
    "sparsity": 0, "rail": 0.1, "edge_gate": 0, "node_gate": 0.0,
    "power": 0, "capacitance": 0.0, "entropy": 0.0001,
    "tanh_sat": 0.0, "skip_linear_l2": 0.0001,
}
TGT_OPTIM = {
    "weight_decay": 0.0001, "grad_clip_norm": 5.0, "epochs": 800,
    "batch_size": 4096, "lr": 0.0012, "reg_warmup_epochs": 100,
    "reg_anneal_epochs": 50, "scheduler_T_0": 50, "scheduler_T_mult": 1,
    "scheduler_eta_min": 1e-05,
}
TGT_PRESET_KEYS = {"use_robust_input", "loss", "out_dim", "write_idx", "read_idx",
                   "write_mode", "read_mode", "leak_mode"}
TGT_STAGE = {"num_inputs": 2, "num_hidden": 8, "num_proj": 0, "num_outputs": 0,
             "hidden_family": "small_world",
             "hidden_kwargs": {"k": 4, "p": 0.2, "seed": 100, "bidirectional": False},
             "edge_repeats": 2, "input_pattern": "all_to_all",
             "output_pattern": "all_to_all", "proj_pattern": "all_to_all",
             "t_span": 1.0, "num_steps": 10}
TGT_PRESET_META = {"use_robust_input": False, "loss": "huber", "out_dim": 1,
                   "write_idx": [0, 1], "read_idx": [0, 1, 2, 3, 4, 5, 6, 7],
                   "write_mode": "dense", "read_mode": "dense",
                   "leak_mode": "non-programmable"}

# deq/config_snapshot.txt (notebook-written config.py)
TGT_DEQ_LAMBDAS = {"sparsity": 0.001, "rail": 0.1, "edge_gate": 0.0005,
                    "node_gate": 0.0, "power": 0.0001, "capacitance": 0.0,
                    "entropy": 0.0001}

def load_config(src):
    ns = {"__name__": "cfg_audit"}
    try:
        exec(compile(src, "config.py", "exec"), ns)
        return ns
    except Exception as e:
        return {"__error__": f"{type(e).__name__}: {e}"}

def cmp_dict(got, want):
    if not isinstance(got, dict):
        return None
    keys_ok = set(got) == set(want)
    vals_ok = all(abs(float(got.get(k, 1e30)) - float(v)) < 1e-12 for k, v in want.items())
    return keys_ok and vals_ok

commits = git("log", "--all", "--format=%H %ad %s", "--date=short").strip().split("\n")
print(f"total commits across all refs: {len(commits)}")

rows = []
for line in commits:
    sha, date, subj = line.split(" ", 2)
    cfg = git("show", f"{sha}:config.py")
    if not cfg:
        continue
    if "skip_linear_l2" not in cfg:
        continue
    ns = load_config(cfg)
    if "__error__" in ns:
        continue
    lam = ns.get("LAMBDAS", {})
    opt = ns.get("OPTIM", {})
    pres = ns.get("PRESETS", {}).get("smooth2d")
    if pres is None:
        continue
    ts = git("show", f"{sha}:train_script.py")
    snap_has_readout = "f.write(f\"readout:" in ts
    snap_has_persistent = "persistent_drive" in ts
    snap_has_drive_mode = "drive_mode" in ts
    row = {
        "sha": sha[:7], "date": date, "subj": subj[:60],
        "lam_smooth2d": cmp_dict(lam, TGT_SMOOTH2D_LAMBDAS),
        "optim_smooth2d": cmp_dict(opt, TGT_OPTIM),
        "lam_deq": cmp_dict(lam, TGT_DEQ_LAMBDAS),
        "preset_meta_ok": all(pres.get(k) == v for k, v in TGT_PRESET_META.items()),
        "preset_keys_ok": TGT_PRESET_KEYS.issubset(set(pres)),
        "n_stages": len(pres.get("stages", [])),
        "stage0_ok": (len(pres.get("stages", [])) > 0 and
                      pres["stages"][0] == TGT_STAGE),
        "snap_readout": snap_has_readout,
        "snap_persistent": snap_has_persistent,
        "snap_drive_mode": snap_has_drive_mode,
        "has_small_world": "small_world" in cfg,
    }
    rows.append(row)

print("\n=== commits whose config.py mentions skip_linear_l2 ===")
hdr = ("sha", "date", "lamS", "optS", "lamDEQ", "presetMeta", "presetKeys", "nSt",
       "stage0", "readout", "persist", "driveMode", "smallworld")
print("{:<8} {:<11} {:<5} {:<5} {:<6} {:<10} {:<10} {:<4} {:<6} {:<7} {:<6} {:<9} {}".format(*hdr))
for r in rows:
    def b(x):
        return "Y" if x is True else ("N" if x is False else "-")
    print("{sha:<8} {date:<11} {lam:<5} {opt:<5} {lamd:<6} {pm:<10} {pk:<10} {ns:<4} {s0:<6} {ro:<7} {pd:<6} {dm:<9} {sw}".format(
        sha=r["sha"], date=r["date"], lam=b(r["lam_smooth2d"]), opt=b(r["optim_smooth2d"]),
        lamd=b(r["lam_deq"]), pm=b(r["preset_meta_ok"]), pk=b(r["preset_keys_ok"]),
        ns=r["n_stages"], s0=b(r["stage0_ok"]), ro=b(r["snap_readout"]),
        pd=b(r["snap_persistent"]), dm=b(r["snap_drive_mode"]),
        sw=b(r["has_small_world"]), subj=r["subj"]))
