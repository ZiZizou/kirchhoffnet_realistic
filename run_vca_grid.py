"""A-E VCA experiment grid (core-vca-expansion plan), friedman2.

Runs five configurations x 3 seeds through train_script.py with the shared
validation protocol, then aggregates best-val from each run's
final_metrics.txt into a mean +/- std table.

| Run | Boundary/Readout VCA | Core VCA | Shunt | Purpose |
|-----|----------------------|----------|-------|---------|
|  A  | OFF                  | OFF      |  --   | Baseline (reproduce ~6.65e-5) |
|  B  | ON  (temporal-readout family) | OFF | -- | init-res boundary/readout VCA |
|  C  | OFF | ON  | -- | core VCA alone — clean test |
|  D  | ON  (temporal-readout)        | ON  | -- | full VCA |
|  E  | OFF | ON  | ON  | input-dependent mixing (routing) |

Protocol (matches the friedman2 baseline regression run):
  800 epochs, batch 4096 (OPTIM default), lr 0.0012, struct_lr_scale 4.0,
  huber, sparse_proj/dense read, small_world k=4 p=0.2 seed=100,
  edge_repeats=2, freeze_read ON, core gating fold into i_edge_const.

Usage:
  python run_vca_grid.py                     # dry-run: print commands
  python run_vca_grid.py --dry-run           # same
  python run_vca_grid.py --launch            # actually run each command
  python run_vca_grid.py --only B,C --seed 41   # subset + single seed
  python run_vca_grid.py --agg ./grid_out    # aggregate already-run results
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import statistics
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Run definitions
# ---------------------------------------------------------------------------

GRID = {
    "A": {
        "name": "baseline",
        "vca": False,
        "temporal_readout": False,   # NOT used for A (matches baseline protocol)
        "vca_core": False,
        "vca_gate_shunt": False,
        "expected_epoch0": "~6.65e-5 baseline",
    },
    "B": {
        "name": "boundary-readout-vca",
        "vca": True,
        "temporal_readout": True,    # readout edge family is the original VCA target
        "vca_core": False,
        "vca_gate_shunt": False,
    },
    "C": {
        "name": "core-vca",
        "vca": True,
        "temporal_readout": False,
        "vca_core": True,
        "vca_gate_shunt": False,
    },
    "D": {
        "name": "full-vca",
        "vca": True,
        "temporal_readout": True,
        "vca_core": True,
        "vca_gate_shunt": False,
    },
    "E": {
        "name": "core-vca-shunt",
        "vca": True,
        "temporal_readout": False,
        "vca_core": True,
        "vca_gate_shunt": True,
    },
}

# Same protocol for every run:
#   friedman2, 800 epochs, lr 0.0012, struct_lr_scale 4.0, huber,
#   small_world k=4 p=0.2 seed=100, edge_repeats=2.
BASE = [
    "train_script.py",
    "--problem", "friedman2",
    "--cell-library", "tanh",
    "--epochs", "800",
    "--lr", "0.0012",
    "--struct-lr-scale", "4.0",
    "--hidden-family", "small_world",
    "--num-hidden", "16",
    "--small-world-k", "4",
    "--small-world-p", "0.2",
    "--small-world-seed", "100",
    "--edge-repeats", "2",
    "--freeze-read",
    "--vca-diag",
]

SEEDS = [0, 1, 2]


def build_command(run_key: str, seed: int, out_root: str) -> list[str]:
    c = GRID[run_key]
    argv = list(BASE)
    out_dir = os.path.join(out_root, f"{run_key}_{c['name']}", f"seed{seed}")
    argv += ["--output", out_dir, "--seed", str(seed)]
    if c["vca"]:
        argv.append("--vca")
    if c["temporal_readout"]:
        argv.append("--temporal-readout")
    if c["vca_core"]:
        argv.append("--vca-core")
    if c["vca_gate_shunt"]:
        argv.append("--vca-gate-shunt")
    return argv


def parse_best_val(out_dir: str) -> float | None:
    path = os.path.join(out_dir, "final_metrics.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            m = re.match(r"best_val:\s*([-+eE0-9.]+)", line.strip())
            if m:
                return float(m.group(1))
    return None


def _seed_dirs(out_root: str, run_key: str) -> list[str]:
    """Run dir + any timestamped re-runs (_ensure_dir appends _<stamp> when the
    requested path already exists). Return the newest per seed."""
    pat = os.path.join(out_root, f"{run_key}_{GRID[run_key]['name']}", "seed*")
    dirs = sorted(glob.glob(pat))
    # group by seed base (strip any _timestamp tail) and keep newest
    by_seed: dict[str, list[str]] = {}
    for d in dirs:
        name = os.path.basename(d)
        if re.match(r"^seed\d+(_\d{8}_\d{6})?$", name):
            seed_base = name.split("_")[0]
            by_seed.setdefault(seed_base, []).append(d)
    out = []
    for seed_base, cands in sorted(by_seed.items()):
        out.append(max(cands, key=lambda d: d))  # timestamp suffix sorts later
    return out


def aggregate(out_root: str, runs: list[str]) -> None:
    print(f"\n=== Aggregation ({out_root}) ===")
    for r in runs:
        vals = []
        for d in _seed_dirs(out_root, r):
            v = parse_best_val(d)
            if v is not None:
                vals.append(v)
        n = len(vals)
        if n == 0:
            print(f"  {r:>4s}  {GRID[r]['name']:<22s}  n/a (no final_metrics.txt)")
            continue
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if n > 1 else 0.0
        lab = GRID[r]["name"]
        print(f"  {r:>4s}  {lab:<22s}  best_val {mean:.6e} +/- {sd:.6e}  (n={n})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", default=True,
                    help="print commands without running (default)")
    ap.add_argument("--launch", action="store_true", default=False,
                    help="actually execute each run sequentially")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated run keys (e.g. 'A,C')")
    ap.add_argument("--seed", type=int, default=None,
                    help="single seed override (default: 0,1,2)")
    ap.add_argument("--out", type=str, default=os.path.join(THIS_DIR, "grid_out"),
                    help="output root (default: ./grid_out)")
    ap.add_argument("--agg", type=str, default=None,
                    help="aggregate existing results under this dir and exit")
    args = ap.parse_args()

    if args.agg:
        aggregate(args.agg, list(GRID.keys()))
        return

    runs = args.only.split(",") if args.only else list(GRID.keys())
    for r in runs:
        r = r.strip()
        if r not in GRID:
            raise SystemExit(f"unknown run key {r!r}; choose from {list(GRID)}")
    seeds = [args.seed] if args.seed is not None else SEEDS

    print(f"[grid] {len(runs)} runs x {len(seeds)} seeds -> {args.out}")
    for r in runs:
        for sd in seeds:
            cmd = build_command(r, sd, args.out)
            cmd_str = " ".join(cmd)
            if args.launch:
                print(f"[run]  {os.path.join(args.out, r, str(sd))}")
                subprocess.run(cmd, cwd=THIS_DIR, check=True)
            else:
                print(f"[DRY]  {cmd_str}")

    if not args.launch:
        print("\n(dry-run; pass --launch to run)")

    if args.launch and args.seed is None:
        aggregate(args.out, runs)


if __name__ == "__main__":
    main()