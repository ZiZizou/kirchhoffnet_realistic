"""hw/audit.py on the paper checkpoint (Phase 3b W4 step 4).  Small row counts: fast."""

import json
import os

import pytest
import torch

from hw import HWConfig
from hw import audit as A

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "assets", "baselines", "paper_trial_0011", "phase_a_best.pt")


@pytest.fixture(scope="module")
def rows():
    return A.canonical_rows(16)


def test_audit_table_on_paper_checkpoint(rows):
    res = A.audit(CKPT, HWConfig(), rows, stiff_rows=8)
    checks = {r[0] for r in res["rows"]}
    assert {"slope gm*max(A,B)", "rail max|x|", "Heun max(dt*rho)", "Isat range",
            "Isat_eff range", "A/B ratio max(A/B,B/A)"} <= checks
    assert all(r[5] in ("PASS", "WARN", "FAIL") for r in res["rows"])
    assert res["y_finite"] and len(res["rails"]) == len(res["stiffness"]) == 7
    # the paper model was trained without the hardware constraints: its Isat
    # sits partly below the 0.6 floor (Phase 3b doc: core Isat 0.18-2.69)
    lo, hi = res["edges"]["core"]["isat"]
    assert lo == pytest.approx(0.176, abs=2e-3) and hi == pytest.approx(2.69, abs=1e-2)
    # Heun is far from its stability limit (dt*rho << 2)
    assert max(r["max_dt_rho"] for r in res["stiffness"]) < 0.5


def test_capture_restores_rhs(rows):
    m = A.load_model(CKPT, None)
    y1 = m(rows)
    A.run_capture(m, rows)
    assert all("rhs" not in st.__dict__ for st in m.net.core.stages)
    torch.testing.assert_close(m(rows), y1, rtol=0, atol=0)


def test_sensitivity_droop_reaches_the_core_read(rows):
    s = A.sensitivity(CKPT, rows, mv=1.0)
    assert s["identity_max_abs"] == 0.0           # tm on at 0 mV == legacy, bit for bit
    assert s["inject"]["dy_per_mv"] > 0 and s["droop"]["dy_per_mv"] > 0
    # trained model: droop is a few % of injection, not ~1e-5 as at raw init
    assert 1e-2 < s["droop_over_inject"] < 1.0
    assert s["inject"]["stage_dx_per_mv"][0] == 0.0  # nothing before stage 0


def test_raw_init_droop_cancels_in_the_repeated_edge_pairs(rows):
    """At raw init every core cell is identical except the STE sign s; with
    edge_repeats=2 a (src,dst) pair with opposite s cancels exactly (tanh is odd),
    so a node whose pairs all have opposite signs has zero core-read gain."""
    torch.manual_seed(0)
    m = A.load_model(None, None, dtype=torch.float64)
    _, recs, _ = A.run_capture(m, rows.double())
    st = m.net.core.stages[1]
    x0, u, _ = recs[1][0]
    g = st.edge_gain("core", u)
    s = torch.sign(st.cell_lib.s_raw.detach())
    src, dst = st.src.tolist(), st.dst.tolist()
    pairs = {}
    for e, (a, b) in enumerate(zip(src, dst)):
        pairs.setdefault((a, b), []).append(float(s[e]))
    dead = [n for n in set(src) | set(dst)
            if all(sum(v) == 0 for k, v in pairs.items() if n in k)]
    assert dead, "expected at least one node with only cancelling pairs at this seed"
    with torch.no_grad():
        d = torch.zeros_like(x0)
        d[:, dead[0]] = 0.0143
        dI = st._compute_i_edge_const(x0 + d, g) - st._compute_i_edge_const(x0, g)
    assert float(dI.abs().max()) < 1e-12


def test_cli(tmp_path):
    out = tmp_path / "a.json"
    rc = A.main(["--checkpoint", CKPT, "--n-rows", "8", "--stiffness-rows", "4",
                 "--json", str(out)])
    assert rc == 0
    d = json.loads(out.read_text())
    assert d["n_rows"] == 8 and d["hw_off"] is True
    rc = A.main(["--checkpoint", CKPT, "--n-rows", "8", "--stiffness-rows", "4", "--strict"])
    assert rc == 1                                  # the unconstrained paper model fails


def test_cli_reads_hw_config_beside_a_hardware_checkpoint(tmp_path):
    """The slurm audit passes no --hw-* flags: it must rebuild the trained hw model
    (here phase3a_core: cap_divider adds per-edge codes, so a legacy build would not load) from
    the hw_config.json the trial wrote, with --kn-isat-max of the trial."""
    from hw.cli import hw_config_from_args
    import argparse
    p = argparse.ArgumentParser()
    from hw import parse_hw_args
    parse_hw_args(p)
    hw = hw_config_from_args(p.parse_args(["--hw-preset", "phase3a_core"]))
    torch.manual_seed(3)
    m = A.load_model(None, hw, arch={"isat_max": 16.0})
    assert any(k.endswith("cap_a_raw") for k in m.state_dict())
    ck = tmp_path / "phase_a_best.pt"
    torch.save(m.state_dict(), ck)
    hw.to_json(str(tmp_path / "hw_config.json"))
    out = tmp_path / "a.json"
    rc = A.main(["--checkpoint", str(ck), "--kn-isat-max", "16", "--n-rows", "8",
                 "--stiffness-rows", "4", "--json", str(out)])
    assert rc == 0
    d = json.loads(out.read_text())
    assert d["hw_off"] is False and d["hw_hash"] == hw.hash()
    # slope bound + ratio form on: the static core slope and A/B obey the limits
    core = d["edges"]["core"]
    assert core["slope"][1] <= hw.units.slope_max + 1e-6
    assert core["gain"][0] >= 0.5 - 1e-6 and core["gain"][1] <= 1.0 + 1e-6


def test_cap_divider_audit_checks_grid_and_counts_codes():
    from hw import HWConstraints
    hw = HWConfig(constraints=HWConstraints(hw_ab_mode="cap_divider", hw_slope_bound=True))
    model = A.load_model(None, hw)
    stats = A.edge_stats(model, [[] for _ in model.net.core.stages], hw.units)
    core = stats["core"]
    assert core["cap_grid_ok"]
    assert core["cap_grid_max_error"] < 1e-6
    assert 1 <= core["cap_grid_n_distinct"] <= 2 ** hw.constraints.hw_ab_bits


def test_slurm_scripts_only_pass_existing_flags():
    """Every --hw*/--zig* flag in scripts/slurm/*.sh exists in the entry parsers."""
    import argparse
    import glob
    import re
    import subprocess
    from hw.cli import add_entry_hw_args
    p = argparse.ArgumentParser()
    add_entry_hw_args(p)
    known = {o for a in p._actions for o in a.option_strings}
    for sh in glob.glob(os.path.join(ROOT, "scripts", "slurm", "*.sh")):
        assert subprocess.run(["bash", "-n", sh]).returncode == 0, sh
        txt = open(sh).read()
        flags = set(re.findall(r"(?<![\w-])(--(?:hw|zig)-[a-z0-9-]*[a-z0-9])", txt))
        assert flags, sh
        assert flags <= known, (sh, sorted(flags - known))


def test_kn_steps_per_stage_flag_sets_stage_num_steps():
    """--kn-steps-per-stage 40 rebuilds the paper grid; absent keeps 10."""
    absent = A._audit_parser().parse_args(["--checkpoint", "unused.pt"])
    arch0 = A.arch_overrides_from_args(absent)
    assert "num_steps_per_stage" not in arch0
    m10 = A.load_model(None, None, arch=arch0)
    assert all(num_steps == 10 for num_steps in m10.net.core.stage_steps)
    assert len(m10.net.core.stage_steps) == len(m10.net.core.stages) == 7

    flagged = A._audit_parser().parse_args(
        ["--checkpoint", "unused.pt", "--kn-steps-per-stage", "40"])
    arch40 = A.arch_overrides_from_args(flagged)
    assert arch40["num_steps_per_stage"] == 40
    m40 = A.load_model(None, None, arch=arch40)
    assert all(num_steps == 40 for num_steps in m40.net.core.stage_steps)
    assert len(m40.net.core.stage_steps) == len(m40.net.core.stages)
