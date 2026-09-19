"""Audit for the aux_core_probe module (plan aux-core-drive,
feature spec aux-core-probe). Covers:

1. Side-effect-free import and required public symbol set.
2. Locked control thresholds (eval-only, NARMA-10).
3. Echo-state aux builder determinism + spectral radius + finite states.
4. Echo-state aux runner determinism + finite states.
5. Shift-register aux builder determinism + valid gain / head_inject
   guards + finite states.
6. Shift-register aux EXACT-CHAIN property: x[t][i] == gain**i * hi * u[t-i]
   within float32 noise (per spec, "interpretable learned delay line;
   exact chain asserted").
7. Slow-feature aux builder determinism + log-spread tau + finite states.
8. Slow-feature aux runner explicit integration path matches
   per-timestep formula (1 - 1/tau)*x_prev + (1/tau)*u_t.
9. aux_drive_concat: aux-only and aux+scalar produce correct drive widths.
10. Edge-count itemizers: each flavor reports aux_recurrent +
    aux_fan_in (items (ii) in spec); main_fan_in derived from drive width.
11. Reference harness: taps8 / scalar / scalar-ESN parity all run,
    finite instruments, taps8 MC > scalar MC on the canonical Gate-1
    best corner (the relation the spec frames as load-bearing).
12. Tap-subspace projection KILL diagnostic (self/random/shift-register/
    echo-state operating points).
13. End-to-end 1-corner aux sweep smoke (row shape, edge budget, verdict).
14. CSV / JSON writers round-trip the sweep output.
15. CLI registration: sweep + smoke subcommands, --dt / --order guards,
    smoke row count, --max-corners grid cap.
16. Verdict determinism: the SUCCESS / KILL verdicts match expected
    pre-registered outcomes on synthetic row lists (no fabrication;
    rows are the same shape _aux_verdict consumes).
17. Methodology + robustness guards: drive-power parity (every leg's
    main-core drive sits at RMS == input_scale), aux_scalar width 9 +
    scalar column == u*input_scale, ref_total_edges tracks
    hidden_dim*n_taps on custom dims, fail-fast on unknown flavor/leg
    and empty data_seeds, washout threaded into the verdict, dead aux
    blocks scale to 0.0 (no divide-by-zero), INSUFFICIENT_DATA reports
    a NaN (not 0.0) edge factor.
18. D1 concat readout leg + per-row copy-flag (tap_r2 on every leg).
19. D2 sparse fan-out (exact k*n_drive nnz, 32-edge SR confirmatory
    case, fail-fast on bad k).
20. D3 compressed-bank reference (PC determinism, sign pinning,
    variance ordering, k guards, tapsvd ref row).
21. D4 residual-distilled teacher (orthogonality gate, determinism,
    teacher_r2 sanity, end-to-end D4 leg + verdict path).
22. Verdict amendment: mixed ref budgets (tapsvd-gated 100-edge budget,
    legacy fallback, rule (i) untouched under compression).
23. Audit-fix regression guards: D4 dict entries, fanout fail-fast at
    all layers, concat-only verdict path, concat-gain note semantics.

Run:
    python -B audit_aux_core.py
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, ".")

import aux_core_probe as ax  # noqa: E402


def assert_close(a, b, msg, tol=1e-5):
    assert abs(float(a) - float(b)) < tol, f"{msg}: {a} != {b}"


def assert_finite(value, msg):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all().item(), f"{msg}: non-finite"
    else:
        v = float(value)
        assert math.isfinite(v), f"{msg}: non-finite {v}"


torch.manual_seed(0)
dev = torch.device("cpu")
dtype = torch.float32


# --- 1. Side-effect-free import and public symbol set --------------------
for name in (
    "echo_state_aux_build", "echo_state_aux_run",
    "shift_register_aux_build", "shift_register_aux_run",
    "slow_feature_aux_build", "slow_feature_aux_run",
    "aux_drive_concat",
    "echo_state_aux_edge_counts",
    "shift_register_aux_edge_counts",
    "slow_feature_aux_edge_counts",
    "main_fan_in_edges",
    "tap_subspace_canonical_corr",
    "AuxProbeRow", "AuxVerdict",
    "run_aux_sweep", "main",
    "PROBE_WASHOUT", "CANONICAL_HIDDEN", "CANONICAL_T_SPAN",
    "CANONICAL_DT_MAX",
    "MAIN_RADIUS", "MAIN_LAMT", "MAIN_INPUT_SCALE",
    "AUX_FLAVORS", "AUX_LEGS", "REF_LEGS",
    "ECHO_N_AUX", "ECHO_RADIUS", "ECHO_LEAK_GRID",
    "SR_N_AUX", "SR_GAIN_GRID",
    "SFB_N_AUX", "SFB_LOG_TAU_RANGE",
    "AUX_SEEDS", "DATA_SEEDS",
    "REF_FAN_IN_EDGES",
    "SUCCESS_MC_MATCH_RATIO", "SUCCESS_EDGE_TOL_FACTOR",
    "SUCCESS_BEAT_SCALAR_MARGIN",
    "TAP_SUBSPACE_KILL_CCORR",
    "VOID_PR_THRESHOLD", "VOID_MC_THRESHOLD",
    "NO_GAIN_OVER_SCALAR_THRESHOLD",
):
    assert hasattr(ax, name), f"missing symbol: {name}"
print("AUDIT1_IMPORT_OK")


# --- 2. Locked control thresholds ----------------------------------------
assert ax.PROBE_WASHOUT == 200, ax.PROBE_WASHOUT
assert ax.CANONICAL_HIDDEN == 25, ax.CANONICAL_HIDDEN
assert ax.CANONICAL_T_SPAN == 1.0, ax.CANONICAL_T_SPAN
assert ax.CANONICAL_DT_MAX <= 0.125, ax.CANONICAL_DT_MAX
# Main core is pinned to the Gate-1 best corner (spec-required: "Main
# core pinned to the Gate-1 best corner").
assert (ax.MAIN_RADIUS, ax.MAIN_LAMT, ax.MAIN_INPUT_SCALE) == (1.0, 1.0, 0.2), (
    ax.MAIN_RADIUS, ax.MAIN_LAMT, ax.MAIN_INPUT_SCALE
)
# Aux flavor set locked per spec.
assert ax.AUX_FLAVORS == ("echo_state", "shift_register", "slow_feature"), (
    ax.AUX_FLAVORS
)
assert ax.AUX_LEGS == ("aux_only", "aux_scalar", "aux_concat"), ax.AUX_LEGS
assert ax.REF_LEGS == (
    "taps8", "scalar", "scalar_esn_parity", "tapsvd",
), ax.REF_LEGS
# Follow-up (aux-core-drive-2): D2 sparse fan-out + D3 compressed bank
# + D4 residual teacher constants.
assert ax.SPARSE_FANOUT_K == (3, 6), ax.SPARSE_FANOUT_K
assert ax.TAPSVD_K == 4, ax.TAPSVD_K
assert ax.D4_RESIDUAL_TEACHER_K == 4, ax.D4_RESIDUAL_TEACHER_K
assert ax.D4_TEACHER_ORTHOG_MAX_CORR == 0.10, ax.D4_TEACHER_ORTHOG_MAX_CORR
assert "residual_distilled" in ax.ALL_FLAVORS, ax.ALL_FLAVORS
print("AUDIT2_THRESHOLDS_OK")


# --- 3. Echo-state aux builder determinism + radius ----------------------
sp1 = ax.echo_state_aux_build(
    n_aux=8, leak=0.5, radius=0.95,
    w_seed=0, win_seed=0, dtype=dtype, device=dev,
)
sp1b = ax.echo_state_aux_build(
    n_aux=8, leak=0.5, radius=0.95,
    w_seed=0, win_seed=0, dtype=dtype, device=dev,
)
assert torch.equal(sp1["W_aux"], sp1b["W_aux"]), "echo W_aux not deterministic"
assert torch.equal(sp1["W_in_aux"], sp1b["W_in_aux"]), "echo W_in not deterministic"
rho = float(torch.linalg.eigvals(
    sp1["W_aux"].to(torch.float64)
).abs().max().item())
assert_close(rho, 0.95, "echo W_aux radius", tol=1e-5)
assert sp1["W_aux"].shape == (8, 8)
assert sp1["W_in_aux"].shape == (8, 1)
assert torch.isfinite(sp1["W_aux"]).all().item()
assert torch.isfinite(sp1["W_in_aux"]).all().item()
# Builder guards.
for bad_leak in (0.0, 1.5, -1.0):
    try:
        ax.echo_state_aux_build(
            n_aux=8, leak=bad_leak, radius=0.95,
            w_seed=0, win_seed=0, dtype=dtype, device=dev,
        )
        raise AssertionError(f"expected ValueError for leak={bad_leak}")
    except ValueError:
        pass
print("AUDIT3_ECHO_BUILDER_OK")


# --- 4. Echo-state aux runner determinism + finite -----------------------
u_seq = torch.randn(30)
e1 = ax.echo_state_aux_run(spec=sp1, u_stream=u_seq)
e2 = ax.echo_state_aux_run(spec=sp1, u_stream=u_seq)
assert torch.equal(e1, e2), "echo run not deterministic"
assert e1.shape == (30, 8)
assert torch.isfinite(e1).all().item()
# All states in [-1, 1] (tanh output after first step).
assert (e1.abs() <= 1.0 + 1e-6).all().item()
print("AUDIT4_ECHO_RUN_OK")


# --- 5. Shift-register aux builder guards --------------------------------
for bad_gain in (0.0, -0.1, 2.0):
    try:
        ax.shift_register_aux_build(
            n_aux=5, gain=bad_gain, head_inject=1.0,
            w_seed=0, dtype=dtype, device=dev,
        )
        raise AssertionError(f"expected ValueError for gain={bad_gain}")
    except ValueError:
        pass
for bad_hi in (-0.1, 11.0):
    try:
        ax.shift_register_aux_build(
            n_aux=5, gain=0.7, head_inject=bad_hi,
            w_seed=0, dtype=dtype, device=dev,
        )
        raise AssertionError(f"expected ValueError for head_inject={bad_hi}")
    except ValueError:
        pass
print("AUDIT5_SR_BUILDER_GUARDS_OK")


# --- 6. Shift-register EXACT chain property ------------------------------
# Per spec: "interpretable learned delay line; exact chain asserted".
# Analytic: x[t][i] = gain**i * head_inject * u[t-i] for t >= i else 0.
gain, hi, n_aux = 0.7, 1.0, 5
u_short = torch.tensor(
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=dtype,
)
sp_sr = ax.shift_register_aux_build(
    n_aux=n_aux, gain=gain, head_inject=hi,
    w_seed=0, dtype=dtype, device=dev,
)
states_sr = ax.shift_register_aux_run(spec=sp_sr, u_stream=u_short)
expected = torch.zeros_like(states_sr)
for t in range(len(u_short)):
    for i in range(n_aux):
        if t >= i:
            expected[t, i] = (gain ** i) * hi * float(u_short[t - i])
assert_close(
    float((states_sr - expected).abs().max().item()), 0.0,
    "shift-register exact chain", tol=1e-5,
)
# Reproducibility across identical spec.
states_sr_b = ax.shift_register_aux_run(spec=sp_sr, u_stream=u_short)
assert torch.equal(states_sr, states_sr_b), "shift-register not deterministic"
print("AUDIT6_SR_EXACT_CHAIN_OK")


# --- 7. Slow-feature aux builder log-spread ------------------------------
sp_sf = ax.slow_feature_aux_build(
    n_aux=8, log_tau_lo=math.log(1.0), log_tau_hi=math.log(40.0),
    w_seed=0, dtype=dtype, device=dev,
)
assert sp_sf["tau"].shape == (8,)
# tau must be log-spread, monotonically increasing, endpoints
# (1, 40) within float32 noise.
assert torch.isfinite(sp_sf["tau"]).all().item()
assert_close(float(sp_sf["tau"][0]), 1.0, "tau[0]=1", tol=1e-4)
assert_close(float(sp_sf["tau"][-1]), 40.0, "tau[-1]=40", tol=1e-2)
monotone_diff = sp_sf["tau"][1:] - sp_sf["tau"][:-1]
assert (monotone_diff > 0).all().item(), "tau not monotone"
# Builder guards.
for bad_lo_hi in [(2.0, 1.0), (math.log(5.0), math.log(5.0))]:
    try:
        ax.slow_feature_aux_build(
            n_aux=8, log_tau_lo=bad_lo_hi[0], log_tau_hi=bad_lo_hi[1],
            w_seed=0, dtype=dtype, device=dev,
        )
        raise AssertionError(f"expected ValueError for lo>=hi {bad_lo_hi}")
    except ValueError:
        pass
print("AUDIT7_SFB_BUILDER_OK")


# --- 8. Slow-feature explicit integration path ---------------------------
T = 16
u_sf = torch.randn(T, dtype=dtype)
states_sf = ax.slow_feature_aux_run(spec=sp_sf, u_stream=u_sf)
assert states_sf.shape == (T, 8)
assert torch.isfinite(states_sf).all().item()
# Per-timestep formula check: x[t+1][i] = (1 - 1/tau[i]) * x[t][i] + (1/tau[i]) * u[t]
for t in range(T - 1):
    expected_next = ((1.0 - 1.0 / sp_sf["tau"]) * states_sf[t]
                     + (1.0 / sp_sf["tau"]) * u_sf[t + 1])
    assert_close(
        float((states_sf[t + 1] - expected_next).abs().max().item()), 0.0,
        f"sf integration t={t}", tol=1e-6,
    )
# Reproducibility.
states_sf_b = ax.slow_feature_aux_run(spec=sp_sf, u_stream=u_sf)
assert torch.equal(states_sf, states_sf_b), "slow-feature not deterministic"
print("AUDIT8_SFB_RUN_OK")


# --- 9. aux_drive_concat shapes ------------------------------------------
aux_states = torch.randn(20, 4)
u_drive = torch.randn(20)
only = ax.aux_drive_concat(aux_states, u_drive, leg="aux_only")
with_s = ax.aux_drive_concat(aux_states, u_drive, leg="aux_scalar")
assert only.shape == (20, 4), only.shape
assert with_s.shape == (20, 5), with_s.shape
assert torch.equal(only, aux_states), "aux_only must equal aux_states"
# Last column is the original u_drive.
assert torch.equal(with_s[:, -1], u_drive), "aux_scalar last col != u"
# Unknown leg should error.
try:
    ax.aux_drive_concat(aux_states, u_drive, leg="aux_bogus")
    raise AssertionError("expected ValueError for unknown leg")
except ValueError:
    pass
print("AUDIT9_CONCAT_OK")


# --- 10. Edge-count itemizers --------------------------------------------
assert ax.echo_state_aux_edge_counts(8) == {"aux_recurrent": 64, "aux_fan_in": 8}
assert ax.shift_register_aux_edge_counts(8) == {
    "aux_recurrent": 7, "aux_fan_in": 1,
}
assert ax.slow_feature_aux_edge_counts(8) == {
    "aux_recurrent": 0, "aux_fan_in": 8,
}
assert ax.main_fan_in_edges(25, 8) == 200, ax.main_fan_in_edges(25, 8)
assert ax.main_fan_in_edges(25, 9) == 225, ax.main_fan_in_edges(25, 9)
print("AUDIT10_EDGES_OK")


# --- 11. Reference harness: taps8 / scalar / ESN parity ------------------
# Each reference row uses the Heun flow-twin at the Gate-1 best corner
# plus the ESN parity diagnostic; spec requires these run and produce
# finite instruments on the same streams as the aux rows.
import narma_experiment as ne  # noqa: E402

torch.manual_seed(0)
u_raw, y_raw = ne._gen_narma_train_streams(
    order=10, seed=0, n_streams=1, n=300,
)
u_stream = u_raw[0]
y_stream = y_raw[0]
u_scaled = ne._scale_drive(
    u_stream, bipolar=True, order=10, input_scale=1.0,
)
ref_taps = ax._reference_row(
    ref_leg="taps8", seed=0, device="cpu",
    hidden_dim=25, n_taps=8,
    radius=ax.MAIN_RADIUS, lamT=ax.MAIN_LAMT,
    input_scale=ax.MAIN_INPUT_SCALE, dt=ax.CANONICAL_DT_MAX,
    washout=ax.PROBE_WASHOUT, max_delay=20,
    jacobian_samples=2, u_scaled=u_scaled, y_seq=y_stream,
)
assert_finite(ref_taps["mc_total"], "ref taps8 mc_total")
ref_scalar = ax._reference_row(
    ref_leg="scalar", seed=0, device="cpu",
    hidden_dim=25, n_taps=8,
    radius=ax.MAIN_RADIUS, lamT=ax.MAIN_LAMT,
    input_scale=ax.MAIN_INPUT_SCALE, dt=ax.CANONICAL_DT_MAX,
    washout=ax.PROBE_WASHOUT, max_delay=20,
    jacobian_samples=2, u_scaled=u_scaled, y_seq=y_stream,
)
assert_finite(ref_scalar["mc_total"], "ref scalar mc_total")
ref_esn = ax._reference_row(
    ref_leg="scalar_esn_parity", seed=0, device="cpu",
    hidden_dim=25, n_taps=8,
    radius=ax.MAIN_RADIUS, lamT=ax.MAIN_LAMT,
    input_scale=ax.MAIN_INPUT_SCALE, dt=ax.CANONICAL_DT_MAX,
    washout=ax.PROBE_WASHOUT, max_delay=20,
    jacobian_samples=2, u_scaled=u_scaled, y_seq=y_stream,
)
# Unknown reference legs fail fast (the harness is derived from the
# leg label internally, so a mislabeled drive cannot be built).
try:
    ax._reference_row(
        ref_leg="taps9", seed=0, device="cpu",
        hidden_dim=25, n_taps=8,
        radius=ax.MAIN_RADIUS, lamT=ax.MAIN_LAMT,
        input_scale=ax.MAIN_INPUT_SCALE, dt=ax.CANONICAL_DT_MAX,
        washout=ax.PROBE_WASHOUT, max_delay=20,
        jacobian_samples=2, u_scaled=u_scaled, y_seq=y_stream,
    )
    raise AssertionError("expected ValueError for unknown ref_leg")
except ValueError:
    pass
assert_finite(ref_esn["mc_total"], "ref esn mc_total")
# Spec's load-bearing claim: taps8 MC > scalar MC on this stream (the
# relation that motivated the aux-core investigation in the first
# place).  Pin it so the aux probe's comparisons are meaningful.
assert ref_taps["mc_total"] > ref_scalar["mc_total"], (
    f"taps8 MC ({ref_taps['mc_total']:.3f}) should exceed scalar MC "
    f"({ref_scalar['mc_total']:.3f}) at the Gate-1 best corner"
)
print(
    f"AUDIT11_REFS_OK taps8_mc={ref_taps['mc_total']:.3f} "
    f"scalar_mc={ref_scalar['mc_total']:.3f} "
    f"esn_mc={ref_esn['mc_total']:.3f}"
)


# --- 12. Tap-subspace projection KILL diagnostic -------------------------
# The KILL signal is operationalized as "taps predict aux" (ridge R^2
# per aux column, averaged): self-subspace should be ~1.0 (KILL);
# random aux vs random taps should be ~0 (no KILL); a literal
# delay-line aux (shift-register) should hit ~1.0 (KILL); a nonlinear
# echo-state aux with leak<1 should sit below the threshold.
import narma_linear_controls as nlc  # noqa: E402
u_audit = torch.randn(200)
audit_taps = nlc.twin_delay_bank(u_audit, n_taps=8)
assert audit_taps.shape == (200, 8)

# (a) Self-subspace: taps predicted from taps -> R^2 near 1.0.
cc_self = ax.tap_subspace_canonical_corr(
    audit_taps, audit_taps, washout=10,
)
assert cc_self > ax.TAP_SUBSPACE_KILL_CCORR, (
    f"self R^2 should trigger KILL, got {cc_self:.3f}"
)

# (b) Independent random aux vs random taps -> R^2 near 0.
cc_rand = ax.tap_subspace_canonical_corr(
    torch.randn(200, 8), torch.randn(200, 8), washout=10,
)
assert cc_rand < 0.5, (
    f"random aux vs random taps should be far below KILL, "
    f"got {cc_rand:.3f}"
)

# (c) Shift-register aux is literally a delay line -> R^2 ~1.0 (KILL).
sp_sr_audit = ax.shift_register_aux_build(
    n_aux=8, gain=0.9, head_inject=1.0, w_seed=0,
    dtype=dtype, device=dev,
)
audit_sr = ax.shift_register_aux_run(spec=sp_sr_audit, u_stream=u_audit)
cc_sr = ax.tap_subspace_canonical_corr(
    audit_sr, audit_taps, washout=10,
)
assert cc_sr > ax.TAP_SUBSPACE_KILL_CCORR, (
    f"shift-register (delay-line copy) should KILL on tap-subspace "
    f"R^2, got {cc_sr:.3f}"
)

# (d) Echo-state aux with leak<1 should sit below the KILL threshold
# (its nonlinear dynamics add information beyond taps).
sp_echo_audit = ax.echo_state_aux_build(
    n_aux=8, leak=0.3, radius=0.95, w_seed=42, win_seed=42,
    dtype=dtype, device=dev,
)
audit_echo = ax.echo_state_aux_run(spec=sp_echo_audit, u_stream=u_audit)
cc_echo = ax.tap_subspace_canonical_corr(
    audit_echo, audit_taps, washout=10,
)
assert cc_echo < ax.TAP_SUBSPACE_KILL_CCORR, (
    f"echo-state with leak=0.3 should sit below KILL, "
    f"got {cc_echo:.3f}"
)
print(
    f"AUDIT12_TAP_PROJ_OK self={cc_self:.3f} random={cc_rand:.3f} "
    f"sr={cc_sr:.3f} echo_l0.3={cc_echo:.3f}"
)


# --- 13. End-to-end 1-corner aux sweep smoke -----------------------------
# Small enough to run in well under 30 seconds on CPU.
out_smoke = Path("./output/audit_aux_core_smoke")
if out_smoke.exists():
    for p in out_smoke.iterdir():
        if p.is_file():
            p.unlink()
out_smoke.mkdir(parents=True, exist_ok=True)
rows, refs, verdicts, elapsed = ax.run_aux_sweep(
    flavors=("echo_state",), legs=("aux_only",),
    aux_seeds=(0,), data_seeds=(0,),
    hidden_dim=25, n_taps=8,
    n_streams=1, train_samples_per_stream=300,
    washout=ax.PROBE_WASHOUT, max_delay=20,
    jacobian_samples=2,
    radius=ax.MAIN_RADIUS, lamT=ax.MAIN_LAMT,
    input_scale=ax.MAIN_INPUT_SCALE,
    device="cpu", dt=ax.CANONICAL_DT_MAX,
    max_corners=1,
)
assert len(rows) == 1, f"expected 1 aux row, got {len(rows)}"
assert len(refs) == 4, (
    f"expected 4 refs (taps8/scalar/ESN/tapsvd), got {len(refs)}"
)
_ref_tags = [r["config_tag"] for r in refs]
assert len(set(_ref_tags)) == 4, _ref_tags  # distinct legs
assert any("tapsvd" in t for t in _ref_tags), _ref_tags
assert len(verdicts) == 1, f"expected 1 verdict, got {len(verdicts)}"
# Aux row is well-formed.
r0 = rows[0]
assert r0.flavor == "echo_state" and r0.leg == "aux_only"
assert r0.n_drive == 8, f"aux_only drive width != 8: {r0.n_drive}"
assert r0.aux_recurrent == 64 and r0.aux_fan_in == 8
assert r0.main_fan_in == 25 * 8
assert r0.main_total_edges == 64 + 8 + 25 * 8
assert r0.ref_total_edges == 200  # 25 * 8
assert math.isfinite(r0.mc_total), r0
print(
    f"AUDIT13_SMOKE_OK mc={r0.mc_total:.3f} ridge={r0.ridge_nrmse:.4f} "
    f"edges={r0.main_total_edges} verdict={verdicts[0].verdict} "
    f"elapsed={elapsed:.1f}s"
)


# --- 14. CSV / JSON / TXT writers ----------------------------------------
ax._rows_to_csv(out_smoke / "aux_rows.csv", rows)
assert (out_smoke / "aux_rows.csv").exists()
ax._verdict_to_json(out_smoke / "aux_verdicts.json", verdicts)
assert (out_smoke / "aux_verdicts.json").exists()
verdict_blob = json.loads(
    (out_smoke / "aux_verdicts.json").read_text()
)
assert verdict_blob[0]["flavor"] == "echo_state"
print("AUDIT14_WRITERS_OK")


# --- 15. CLI registration + guards ---------------------------------------
for mode in ("sweep", "smoke"):
    res = subprocess.run(
        [sys.executable, "-B", "aux_core_probe.py", mode, "--help"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, f"{mode} --help failed: {res.stderr}"
# --dt above the fixed-dt bound is rejected, not silently clipped.
res = subprocess.run(
    [sys.executable, "-B", "aux_core_probe.py", "smoke",
     "--dt", "0.25", "--output", str(out_smoke / "cli_dt")],
    capture_output=True, text=True,
)
assert res.returncode != 0, "--dt 0.25 should violate the fixed-dt rule"
# --order 20 is rejected up front (order-10 pre-registration).
res = subprocess.run(
    [sys.executable, "-B", "aux_core_probe.py", "smoke",
     "--order", "20", "--output", str(out_smoke / "cli_order")],
    capture_output=True, text=True,
)
assert res.returncode != 0, "--order 20 should be rejected"
# --smoke mode produces a verdict file.
res = subprocess.run(
    [sys.executable, "-B", "aux_core_probe.py", "smoke",
     "--output", str(out_smoke / "cli_smoke"),
     "--flavor", "echo_state"],
    capture_output=True, text=True,
)
assert res.returncode == 0, res.stderr[-2000:]
import csv as _csv
with (out_smoke / "cli_smoke" / "aux_rows.csv").open() as f:
    csv_rows = list(_csv.DictReader(f))
assert len(csv_rows) == 1, f"smoke row count: {len(csv_rows)}"
assert csv_rows[0]["leg"] == "aux_only"
assert csv_rows[0]["flavor"] == "echo_state"
# sweep --max-corners caps the grid before running.
res = subprocess.run(
    [sys.executable, "-B", "aux_core_probe.py", "sweep",
     "--max-corners", "2", "--output", str(out_smoke / "cli_sweep")],
    capture_output=True, text=True,
)
assert res.returncode == 0, res.stderr[-2000:]
with (out_smoke / "cli_sweep" / "aux_rows.csv").open() as f:
    sweep_rows = list(_csv.DictReader(f))
assert len(sweep_rows) == 2, f"sweep --max-corners 2 rows: {len(sweep_rows)}"
print("AUDIT15_CLI_OK")


# --- 16. Verdict determinism (synthetic rows) ----------------------------
# Build the success / kill cases directly from AuxProbeRow-shaped stubs.
@dataclass
class RowStub:
    flavor: str
    leg: str
    mc_total: float
    state_pr: float
    aux_void_kill: bool
    seed: int = 0
    tanh_engaged: bool = True
    states_finite: bool = True
    aux_seed: int = 0
    aux_value: float = 0.5
    main_total_edges: int = 272
    ref_total_edges: int = 200
    ridge_nrmse: float = float("nan")
    ridge_concat_nrmse: float = float("nan")
    config_tag: str = "stub"
    # The real AuxProbeRow has more fields; we only consult these.
    pass


# SUCCESS path 1: aux MC >= 0.85 * taps8 MC at <= edge cost.
v1 = ax._aux_verdict(
    flavor="echo_state",
    aux_rows=[
        RowStub(
            flavor="echo_state", leg="aux_only",
            mc_total=10.0, state_pr=3.0, aux_void_kill=False,
            main_total_edges=272,
        ),
    ],
    taps8_mc=10.0, scalar_mc=4.0, scalar_esn_mc=8.0,
    aux_states_by_leg={},
    taps8_drive=torch.zeros(20, 8),
)
assert v1.verdict == "SUCCESS", v1
assert v1.aux_matches_taps8, v1
assert v1.reasons and any(
    "SUCCESS path 1" in r for r in v1.reasons
), v1.reasons

# KILL: aux MC too low (below scalar by margin, below taps8).
v2 = ax._aux_verdict(
    flavor="shift_register",
    aux_rows=[
        RowStub(
            flavor="shift_register", leg="aux_only",
            mc_total=1.0, state_pr=2.0, aux_void_kill=False,
            main_total_edges=208,
        ),
    ],
    taps8_mc=10.0, scalar_mc=4.0, scalar_esn_mc=8.0,
    aux_states_by_leg={},
    taps8_drive=torch.zeros(20, 8),
)
assert v2.verdict == "KILL", v2

# KILL: aux void/dead (PR below threshold).
v3 = ax._aux_verdict(
    flavor="echo_state",
    aux_rows=[
        RowStub(
            flavor="echo_state", leg="aux_only",
            mc_total=2.0, state_pr=0.5, aux_void_kill=True,
            main_total_edges=272,
        ),
    ],
    taps8_mc=10.0, scalar_mc=4.0, scalar_esn_mc=8.0,
    aux_states_by_leg={},
    taps8_drive=torch.zeros(20, 8),
)
assert v3.verdict == "KILL", v3
assert v3.aux_void, v3

# KILL: aux+scalar gives no gain over scalar.
v4 = ax._aux_verdict(
    flavor="shift_register",
    aux_rows=[
        RowStub(
            flavor="shift_register", leg="aux_only",
            mc_total=2.0, state_pr=2.0, aux_void_kill=False,
            main_total_edges=208,
        ),
        RowStub(
            flavor="shift_register", leg="aux_scalar",
            mc_total=4.05, state_pr=2.0, aux_void_kill=False,
            main_total_edges=25 * 9 + 1 + 7,
        ),
    ],
    taps8_mc=10.0, scalar_mc=4.0, scalar_esn_mc=8.0,
    aux_states_by_leg={},
    taps8_drive=torch.zeros(20, 8),
)
assert v4.verdict == "KILL", v4
assert v4.no_gain_over_scalar, v4

# INSUFFICIENT_DATA: no engaged+finite aux rows survive.
v5 = ax._aux_verdict(
    flavor="echo_state",
    aux_rows=[
        RowStub(
            flavor="echo_state", leg="aux_only",
            mc_total=float("nan"), state_pr=float("nan"),
            aux_void_kill=True, tanh_engaged=False,
            states_finite=False, main_total_edges=272,
        ),
    ],
    taps8_mc=10.0, scalar_mc=4.0, scalar_esn_mc=8.0,
    aux_states_by_leg={},
    taps8_drive=torch.zeros(20, 8),
)
assert v5.verdict == "INSUFFICIENT_DATA", v5

print("AUDIT16_VERDICT_DETERMINISM_OK")


# --- 17. Methodology + robustness guards ---------------------------------
# (a) Drive-power parity: every leg's main-core drive sits at
# RMS == input_scale (the pinned Gate-1 operating point).
u_par = torch.randn(100)
aux_par = torch.randn(100, 8)
d_only, s_only = ax.parity_scale_drive(
    aux_par, u_par, leg="aux_only", input_scale=0.2,
)
assert d_only.shape == (100, 8)
assert_close(
    float(d_only.pow(2).mean().sqrt().item()), 0.2,
    "aux_only parity RMS", tol=1e-6,
)
assert math.isfinite(s_only) and s_only > 0
d_scal, s_scal = ax.parity_scale_drive(
    torch.cat([aux_par, u_par.unsqueeze(-1)], dim=1), u_par,
    leg="aux_scalar", input_scale=0.2,
)
assert d_scal.shape == (100, 9)
assert_close(
    float(d_scal[:, :-1].pow(2).mean().sqrt().item()), 0.2,
    "aux_scalar block parity RMS", tol=1e-6,
)
assert torch.equal(
    d_scal[:, -1], (u_par * 0.2).to(dtype=d_scal.dtype),
), "aux_scalar column must equal u*input_scale"
# Dead (zero-RMS) blocks scale to 0.0 instead of dividing by zero.
d_dead, s_dead = ax.parity_scale_drive(
    torch.zeros(50, 8), torch.randn(50), leg="aux_only",
    input_scale=0.2,
)
assert s_dead == 0.0 and torch.equal(d_dead, torch.zeros(50, 8))
# Guards.
for kw in ({"leg": "aux_bogus", "input_scale": 0.2},
           {"leg": "aux_only", "input_scale": 0.0},
           {"leg": "aux_only", "input_scale": float("nan")}):
    try:
        ax.parity_scale_drive(aux_par, u_par, **kw)
        raise AssertionError(f"expected ValueError for {kw}")
    except ValueError:
        pass

# (b) aux_scalar sweep row: 9-wide drive, 25x9 main fan-in.
rows_s, _, _, _ = ax.run_aux_sweep(
    flavors=("shift_register",), legs=("aux_scalar",),
    aux_seeds=(0,), data_seeds=(0,),
    hidden_dim=25, n_taps=8,
    n_streams=1, train_samples_per_stream=300,
    washout=ax.PROBE_WASHOUT, max_delay=20,
    jacobian_samples=1,
    radius=ax.MAIN_RADIUS, lamT=ax.MAIN_LAMT,
    input_scale=ax.MAIN_INPUT_SCALE,
    device="cpu", dt=ax.CANONICAL_DT_MAX,
    max_corners=1,
)
assert len(rows_s) == 1 and rows_s[0].n_drive == 9, rows_s[0].n_drive
assert rows_s[0].main_fan_in == 25 * 9
assert rows_s[0].main_total_edges == 7 + 1 + 25 * 9  # sr rec + fan + main
assert math.isfinite(rows_s[0].drive_scale) and rows_s[0].drive_scale > 0

# (c) ref_total_edges tracks hidden_dim * n_taps on custom dims.
rows_c, _, _, _ = ax.run_aux_sweep(
    flavors=("slow_feature",), legs=("aux_only",),
    aux_seeds=(0,), data_seeds=(0,),
    hidden_dim=10, n_taps=4,
    n_streams=1, train_samples_per_stream=300,
    washout=ax.PROBE_WASHOUT, max_delay=20,
    jacobian_samples=1,
    radius=ax.MAIN_RADIUS, lamT=ax.MAIN_LAMT,
    input_scale=ax.MAIN_INPUT_SCALE,
    device="cpu", dt=ax.CANONICAL_DT_MAX,
    max_corners=1,
)
assert rows_c[0].ref_total_edges == 40, rows_c[0].ref_total_edges
assert rows_c[0].main_fan_in == 10 * 8

# (d) Fail-fast: unknown flavor / leg raise before burning compute on
# refs; empty data_seeds raises instead of IndexError in aggregation.
for bad_kw in ({"flavors": ("echo_bogus",), "legs": ("aux_only",)},
               {"flavors": ("echo_state",), "legs": ("aux_bogus",)}):
    try:
        ax.run_aux_sweep(
            aux_seeds=(0,), data_seeds=(0,),
            hidden_dim=10, n_taps=4, n_streams=1,
            train_samples_per_stream=50, washout=10, max_delay=5,
            jacobian_samples=1, device="cpu",
            max_corners=1, **bad_kw,
        )
        raise AssertionError(f"expected ValueError for {bad_kw}")
    except ValueError:
        pass
try:
    ax.run_aux_sweep(
        flavors=("echo_state",), legs=("aux_only",),
        aux_seeds=(0,), data_seeds=(),
        hidden_dim=10, n_taps=4, n_streams=1,
        train_samples_per_stream=50, washout=10, max_delay=5,
        jacobian_samples=1, device="cpu", max_corners=1,
    )
    raise AssertionError("expected ValueError for empty data_seeds")
except ValueError:
    pass

# (e) Washout threads into the verdict (non-default washout must not
# silently use PROBE_WASHOUT in the tap-subspace diagnostic).
v_w = ax._aux_verdict(
    flavor="echo_state",
    aux_rows=[
        RowStub(
            flavor="echo_state", leg="aux_only",
            mc_total=10.0, state_pr=3.0, aux_void_kill=False,
            main_total_edges=272,
        ),
    ],
    taps8_mc=10.0, scalar_mc=4.0, scalar_esn_mc=8.0,
    aux_states_by_leg={},
    taps8_drive=torch.zeros(20, 8),
    washout=5,
)
assert v_w.verdict == "SUCCESS", v_w

# (f) INSUFFICIENT_DATA carries a NaN (not 0.0) edge factor.
assert math.isnan(v5.aux_edge_cost_factor), v5.aux_edge_cost_factor

print("AUDIT17_METHODOLOGY_OK")


# --- 18. D1: concat readout leg + per-row copy-flag -----------------------
# aux_concat drives the main core with aux states only, but the readout
# sees [main_states; aux_states] (33 features at h=25, n_aux=8); the
# per-row tap_r2 copy-flag must be present on every leg.
rows_d1, _, _, _ = ax.run_aux_sweep(
    flavors=("echo_state",), legs=("aux_concat",),
    aux_seeds=(0,), data_seeds=(0,),
    hidden_dim=25, n_taps=8,
    n_streams=1, train_samples_per_stream=300,
    washout=ax.PROBE_WASHOUT, max_delay=20,
    jacobian_samples=1,
    radius=ax.MAIN_RADIUS, lamT=ax.MAIN_LAMT,
    input_scale=ax.MAIN_INPUT_SCALE,
    device="cpu", dt=ax.CANONICAL_DT_MAX,
    max_corners=1,
)
assert len(rows_d1) == 1 and rows_d1[0].n_drive == 8, rows_d1
assert rows_d1[0].main_fan_in == 25 * 8
assert math.isfinite(rows_d1[0].ridge_concat_r2), rows_d1[0]
assert math.isfinite(rows_d1[0].tap_r2), rows_d1[0].tap_r2
assert 0.0 <= rows_d1[0].tap_r2 <= 1.0, rows_d1[0].tap_r2
# Row-path tap_r2 must equal a direct recomputation on the same
# (raw aux, taps) pair — this pins the wiring, not a threshold: on the
# autocorrelated NARMA stream every echo/SF row sits at 0.86-0.92
# (measured), i.e. the 0.90 KILL line cuts through the echo family and
# echo verdicts can flip by seed.  The below-threshold directional pin
# lives on white noise (AUDIT12d, 0.760), where it genuinely holds.
import post_twin_gates as _ptg  # noqa: E402
_taps_direct = _ptg.flow_twin_drive(
    u_scaled, harness="taps8", n_taps=8,
    input_scale=ax.MAIN_INPUT_SCALE,
)
_sp_direct = ax.echo_state_aux_build(
    n_aux=8, leak=0.3, radius=0.95, w_seed=0, win_seed=0,
    dtype=dtype, device=dev,
)
_aux_direct = ax.echo_state_aux_run(spec=_sp_direct, u_stream=u_scaled)
_r2_direct = ax.tap_subspace_canonical_corr(
    _aux_direct, _taps_direct, washout=ax.PROBE_WASHOUT,
)
assert abs(rows_d1[0].tap_r2 - _r2_direct) < 1e-9, (
    rows_d1[0].tap_r2, _r2_direct,
)
print(
    f"AUDIT18_D1_CONCAT_OK concat_r2={rows_d1[0].ridge_concat_r2:.4f} "
    f"tap_r2={rows_d1[0].tap_r2:.3f}"
)


# --- 19. D2: sparse fan-out ------------------------------------------------
# Exact nnz == k * n_drive (Gate-3 sparse_random_W_in reuse); SR at k=3
# totals 7 + 1 + 24 = 32 edges (spec's pre-registered confirmatory case).
for k in (3, 6):
    W_sp = ax.ptg.sparse_random_W_in(
        n_nodes=25, n_in=8, k_per_tap=k, seed=0,
        dtype=dtype, device=dev,
    )
    assert W_sp.shape == (25, 8)
    assert int((W_sp != 0).sum().item()) == k * 8, (W_sp != 0).sum()
rows_d2, _, _, _ = ax.run_aux_sweep(
    flavors=("shift_register",), legs=("aux_only",),
    aux_seeds=(0,), data_seeds=(0,),
    hidden_dim=25, n_taps=8,
    n_streams=1, train_samples_per_stream=300,
    washout=ax.PROBE_WASHOUT, max_delay=20,
    jacobian_samples=1,
    radius=ax.MAIN_RADIUS, lamT=ax.MAIN_LAMT,
    input_scale=ax.MAIN_INPUT_SCALE,
    device="cpu", dt=ax.CANONICAL_DT_MAX,
    max_corners=1, fanout_kinds=("sparse_random",),
)
assert len(rows_d2) == 1, rows_d2
r2d2 = rows_d2[0]
assert r2d2.fanout_kind == "sparse_random" and r2d2.fanout_k == 3, r2d2
assert r2d2.fanout_edges == 3 * 8, r2d2.fanout_edges
assert r2d2.main_total_edges == 7 + 1 + 3 * 8, r2d2.main_total_edges
assert math.isfinite(r2d2.mc_total), r2d2
# Invalid k fails fast.
try:
    ax._aux_probe_row(
        flavor="shift_register", leg="aux_only", seed=0, device="cpu",
        hidden_dim=4, n_taps=8,
        aux_param="gain0.9", aux_value=0.9, aux_seed=0,
        aux_spec=ax.shift_register_aux_build(
            n_aux=8, gain=0.9, head_inject=1.0, w_seed=0,
            dtype=dtype, device=dev,
        ),
        drive=torch.zeros(50, 8), drive_scale=1.0,
        u_scaled=torch.randn(50),
        radius=1.0, lamT=1.0, input_scale=0.2, dt=0.125,
        washout=10, max_delay=5, jacobian_samples=1,
        y_seq=torch.randn(50), fanout_kind="sparse_random", fanout_k=9,
    )
    raise AssertionError("expected ValueError for fanout_k > hidden_dim")
except ValueError:
    pass
print("AUDIT19_D2_SPARSE_FANOUT_OK")


# --- 20. D3: compressed-bank reference (tapsVD4) ---------------------------
# Determinism: same stream -> identical PC bank (SVD sign/ordering
# pinned); drive width 4; ref row reports finite instruments; edge cost
# 25*4 = 100 vs the full 25*8 = 200.
u_tvd = torch.randn(400, dtype=dtype)
pc1, mean1, comps1 = ax.tapsvd_pc_bank(u_tvd, k=4, n_taps=8)
pc2, mean2, comps2 = ax.tapsvd_pc_bank(u_tvd, k=4, n_taps=8)
assert torch.equal(pc1, pc2), "tapsvd PC bank not deterministic"
assert torch.equal(comps1, comps2), "tapsvd components not deterministic"
assert pc1.shape == (400, 4) and comps1.shape == (4, 8)
# Sign pinning: largest-|.| element of each component is positive.
for i in range(4):
    j = int(comps1[i].abs().argmax().item())
    assert comps1[i, j] > 0, (i, comps1[i, j])
# Ordering: PC bank columns have non-increasing variance.
var = pc1.var(dim=0)
assert (var[1:] <= var[:-1] + 1e-6).all().item(), var
# k guards.
for bad_k in (0, 9):
    try:
        ax.tapsvd_pc_bank(u_tvd, k=bad_k, n_taps=8)
        raise AssertionError(f"expected ValueError for k={bad_k}")
    except ValueError:
        pass
ref_tvd = ax._reference_row(
    ref_leg="tapsvd", seed=0, device="cpu", hidden_dim=25, n_taps=8,
    radius=ax.MAIN_RADIUS, lamT=ax.MAIN_LAMT,
    input_scale=ax.MAIN_INPUT_SCALE, dt=ax.CANONICAL_DT_MAX,
    washout=ax.PROBE_WASHOUT, max_delay=20,
    jacobian_samples=1, u_scaled=u_scaled, y_seq=y_stream,
)
assert_finite(ref_tvd["mc_total"], "ref tapsvd mc_total")
assert ref_tvd["drive"].shape[1] == 4, ref_tvd["drive"].shape
print(
    f"AUDIT20_D3_TAPSVD_OK tapsvd_mc={ref_tvd['mc_total']:.3f} "
    f"(taps8_mc={ref_taps['mc_total']:.3f})"
)


# --- 21. D4: residual-distilled teacher -----------------------------------
# (a) Teacher orthogonality: max |corr| teacher vs main states < 0.10
# (asserted by construction in residual_teacher_build).
teach = ax.residual_teacher_build(
    u_stream=u_scaled,
    hidden_dim=25, n_taps=8, input_scale=ax.MAIN_INPUT_SCALE,
    teacher_k=ax.D4_RESIDUAL_TEACHER_K,
)
assert teach["r"] == ax.D4_RESIDUAL_TEACHER_K, teach["r"]
assert math.isfinite(teach["max_abs_corr"]), teach
assert teach["max_abs_corr"] < ax.D4_TEACHER_ORTHOG_MAX_CORR, (
    teach["max_abs_corr"]
)
# resid_r2_mean is main-EXPLAINED variance: ~0 by construction (the
# teacher is the projected-out residual; a large value would flag a
# broken projection, contradicting the orthogonality gate above).
assert math.isfinite(teach["resid_r2_mean"]), teach
assert teach["resid_r2_mean"] < 0.05, teach["resid_r2_mean"]
# (b) Determinism.
teach_b = ax.residual_teacher_build(
    u_stream=u_scaled,
    hidden_dim=25, n_taps=8, input_scale=ax.MAIN_INPUT_SCALE,
    teacher_k=ax.D4_RESIDUAL_TEACHER_K,
)
assert torch.equal(teach["teacher"], teach_b["teacher"]), "teacher non-det"
# (c) teacher_r2 sanity: exact-copy states -> R^2 ~1; random -> low.
sr_echo = ax.echo_state_aux_build(
    n_aux=8, leak=0.3, radius=0.95, w_seed=0, win_seed=0,
    dtype=dtype, device=dev,
)
echo_states = ax.echo_state_aux_run(spec=sr_echo, u_stream=u_scaled)
r2_mean, r2_max = ax.teacher_r2(
    echo_states, teach["teacher"], washout=ax.PROBE_WASHOUT,
)
assert math.isfinite(r2_mean) and math.isfinite(r2_max), (r2_mean, r2_max)
print(
    f"AUDIT21_D4_TEACHER_OK r={teach['r']} "
    f"max|corr|={teach['max_abs_corr']:.4f} "
    f"echo_teacher_r2={r2_mean:.4f}"
)
# (d) End-to-end D4 leg: run one residual_distilled corner; selection
# by teacher R^2 must produce a finite verdict path.
rows_d4, _, verdicts_d4, _ = ax.run_aux_sweep(
    flavors=(ax.D4_FLAVOR,), legs=("aux_only",),
    aux_seeds=(0,), data_seeds=(0,),
    hidden_dim=25, n_taps=8,
    n_streams=1, train_samples_per_stream=300,
    washout=ax.PROBE_WASHOUT, max_delay=20,
    jacobian_samples=1,
    radius=ax.MAIN_RADIUS, lamT=ax.MAIN_LAMT,
    input_scale=ax.MAIN_INPUT_SCALE,
    device="cpu", dt=ax.CANONICAL_DT_MAX,
    max_corners=1,
)
assert len(rows_d4) == 1 and rows_d4[0].flavor == ax.D4_FLAVOR, rows_d4
assert math.isfinite(rows_d4[0].d4_teacher_r2_mean), rows_d4[0]
assert len(verdicts_d4) == 1, verdicts_d4
assert verdicts_d4[0].verdict in (
    "SUCCESS", "KILL", "INSUFFICIENT_DATA",
), verdicts_d4[0]
print(
    f"AUDIT21_D4_LEG_OK teacher_r2={rows_d4[0].d4_teacher_r2_mean:.4f} "
    f"mc={rows_d4[0].mc_total:.3f} verdict={verdicts_d4[0].verdict}"
)


# --- 22. Verdict amendment: mixed ref budgets -------------------------------
# The compressed tapsVD4 budget (100) applies only when tapsvd matches
# taps8; otherwise the full 200 stands.  (i) stays unconditional.
_base_kwargs = dict(
    flavor="echo_state",
    aux_rows=[RowStub(
        flavor="echo_state", leg="aux_only",
        mc_total=9.0, state_pr=3.0, aux_void_kill=False,
        main_total_edges=272,  # factor 1.36 vs 200, 2.72 vs 100
    )],
    taps8_mc=10.0, scalar_mc=8.0, scalar_esn_mc=8.0,
    aux_states_by_leg={},
    taps8_drive=torch.zeros(20, 8),
)
# tapsvd matches -> ref budget 100 -> factor 2.72 > 1.5 -> no SUCCESS p1
# (and scalar_mc=8.0 keeps p2 out of reach: 9.0 < 1.5*8.0).
v_tvd_match = ax._aux_verdict(
    **_base_kwargs, tapsvd_mc=9.5, tapsvd_edges=100,
)
assert v_tvd_match.verdict == "KILL", v_tvd_match
assert abs(v_tvd_match.aux_edge_cost_factor - 272 / 100) < 1e-9, v_tvd_match
# tapsvd does NOT match -> ref budget 200 -> factor 1.36 <= 1.5 -> p1 fires.
v_tvd_miss = ax._aux_verdict(
    **_base_kwargs, tapsvd_mc=5.0, tapsvd_edges=100,
)
assert v_tvd_miss.verdict == "SUCCESS", v_tvd_miss
assert abs(v_tvd_miss.aux_edge_cost_factor - 272 / 200) < 1e-9, v_tvd_miss
# No tapsvd info at all -> legacy behaviour (200 budget).
v_legacy = ax._aux_verdict(**_base_kwargs)
assert v_legacy.verdict == "SUCCESS", v_legacy
assert abs(v_legacy.aux_edge_cost_factor - 272 / 200) < 1e-9, v_legacy
# Rule (i) untouched: a tap-copy aux still KILLs with a compressed ref.
# Self-subspace aux (aux == taps drive) hits the copy flag regardless.
# NOTE: washout=5 here (the default PROBE_WASHOUT=200 would leave zero
# post-washout rows on a T=30 stub drive and the diagnostic correctly
# returns NaN instead of a copy flag).
_copy_drive = torch.randn(30, 8, generator=torch.Generator().manual_seed(7))
v_copy = ax._aux_verdict(
    flavor="shift_register",
    aux_rows=[RowStub(
        flavor="shift_register", leg="aux_only",
        mc_total=9.0, state_pr=3.0, aux_void_kill=False,
        main_total_edges=208,
    )],
    taps8_mc=10.0, scalar_mc=8.0, scalar_esn_mc=8.0,
    aux_states_by_leg={"aux_only": _copy_drive.clone()},
    taps8_drive=_copy_drive,
    washout=5,
    tapsvd_mc=9.5, tapsvd_edges=100,
)
assert v_copy.verdict == "KILL" and v_copy.tap_subspace_kill, v_copy
print("AUDIT22_MIXED_REF_BUDGETS_OK")


# --- 23. Audit-fix regression guards -------------------------------------
# (a) D4 shares the echo runner/counter dicts (no KeyError path).
assert ax.AUX_RUNNERS[ax.D4_FLAVOR] is ax.echo_state_aux_run
assert ax.AUX_EDGE_COUNTERS[ax.D4_FLAVOR](8) == {
    "aux_recurrent": 64, "aux_fan_in": 8,
}
# (b) Unknown fanout kinds fail fast at all three layers (never a
# silently-dense row under a sparse label).
try:
    ax.run_aux_sweep(
        flavors=("echo_state",), legs=("aux_only",),
        aux_seeds=(0,), data_seeds=(0,),
        hidden_dim=10, n_taps=4, n_streams=1,
        train_samples_per_stream=50, washout=10, max_delay=5,
        jacobian_samples=1, device="cpu", max_corners=1,
        fanout_kinds=("sparse_bogus",),
    )
    raise AssertionError("expected ValueError for bad fanout_kinds")
except ValueError:
    pass
try:
    ax._aux_probe_instrument(
        drive=torch.zeros(20, 4), hidden_dim=10, n_drive=4,
        w_seed=0, win_seed=0, radius=1.0, lamT=1.0,
        dt=0.125, device="cpu", washout=5, max_delay=5,
        jacobian_samples=1, y_seq=torch.randn(20),
        u_scaled=torch.randn(20), fanout_kind="sparse_bogus",
    )
    raise AssertionError("expected ValueError for bad instrument fanout")
except ValueError:
    pass
# (c) Concat-only legs produce a real verdict (aux_concat competes for
# the winner; previously only aux_only/aux_scalar were considered).
v_concat_only = ax._aux_verdict(
    flavor="echo_state",
    aux_rows=[RowStub(
        flavor="echo_state", leg="aux_concat",
        mc_total=10.0, state_pr=3.0, aux_void_kill=False,
        main_total_edges=272,
    )],
    taps8_mc=10.0, scalar_mc=4.0, scalar_esn_mc=8.0,
    aux_states_by_leg={},
    taps8_drive=torch.zeros(20, 8),
)
assert v_concat_only.verdict == "SUCCESS", v_concat_only
# (d) Concat-gain note threshold pinned; D1 note is informational only
# (it rides alongside the MC-path verdict, never deciding it).
assert ax.CONCAT_GAIN_NOTE_DELTA == 0.02, ax.CONCAT_GAIN_NOTE_DELTA
v_gain = ax._aux_verdict(
    flavor="shift_register",
    aux_rows=[
        RowStub(
            flavor="shift_register", leg="aux_only",
            mc_total=9.0, state_pr=3.0, aux_void_kill=False,
            main_total_edges=208,
            ridge_nrmse=0.60, ridge_concat_nrmse=0.60,
        ),
        RowStub(
            flavor="shift_register", leg="aux_concat",
            mc_total=9.0, state_pr=3.0, aux_void_kill=False,
            main_total_edges=208,
            ridge_nrmse=0.60, ridge_concat_nrmse=0.50,
        ),
    ],
    taps8_mc=10.0, scalar_mc=4.0, scalar_esn_mc=8.0,
    aux_states_by_leg={},
    taps8_drive=torch.zeros(20, 8),
)
# Gain 0.10 > 0.02 -> note fires; verdict SUCCESS comes from path 1
# (9.0 >= 8.5 at factor 1.04), not from the note.
assert any("concat readout gain" in r for r in v_gain.reasons), v_gain
assert v_gain.verdict == "SUCCESS", v_gain
# Same rows without the concat edge -> no note.
v_nogain = ax._aux_verdict(
    flavor="shift_register",
    aux_rows=[
        RowStub(
            flavor="shift_register", leg="aux_only",
            mc_total=9.0, state_pr=3.0, aux_void_kill=False,
            main_total_edges=208,
        ),
    ],
    taps8_mc=10.0, scalar_mc=4.0, scalar_esn_mc=8.0,
    aux_states_by_leg={},
    taps8_drive=torch.zeros(20, 8),
)
assert not any("concat readout gain" in r for r in v_nogain.reasons)
print("AUDIT23_FIX_REGRESSIONS_OK")


print("\nALL_AUX_CORE_AUDITS_OK")
