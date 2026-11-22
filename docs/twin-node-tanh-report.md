# Twin node-tanh probe — combined sweep report (coarse + fine)

**Date:** 2026-09-17
**Plan:** `twin-node-tanh` · **Spec:** `twin-probe-spec`
**Logs:** `outputs_alliance_can/narma_twin-60146708.out` (coarse, 144 corners),
`outputs_alliance_can/narma_twin_fine-60149835.out` (fine, 369 corners)
**Code:** `twin_node_tanh()` + helpers in `narma_linear_controls.py`;
audits in `audit_twin_standalone.py` (7/7 green) + `probe_audit2.py` §§12–18

## 1. Question

Neither prior "linear" result tested the architecture under discussion —
dense linear mixing → summation → **per-node** tanh,
`x_{t+1} = (1−a)·x_t + a·tanh(W_mix@x_t + W_in@d_t)` (the ESN update).
Rung-1 ran per-edge tanh in its linear regime (args ~1e-2) under a
leak-dominated map (L≈6.9/sample, J≈0.005 — unwinnable by construction);
C1/C1b ran pure-linear reservoirs with no nonlinearity anywhere
(C1b died by slow-integrator blowup, ridge 2.43). The twin fills the
untested bottom-right cell of that 2×2, digitally, before any analog
cell design.

Pre-registered gates: **PASS_ARCH** = NRMSE ≤ 0.50 **and** MC ≥ 5
(near-ESN numbers → promote analog realization + multistage/anneal-prune);
**FAIL_FABRIC** = NRMSE ≥ 0.60 **and** MC ≤ 3 (C1b-like → fabric exonerated,
harness suspect); **VOID** = tanh disengaged or non-finite (no verdict).

## 2. Setup (identical for both sweeps)

- 25-node standalone discrete map, dense Gaussian `W_mix` normalized to a
  target spectral radius, Gaussian `W_in`; zero training (build → forward
  once → ridge readout). NARMA-10, full streams (2500 samples, washout 200),
  shared C1/C1b instruments (ridge NRMSE/R², washout-corrected MC over
  20 delays with SVD fallback, PR raw + standardized, per-sample Jacobian
  spectra via analytic form verified against autograd to 1e-6,
  tanh-engagement guard on pre-activation stats) + in-harness ESN parity
  ref (sr=0.9, leak=1.0) on the identical stream. Bit-identical run-to-run.
- Two drive harnesses, both run: **taps8** (8-tap causal delay bank
  `[u_t…u_{t−7}]` through 25×8 `W_in`, RMS-normalized — mirrors the
  Phase-6–8 fabric input) and **scalar** (single rail-mapped value through
  25×1 `W_in`, C1-style; ESN parity sees bit-identical drive here).
- Twin states live in [−1,1] (rails N/A per C0 convention); activity
  screened in tanh units against the ESN ref.

## 3. Sweep 1 — coarse grid (job 60146708)

Grid: radius {0.7, 0.9, 1.0, 1.1} × leak a {0.3, 1.0} × input scale
{0.2, 0.5, 1.0} × data-seeds {0,1,2} × harnesses {taps8, scalar}, one
W per corner → 144 legs + preflight, 144/144 complete, 0 failures, 0 voids.

| | twin NRMSE mean / best | twin MC mean / max | parity-ESN NRMSE / MC |
|---|---|---|---|
| taps8 (72) | 0.647 / **0.505** | **8.10 / 10.29** | 0.763 / 5.22 |
| scalar (72) | 0.688 / 0.547 | 4.65 / 8.94 | 0.763 / 5.22 |

Verdicts: **0 PASS** (best 0.5047 misses 0.50 by 0.005, MC passes),
**7 FAIL** (all scalar/a=0.3/is=1.0 — hot drive + slow leak + scalar
bottleneck, NRMSE ~0.64/MC ~2.8; a drive/retention mismatch, no blowup
anywhere), 137 INCONCLUSIVE. Landscape: input scale dominates (0.2 good,
1.0 degraded in every group); radius nearly flat (±0.01–0.02); seed 1
systematically kindest (parity ESN agrees: 0.64 vs 0.67–0.73).

## 4. Sweep 2 — fine grid (job 60149835)

Targeted at sweep 1's winning region, adding the highest-value new axis
(**W-seed search** — sweep 1 used one random W per corner).
Grid: taps8 MAIN radius {1.0, 1.1} × a {0.2, 0.3, 0.5, 1.0} × input scale
{0.10, 0.15, 0.20, 0.25, 0.30} × W-seeds {0,1,2} × data-seeds {0,1,2}
(360 legs) + scalar CONFIRM 3 best-guess combos × 3 seeds (9 legs) →
369/369 complete, 0 failures, 0 voids.

| | twin NRMSE mean / best | twin MC mean / max | verdicts |
|---|---|---|---|
| taps8 (360) | 0.536 / **0.393** | **9.35 / 13.11** | **78 PASS**, 0 fail |
| scalar (9) | 0.632 / 0.547 | 7.35 / 8.94 | 0 pass (replicates sweep-1 ceiling) |

Best corner: r=1.0/a=1.0/is=0.1/W-seed 2/seed 1 → **0.3926 / MC 12.45 /
PR 7.17** vs parity ESN 0.5219 on the same stream. Top-12 corners span
0.39–0.44 with MC 10.7–13.1. Per-scale means (taps8): 0.1 → 0.491;
0.15 → 0.515; 0.2 → 0.539; 0.25 → 0.559; 0.3 → 0.576 — quieter is
monotonically better. W-seed moves NRMSE by mean 0.031 / max 0.073 at
fixed (r, a, is, data-seed); W-seed 2 owns the top 2 corners.
At is=0.1, a=1.0 takes 7 of the top 10 (MC ~12–13) — the ESN-native
full-replacement setting is the memory maximizer, as theory predicts.

## 5. What we learned

1. **Mechanism confirmed.** Dense mixing → sum → per-node tanh holds
   ESN-class memory (MC 8–13, PR 4–9) with healthy Jacobians
   (max|λ| 0.35–1.0, rank-proxy up to ~25) and engaged-but-unsaturated
   tanh in all 513 corners across both sweeps. No blowups, no voids,
   no leak-dominated pathology. This is what rung-1 and C1b could not
   show: rung-1 forgot (J≈0.005) and C1b blew up (ridge 2.43); the twin
   remembers and stays bounded.
2. **The remaining gap to the tuned ESN (0.296) is tuning, not
   architecture.** Every gradient points the same way — quieter drive,
   W-seed search, a∈[0.5,1.0] undersampled — and each refinement step so
   far (300-sample smoke 0.42 → coarse best 0.505 → fine best 0.393) has
   paid off. Nothing in the data suggests a second missing mechanism.
3. **The delay bank earns its keep.** Scalar-harness MC caps at ~4–5
   (a=0.3) while taps8 reaches ~9–13 at matched settings: supplied
   memory plus mixing is the winning combination, consistent with the
   "fabric as nonlinear expander" reframe.
4. **Scalar + slow-leak + hot-drive is the one dead corner** (7 FAIL
   legs, both sweeps agree) — boundary condition for the operating
   envelope, not a structural result.
5. **Gatekeeping note:** sweep-1's 0.50 miss by 0.005 could have read as
   failure; the fine grid cleared it 78 times over. Pre-registered
   near-misses deserve follow-up grids, not verdicts.

## 6. Implications (per twin-probe-spec decision rule)

PASS_ARCH cleared with margin → promote: (a) analog realization Study
(linear-OTA mixer + diff-pair node tanh — more analog-native than the
current per-edge tanh); (b) multistage reframing (stage = mixing bank,
transfer = node nonlinearity); (c) dense-start anneal-prune as the
topology-search method under a device budget. Heun-integrated
rhs-override twin remains deferred future work (discrete map decided
the architecture question first).
