# Spec: regime-transplant-grid

Scope: feature

# Regime-Transplant Grid (feature spec for plan narma-regime-transplant)

Scope: feature — reusable for any fabric task whose E0 grids only sampled the
linear-init corner and need a calibrated jump to a new operating regime.

## Diagnosis it encodes

ESN-25 (same node budget) scores NRMSE 0.3980 / MC 14.65 from the fast,
lightly-driven corner (sr=0.9, leak=1.0 convex-blend = zero direct retention,
input-scaling=0.2) while the fabric's defaults sit in the opposite corner
(slow leak τ~35, full-rail swing). Every fabric E0 grid varied *within* the
fabric's corner. The trained k2 checkpoint shows the optimizer walking toward
higher gain (boundary gm 5×, Isat decoupled −3.35/−2.67) and stalling an order
of magnitude below the saturation cliff: repelled, not frozen.

## Regime translation (fabric coordinates)

Per-sample retention ≈ e^{−leak·t_span} (t_span=1.0, C_eff=1). Target
retention 0.37–0.82 (τ 1–5 samples) → leak 0.2–1.0 → raw_leak −1.5 to +0.5
via softplus-inverse. (NOT raw 0..+2: that gives retention 0.50–0.12, faster
than the ESN winner's effective regime.)

- **leak:** raw −1.5 / −0.5 / +0.5 (fast, ~5–20× faster forgetting than default −3.0)
- **swing:** 0.1 / 0.2 / 0.3 of rails (below the historical 0.25 floor)
- **gm:** raw −3.5 / −3.0 (moderate curvature, gm ≈ 0.3–0.5, no saturation at reduced swing)
- **Isat:** coupled (= gm) vs high (−2) vs low (−6) at fixed gm — both decoupled
  quadrants (optimizer's revealed preference is high-Isat/low-gm; test the
  advisor's high-gm/low-Isat too)
- **Priority canary:** high core Isat (raw −2) + moderate core gm (raw −3.5),
  leak raw −0.5, swing 0.2 — asks whether the core becomes useful
  (gate-zero delta flips sign) when its currents are strong at moderate curvature

## Harness plumbing (probe-only, zero dynamics changes)

1. Isat-decoupled fill: `--isat-grid` (or per-corner isat_init passthrough);
   default preserves coupled behavior byte-identically.
2. Swing floor: extend `--drive-grid` below 0.25 (0.1).
3. Per-library fills for the anchor/means corners (core vs boundary vs output
   differ); coupled single-fill path untouched.

## Exact-tensor anchor (instrument calibration)

Load the k2 epoch-54 checkpoint's real dynamics tensors (core gm −4.10/Isat
−4.09; boundary gm −3.35/Isat −2.67; output gm −3.73/Isat −3.72; leak −3.56;
readout refit by ridge, not copied) as an E0 corner. Expect ridge 0.60–0.68
(MC ≈2.6). Reads ≈0.64 → instrument calibrated, historical all-FAILs stand.
Reads ≈0.72 → instrument blind, all historical FAILs voided, stop and fix
the meter. The anchor validates the judge, not the student: it does not test
any training proposal and does not declare the architecture dead anywhere.

## Gates (unchanged)

MC ≥ 4, rails ≤ 5%, ridge < 0.6. Full grid ≈54 corners × ~80 s ≈ 1.2 GPU-hr.

## Canary outcome table (pre-registered)

- Anchor ≈0.64 + canary passes → expand to full grid, then short k=2 train.
- Anchor ≈0.64 + canary fails → transplant wounded; re-examine the
  ESN→fabric translation, do not burn GPU on the full grid.
- Anchor ≈0.72 → stop; fix the instrument; asterisk all historical FAILs.
- Anchor ≈0.64 + gate-zero flip without gate pass → core-usefulness confirmed;
  full grid with core-Isat as lead axis.