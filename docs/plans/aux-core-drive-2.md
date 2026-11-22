---
plan name: aux-core-drive-2
plan description: Concat readout, sparse fan-out, compressed bank, residual-distilled aux
plan status: implemented-pending-test
---

## Idea
Follow-up to aux-core-drive. Reopen the aux-core question on four legs the
twin probe left unrun, without touching the KILL rules that closed
aux-core-drive. The SR grievance is addressed structurally rather than by
weakening rule (i): a tap-copy still dies wherever it appears, but D2/D3
change what "the bank" costs, D1 changes what the readout can see, and D4
gives aux an objective distinct from main's. All eval-only; no fabric, no
feedback, no gradient training.

## Implementation
- D1 concat readout leg (aux_concat): ridge over [main_states(25); aux_states(8)]
  = 33 features, main-only readout kept as paired control; per-row copy-flag
  (tap_r2) reported on every leg incl. concat; rule (i) still fatal
- D2 sparse main fan-out (k in {3,6}, Gate-3 sparse_random_W_in reuse,
  radius-restored): replaces the dense 25 x n_aux fan-in; same drive-power
  parity and verdict paths incl. rule (i)
- D3 compressed-bank reference (tapsVD4): SVD top-4 PCs of the taps8 bank
  (deterministic, sign-pinned), drive = 4 PC cols at parity + dense 25x4
  fan-in (100 edges); reference-only, never enters _aux_verdict
- D4 residual-distilled aux: offline Phase A teacher = top-4 residual
  directions of taps8 PCs after projecting out the main+scalar state span;
  echo hyperparams selected by mean teacher R^2 (existing leak/radius/seed
  axes only); then standard sweep + verdicts
- Verdict amendment (additive only, rule (i) untouched): tapsVD4 refs gain
  tapsvd_mc/tapsvd_edges; flavor ref budget = min(full, compressed) when the
  compressed bank matches taps8 within 0.85; SUCCESS/KILL thresholds unchanged
- Audit extensions in audit_aux_core.py sections 18-22: D1 concat row,
  D2 exact nnz + SR@k3 == 32 edges, D3 determinism/sign/variance ordering +
  ref row, D4 teacher orthogonality (< 0.10) + determinism + end-to-end leg,
  mixed-ref-budget verdict aggregation with rule (i) untouched
- Docs: per-leg verdicts + edge tables recorded in the progress summary after
  the first full run; fabric integration still requires a SUCCESS under the
  kept rules

## Required Specs
<!-- SPECS_START -->
- aux-core-probe
<!-- SPECS_END -->

## Pre-registered expectations (directional, non-binding)
- D2-SR dies on (i) again (confirmatory); D2-echo / D2-SF are the
  cost-undercut candidates.
- D3 either matches taps8 (baseline moves to 100 edges) or it does not
  (bank is incompressible — strengthens the original KILL).
- D4 is the only leg that can produce a non-copy aux with taps8-grade MC;
  a D4 SUCCESS is the sole path to fabric integration in this plan.

## Explicitly out of scope
Gradient-optimized aux weights; M>1 input scaling sims; any
DifferentialStage surgery before a SUCCESS.
