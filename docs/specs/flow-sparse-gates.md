# Spec: flow-sparse-gates

Scope: feature

# Spec: flow-sparse-gates (feature spec for plan post-twin-gates)

Scope: feature

## Purpose

Pin the three post-twin fatal-risk gates that stand between the confirmed
digital twin and any multistage work. Each gate is eval-only, zero-training,
and carries a pre-registered fatal verdict that stops its downstream line.
Multistage (frozen baseline, then training) stays stubbed until all three
pass and emit a buildable spec.

## Gate 1 — Heun flow-twin (realizability gate)

- Standalone Heun integrator for `dx/dt = −λx + λ·tanh(W_mix@x + W_in@d)`,
  drive held constant within a sample. Standalone, NOT stage rhs-override
  (keeps cell-lib/stage state untouched; analytic per-substep Jacobian).
- Fixed-dt rule: t_span=1.0 held; num_steps scaled so dt ≤ 0.125 at every λT.
  Audit asserts the dt bound in every leg.
- Sweep: λT {0.3, 1, 3, 10} × radius {0.9, 1.1} × input scale {0.1, 0.2} ×
  W-seeds {0,1,2} × data-seeds {0,1,2}, taps8 main + scalar confirm
  (λT {1,3}, r=1.1, is=0.2). Same instruments + ESN parity + engagement
  guard as the discrete twin.
- Verdicts: holds at λT≤3 with MC≥8 → flow-viable, proceed; holds only at
  λT≥10 → viable with recorded speed requirement (OTA gain/capacitance
  floor for the cell spec); collapses everywhere → digital-only, redirect
  program, cancel multistage.

## Gate 2 — sparsity/device-budget + mismatch (parallel with Gate 1)

- Sparsity masks on W_mix at matched edge counts {~100 torus-equivalent,
  200, 400, 625 dense}, magnitude-first: (a) random-sparse, 3 mask-seeds
  each; (b) magnitude-pruned-from-dense with spectral-radius restoration
  after pruning (unrescaled pruning is a protocol violation — it confounds
  sparsity with spectral collapse). Plus one low-rank+sparse row (rank-2
  global rails + local ring) at ~200 edges.
- Mismatch: build-time multiplicative Gaussian noise on W_mix and W_in,
  σ {0, 0.02, 0.05, 0.10}, 3 noise draws × 3 W-seeds at the best corner.
- Heterogeneous node operating points (per-node tanh gain/bias jitter) as
  one row at the best corner (free physically; doubles as
  mismatch-tolerance probe).
- Verdicts: graceful degradation to ~200 edges → anneal-prune becomes the
  topology-search method with a concrete edge target; cliff below 500 →
  hard physical decision before any training code. Mismatch holds at
  5–10% → silicon-plausible; collapses → precision redirect.

## Gate 3 — sparse structured W_in

- Dense Gaussian (banked reference) vs sparse-random (each tap → k random
  nodes, k∈{3,6}) vs sparse-structured (each tap → localized node group
  matching torus geometry), all with dense W_mix distributing. ~20 legs at
  the best corner.
- Verdict: structured holds → boundary spec is localized tap groups +
  dense mixer; only-dense holds → input routing joins the device-budget
  problem. Also revisits the killed Round-2 sparse-drive result where it
  gets a fair trial.

## Output contract

- Per-gate verdicts recorded against the fatal criteria above; on all-pass,
  emit the buildable spec: (λT operating point, edge-count target,
  mismatch tolerance, W_in geometry) as the sole input to the
  frozen-multistage experiment. No training anywhere in this plan.