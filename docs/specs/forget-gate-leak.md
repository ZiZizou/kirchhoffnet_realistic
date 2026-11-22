# Spec: forget-gate-leak

Scope: feature

# Forget-Gate Leak (feature spec for plan narma-forget-gate)

Scope: feature — reusable for any fabric task needing state/input-dependent retention.

## Context

Static per-node leak (`softplus(raw_leak)`, `raw_leak_init = -3.0` → 0.0486)
cannot express timescale selection: every node forgets at one learned rate.
The LSTM borrow is a per-node, per-sample retention multiplier computed from
both the fresh input drive and the node's own history-bearing state.

## Gate math (softplus-unbounded leg)

Per node `j`, per sample `t`:

```text
s_j(t)      = a_j · V_u(t) + c_j · x_j(t) + b_j
leak_eff_j(t) = softplus(s_j(t))          // (0, ∞): floored at 0, unbounded above
```

- `V_u(t)`: scalar input drive for the sample (`u ∈ [0,0.5] → V_u ∈ [−3,+3]`).
  Shared by all nodes, history-free ("what just arrived").
- `x_j(t)`: node `j`'s own ODE voltage. History-bearing ("what this node
  remembers"). This feedback — state modulating its own decay rate inside
  `rhs()` — is what makes it a genuine forget gate rather than
  input-dependent leak.
- `(a_j, c_j, b_j)`: per-node input sensitivity / self-state sensitivity /
  baseline openness. 25 hidden nodes → 75 new params.
- Reuses the existing leak link function: today's static leak is
  `softplus(raw_leak_j)`, so the gate is `raw_leak_j(t)` made dynamic.
  No new nonlinearity machinery.

## Init (free, near-canonical)

- `b_j = −3.0 + small noise` (starts near today's 0.0486 regime, free to move).
- `a_j, c_j ~ N(0, 0.1)` (near-static at step 0, gate authority grows in training).
- No identity-at-init constraint: day-0 is near-canonical, not byte-identical.

## Gradient profile

`softplus′(s) = σ(s)`: upper side → 1 (fast-forget direction always has full
gradient — no vanishing). Low side thins (`σ(−3) ≈ 0.047` at init), so the
"remember harder" direction learns slowly at first. Accepted for this leg.

## Integration point

`DifferentialStage.rhs()` where `leak_term = leak * x` is computed
(`differential_stage.py`, `_effective_leak` call site): when forget-gate mode
is on, replace the static per-node leak with `leak_eff(t)` computed from the
current `(V_u, x)` each Heun step. Mode off (default) → byte-identical `rhs`.

## Stability posture (unbounded — monitor, don't clamp)

No upper clamp on this leg. The existing instruments are the guardrail:

- rail% over threshold or non-finite BPTT → leg killed (same finiteness guard
  that caught the S3 dense-AMP overflow).
- Jacobian max|J| spot-check against the C1b band (monitor-only).
- If the leg dies by rails/non-finites, that is the pre-registered evidence
  motivating the bounded-tanh follow-up (`λ = 1 + 0.8·tanh(s)`, range
  [0.2, 1.8]→[0.2, 2]).

## Explicitly deferred

- GRU-tied variant (one `λ_j` scaling both leak and boundary drive current):
  spec'd, not built — entangles retention + intake (attribution confound).
- Sigmoid-rescaled ablation (`σ` to [0.2, 2]): one-leg comparison for
  LSTM-literature comparability, only if the softplus leg moves.
- Bounded-tanh primary: becomes the next plan iff this leg dies by
  instability or saturates without memory gains.