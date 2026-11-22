# Spec: aux-core-probe

Scope: feature

# Spec: aux-core-probe (feature scope for plan aux-core-drive)

Reusable contract for a learnable aux-core input stage that replaces the
static multi-tap delay bank. Applies to the twin-level probe first; same
legs/verdicts carry over to any future fabric-integrated aux core.

## Problem

Supplying explicit delay taps is not a crime at one input, but cost scales
as 8M routed voltages + (nodes × 8M) dense mixer weights for M inputs.
The static bank is unprunable and unoptimizable. Proven pulsing: taps8
drive is load-bearing (twin taps8 MC ~9–13 vs scalar ~4–5, both maps;
fabric sparse-drive and small-world legs failed), so the replacement must
*earn* its place against the bank, not merely exist beside it.

## Architecture (strict feedforward; feedback explicitly out of scope)

Input → aux core → main core → output, plus an optional direct
input→main tap (aux+scalar leg). No main→aux path. Readout may be
main-only or concatenated [main; aux] (must be declared per leg).

## Aux flavors (all three; echo-state first)

1. **Echo-state aux**: 8 nodes (matches tap count for fair comparison),
   Gaussian W at radius ~0.95, scalar input, leak sweep {0.3, 1.0}.
2. **Shift-register aux**: chain topology, learned per-edge gain, head
   injection (interpretable learned delay line; exact chain asserted).
3. **Slow-feature bank**: decoupled leaky integrators at log-spread τ
   (attacks timescale-diversity deficit directly).

## Legs (eval-only, zero training; same streams/seeds everywhere)

Per flavor: aux-only drive (8 cols), aux+scalar drive (9 cols).
References (every sweep): taps8 drive, scalar drive, scalar-ESN parity.
Main core pinned to the Gate-1 best corner (λT/radius/scale); aux
hyperparameters swept small (3 aux-seeds × 2–3 settings).

## Instruments (reuse twin instruments, no new metrics)

Ridge NRMSE/R², washout-corrected MC total + per-delay curve, PR,
Jacobian spectra, tanh-engagement guard, ESN parity. Per-leg rows carry
config tag + PASS/FAIL + itemized edge counts (aux recurrent + fan-in +
fan-out separately, for the device-budget comparison).

## Pre-registered verdicts (per flavor)

- **SUCCESS**: aux leg matches taps8 MC within tolerance at ≤ edge cost,
  or beats scalar by margin ≫ cost.
- **KILL**: (i) aux states project onto the tap subspace
  (canonical-correlation ≈ delay-line copy = expensive taps);
  (ii) aux void/dead; (iii) no gain over main+scalar.
- No flavor may claim victory on MC total alone without the
  beyond-window (k > n_taps) breakdown reported alongside.

## Output contract

Per-flavor verdicts + edge-count tables land in the leg output dir next
to (never inside) training tables. Fabric integration
(`DifferentialStage` surgery) is earned only by a SUCCESS verdict, never
started speculatively. Feedback coupling is a separate future plan.