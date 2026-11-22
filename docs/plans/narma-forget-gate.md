---
plan name: narma-forget-gate
plan description: Gated retention leak design
plan status: active
---

## Idea
Implement Step-6 forget-gate leak from docs/narma-revised-plan.md §11-Q1 as softplus-unbounded dynamic leak (leak_eff_j(t) = softplus(a_j·V_u + c_j·x_j + b_j), 75 params, free init b≈−3.0), gated behind Round-2 VCA verdicts per §13.5: design + audit now, E0 probe then one short k=2 train only if VCA moves nothing. Unbounded stability handled by monitor-only guardrails (rail%/finiteness kill + Jacobian spot-check); tanh-bounded follow-up explicitly deferred.

## Implementation
- Add forget_gate mode to DifferentialStage: per-node (a,c,b) params, rhs() computes leak_eff(t)=softplus(a*V_u + c*x + b) per Heun step, off-default byte-identical
- Thread factory: _build_fabric_net(forget_gate=...) + CLI flags + cond-name suffix + CSV column, following §13.2 VCA-plumbing pattern
- Write audit script: off-state byte-identity, on-state forward-finite, 75-param accounting, init values land (b≈−3), 1-epoch train smoke, Jacobian spot-check monitor-only
- E0 frozen-gate probe on VCA-winning corner: ridge/MC/PR/rails/IPC-2/product-pairs + gate-active vs gate-frozen-open delta
- Gated train (only on §13.5 'VCA moves nothing' fork): one k=2 short train, temporal readout, fp32 per S3 AMP lesson
- Pre-registered kill evaluation: no MC/PR/product-pair movement vs static control, or rail/non-finite death → close plan, register bounded-tanh follow-up

## Required Specs
<!-- SPECS_START -->
- advisor-probe-suite
- canonical-ctle-bench
- lstm-baseline-complexity
- multistage-sequence
- narma-e0-gate
- plain-mlp-config
- forget-gate-leak
<!-- SPECS_END -->