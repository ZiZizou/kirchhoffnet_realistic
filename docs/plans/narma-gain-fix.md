---
plan name: narma-gain-fix
plan description: Spectral-radius gain normalization fix
plan status: active
---

## Idea
The advisor claims the fabric's MC ceiling is an operating-point error, not a structural limit: rung-1 matched the ESN leak but not the mixing term, leaving gm·isat ~0.005–0.2 where the per-sample Jacobian retains ~0.5% of state (MC 0.003). ESN's memory comes from both retention (~0.9 via leak) and mixing (ρ(W)≈0.9); KNet has never tested retention≈0.9 + mixing≈0.9 without saturation because gm and Isat were varied independently and the small-signal conductance is their product, not either alone. The fix is a one-time init-time rescaling of Isat (or global gain) so that the Heun-map Jacobian at x=0 has ρ≈0.9, mirroring ESN practice. Gating (hidden-VCA, dynamic leak) and multi-stage are then retested on a live network. This plan first locally verifies the four load-bearing factual claims (carryover, gm·isat product, ESN leak convention, rung-1 parameter mismatch) with code inspection + deterministic CPU smokes, then implements the spectral-normalization init hook, re-runs the corrected rung-1, and only then re-enables gated/depth experiments.

## Implementation
- Step 0 — Claim triage (read-only, local CPU): verify E2 carryover + Jacobian step-count dependence from code/logs; prove small-signal gain is gm·isat (FreeTanhLibrary._tanh_core) and that default init gives product ~0.006; audit ESN leak convention (narma_experiment.ESN leak=1.0 => retention 0 vs 0.9) and rung-1's ESN-mapped leak L=-ln(1-l)/ts vs the advisor's 'leak 6.9 vs 0.05' reading; reproduce rung-1 numerically (MC 0.003, |J|~0.005) and the ESN reference MC 12 on identical streams — decide if advisor's 'retention 0.001 + mixing ~0' description matches code
- Step 1 — Design spectral-normalization hook: define contract (build net -> linearize Heun map J=∂Heun(x,u=0)/∂x at x=0 -> compute ρ(J) via eig or power iteration -> scale Isat_raw by target_rho/current_rho -> rebuild or in-place rescale); choose scope (new function normalize_spectral_radius(net, target=0.9) in topology.py or narma_advisor_probes.py, exposed via --gain-norm / --target-rho flag); decide Isat vs gm vs global gain as the rescaled knob and how to respect gm_min/max rails and x_max compliance; handle multi-stage (per-stage vs global)
- Step 2 — Implement hook + CLI: add normalize_spectral_radius() with single-sample Jacobian (reuse _one_sample_transition/jacobian_eigs path), safe fallback when eig fails, deterministic seed handling, logging; wire through config.make_narma_preset / topology.build_net_from_config and narma_experiment._build_fabric_net; add --spectral-radius-target / --spectral-norm flag to narma_experiment.py, narma_advisor_probes.py e0_sweep, and narma_linear_controls rung-1 entry; keep backward compat (no flag => legacy behavior)
- Step 3 — Corrected rung-1 reproduction: re-run rung-1 with retention 0.9 (leak 0.1, matching ESN spectral-radius-only retention) and ρ(J)=0.9 via the new hook, input_scale lowered to keep |x|<0.9*x_max; report MC, ridge, PR, rail_frac, J spectrum vs ESN reference on same streams; pass criterion MC>5 and ridge <0.4 rail-free => pipeline capable; if MC stays <3, pipeline-limited verdict
- Step 4 — E0 operating-point re-screen (if Step 3 passes): 48-corner gm×isat×drive grid with spectral normalization on (target 0.85-0.95) vs off; score MC/PR/ridge/rails via e0_sweep; verify the 'retention 0.9 + mixing 0.9 + no saturation' corner exists rail-free; audit that hidden-conditioned VCA and dynamic leak now see O(1) hidden signal (hidden RMS, gate variance)
- Step 5 — Gating retrial on live network: re-run 20-epoch spike trains B/C/D (dynamic leak, hidden-VCA) from a spectral-normalized init; check trajectories are no longer digit-identical, rail_frac behaves, MC does not collapse 3.0->2.0 while NRMSE improves; decision gate for Step-6 trainable forget gate (softplus-unbounded leak) now that gates have signal
- Step 6 — Audits + batch wiring: extend probe_audit1/2 and add probe_audit_gain_norm.py (init product -> Jacobian scaling -> MC lift, epoch-0 identity when target==measured, multi-stage split t_span); update batch scripts (narma_gating_e0_batch, narma_gating_spike_batch, narma_gating_depth_batch) with LEG_TIMEOUT bump to 12000s and new flags; document results in narma-progress-summary.md §10

## Required Specs
<!-- SPECS_START -->
- advisor-probe-suite
- canonical-ctle-bench
- lstm-baseline-complexity
- multistage-sequence
- narma-e0-gate
- per-node-diagnostics
- plain-mlp-config
- narma-gain-norm
<!-- SPECS_END -->