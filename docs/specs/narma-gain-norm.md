# Spec: narma-gain-norm

Scope: feature

# Feature Spec: NARMA Gain Normalization (Spectral-Radius Init)

## Context
The fabric's MC ceiling (rail-free MC ≤2.3, MC>3 only with ~30% rails) correlates with a per-sample Jacobian max|λ| ≈0.005 in the native-linear rung-1 screen (MC 0.003). Small-signal gain is Isat·gm (FreeTanhLibrary._tanh_core: i≈isat·gm·ΔV), not gm alone. Default init gm≈isat≈0.077 gives product ≈0.006, far below the regime where recurrent mixing retains state. ESN practice scales W to ρ≈0.9; KNet never has.

## Goal
Provide a one-time, deterministic, CPU-only init rescaling that puts the Heun-map Jacobian at x=0,u=0 on the edge of chaos (target ρ≈0.9) without entering compliance, then retest the pipeline's memory capacity and gating levers on a live network.

## Contract

### 1. Core Function
`normalize_spectral_radius(net: KirchhoffNetWithIO, target_rho: float = 0.9, knob: str = "isat") -> dict`
- Build-time hook: call once after `topology.build_net_from_config` / `config.make_narma_preset`.
- Linearize the single-sample Heun map `J = ∂Heun(x,u=0)/∂x` at `x=0` (reuse `narma_advisor_probes._one_sample_transition` + `torch.autograd.functional.jacobian`; power-iteration fallback if eig fails).
- Compute `rho = max|eig(J)|` (or power-iteration spectral estimate).
- Scale knob: `isat_raw` (default), or `gm_raw`, or global gain. Preferred: `isat_raw` via sigmoid inverse: `isat_new = isat_old * (target_rho / max(rho,1e-9))` clamped to [isat_min,isat_max] rails; log the ratio.
- Return `{rho_before, rho_after, scale, knob, target_rho, n_nodes, method}` for audit.
- Deterministic: same seed + target -> same `isat_raw` up to float32 rounding.
- Multi-stage: per-stage normalization (each stage's J scaled to target^(1/num_stages)) OR global last-stage J; spec picks per-stage for depth>1 (t_span split already).

### 2. CLI
- `narma_experiment.py`: `--spectral-norm` (bool) + `--spectral-radius-target` (float, default 0.9) + `--spectral-knob {isat,gm,gain}`.
- `narma_advisor_probes.py e0`: same flags threaded through `e0_sweep` and `_e0_row` (config_tag suffix `_rho<target>_knob<k>`).
- `narma_linear_controls.py`: same flags for rung-1 re-run.
- No flag => legacy behavior (epoch-0 bit-identical; audit asserts scale≈1 when rho≈target).

### 3. Edge Cases
- `rho≈0` or non-finite eig: halve/do not scale, warn, return NaN rho.
- Already at target within tol 0.005: no-op.
- Compliance: after scaling, verify `|x|<0.9*x_max` on a 1-stream washout; if rails >5%, warn and optionally reduce input_scale (caller decides).
- Backward compat: no spec change to `cell_library.py` defaults; overrides are post-build fills only (probe_audit1 asserts defaults preserved).

### 4. Verification
- Unit: given synthetic `gm_raw=-5,isat_raw=-5` net, `normalize_spectral_radius` scales isat so measured `rho_after` in [target-0.02, target+0.02]; epoch-0 identity when target==rho_before.
- E0 probe: corrected rung-1 (`leak=0.1, t_span=1.0, num_steps=8, input_scale` tuned) lifts MC 0.003 -> ≥5 and ridge <0.6 rail-free if pipeline is capable. If MC stays <3, verdict is pipeline-limited (not operating-point).
- Gating sanity: hidden RMS O(1) after norm, `vca_W_hidden` gate variance >0 (vs ~0 before), spike B/C/D trajectories diverge.

### 5. Non-Goals
- Training-time adaptation (init only).
- Changing leak (leak stays 0.05-0.1 per corrected rung-1; mixing is the scaled axis).
- Two-core/feedback/systolic structures (deferred).

## Dependencies
- `differential_stage._one_sample_transition`, `jacobian_eigs`, `topology.build_net_from_config`, `config.make_narma_preset`.