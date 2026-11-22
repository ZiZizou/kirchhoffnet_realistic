# Spec: core-decomp-legs

Scope: feature

# Shunt-vs-Tanh Decomposition Legs (feature spec for plan canary-core-split)

Reusable leg definitions for decomposing a gate-zero flip into branch-level attribution and re-testing with init diversity. Scoped to single-stage NARMA fabric E0 corners; probe-side fills only, zero dynamics changes.

## Arm definitions (post-build fills on the exact priority corner net)

All arms start from the identically built + `apply_gain_override` corner net (same seed → same `s_raw`, same streams). Each arm re-collects states and re-scores the full E0 instrument set (ridge / MC / PR / rails / Jacobians) plus the gate-zero pass.

- **Arm A (shunt-only kill):** fill `g_resistive_raw` → −20.0 on the core `cell_lib` (G ≈ 2e-9, tanh branch untouched). Live tanh + dead shunt.
- **Arm B (tanh-only kill):** fill core `gm_raw` + `isat_raw` → −20.0 (tanh current → ~0, shunt live at init G). Dead tanh + live shunt.
- **Arm C (control):** existing `_gate_zero_eval_forward` (`z_logits` → −12, restored after). Must reproduce the recorded gate-zero numbers (canary: ridge 0.8122 / MC 0.86); if it does not, the leg is void (build non-determinism) — stop, do not interpret A/B.

Boundary/output libraries are NEVER touched by the arms (drive path identical across arms).

## Pre-registered reading

- A kills the bimodality (nodes spread, MC jumps), B doesn't → shunt is the homogenizer → Phase-2 fix is jittered-G init (+ shunt-reduction arm at raw −7.3).
- B kills it, A doesn't → tanh degree-weighted injection creates the split → jittered gm/Isat init. (Prior: tanh dominates ~30× per-unit-ΔV at the canary corner, so expect B.)
- Both partially kill it → branches interact (shunt drains what tanh injects unevenly) → jitter both.
- Neither kills it → split is in the boundary injection itself → new hypothesis, report back.
- A kills everything including memory (MC → ~0) → shunt was carrying the memory, tanh decorative — strangest outcome, highest value.

## Circulation grid (Phase 3, gated on arm verdict)

Torus only, x_max pinned at 3.0: `{jitter-G, jitter-gm, both}` × `{retention 0.6, 0.3 via raw: leak tokens}` × `{swing 0.1, 0.2 via drive}` ≈ 12 corners. Gates: MC ≥ 4, rails ≤ 5%, ridge < 0.6 (C0-anchored). First grid to sample the circulation regime with mixing diversity. On pass: transplant-54 grid gains the winning jitter axis, then a short k=2 train from the winner. On fail: report rails-vs-mixing verdict; revival ladder (x_max sweep → hetero-C → AGC) activates in that order.