# Spec: per-node-diagnostics

Scope: repo

# Per-Node Diagnostics (repo scope)

Probe-side-only instrument for decomposing trajectory-global E0/canary metrics (rail fraction, PR, MC) into per-node structure. Answers "which nodes rail, which are dead, and which local quantity predicts the split" before any dynamics change.

## Instrument: `node_diagnostics(net, states_full, u_seq, x_max)`

Inputs: built single-stage net (post `apply_gain_override`), collected full states `(T, N)`, input stream, rail magnitude. Eval-only, no weight mutation.

Per hidden node `j`, report:
1. `max_abs_ratio[j]` = max_t |x_j(t)| / x_max — the bimodality histogram (hot vs dead).
2. `shunt_G_sum[j]` = Σ softplus(g_resistive_raw) over core edges incident (in+out) on `j` — tests the shunt-homogenizer story (railed nodes ↔ low total G, dead nodes ↔ high total G).
3. `sign_balance[j]` = Σ sign(s_raw) over incoming core edges — tests the sign-pattern story.
4. `boundary_current[j]` = boundary OTA current into `j` at init-state (first-sample, zero-state) — tests the heterogeneous-drive story; uniform at init by construction, so a null here is expected and informative.
5. Pearson correlations of (2)–(4) against (1). One of these explains the split before any integration is run.

## Provenance rules

- `s_raw` is never touched by `apply_gain_override`; every E0 corner rebuilt with the same build seed shares one sign pattern (`torch.manual_seed(seed)` in `_build_fabric_net` + E0 factories). Dump `s_raw` once per leg; rail variance across same-seed corners cannot be sign-driven.
- Gate-zero (`z_logits` → −12) kills tanh AND shunt in both frozen and dynamic paths (edge_mask multiplies both; `differential_stage.py` rhs + `_compute_i_edge_const`). Arm A (shunt-only kill: `g_resistive_raw` → −20) and Arm B (tanh-only kill: core `gm_raw`/`isat_raw` → −20) are therefore a clean decomposition with gate-zero as the full-kill control. All three are post-build fills — no dynamics code.
- Output: one JSON row per corner (`node_diagnostics.json` next to `canary.json`), keyed by `config_tag`; resume-safe (pure function of built net + streams).

## Gates

No pass/fail on the instrument itself. It feeds the Phase-1 arm reading: whichever of shunt_G_sum / sign_balance / boundary_current correlates with max_abs_ratio names the Phase-2 jitter target (jittered-G vs jittered-gm/Isat vs balanced-sign init).