# Spec: core-refresh-semantics

Scope: feature (reusable across NARMA and static-task fabric experiments)

## Core refresh interval

`core_refresh_interval = k` controls how often the frozen core nonlinear current
`i_edge_const` is recomputed from the evolving state `x` (counted in Heun steps,
not individual `rhs()` calls):

- `k = 8`: with `num_steps=8`/sample this is exactly 1 eval/sample = legacy
  `freeze_read=True` behavior in `_forward_heun_sequence` (baseline, cheapest).
- `k = 4`: recompute every 4 Heun steps = 2 evals/sample (cheap recurrent).
- `k = 2`: recompute every 2 Heun steps = 4 evals/sample (denser recurrent).
- `k = 1`: recompute every Heun step = near-dynamic at `num_steps`
  evals/sample (one recompute per step, reused for both `rhs()` calls) —
  approximately, not byte-identically, legacy `freeze_read=False`, which
  costs `2*num_steps` evals/sample (fresh `cell_lib` call in every `rhs()`).
  Deferred unless `k ∈ {8,4,2}` fail.

Counting is in **Heun steps** (each step = 2 `rhs()` calls: `k1=rhs(x)`,
`k2=rhs(x+dt*k1)`), not individual `rhs()` calls. Per-rhs refresh would double
`cell_lib` evals for marginal gain since `x_pred ≈ x + O(dt)`.

When `k >= num_steps` the core current is recomputed once at step 0 and never
again within the window (equivalent to legacy `freeze_read=True` at the
`num_steps` value chosen).

## Independence guarantees

- `i_boundary_const` / `i_readout_const` keep following their own
  `freeze_boundary` / `freeze_temporal_read` flags; refresh `k` applies to the
  core family only.
- The `FreeTanhLibrary` resistive shunt (`resistive_current`) always uses
  evolving voltages and bypasses freezing, at every `k`.
- Core VCA gate (`vca_gate_core`), when enabled, is folded into `i_edge_const`
  at each recompute; boundary/readout VCA gates stay per-step dynamic.
- Default when unset preserves legacy behavior byte-for-byte
  (`k>=num_steps` ≡ legacy `freeze_read=True` frozen; `k=1` ≈ legacy
  `freeze_read=False` numerically at half the eval cost, see above).

## Equivalence checks

- `k >= num_steps` loss trajectory must match legacy `freeze_read=True`
  seed-for-seed (same init, same chunk order).
- `k=1` loss trajectory must track legacy `freeze_read=False` closely
  (near-dynamic; exact match not required since the frozen-within-step
  current skips the `x_pred` re-evaluation).
- Remainder handling: recompute at step 0 and every subsequent `k` steps; if
  `num_steps % k != 0` the final tail steps use the last computed `i_edge_const`.
  Warn on `k=1` with `dt > 0.2` (stiffness risk for `tanh_free` `gm≈10`).

## NARMA-10 grid (reference instantiation)

- `order=10`, `tanh_free`, `t_span=1.0`, `num_steps=8` (`dt=0.125`, matches
  the static-task `num_steps/t_span ≈ 10` discipline),
  `n-streams=4`, `train-samples=2500`, `tbptt-chunk=25`,
  `--compile --fabric-only`, `seeds={0,1,2}`, `epochs=80`,
  `val_every=5`, early-stop patience 15.
- Refresh ladder `{8, 4, 2}`; deferred fallback `k=1` only if all three fail
  the 0.15–0.29 NRMSE band.
- Log per condition: refresh `k`, `cell_lib` evals/sample, wall-clock,
  `NRMSE`/`R²`/`MC`.
- Success band: fabric `NRMSE ∈ [0.15, 0.29]` (standard-RNN parity) to keep
  pushing signal-path; `k=2 ≈ k=1` implies cheap recurrence wins.
