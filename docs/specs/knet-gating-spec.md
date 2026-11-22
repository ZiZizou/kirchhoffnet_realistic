# Spec: knet-gating-spec

Scope: feature

# Gated-memory feature spec (plan knet-gated-memory)

## Scope
NARMA-10 recurrent memory/nonlinearity fix for KirchhoffNet. Covers the
Phase-3 spike (dynamic leak + hidden-conditioned VCA), the E0 extension
axes, and the native-linear rung-1 control. Phases 4/5 are gated on the
spike verdict and NOT in scope for implementation under this spec.

## CLI contracts
- `--dynamic-leak`: enables `leak_eff_j = softplus(a_j*x_j + b_j*u + c_j)`
  per node. Disabled = legacy `leak_floor + softplus(raw_leak)` path,
  bit-identical forward.
- `--vca-use-hidden`: VCA gate becomes
  `gate_e = 2*sigmoid(b_e + ([u;x]^T W) v_e)` with
  `W: (in_dim + num_nodes, rank)`. Disabled = legacy input-only VCA,
  bit-identical forward.
- E0 extension: `gm_raw x isat_raw` axes at fixed `leak_constant=0.1`,
  implemented by extending `narma_advisor_probes.e0_sweep` (no new script).
- Rung-1: native-linear control via small `gm`/`Isat` init + far rails +
  ESN-matched leak/input-swing; ESN winner recipe read from
  `run_baselines` logs per seed (never hardcoded).

## Epoch-0 equality invariants (must hold, audited)
- `dyn_leak_a=0, dyn_leak_b=0, dyn_leak_c=raw_leak` at init; spike-off and
  spike-on-at-init forwards match to float tolerance on a fixed seed.
- Hidden block of extended `vca_W` is all-zero at init; gate == 1.0
  everywhere at epoch 0; `vca_W.grad` nonzero after one backward.
- `cell_library.py` init defaults are NEVER edited; all overrides live in
  probe/build-time hooks (`apply_gain_override` style).

## Metric gates
- Operating point target: `PR > 5`, `MC > 4`, `rails < 5%`, `ridge < 0.6`
  (reuse existing E0 PASS/FAIL tags in `advisor-probe-suite`).
- Rung-1 target: MC ~= 14 (ESN-25 parity) with linear ridge readout.
- Spike go/no-go: 20-epoch NARMA-10 train from best-known point with both
  flags on; compare MC/NRMSE vs baseline.

## Execution discipline
- Thin CPU smokes locally; full grids on Alliance GPU via gitignored Slurm
  batch scripts (established pattern). No heavy runs on local CPU.
- NARMA protocol pinned: 4x2500 train, washout 200, test 1000, AdamW
  lr=1e-3, truncated BPTT chunk 25, t_span=1.0, num_steps=8.