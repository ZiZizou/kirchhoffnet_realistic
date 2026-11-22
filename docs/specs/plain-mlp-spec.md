# Spec: plain-mlp-spec

Scope: feature

# Plain MLP Control — Feature Spec

## Purpose
Provide a clean single-head MLP control for the CTLE DAgger pipeline to isolate whether the MoE advantage comes from routing/specialization vs trunk/training setup. Separate file `generative-distillation-plain-mlp.py` shares all infra via `ctle_dagger_common.py` so MoE and plain runs are exactly comparable.

## Non-Goals
- No change to `mlp_bayes_opt.py` (out-of-BO for now).
- No `--moe-num-experts 1` workaround (leaves gate/regime loss active, cross-entropy invalid with 3-class labels).

## Architecture Contract

### Shared infra (from `ctle_dagger_common.py`)
- `SPEC_RANGES`, `PARAM_LOG_BOUNDS`, `PARAM_COLS`, `SPEC_INPUT_COLS`
- `HybridHurdleModel`, `ZIGHurdleWrapper`, `ConditionalSplineFlow` family, `FlowInference`, `FlowTeacherLabeler`
- Scalers: `zig_scaler_X/Y_p`, `flow_scaler_X/C`, `eye_scale_*`, `PER_TARGET_HURDLE`
- `RegimeAwareLoss` (HybridHurdle + Q75, manifold/physics/invalid), `RampedRegimeLoss`
- `DistillationDataset`, `StudentEvaluator`, `TeacherLabeler`, `sample_validation_specs`, `filter_by_zig_*`, `empirical_fallback_label`, `compute_forward_errors*`, diagnostics, plotting

Import must be side-effect free: heavy loads/dataset building/DAgger loop behind `if __name__ == "__main__"` in common; plain and nuance files both `from ctle_dagger_common import *`.

### PlainMLP class (new, in plain file only)
```
PlainMLP(trunk_width=45, trunk_layers=3, activation=SiLU, use_layernorm=False, use_log_features=True)
  input:  [power, jitter, height, width] raw
  scale_input: Q75 StandardScaler for power + log10/eye_scale for others → 4 dims;
               if use_log_features: cat 4 raw log dims → 8 dims (mirrors RegimeAwareMoE)
  trunk:  Sequential[ Linear(8→W)+Act (+LN if enabled) ] × L
  head:   Linear(W→7)
  buffers: log_lo/log_hi (7) from PARAM_LOG_BOUNDS, not trainable; scaler constants for scale_input
  forward(x) -> logits (B,7)  # no return_regime_loss, no gate, no regime_classifier
  predict(x) -> dict name→physical (sigmoid+bounds)
  get_bounded_output(x) -> (logits, bounded_log, physical)
```
Defaults match MoE trunk (SiLU, no LN, log_features True) so `W=45 L=3` is the informative first comparison.

### Param helpers
- `plain_param_count(W,L) = 9*W + (L-1)*(W*W+W) + (7*W+7)`  # trunk+head, no routing
- `moe_param_count` already in `mlp_bayes_opt.py:152` for reference: `9W+(L-1)(W²+W)+16E+E(7W+7)`
- `derive_plain_width(budget, L)` → max W s.t. plain_param_count ≤ budget (solve + step down)
- Reference numbers: MoE 45×3×3=5559, plain same-width 45=4867, plain 48=5479 (≈matched ≤), 49=5691 (≈matched ≥)

## CLI Contract (plain file)

Keep shared flags: `--dagger-iterations`, `--epochs-per-iter`, `--common-eval-size`, `--earlystop-eval-every`, `--lr`, `--weight-decay`, `--batch-size`, `--device`, `--output`, `--data-dir`, `--teacher-dir`, `--seed`.

Add plain trunk overrides (defaults = best MoE shape):
- `--plain-trunk-width` (default 45), `--plain-trunk-layers` (default 3)
- `--plain-activation` {silu, gelu} default silu, `--plain-use-layernorm` flag default False
- `--plain-use-log-features` flag default True (when False input_dim=4)
- `--param-budget` int optional: when set, auto-derive W via `derive_plain_width(budget, L)` and override explicit width; log `derived width N for budget B (L=...)`

Remove `--moe-num-experts` (plain has 1 implicit head). Keep BO override parsing pattern (parse early, re-apply after defaults).

## Training Loop Delta

- No `get_regime_label`, no `gate`/`regime_classifier`, no `F.cross_entropy`.
- Inner epoch: `logits = student(specs)` → `base_dict = criterion(student, targets, params, logits)` → `ramped = ramped_criterion(..., logits, base_dict)` → `loss = ramped['total']` (no `+0.1*regime`).
- Remove `epoch_losses['regime']` and regime logging; keep imit/spec/phys/invalid/manifold.
- Optimizer sees only trunk+head+buffers.

## Output / Comparability

- `MODEL_NAME='dagger_student_plain'`, default `OUTPUT_DIR` → `dagger_output_plain` (or CLI `--output`).
- Reuse `initial_dataset_cache` (hash on N_EMPIRICAL_SAMPLES etc., not arch) and `COMMON_EVAL_SPECS` seed 24681012 so failure rates are apple-to-apple.
- Logs must emit `Student model: X params (plain W×L …)` for BO-style parsing.

## Verification

- `python -m py_compile ctle_dagger_common.py generative-distillation-plain-mlp.py` passes (via %TEMP% cfile on SSHFS if needed).
- Smoke: `--dagger-iterations 1 --epochs-per-iter 2 --common-eval-size 200 --device cpu` → no regime loss, param count matches helper, `Test failure rate:` and `dagger_student_plain_predictions.csv` in isolated dir.
- Two-command comparison:
  `python generative-distillation-plain-mlp.py --plain-trunk-width 45 --plain-trunk-layers 3 ...`
  `python generative-distillation-plain-mlp.py --param-budget 5559 --plain-trunk-layers 3 ...` (→ W=48)
  Compare both against MoE `width=45 layers=3 experts=3` failure rate to fill table: same-width vs same-budget.

## Risks

- Extracting common must not break nuance file's current behavior; guard with import-only refactor first, smoke both files.
- `scale_input` mismatch (8 vs 4) would confound result — enforce plain mirrors MoE's 8-dim default.