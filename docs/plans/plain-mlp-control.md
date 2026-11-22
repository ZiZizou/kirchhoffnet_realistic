---
plan name: plain-mlp-control
plan description: Plain MLP control via shared import
plan status: active
---

## Idea
Create a plain single-head MLP control for the CTLE DAgger experiment that shares all infra with the existing MoE pipeline via a shared import module, instead of duplicating ~3200 lines.

Current `generative-distillation-improved-dagger-nuance-mlp.py` always instantiates `RegimeAwareMoE` (shared trunk SiLU 44×3, gate 8→3, regime_classifier 8→3, 3×Linear(W→7) experts, 8-dim Q75+log input, regime cross-entropy 0.1×) and is driven by `mlp_bayes_opt.py` sampling 2-4 experts. Using `--moe-num-experts 1` is invalid because `forward(return_regime_loss=True)` still computes 3-class regime labels and leaves gate/regime loss active.

The control must be a separate short file (e.g. `generative-distillation-plain-mlp.py`) that preserves everything except MoE: same 8-feature preprocessing/scale_input, same trunk width/layers/activation, single `Linear(W→7)` head, same bounded 7-param sigmoid transform, same `RegimeAwareLoss`, `StudentEvaluator`, `DistillationDataset`, teacher labeling, DAgger schedule/cache/validation/evaluation. Best first comparison uses the best MoE shape width=45 layers=3 (MoE 5559 params vs plain 4867 same-width; param-matched plain 48→5479 / 49→5691). The file must support auto-deriving W from a CLI `--param-budget` flag (solve plain param formula), allow trunk/activation/layernorm overrides matching MoE defaults (SiLU, no LN, use_log_features=True), write to isolated output dir, and stay out of `mlp_bayes_opt.py` (out-of-BO for now).

## Implementation
- Extract shared infra into `ctle_dagger_common.py`: identify pure definitions vs side-effecting top-level code in `generative-distillation-improved-dagger-nuance-mlp.py` (imports, SPEC_RANGES/PARAM_LOG_BOUNDS, ZIG/spline/flow classes, scaler loads, RegimeAwareLoss, DistillationDataset, StudentEvaluator, sampling/labeling/helpers, DAgger utils, constants). Move them behind `if __name__=="__main__"` guard so import is side-effect free; keep original file importing from common for single source (or leave original untouched and have common be a copy if safer — decide during impl).
- Create `generative-distillation-plain-mlp.py` (short name) that imports everything from `ctle_dagger_common.py` and defines only `PlainMLP`: same `scale_input` (4 Q75 + 4 log →8 dims), trunk `L×Linear(W→W)+SiLU` (no LN by default), one `Linear(W→7)` head, `log_lo/hi` buffers, `forward(x)->logits`, `predict`/`get_bounded_output`. No gate, no regime_classifier, no `get_regime_label`, no `return_regime_loss`. Add `plain_param_count(W,L)=9W+(L-1)(W²+W)+(7W+7)` and `derive_plain_width(budget,L)` solver mirroring `mlp_bayes_opt.py:moe_param_count` logic.
- Wire CLI for plain file: keep shared DAgger flags (`--dagger-iterations`, `--epochs-per-iter`, `--common-eval-size`, `--earlystop-eval-every`, `--lr`, `--weight-decay`, `--batch-size`, `--device`, `--output`, `--data-dir`, `--teacher-dir`, `--seed`) and add plain trunk overrides (`--plain-trunk-width`, `--plain-trunk-layers`, `--plain-activation` SiLU/GELU, `--plain-use-layernorm` flag, `--plain-use-log-features` flag) defaulting to best MoE shape width=45 layers=3 SiLU no-LN log_features=True; add `--param-budget` that when set auto-computes width via solver (CLI wins over explicit width) and logs derived width vs budget.
- Strip MoE auxiliary objective from training loop: in plain file's copy of the DAgger inner loop/`train_epoch`, remove `logits, regime_loss = student(..., return_regime_loss=True)` + `+0.1*regime_loss`; loss is only `ramped_criterion` (RampedRegimeLoss around RegimeAwareLoss). Remove `epoch_losses['regime']`/logging, keep warmup ramp for ZIG/physics/manifold. Ensure optimizer param groups cover only trunk+head.
- Wire DAgger loop/cache/eval to PlainMLP: instantiate `student = PlainMLP(trunk_width, trunk_layers, activation, use_layernorm, use_log_features)`, set `MODEL_NAME='dagger_student_plain'`, default `OUTPUT_DIR` suffix `dagger_output_plain` (or `OUTPUT_DIR` CLI), reuse same `initial_dataset_cache` hash and `COMMON_EVAL_SPECS` seed 24681012 so failure rates are directly comparable. Keep `filter_by_zig_*`, `empirical_fallback_label`, tiered teacher budget, carry-forward guardrail unchanged.
- Verify: `python -m py_compile` both files, one-epoch CPU smoke (`--dagger-iterations 1 --epochs-per-iter 2 --common-eval-size 200 --device cpu`) check log shows `Student model: X params (plain W×L)` matches helper, no regime loss, `Test failure rate` emitted, predictions CSV written to isolated dir; run same-width (45) and budget-matched (48) commands and record comparison table row for MoE 5559 vs plain 4867/5479.

## Required Specs
<!-- SPECS_START -->
- lstm-baseline-complexity
- plain-mlp-spec
<!-- SPECS_END -->