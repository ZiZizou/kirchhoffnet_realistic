---
plan name: audit-plain-ctl
plan description: Audit and fix CTLE plain control bugs
plan status: active
---

## Idea
Audit `ctle_dagger_common.py` (130 KB shared infra extracted from `generative-distillation-improved-dagger-nuance-mlp.py` 3232-line self-contained script) and new `generative-distillation-plain-mlp.py` (5.8 KB control) for divergences, formula errors, and fragile active-context patterns before running head-to-head DAgger comparisons.

Scope:
- Compare common vs nuance line-by-line for behavioral parity (teacher blocks, scaling, loss, dataset, evaluator, DAgger schedule, early-stop, failure handling).
- Verify PlainMLP vs RegimeAwareMoE trunk equivalence and param helpers vs spec table (MoE 45x3x3=5559, Plain 45x3=4867, 48x3=5479, 49x3=5691) and `--param-budget` auto-derive.
- Check boring infra: logging/Timer, HPARAMS_DEFAULTS defaults, DEVICE/seed handling, CSV loading, scaler attach, output isolation (`dagger_output_plain`), BO non-interference.
- Classify findings by severity (P0 training-correctness, P1 param-count/reporting, P2 robustness/ergonomics) and propose minimal fixes that keep `generative-distillation-improved-dagger-nuance-mlp.py` untouched for mlp_bayes_opt.py stability.

## Implementation
- Re-read all three files with offset to avoid truncation and diff sections (HybridHurdleModel, ZIGHurdleWrapper, spline flow, FlowInference, FlowTeacherLabeler, RegimeAwareMoE, PlainMLP, BoundedMLP legacy stub, RegimeAwareLoss/RampedRegimeLoss, DistillationDataset, StudentEvaluator, filters, sampling helpers, setup, run_dagger_training, plain CLI/build_student/main). Capture line numbers for each finding.
- Fix P0 `derive_plain_width` formula: `b = 9+7` is wrong for L>1. Correct to `b = L+15` (since plain = (L-1)W^2+(L+15)W+7) or solve generally via quadratic in `plain_param_count`. Verify against closed-form: 5559->48, 4867->45, small budgets return -1, and smoke for L=2 vs L=3. Add unit test.
- Fix param-count logging mismatch: plain `main` and `run_dagger_training` return count `requires_grad` filtering vs nuance historical `sum(p.numel())` (includes log_lo/hi buffers, +14). Unify to one definition (document trainable vs total) and align spec table to same definition; update smoke to assert both.
- Harden active-context (`_active_ctx`) fragility: `filter_by_zig_*`, `compute_forward_errors`, `StudentEvaluator.identify_failures`, `log_failure_breakdown` now read VALIDITY_THRESHOLD etc from context and crash if called before `setup()`/without fallback. Add safe defaults + explicit `setup()` guard or pass thresholds explicitly; remove duplicate `spec_to_log` vs helper mismatch.
- Audit scaler attachment: RegimeAwareMoE/PlainMLP now require `attach_scaler()` before forward (original nuance read flow_scaler_C at __init__). Ensure `setup()`+`run_dagger_training` always attach, update BoundedMLP stub doc to state it is intentionally not usable via common (or restore faithful impl), and prevent silent forward-without-attach.
- Tighten setup/CLI parity: ensure HPARAMS_DEFAULTS vs plain defaults stay aligned (nuance MOE_TRUNK_WIDTH 44 vs plain 45 intentional but comment says W=44), DEVICE auto logic mirrors nuance, seed handling matches, and `--param-budget` wins over `--plain-trunk-width` is logged. Add help text + error for budget too small.
- Produce audit report with severity, repro steps, and before/after code snippets; attach as spec `audit-plain-ctl-findings` and link to plan. Gate DAgger head-to-head runs until P0/P1 fixes land.

## Required Specs
<!-- SPECS_START -->
- lstm-baseline-complexity
<!-- SPECS_END -->