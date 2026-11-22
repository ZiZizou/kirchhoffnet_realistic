---
plan name: plain-mlp-config
plan description: Enrich plain MLP config save
plan status: done
---

## Idea
Modify generative-distillation-plain-mlp.py so the trained PlainMLP saves a self-contained teacher_config.json (schema v2) that captures full architecture (W/L, activation, layernorm, log_features, preprocessing, param_budget, param_count) plus scaler constants (scaler_p_scale/mean, eye_scale_j/h/w), input_log_min/max, param_log_bounds, checkpoint name and meta, written once after DAgger. Update ctle_dagger_common.load_plain_mlp_teacher to auto-attach scaler from that JSON (and accept embedded dict checkpoints) so downstream loaders need only ckpt_path, with shape-inference fallback for legacy runs. Industry standard is sidecar config.json + raw state_dict (HF-style); keep .pt as raw state_dict, no rename.

## Implementation
- Add helper build_plain_teacher_config(student, ctx, args) in generative-distillation-plain-mlp.py (or shared) gathering W/L/activation/LN/log_features/preprocessing/param_budget, plain_param_count(), ctx scaler_y_p/eye_scale_*, input_log_min/max, PARAM_LOG_BOUNDS, meta (seed/git/timestamp)
- Edit generative-distillation-plain-mlp.py:main() to build config after run_dagger_training returns and atomically write <output>/teacher_config.json with schema_version 2 (retain v1 keys, add scaler/input_log_min/max/param_log_bounds/checkpoint/meta)
- Extend ctle_dagger_common.py:load_plain_mlp_teacher to read v2 sidecar (and embedded {'config':...} dict), construct PlainMLP and auto attach_scaler from JSON, removing caller guessing; keep _plain_mlp_arch_from_state fallback for pre-v2
- Add dict-check for torch.load returning {'state_dict':..., 'config':...} so future embedded checkpoints also auto-configure; preserve raw OrderedDict path and requested!=actual warning
- Verify with py_compile and smoke test: 1-epoch DAgger stub, load_plain_mlp_teacher(ckpt, device=cpu) with empty kwargs reproduces predict within 1e-6, and old ckpt without JSON still loads via inference

## Required Specs
<!-- SPECS_START -->
- plain-mlp-config
- lstm-baseline-complexity
- plain-mlp-load
<!-- SPECS_END -->