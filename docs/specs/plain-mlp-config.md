# Spec: plain-mlp-config

Scope: repo

# Plain MLP Config Save/Load Contract (repo scope)

## Purpose
Make `generative-distillation-plain-mlp.py` checkpoints loadable without guessing architecture or scaler settings. Single sidecar `teacher_config.json` (HuggingFace-style `state_dict + config.json`) is authoritative.

## Current state
- `generative-distillation-plain-mlp.py:134-149` writes 7-key JSON post-DAgger (trunk_width/layers, activation, use_layernorm, use_log_features, input_preprocessing, param_budget).
- `ctle_dagger_common.py:893-997` infers from `state_dict` shapes (`_plain_mlp_arch_from_state`) and requires external `attach_scaler()` with `scaler_p_*`, `eye_scale_*`, `input_log_min/max` from `setup():2424-2445`.

## Target contract
**File:** `<output_dir>/teacher_config.json` (keep name, co-located with `dagger_student_plain.pt` as `load_plain_mlp_teacher:944` expects). Write once, after `run_dagger_training` returns.

**Schema v2 (`schema_version: 2`):**
- `trunk_width`, `trunk_layers`, `activation` (`silu|gelu`), `use_layernorm` (bool), `use_log_features` (bool), `input_preprocessing` (`knet|q75`), `param_budget` (int|null), `plain_param_count` (int)
- `scaler`: `{scaler_p_scale, scaler_p_mean, eye_scale_j, eye_scale_h, eye_scale_w}` (floats from `ctx['scaler_y_p']`, `ctx['eye_scale_*']`)
- `input_log_min`, `input_log_max`: list[4] float or null (from `ctx['input_log_min/max']`; required when `input_preprocessing==knet`)
- `param_log_bounds`: dict `{fW:[lo,hi], ...}` (from `PARAM_LOG_BOUNDS`)
- `checkpoint`: `"dagger_student_plain.pt"` (basename)
- `meta`: `{seed, trained_at (ISO), git_commit|null}`
- Keep all v1 keys for backward compat.

**.pt artifact:** stays raw `state_dict` (`torch.save(state_dict)`). Loader also accepts `{'state_dict':..., 'config':...}` dict if ever produced; check `isinstance(state,dict) and 'state_dict' in state:964`.

## Loader behavior (`ctle_dagger_common.py:925 load_plain_mlp_teacher`)
- If sidecar v2 present, override kwargs, construct `PlainMLP(...)`, auto `attach_scaler(**scaler, input_preprocessing, input_log_min/max)` before return. No caller guessing.
- Validate `requested vs actual` shape mismatch still warns (`971`) but auto-uses checkpoint shape.
- Fallback: inference via `_plain_mlp_arch_from_state` for pre-v2 runs (input_dim 4|8, LN detection via 1-D weights).
- Embedded dict `config` takes precedence over sidecar if both present.

## Non-goals
- No provisional pre-DAgger write.
- No change to `run_dagger_training` internal save besides sidecar (bundle `.bundle.pt` optional, not required for v2).
- No rename of `teacher_config.json`.

## Verification
- `python -m py_compile` both files.
- Smoke: train 1-epoch stub, `load_plain_mlp_teacher(ckpt, device='cpu')` with empty kwargs reproduces `model.predict` within 1e-6.
- Old ckpt without JSON still loads via inference.