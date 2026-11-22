# Spec: plain-mlp-load

Scope: feature

# Feature: Plain MLP self-contained load (no guessing)

## User story
As a downstream consumer (dagger-mlp-distillation, fixed-mlp, BO, notebook), I can load a trained PlainMLP teacher from only its checkpoint path: `load_plain_mlp_teacher("outputs/dagger_output_plain/dagger_student_plain.pt")` reconstructs architecture + preprocessing without CLI flags.

## Current pain
- `teacher_config.json` is 7 keys, missing scaler constants and `input_log_min/max`.
- `load_plain_mlp_teacher` needs caller-supplied `trunk_width/layers/activation/use_layernorm/input_preprocessing` or it infers from `state_dict` shapes and still needs manual `attach_scaler()`.
- `run_dagger_training` saves raw `state_dict`; no single file that is sufficient.

## Feature scope (this plan)
1. **Writer** (`generative-distillation-plain-mlp.py:main`): after `run_dagger_training` returns `result["output_dir"]`, build v2 config dict and atomically write `teacher_config.json` there. Write once, post-DAgger.
2. **Reader** (`ctle_dagger_common.py:load_plain_mlp_teacher`): if co-located `teacher_config.json` has `schema_version>=2`, use it to construct `PlainMLP` and auto `attach_scaler`. Also accept `torch.load` dict of form `{"state_dict": ..., "config": {...}}`.
3. No rename; `.pt` stays raw `state_dict`; sidecar is authoritative.

## Config payload (v2)
```json
{
  "schema_version": 2,
  "trunk_width": 48,
  "trunk_layers": 3,
  "activation": "silu",
  "use_layernorm": false,
  "use_log_features": false,
  "input_preprocessing": "knet",
  "param_budget": 5559,
  "plain_param_count": 5479,
  "scaler": {"scaler_p_scale": 0.5, "scaler_p_mean": -2.0, "eye_scale_j": 1.8, "eye_scale_h": 1.2, "eye_scale_w": 1.4},
  "input_log_min": [-4, 0, -0.3, 0],
  "input_log_max": [-1, 3, 2.3, 2.7],
  "param_log_bounds": {"fW": [-6,1], ...},
  "checkpoint": "dagger_student_plain.pt",
  "meta": {"seed": 0, "trained_at": "2026-09-02T...", "git_commit": "abc123|null"}
}
```
- `input_log_min/max` are `ctx["input_log_min/max"]` (numpy → list), null when not applicable.
- `scaler` from `ctx["scaler_y_p"]` (mean_/scale_[0]) and `ctx["eye_scale_*"]`.
- `plain_param_count` from `plain_param_count(W,L)`.
- Preserve all v1 keys; add new keys.

## Loader contract
- `load_plain_mlp_teacher(ckpt_path, device=None, **kwargs)`:
  1. Resolve `cfg_path = dirname(ckpt)/teacher_config.json`. If exists and `schema_version>=2`, ignore kwargs for arch/scaler, use JSON.
  2. Also if `torch.load(ckpt)` returns dict with `"config"` key containing v2-like dict, prefer embedded config over sidecar.
  3. Construct `PlainMLP(trunk_width, trunk_layers, activation=SiLU|GELU, use_layernorm, use_log_features)`.
  4. `model.attach_scaler(scaler_p_scale, scaler_p_mean, eye_scale_j/h/w, input_preprocessing, input_log_min/max)`.
  5. Validate `requested vs actual` shape inference; warn but use checkpoint shape (existing `971` behavior).
  6. `load_state_dict` (handle `{"state_dict": ...}` wrapped), freeze, eval, return.
- Legacy: if no v2 config, fall back to existing kwarg/inference path.

## Acceptance criteria
- `teacher_config.json` written after DAgger, valid JSON, contains all v2 keys, `plain_param_count == plain_param_count(W,L)`.
- `load_plain_mlp_teacher(ckpt)` with empty kwargs loads and `model.predict(torch.randn(2,4))` succeeds without `attach_scaler` call.
- Round-trip: `pred_before == pred_after` within 1e-6 (same device, eval mode).
- Old checkpoint without JSON still loads via `_plain_mlp_arch_from_state` inference.
- `py_compile` passes; 1-epoch smoke test passes.

## Out of scope
- Changing `run_dagger_training` to write bundle `.bundle.pt` (optional future).
- Pre-DAgger provisional write.
- Modifying `PARAM_LOG_BOUNDS` values.