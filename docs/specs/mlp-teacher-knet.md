# Spec: mlp-teacher-knet

Scope: feature

# MLP-Teacher KNet Distillation — Feature Spec

## Purpose
Train the analog KirchhoffNet student (LocalKirchhoffStudentWrapper) via DAgger using a frozen PlainMLP as teacher instead of the generative ConditionalSplineFlow+ZIG teacher. Isolates whether KNet can distill a compact MLP policy (W48/L3, knet preprocessing) under the same ZIG-judged evaluation, enabling MLP→KNet vs Flow→KNet comparison with identical student/loop/eval.

## Non-Goals
- No change to `generative-distillation-plain-mlp.py` training logic (except emitting `teacher_config.json` for future auto-inference; fallback defaults cover the existing checkpoint).
- No MLP→MLP variant; student remains KNet.
- No change to ZIG/HybridHurdleModel, RegimeAwareLoss math, or COMMON_EVAL_SPECS failure definition.

## Teacher Contract

### Checkpoint assumed
User run: `python generative-distillation-plain-mlp.py --param-budget 5559 --plain-trunk-layers 3 --dagger-iterations 4 --epochs-per-iter 100 --common-eval-size 1000 --output outputs/dagger_output_plain_mlp_w48_preprocessing_knet --device cuda --seed 100 --input-preprocessing knet`
→ derived `W=48, L=3, activation=SiLU, use_layernorm=False, use_log_features=False (knet overrides)`, ~5479 params (`plain_param_count(48,3)` in ctle_dagger_common.py:171), file `outputs/dagger_output_plain_mlp_w48_preprocessing_knet/dagger_student_plain.pt` (state_dict).

### Loading
`ctle_dagger_common.py` exposes:
```
load_plain_mlp_teacher(ckpt_path: str|Path, device, *, trunk_width=48, trunk_layers=3, activation=SiLU, use_layernorm=False, input_preprocessing="knet") -> PlainMLP (eval, frozen)
```
- Instantiate `PlainMLP(trunk_width, trunk_layers, activation, use_layernorm, use_log_features=(input_preprocessing=="q75"))` then `load_state_dict(torch.load(ckpt, map_location=device))`, `.eval()`, `requires_grad=False`.
- If `outputs/.../teacher_config.json` exists (new runs), prefer its `trunk_width/layers/activation/use_layernorm/input_preprocessing` over CLI defaults; log which source was used.
- Caller must call `teacher.attach_scaler(scaler_p_scale, scaler_p_mean, eye_scale_j/h/w, input_preprocessing, input_log_min/max)` with the same constants `setup()` derives from df so `teacher.scale_input` matches student preprocessing (knet: `2*(log10(x)-lo)/(hi-lo)-1` clipped to [-4,4], not Q75 8-dim).

### Labeling interface
`ctle_dagger_common.py` exposes:
```
class MlpTeacherLabeler:
  def __init__(self, teacher: PlainMLP, device, batch_size=1024)
  def label_single(self, spec: np.ndarray shape (4,) [power,jitter,height,width]) -> np.ndarray shape (7,)  # physical params
  def label_batch(self, specs: np.ndarray shape (N,4)) -> np.ndarray shape (N,7)
```
- Implementation: batched `teacher.predict(torch.from_numpy(specs).to(device))` → stack `dict[name]` in `PARAM_COLS` order (`fW,current,ind,Rd,Cs,Rs,VDD`), no candidate sampling/ranking, deterministic. Chunk by `batch_size` to bound VRAM.
- `create_mlp_distillation_dataset(df, n_samples, boundary_ratio, teacher_labeler)` mirrors `create_flow_distillation_dataset` spec sampling (`sample_target_specs` + per-sample noise `power*U[0.95,1.05]`, `height/width*U[0.9,1.1]` or `U[0,2]` if ≤1, `jitter*U[0.95,1.05]` clipped [1.57,100]) → single `label_batch(noisy_specs)` → `filter_by_zig_validity(threshold=0.5)` → replace invalid rows via `empirical_fallback_label(k=3)` → return `List[dict{power,height,width,jitter,params}]` consumable by `DistillationDataset`.

## Architecture Contract (shared infra reuse)

Reuse from `ctle_dagger_common.py` unchanged: `PARAM_COLS/BOUNDS`, `SPEC_RANGES`, `HybridHurdleModel/ZIGHurdleWrapper`, `RegimeAwareLoss`/`RampedRegimeLoss` (HybridHurdle+Q75, physics/manifold/invalid), `DistillationDataset` (hard-buffer WeightedRandomSampler), `StudentEvaluator` (ZIG validity + adaptive degradation `DEGRADE_REL_THRESHOLD=0.20` on ≥`MIN_DEGRADED_DIMS=2` dims), `sample_validation_specs`, `filter_by_zig_*`, `empirical_fallback_label`, `compute_forward_errors*`, `run_diagnostics`, `setup()` df/scaler/ZIG loading, `COMMON_EVAL_SPECS` seed 24681012.

Extensions in `ctle_dagger_common.py` (side-effect free import):
- `load_plain_mlp_teacher`, `MlpTeacherLabeler`, `create_mlp_distillation_dataset` as above.
- `setup()` already computes `input_log_min/max` for knet; ensure `ctx` carries them for `attach_scaler`.

## New Script Contract: `dagger-mlp-distillation-kirchhoffnet.py`

Structure clones `dagger-nuance-distillation-kirchhoffnet.py` (~3400 lines) with these deltas:

| Area | Generative (flow) | MLP-teacher |
|------|-------------------|-------------|
| Paths | `TEACHER_DIR` (flow_scaler_X/C) | `MLP_TEACHER_CKPT` (default `outputs/dagger_output_plain_mlp_w48_preprocessing_knet/dagger_student_plain.pt`) |
| Teacher load | `FlowTeacherLabeler(TEACHER_DIR, DEVICE)` | `load_plain_mlp_teacher(MLP_TEACHER_CKPT, DEVICE, width/layers/act/LN)` + `MlpTeacherLabeler(teacher)` |
| Initial dataset | `create_flow_distillation_dataset(df, n_samples=N_EMPIRICAL_SAMPLES, n_candidates=N_CANDIDATES_INITIAL, boundary_ratio, teacher_labeler)` | `create_mlp_distillation_dataset(df, n_samples=N_EMPIRICAL_SAMPLES, boundary_ratio, teacher_labeler)` (no candidate budget) |
| DAgger labeling | interior `N_CANDIDATES_PER_SPEC=3000` + boundary `N_CANDIDATES_BOUNDARY=10000` via flow sampling | single `MlpTeacherLabeler.label_batch(failed_specs)` for both interior/boundary; keep ZIG validity gate `threshold=0.5` + k-NN fallback, unified valid/total logging |
| Config hash | `N_CANDIDATES_* + BOUNDARY_RATIO + TEACHER_DIR + df fingerprint` | `mlp_ckpt (mtime+size) + W/L/preprocessing + N_EMPIRICAL_SAMPLES + BOUNDARY_RATIO + df fingerprint`; hash prefix `initial_dataset_mlp_<hash>.pkl`; old flow caches never reused |
| CLI | `--teacher-dir`, `--n-candidates-*` | `--mlp-teacher-ckpt`, `--mlp-teacher-width` (def 48), `--mlp-teacher-layers` (def 3), `--mlp-teacher-activation {silu,gelu}` (def silu), `--mlp-teacher-use-layernorm` flag, `--mlp-teacher-batch-size` (def 1024); retain all KN_* topology flags, `--dagger-iterations/epochs-per-iter/common-eval-size/batch-size/lr/weight-decay/device/seed/output/data-dir`, `--count-params-only` preflight before teacher load |
| Logging | `[CFG] TEACHER_DIR` | `[CFG] MLP_TEACHER_CKPT + MLP_TEACHER_WIDTH/LAYERS/PREPROCESSING + dataset hash` |
| Checkpoint | `dagger_checkpoint.pt` with `dataset/train_indices/val_indices/hard_buffer/histories/rng/common_eval_specs` | same schema, plus `mlp_teacher_ckpt` field in payload for provenance |
| Eval | `StudentEvaluator` via ZIG | unchanged (ZIG remains oracle) |

Student: `LocalKirchhoffStudentWrapper` (dagger-nuance-distillation-kirchhoffnet.py:1535) with `KN_*` defaults: `KN_NUM_STAGES=4, KN_NUM_HIDDEN=14, small_world k=4 p=0.2 seed=1, edge_repeats=2, cell_library=tanh_free, leak=non-programmable, interstage=residual-relu-tanh, freeze_read=True, temporal_readout=True, boundary_fan_out={"0":[2,4],"1":[1,3],"2":[12,5],"3":[7,9]}, x_max=4.0, VCA True/rank2/core/separate_core_bus` plus BO overrides; param breakdown via `parameter_breakdown()`.

Preserve: Tee log mirroring, `_log_hyperparameters()`, `Timer`, `DistillationDataset` growth, failure-capped retry (`FAILURE_CAP_RATIO=0.40` capped to highest-error), `COMMON_EVAL_SPECS` shared eval + per-epoch `identify_failures` + `EARLYSTOP_*`/`DIVERGENCE_*` carry-forward, ` CosineAnnealingLR` with per-group scales (mapper 1.0/struct 4.0/dyn 1.0 incl. `vca_W*`), NaN/grad guards, diagnostics plots, `*_predictions.csv`.

## Plain Script Minimal Patch

`generative-distillation-plain-mlp.py:main()` after `torch.save` also `json.dump({"trunk_width":W,"trunk_layers":L,"activation":act,"use_layernorm":LN,"input_preprocessing":preproc,"param_budget":budget}, teacher_config.json)`. Load path tolerates absence (fallback to W48/L3/knet).

## Verification

- `python -m py_compile ctle_dagger_common.py generative-distillation-plain-mlp.py dagger-mlp-distillation-kirchhoffnet.py` passes (use %TEMP% cfile on SSHFS).
- `--count-params-only` prints `trainable params: N` for KNet student without loading teacher/ZIG.
- CPU smoke: `python dagger-mlp-distillation-kirchhoffnet.py --mlp-teacher-ckpt outputs/dagger_output_plain_mlp_w48_preprocessing_knet/dagger_student_plain.pt --dagger-iterations 1 --epochs-per-iter 2 --common-eval-size 100 --batch-size 64 --device cpu --output /tmp/smoke_mlp_knet` → initial dataset builds via batched MLP (no candidate loop), ZIG validity gate logs, one DAgger iteration completes, `dagger_curves.png` + `dagger_student_mlp_predictions.csv` emitted, second invocation hits `initial_dataset_mlp_<hash>.pkl` cache.
- Determinism: two `label_batch` calls on same specs produce bitwise-identical params; `DistillationDataset` hard-buffer weighting and `COMMON_EVAL_SPECS` seed match flow script for comparable failure rates.

## Risks

- Preprocessing mismatch (teacher trained knet vs student called q75) → silent accuracy drop; mitigate by forcing both `attach_scaler(..., input_preprocessing="knet", input_log_min/max)` and asserting equality.
- Checkpoint drift: if plain script re-saved with different W/L, default W48 breaks; mitigate via `teacher_config.json` auto-inference with fallback warning.
- Forgetting to invalidate flow caches → stale flow-labeled data reused for MLP run; mitigated by hash including `mlp_ckpt` identity and new `mlp` prefix.