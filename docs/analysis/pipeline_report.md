# HR-KNet CTLE pipeline — training / task / verification analysis

**Scope:** read-only analysis of the *training, task, and verification* side of
`C:\Users\Atharva\Documents\kirchhoffnet_realistic` for the paper
*HR-KNet: A hardware-realistic analog accelerator and design framework*.
Every claim below is cited as `file:line`. Where something does not exist in the
repo it is explicitly marked **not found** rather than guessed.

---

## 0. Executive summary (TL;DR)

| Question | Answer |
|---|---|
| CTLE task | 4 measured specs → 7 CTLE device knobs, log-bounded; specs sampled from a Cadence-generated GF22 CTLE dataset (83,344 rows) |
| Where is the CTLE circuit? | **Not in this repo.** It is an external GF22-FDX Maestro testbench `ECE740 / tb_channel_CTLE_ML_{tb}` driven by OCN templates that live at `/home/annaik/Documents/train_ctle/ocn_template_ideal_bob` |
| "Cadence-verified" refers to | **The CTLE circuit**, not the KNet. The KNet is a PyTorch behavioural model; only its 7 output knob values are pushed into Cadence/Spectre |
| Verilog-A / netlists / `.scs` / `.va` | **not found anywhere in the repo** (repo-wide search incl. `output*/`, `outputs*/`, `tmp/`, excluding venvs) |
| Paper's final HR-KNet | Phase-A KNet BO trial_0011 — 5,649 params, `stages=7, hidden=11, k=2, rank=3, shared-x2 readout, GLN, t_span=4.660583, gm_max=4.241970, isat_max=28.68855`. Its checkpoint `phase_a_best.pt` is **not in this checkout** |
| Retraining with SI units | Requires replacing `config.py:302-318` (PHYS) and `config.py:270-273` (cell gm/Isat bounds) — both are explicitly documented as *not* SI-calibrated |
| Compute for one retrain | Phase-A KNet ≈ 19–21 s/epoch → ~1.6–1.8 h/trial; 30 trials ≈ 50–55 H100-hours (SLURM wall request 24 h, `run_phase_a_knet_bo.sh:6`) |

---

## 1. The CTLE problem definition

### 1.1 Inputs — 4 specifications

Order and identity (this ordering is load-bearing; it differs between files in
one place, see §1.6):

* `ctle_dagger_common.py:85` — `SPEC_INPUT_COLS = ['power', 'stage_2_jitter', 'stage_2_eye_max_height', 'stage_2_eye_max_width']`
* `ctle_dagger_common.py:78-83` — empirical ranges:
  `power (0.0012, 0.012)`, `stage_2_eye_max_height (0.0, 88.4)`,
  `stage_2_eye_max_width (0.0, 98.5)`, `stage_2_jitter (1.57, 100.0)`
  (power in W, jitter/height/width in % of UI)
* `train_ctle.py:105-109` — same ranges, different column order
  (`power, jitter, height, width`), power sampled log-uniformly
  (`train_ctle.py:481`).
* Specs are **measured at the stage-2 output** of the 2-stage CTLE — see the
  paper figure caption `tmp/analysis/paper_raw.txt:575-576`
  ("The 4 input specs (power, jitter, eye height/width) are measured at the
  stage-2 output").

**Normalisation.** The "knet" preprocessing is:

```
logs   = log10(specs)
scaled = 2*(logs - lo)/(hi - lo) - 1
scaled = clamp(scaled, -input_rail, +input_rail)   # input_rail = 4.0
```

* `fixed-mlp-distillation-kirchhoffnet.py:263-268` (`KirchhoffStudent.scale_input`)
* `ctle_dagger_common.py:3551-3557` — `lo/hi` are the dataset log-min/max with a
  **5 % pad** of the per-dim log range
* `fixed-mlp-distillation-kirchhoffnet.py:214-219` (`input_log_bounds`), pad
  fraction from `--kn-input-log-pad-frac` (default 0.05,
  `fixed-mlp-distillation-kirchhoffnet.py:131`)
* The bounds are stored in the canonical `.npz` so every BO trial sees byte-identical
  preprocessing: `ctle_dagger_common.py:2032-2033`, read back at
  `ctle_dagger_common.py:2927-2930`.
* A legacy "q75" preprocessing (log → StandardScaler/Q75 → 8 features) also exists
  (`ctle_dagger_common.py:3566-3569`, `ctle_dagger_common.py:1478-1499`); the
  fixed-distillation harness *rejects* anything but `knet`
  (`fixed-mlp-distillation-kirchhoffnet.py:484-488`).

**Boundary / interior split of the spec space** (used for the stratified reporting
and for over-weighting hard cases):

* `ctle_dagger_common.py:99-110` — `BOUNDARY_ABS_TOLERANCES` =
  height < 5.0, width < 5.0, jitter > 90.0, power < 0.003 (plus relaxed variants)
* `ctle_dagger_common.py:2483-2487` (`is_boundary_spec`) and
  `ctle_dagger_common.py:2280-2283` (in-evaluator mask)
* `ctle_dagger_common.py:2339-2364` (`sample_validation_specs`) samples
  `boundary_ratio` of specs from the boundary pool (default 0.5, `ctle_dagger_common.py:117`).

### 1.2 Outputs — 7 CTLE knobs

* `ctle_dagger_common.py:87` — `PARAM_COLS = ['fW', 'current', 'ind', 'Rd', 'Cs', 'Rs', 'VDD']`
  (differential-pair width factor, tail bias current, peaking inductor,
  load resistor, degeneration capacitor, degeneration resistor, supply)
* `ctle_dagger_common.py:89-97` — `PARAM_LOG_BOUNDS` (physical, log10):
  | knob | min | max |
  |---|---|---|
  | `fW` | 1e-6 | 10.0 |
  | `current` | 5e-4 | 2.5 (A) |
  | `ind` | 1e-12 | 3.0 (H) |
  | `Rd` | 10 | 1500 (Ω) |
  | `Cs` | 1e-15 | 1e-9 (F) |
  | `Rs` | 10 | 1500 (Ω) |
  | `VDD` | 0.6 | 1.2 (V) |
* `ctle_dagger_common.py:229-236` (`params_from_logits`) and `:239-247`
  (`params_to_log_logits`) — the sigmoid-in-log-space map.
* Duplicated verbatim in `spice_failure_check.py:71-79` and `train_ctle.py:115-122`.
  ⚠ A **third, inconsistent** copy lives in `ctle_kirchhoff_bo.py:41-43`
  (`fW [-12,-8]`, `current [-7,-2]`, `ind [-12,-7]`, `Rd [1,6]`, `Cs [-15,-9]`,
  `Rs [0,6]`, `VDD [-1,1]` — all different). See §6.

### 1.3 Dataset source

**Primary CTLE dataset (Cadence-generated, derived):**

* `augmented-cvae-ctle/ctle_ml_dataset.csv` — **83,344 data rows + header**
  (83,345 lines). Columns: `power, stage_1/2_28G_gain_difference,
  stage_1/2_eye_max_height, stage_1/2_eye_max_width, stage_1/2_jitter,
  fW, current, ind, Rd, Cs, Rs, VDD`.
  This matches the paper's "83,344 simulated two-stage-CTLE records"
  (`tmp/analysis/paper_raw.txt:697`).
* Raw simulator logs (the true Cadence provenance): `augmented-cvae-ctle/dataset_log_*.csv`
  (15 files, `ctle_dagger_common.py:153-162`). Header shows columns such as
  `eye_maxHeight_norm Vout_2 56G`, `eye_p2pJitterAverage_norm stage 2`,
  `eye_maxWidth_norm Vout_2 56G`, `power`, plus `batch`, `tb` (testbench id),
  `failure_mode`, `predicted_feasibility`, `sample_strategy`, `sampler`,
  `thompson_score`, `reward` → these are **active-learning / Bayesian-optimisation
  driven Cadence runs** over 8+ testbenches (`tb` ∈ {3,5,…}).
* Column renaming: `ctle_dagger_common.py:164-168` (`COL_MAPPING`) maps the raw
  Spectre measurement names onto `stage_2_eye_max_height/width`, `stage_2_jitter`.
* Loading + filtering: `ctle_dagger_common.py:3533-3550` — concatenates the 15
  CSVs, selects the 11 required columns, and drops rows with
  non-positive jitter/power/height/width or any NaN.
  From the real logs this yields **11,161 filtered rows out of 23,854**
  (`outputs_alliance_can/kn_bayes_opt/ctle_knet_e400/trial_0011.log.txt:40-45`).
* Default data dir: `ctle_dagger_common.py:75` —
  `/home/annaik/Documents/augmented-cvae-ctle/` (the repo copy is `augmented-cvae-ctle/`).

**The ZIG forward surrogate** (the model that maps device parameters → the 4 specs):

* `ctle_dagger_common.py:264-340` — `HybridHurdleModel` (hurdle/zero-inflated
  "ZIG" model: a validity head + per-metric regression heads).
* `ctle_dagger_common.py:341-391` — `ZIGHurdleWrapper`.
* Loaded from a *teacher artifact directory*
  `/home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2/`
  (`ctle_dagger_common.py:74`, loaded at `:3505-3531`): `hybrid_hurdle_ctle_model.pt`,
  `hybrid_hurdle_scaler_X.pkl`, `hybrid_hurdle_scaler_y_power.pkl`,
  `hybrid_hurdle_config.pkl`, `flow_scaler_X.pkl`, `flow_scaler_C.pkl`.
  **This directory is not in the repo.**

**The upstream generative flow teacher** (~1 M parameters, `I → 7 params`):

* `ctle_dagger_common.py:565-633` (`ConditionalSplineFlow`), `:635-808`
  (`FlowInference.sample_and_rank`), `:839-892` (`FlowTeacherLabeler`).
* Same teacher directory as the ZIG.

**The canonical (fixed) 20,000-spec split used by the paper's final pipeline:**

* `ctle_dagger_common.py:1871-2062` — `create_canonical_flow_dataset`:
  samples 20,000 specs (`--phase-a-n-samples`, default 20000,
  `dagger-nuance-distillation-kirchhoffnet.py:750`), labels each with the flow
  teacher's **top-8** candidates (`PHASE_A_TOP_K = 8`, `ctle_dagger_common.py:1810`),
  filters by ZIG validity at 0.9 (relaxed 0.5), and **breaks ties by minimum
  predicted power** (`PHASE_A_TIE_BREAK = "min_power"`, `ctle_dagger_common.py:1809`,
  tie-break code `:1965-1969`).
* Split: 80/10/10 → **16,000 train / 2,000 val / 2,000 test**
  (`ctle_dagger_common.py:2001-2007`), matching the paper
  (`tmp/analysis/paper_raw.txt:697`: "fixed 20,000-specification canonical set with a
  16,000/2,000/2,000 train/validation/test split").
* Schema-2 additionally appends the frozen MLP teacher's logits
  (`mlp_logits_trial0019`), the flow/MLP label powers, and `power_norm_const`
  (`ctle_dagger_common.py:2035-2055`).

**Is there a CTLE circuit model / netlist / Cadence-generated dataset generator in the repo?**
**No.** There is no SPICE netlist, no behavioural CTLE model, and no dataset
generator script; the 83 k rows are *outputs* of an external Cadence flow
(the `dataset_log_*.csv` files were produced elsewhere and copied in).

### 1.4 Loss functions

| Path | Loss | Where |
|---|---|---|
| **Fixed distillation (paper §III-B Eq. 8/11)** | `MSE(student_logits, teacher_logits)` | `fixed-mlp-distillation-kirchhoffnet.py:519` (`criterion = nn.MSELoss()`), `:540` |
| **Phase-A (paper §IV-B, "C-full")** | 4-term guided loss: `1.0·MSE(student, trial_0019 teacher logits) + 1.0·ZIG-forward-MSE + 0.0005·|power|/train-mean + 0.3·validity-NLL(ramp 10+30)` | `ctle_dagger_common.py:2885-2890` (weights), `:2997-3001` (ramp), described in `run_train_knet_single_shot_distillation.sh:14-18, 49-51` |
| Phase-A legacy (no teacher) | Huber on log-params | `ctle_dagger_common.py:2825-2863` docstring |
| **DAgger (older path)** | `RegimeAwareLoss` = α_spec·L_spec + β_phys·L_phys + γ_mono·L_mono + α_invalid·L_invalid + α_manifold·L_manifold | `ctle_dagger_common.py:1536-1676`; weights `:600-606` |
| Synthetic benchmarks | Huber / MSE | `train.py:1-13`, `train_script.py:1-35` |

AdamW with three LR groups (mapper / struct / dynamic) is used for the KNet
(`ctle_dagger_common.py:3372-3412`, `_build_phase_a_optimizer`; group split logged
e.g. `outputs_alliance_can/kn_bayes_opt/phase_a_mlp_distill_ctle/trial_0011.log.txt`).

### 1.5 The "failure rate" metric

**One fixed forward rule, used identically for the surrogate and for Cadence:**

> A design **fails** when at least **two** of the four specifications degrade by
> **≥ 20 %** in the bad direction (higher power or jitter, lower eye height or
> width) relative to the target.

Code:

* `ctle_dagger_common.py:2490-2529` — `compute_forward_errors`:
  one-sided relative degrades (`:2509-2512`), per-dim boolean at
  `degrade_thr` (`:2513-2518`), `failure_mask = degraded_dims >= min_bad` (`:2521`).
* `ctle_dagger_common.py:122-123` — `DEGRADE_REL_THRESHOLD = 0.20`,
  `MIN_DEGRADED_DIMS = 2`.
* `ctle_dagger_common.py:2240-2296` — `StudentEvaluator.identify_failures`:
  forward the student → decode 7 params → ZIG predicts the 4 specs → apply the
  rule. **Additionally**, a spec counts as failed if the ZIG validity probability
  `p_valid < VALIDITY_THRESHOLD (0.5)` or any value is NaN
  (`ctle_dagger_common.py:2275-2276`).
* Stratified reporting: boundary vs interior (`:2280-2285`, `:3894-3904`).
* Surrogate ("ZIG") rate = this rule applied to ZIG-predicted specs.
  Cadence rate = the same rule applied to **Spectre-measured** specs
  (`spice_failure_check.py:1516-1583`).

⚠ **Threshold inconsistency:** the Cadence scorer defaults to
`--degrade-thr 0.10` (`spice_failure_check.py:262-263`) while the surrogate
uses 0.20 (`ctle_dagger_common.py:122`, and the fixed harness default
`--zig-degrade-threshold 0.20` at `fixed-mlp-distillation-kirchhoffnet.py:111`).
See §6.

### 1.6 Minor inconsistencies found

* Column order of the 4 specs differs: `power, jitter, height, width`
  (`train_ctle.py:101`) vs `power, stage_2_jitter, stage_2_eye_max_height,
  stage_2_eye_max_width` (`ctle_dagger_common.py:85`). The DAgger runner re-orders
  explicitly at `ctle_dagger_common.py:4066-4069`
  (`specs_eval[:, [0, 3, 1, 2]]`) — easy to get wrong.
* `PARAM_LOG_BOUNDS` has three divergent copies (§1.2).
* `ctle_dagger_common.py:78-83` `SPEC_RANGES` uses key `stage_2_eye_max_height`
  etc.; `train_ctle.py:105-109` uses `height/width/jitter` with the same values.

---

## 2. End-to-end training flow

### 2.1 Which entry point produces the paper's final HR-KNet CTLE model?

The paper's final model is produced by the **Phase-A BO** over a *frozen,
fixed* canonical dataset, with the KNet trained by MLP-guided distillation:

```
# 1) build the shared canonical dataset (once)
python dagger-nuance-distillation-kirchhoffnet.py \
  --prepare-canonical-dataset $CANONICAL \
  --phase-a-mlp-teacher-ckpt $TEACHER_CKPT \
  --phase-a-n-samples 20000 --phase-a-boundary-ratio 0.5 \
  --phase-a-power-norm train_mean --seed 100 \
  --teacher-dir /home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2 \
  --data-dir /home/annaik/Documents/augmented-cvae-ctle \
  --device cuda --output $PREPARE_SCRATCH
```
`run_train_knet_single_shot_distillation.sh:53-66` (dataset build), and the same
block with `--ctle-canonical-dataset` in `run_phase_a_knet_bo.sh:37-48`.

```
# 2) BO over topology/solver/device + train the candidate
python kn_bayes_opt.py --dataset ctle --ctle-phase-a \
  --ctle-canonical-dataset "$CANONICAL" --ctle-objective validation \
  --param-budget 7000 --param-tolerance 0.15 \
  --epochs 300 --n-trials 30 --seed 100 --device cuda \
  --output outputs/phase_a_knet_bo
```
`run_phase_a_knet_bo.sh:50-59`.

```
# 3) the single-shot "C-full" run that produced the paper's model
python dagger-nuance-distillation-kirchhoffnet.py --canonical-dataset "$CANONICAL" \
  --phase-a-epochs 300 --kn-num-stages 6 --kn-num-hidden 12 --kn-small-world-k 4 \
  --kn-vca-rank 2 --kn-x-max 4.0 --kn-gm-max 1.351251 --kn-isat-max 1.364811e+01 \
  --kn-moe-num-experts 3 --kn-moe-gate-rank 2 --kn-moe-top-k 1 \
  --kn-mapper-lr-scale 1.0 --kn-struct-lr-scale 4.0 --kn-dyn-lr-scale 1.0 \
  --lr 6.002040e-03 --weight-decay 7.893409e-04 --batch-size 512 \
  --earlystop-eval-every 5 --phase-a-validity-weight 0.3 \
  --phase-a-validity-ramp-start 10 --phase-a-validity-ramp-epochs 30 \
  --phase-a-mlp-weight 1.0 --phase-a-fwd-weight 1.0 --phase-a-power-weight 0.0005 \
  --kn-readout shared-x2 --kn-learnable-clip-sharpness --kn-gln-rails \
  --kn-gln-B 4 --kn-gln-rank 2 --kn-gln-families boundary,readout \
  --t-span 5.084841 --boundary-fan-out '{"0":[0,4],"1":[1,5],"2":[2,6],"3":[3,7]}' \
  --device cuda --seed 100 --phase-a-mlp-teacher-ckpt "$TEACHER_CKPT" \
  --output .../trial_0005
```
`run_train_knet_single_shot_distillation.sh:68-77`.

### 2.2 The complete set of entry points

| Script | Role |
|---|---|
| `dagger-nuance-distillation-kirchhoffnet.py` (3783 lines) | **Main KNet CTLE harness.** Two modes: (a) *Phase-A* (static canonical dataset, `--phase-a-*`, `--canonical-dataset`, `--prepare-canonical-dataset`); (b) *DAgger* (`--dagger-iterations`, `--epochs-per-iter`). Contains its own copy of the flow teacher, ZIG model and `LocalKirchhoffStudentWrapper` (`:1692-1990`). |
| `dagger-nuance-distillation-kirchhoffnet.ipynb` / `.py` | Notebooks/variants of the above |
| `ctle_dagger_common.py` (4164 lines) | Shared infra: `setup()` (`:3484-3595`), `run_dagger_training()` (`:3601-4164`), `run_phase_a_training()` (`:2825-3308`), `PlainMLP` (`:1382-1529`), `RegimeAwareMoE` (`:1206-1347`), `HybridHurdleModel` (`:264-340`), `ConditionalSplineFlow` (`:565-633`), `StudentEvaluator` (`:2223-2296`), `create_canonical_flow_dataset` (`:1871-2062`). |
| `ctle_kirchhoff_bo.py` (653 lines) | A **separate, older** BO variant: adjoint-trained (`torchdiffeq.odeint_adjoint`, dopri5), bare-repo KCL circuit (`:1-12`, `:63-...`), its own `PARAM_LOG_BOUNDS` (`:41-43`). Copies `phase_a_best.pt` → `best_model.pt` (`:570`). **Not** the paper's model. |
| `train_ctle.py` (1051 lines) | Older standalone 4-phase KD trainer from a `RegimeAwareMoE` teacher with **LHS-sampled synthetic** specs (`train_ctle.py:1-20`, `:468-488`, `:524-585`). Uses a 3-stage *grid* topology (`make_ctle_preset`, `:142-311`). |
| `fixed-distillation-bayes-opt.py` (1156 lines) | **Clean fixed-dataset BO** (no DAgger): multi-objective TPE over (ZIG failure rate, teacher-logit MSE). Entry for both KNet and MLP. |
| `fixed-mlp-distillation-kirchhoffnet.py` (614 lines) | The shared fixed-distillation harness (KNet or MLP student, MSE on teacher logits). |
| `fixed-knet-bayes-opt.py` / `fixed-plainmlp-bayes-opt.py` (10 lines each) | Thin `runpy` wrappers injecting `--student-kind {knet,mlp}`. |
| `fixed-mlp-distillation-plainmlp.py` (18 lines) | Thin wrapper injecting `--student-kind mlp`. |
| `compare-fixed-distillation-bo.py` (51 lines) | Compares two BO `bo_summary.json` files and exits non-zero unless the KNet wins on both metrics. |
| `kn_bayes_opt.py` (2143 lines) | The **synthetic + CTLE BO driver** for KNet (15-dim TPE). |
| `mlp_bayes_opt.py` (1162 lines) | The MLP / MoE-MLP BO driver (teacher search + baseline). |
| `generative-distillation-improved-dagger-nuance-mlp.py` | MoE student DAgger trainer (the paper's *digital baseline*, `RegimeAwareMoE`). |
| `generative-distillation-plain-mlp.py` | PlainMLP DAgger trainer (the paper's *teacher* family). |
| `generative-distillation-plain-mlp-bayes-opt.py` | BO over PlainMLP architectures (teacher search, 25 k budget). |
| `train.py` / `train_script.py` | Generic KNet trainer (synthetic benchmarks). `train.py` = loss/regularizers/schedules; `train_script.py` = CLI entry, saves `model.pt`, `config_snapshot.txt`, `final_metrics.txt` (`train_script.py:4573`, `:4538-4546`, `:726-728`). |
| `bo_param_sampling.py` (443 lines) | Shared joint feasible-architecture sampler + resume fingerprint guard. |
| `generate_report.py` (888 lines) | Builds `comparison_report.html` from *external* result dirs (`C:\Users\Atharva\Downloads\results_many_models_realizable_knet_nb`, `generate_report.py:8-13`). Synthetic-benchmark reporting only; **no CTLE/Cadence content**. |
| `spice_failure_check.py` (2003 lines) | The Cadence/Spectre verification harness (§3). |

### 2.3 The distillation teacher

Two-level teacher chain (paper §III-A/III-B):

```
~1M-param conditional spline flow  (I → 7 CTLE params)      ← "flow teacher"
        │  ZIG validity + consistency filtering, k-NN fallback
        ▼
~25k-param PlainMLP / RegimeAwareMoE  (4 specs → 7 params)   ← "digital teacher"
        │  frozen; labels each spec row exactly once
        ▼
~5.6k-param HR-KNet (KNet fabric)                            ← "analog student"
```

* Flow teacher: `ctle_dagger_common.py:565-633`, `:839-892`;
  artifact dir `/home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2/`
  (`ctle_dagger_common.py:74`).
* The **frozen PlainMLP teacher** actually used by the paper's pipeline:
  `outputs/generative_plain_bayes_opt/plain_knet_budget25000_fproxy/trial_0019/dagger_student_plain.pt`
  — W=96, L=3, SiLU, LayerNorm=True, knet preprocessing
  (`run_train_knet_single_shot_distillation.sh:36`;
  `outputs_alliance_can/kn_bayes_opt/phase_a_mlp_distill_ctle/trial_0011.log.txt:44-46`).
* Teacher search (paper §III-A): 20 trials, 2 workers, seed 100, 25 k-param budget,
  0.25 penalty past budget, 8 DAgger iterations × 100 epochs, Huber loss on teacher
  labels, best-validation trial (5 % failure) frozen
  (`tmp/analysis/paper_raw.txt:660-668`).
  In this repo the MLP BO driver is `mlp_bayes_opt.py` /
  `generative-distillation-plain-mlp-bayes-opt.py`.
* Teacher loading + scaler attachment: `ctle_dagger_common.py:926-1063`
  (`load_plain_mlp_teacher`), `:1064-1107` (`MlpTeacherLabeler`),
  `fixed-mlp-distillation-kirchhoffnet.py:393-410`.
* Schema-2 teacher metadata (`teacher_config.json`) example:
  `outputs/smoke_mlp_teacher/teacher_config.json`
  (`{"trunk_width":48,"trunk_layers":3,"activation":"silu","use_layernorm":false,
  "input_preprocessing":"knet","param_budget":5559}`).

### 2.4 BO search space — every searched hyper-parameter and range

#### (a) `kn_bayes_opt.py` — 15 dims (synthetic + CTLE)

The 15 dims are enumerated verbatim in
`outputs_alliance_can/kn_bayes_opt/friedman1_knet_e400/best_hyperparams.txt:17`:

> `fanout_count, num_hidden, small_world_k, num_stages, t_span, lr, vca_rank,
> weight_decay, batch_size, x_max, gm_max, isat_max, device_l2_lambda,
> freeze_boundary, freeze_temporal_read`

| Dim | Type | Range / choices | Source |
|---|---|---|---|
| `num_hidden` | joint categorical (arch tuple) | hidden ∈ [8, 32] (generic) / prior-biased set | `bo_param_sampling.py:332-433`; `kn_bayes_opt.py:1750-1752` |
| `num_stages` | joint categorical | [1, 10] generic; CTLE prior (3,4,5) | `bo_param_sampling.py:335`; `fixed-distillation-bayes-opt.py:58` |
| `small_world_k` | joint categorical | {2,4,6,8} generic; CTLE prior (4,6) | `bo_param_sampling.py:335`; `fixed-distillation-bayes-opt.py:59` |
| `vca_rank` | joint categorical | [1, 8] generic; CTLE prior (2,3,4) | `bo_param_sampling.py:336`; `fixed-distillation-bayes-opt.py:60` |
| `fanout_count` | fixed per study | 2 (CTLE) | `bo_param_sampling.py:337` |
| `small_world_p` | **fixed** | 0.2 | `kn_bayes_opt.py:1755` |
| **`t_span`** | float | 0.5 – 10.0 (generic); CTLE clamps to **3.0 – 7.0** | `kn_bayes_opt.py:815-816`; CTLE clamp `:1375` |
| `num_steps` | derived | `round(STEPS_PER_T_SPAN × t_span)` (70 steps at t_span 7) | `kn_bayes_opt.py:1758`; `config.py:531-535` |
| `lr` | float (log) | 1e-4 – 1e-2 (generic); **2e-4 – 5e-3** (fixed-distillation) | `kn_bayes_opt.py:823-824`; `fixed-distillation-bayes-opt.py:320` |
| `weight_decay` | float (log) | 1e-6 – 1e-2 | `kn_bayes_opt.py:825-826`; `fixed-distillation-bayes-opt.py:364` |
| `batch_size` | categorical | {128,256,384,512} (fixed-dist. KNet); {256,512,1024} (MLP BO) | `fixed-distillation-bayes-opt.py:315-318`; `mlp_bayes_opt.py:951` |
| **`x_max` (rail)** | float | **0.5 – 8.0** generic; **2.0 – 8.0** fixed-distillation | `kn_bayes_opt.py:830-834`; `fixed-distillation-bayes-opt.py:362` |
| **`gm_max`** | float (log) | **1.0 – 50.0** | `kn_bayes_opt.py:836-841`; `config.py:325` |
| **`isat_max`** | float (log) | **1.0 – 50.0** | `kn_bayes_opt.py:843-848`; `config.py:326` |
| `device_l2_lambda` | float | 0.0 – 1e-3 | `kn_bayes_opt.py:862-867`, `:1774-1775` |
| `freeze_boundary` | categorical | {0,1} | `kn_bayes_opt.py:1776-1777` |
| `freeze_temporal_read` | categorical | {0,1} | `kn_bayes_opt.py:1778-1779` |
| `solver_scale` | float | **0.5 – 2.0** (param-count-neutral; scales per-stage t_span and num_steps together) | `fixed-distillation-bayes-opt.py:363`; `docs/specs/fixed-bo-round1.md:33` |
| `readout` | **fixed per study** | `temporal` / `shared` / `shared-x2` | `kn_bayes_opt.py:732`; `bo_param_sampling.py:359-360` |
| GLN | **fixed per study** | `--gln-B 4 --gln-rank 2 --gln-families boundary,readout` | `kn_bayes_opt.py:752-766` |
| `gm_max`/`isat_max` **not** searched in fixed-distillation round 1 (deferred) | — | — | `docs/specs/fixed-bo-round1.md:35-36` |

**Not searched but hard-wired in CTLE runs** (important for a physical retrain):

* `leak` = fixed 0.0486 (`--leak non-programmable`) — README §9; `config.py` INIT.
* `C_eff` = **1.0 (unitless)** — `config.py:305`.
* `clip_current` = 0.05, `clip_softness` = 0.02 — `config.py:306-308`.
* `edge_repeats` = 2 — `bo_param_sampling.py:283`.
* `small_world_seed` = 1 — `bo_param_sampling.py:279`.
* `interstage_activation` = `residual-relu-tanh` — `bo_param_sampling.py:286`.
* `freeze_read` = True — `bo_param_sampling.py:290`.
* `cell_library` = `tanh_free` — `bo_param_sampling.py:289`.
* `vca_gate_shunt` = False, `vca_bias` = False, `vca_separate_core_bus` = True,
  `vca_core_enabled` = True — `bo_param_sampling.py:296-297`.

#### (b) `mlp_bayes_opt.py` — teacher / baseline search

* `lr` 1e-4 – 1e-2 (log), `weight_decay` 1e-6 – 1e-2 (log), `batch_size`
  {256,512,1024}, `activation` {silu,gelu}
  (`mlp_bayes_opt.py:948-952`, `:408-411`).
* Architecture is *solved* for width to keep the parameter budget exact
  (`bo_param_sampling.py:105-145`); for the CTLE MoE the feasible set is
  `(layers ∈ [2,3], experts ∈ [2,4], width ∈ [32,64])`
  (`bo_param_sampling.py:196-207`).

#### (c) BO mechanics

* Sampler: `TPESampler(multivariate=True, group=True)` (`README` §14;
  `docs/specs/fixed-bo-round1.md:46`). `fixed-distillation-bayes-opt.py:68`
  tracks a `SAMPLER_REV = 3` fingerprint.
* Pruner: `NopPruner` for synthetic; `SuccessiveHalvingPruner` with rungs
  `1,2,4` DAgger iterations for CTLE (`kn_bayes_opt.py:922-933`).
  The fixed-distillation BO uses a **rank-based two-rung halving** instead
  (`fixed-distillation-bayes-opt.py:386-421`, `docs/specs/fixed-bo-round1.md:54-59`).
* **Finite penalties, never `inf`/`TrialPruned`** — `(1.0, 1e6)`
  (`fixed-distillation-bayes-opt.py:41`, `bo_param_sampling.py:36-46`), because
  pruned/None values crash multi-objective TPE (optuna#5260 / #3676).
* Over-budget objective: `base × (params/budget)²` (`bo_param_sampling.py:39-46`).
* Objective for CTLE: `--ctle-objective {validation,test}`, default `validation`
  (`kn_bayes_opt.py:956`); the legacy path returned the *test* rate
  (`kn_bayes_opt.py:1701-1707`) — see the test-leakage caveat in §6.

### 2.5 The sh scripts and SLURM headers

| Script | SBATCH | Purpose |
|---|---|---|
| `run_phase_a_knet_bo.sh` | `def-shiyusu`, 1× `nvidia_h100_80gb_hbm3_3g.40gb`, 6 CPU, 32 G, **`--time=24:00:00`** (`:1-7`) | Phase-A KNet BO (30 trials × 300 ep) + canonical dataset build (`:37-59`) |
| `run_train_knet_single_shot_distillation.sh` | same GPU, **`--time=03:00:00`** (`:1-7`) | Canonical dataset build + one "C-full" MLP-guided KNet trial (`:53-77`) |
| `run_knet_HiPPO.sh` | `--time=4:00:00` (`:6`) | **Misleading name**: body is `python test_memory_knet.py` (`:36`) — that file is **not found** in the repo; the header comment is a copy of the Phase-A header |
| `narma_*_batch.sh` (23 files), `post_twin_gates_batch.sh` | — | NARMA reservoir research, **not** CTLE |
| `run_dagger_distillation.sh-57774495.out` | (log, not a script) | Contains the exact cold-start DAgger command (`:13-24`) |

The remaining `narma_*`/`post_twin_*` batch scripts and `*_batch.sh` files are
unrelated to the CTLE/Cadence pipeline.

---

## 3. Existing Cadence / SPICE integration

### 3.1 `spice_failure_check.py` — the *only* Cadence-touching file in the repo

Module docstring `spice_failure_check.py:1-41`. Three stages:

**Stage 1 — inference (host, PyTorch).**
Load one checkpoint → forward N target specs → 7 physical params.

* `--model-kind {knet, mlp, moe, knet-distill}` (`:114-121`)
* `MlpTeacher` loader `:413-585`; `MoeStudent` loader `:586-816`;
  `KNetDistillWrapper` `:828-1078`; `load_knet_distill()` `:1084-1180`;
  `predict_knet_distill()` `:1183-1193`.
* The KNet is rebuilt from **explicit CLI flags** via
  `topology.build_net_from_config` + `cell_library.make_cell_library`
  (`:1091-1092`, `:1129-1169`) and `load_state_dict(..., strict=True)` (`:1173`).
* `input_log_min/max` come from the canonical `.npz` (`:1109-1122`) — no silent
  defaults.
* Alternative bridge: `--pred-params-csv <csv>` with (N,7) log10 params skips
  Stage 1 entirely (`:223-227`).

**Stage 2 — Cadence Spectre via Maestro/Ocean (Linux `ssr0` only).**

* `write_spec_ocn()` `:1212-1279` — regex-rewrites `template_{tb}.ocn`:
  swaps the schematic lib/cell/view to `ECE740 / tb_channel_CTLE_ML_{tb_idx}` /
  `maestro` (`:1236-1252`), forces batch mode (`:1253-1263`), rewrites the seven
  `desVar`s (`:1264-1270`), and repoints the CSV export (`:1271-1278`).
* `run_simulation()` `:1282-1319` — writes `run_ocean.csh`:
  `cd /home/annaik/workarea_GF22_FDX_EXT` → `source <pdk>` → `ocean -nograph` →
  `load "<ocn>"` → `exit()`; launched via `/bin/csh` with a timeout (`:1290-1311`).
* PDK: `--pdk-source-cmd` default
  `source /home/annaik/workarea_GF22_FDX_EXT/kit.gf22fdx_ext.v1.0_4.1_stack19_oa.csh`
  (`:253-255`) → **GF22 FDX EXT**, i.e. the 22 nm node.
* Testbenches: `--tb-ids 1,2,3,4,5,6,7,9` (`:242-243`), one TB pinned per worker
  so two sims never share a TB (`docs/specs/spice-failure-spec.md:30`).
* Result parsing: `_wait_for_csv()` `:1322-1328`, `_parse_ctle_csv()` `:1331-1357`
  (re-implementation of `ocean_utils_ctle.parse_ctle_csv`, `:1332`).
* Worker `_ocean_worker()` `:1438-1489`; per-spec artefacts
  `params.json`, `target.json`, `spec.ocn`, `result.csv`, `parsed.json`
  (`docs/specs/spice-failure-spec.md:27`).
* **Local/mock mode**: without `--enable-ocean` a deterministic mocked parser
  `_mocked_parser()` `:1417-1431` is used; `--mock-template` writes a synthetic
  OCN. Both are explicitly gated so mocks cannot pose as Spectre
  (`docs/specs/spice-failure-spec.md:40-41`; `--score-force-mocked` `:291-293`).

**Stage 3 — scoring.**
* `_one_sided_degrade()` `:1516-1525`, `score_spec()` `:1528-1583` — same rule as
  the surrogate, but with `--degrade-thr` (default **0.10**) and
  `--min-degraded-dims` (default 2) (`:262-265`). Missing key / `eval err` /
  `sim err` / NaN / non-positive → **failure with `error=1`** (`:1544-1569`).
* `write_results_csv()` `:1586-1605`, `summarize()` `:1607-1645`
  (`results.csv` + `summary.json` with `n, failures, failure_rate, error_rate,
  dim_degrade_rates[4], ckpt_sha256, template_sha256, degrade_thr, key_map, …`).
* Extra modes: `--offline-csv` `:1691`, `--score-run-dir` `:1725` (zero Ocean
  calls, refuses mocked entries), `--dump-keys` `:1789`.

### 3.2 Repo-wide search for netlists / Verilog-A / OCEAN / SKILL

Searched the whole repo (including `output*/`, `outputs*/`, `outputs_alliance_can/`,
`tmp/`, `Paper/`, `augmented-cvae-ctle/`, `docs/`, `augmented-cvae-ctle/`),
excluding `venv/`, `.venv_simple_ctle/`, `site-packages`, `__pycache__`,
`node_modules`, `.git`, and the vendored `kirchhoffnet/` upstream checkout:

| Artifact type | Result |
|---|---|
| `*.ocn` files | **not found** (only an embedded Python string, the mock template `spice_failure_check.py:1362-1413`) |
| `*.scs` (Spectre model files) | **not found** (only the string `modelFile('$SPECTRE_MODEL_PATH/design_wrapper.lib.scs" "tt_pre")` inside the mock template, `spice_failure_check.py:1377`) |
| `*.va` (Verilog-A) | **not found** |
| `*.sp`, `*.cir`, `*.spi`, `*.net`, `*.netlist` | **not found** |
| `*.lib`, `*.mdl`, `*.cdsl`, `*.apt` | **not found** |
| OCEAN / SKILL scripts (`*.il`, `*.skill`) | **not found** |
| grep for `verilog|Verilog-A|behavioral model|veriloga` | **no matches anywhere in the repo** |
| grep for `spectre|Spectre|ocean -nograph|ngspice|PySpice|OCEAN` | only `spice_failure_check.py` (14 hits), the vendored `kirchhoffnet/README.md:41` and `kirchhoffnet/bare/README.md:6` (both say "we would like to redo this in SPICE/Spectre — future work"), and the paper text itself |
| SPICE/simulator python packages in `requirements_*.txt` | **none** (`requirements_local.txt`, `requirements_all_can.txt` — torch/optuna/pandas/sklearn/… only) |

**Conclusion:** there is **no Verilog-A behavioural model of the KNet OTA /
sampling switch in this repository**, and **no netlist export**. The only
"circuit" artefacts are (a) the mock OCN template embedded in
`spice_failure_check.py:1362-1413`, and (b) references to files that live
*outside* the repo on the Linux cluster:

* `/home/annaik/Documents/train_ctle/ocn_template_ideal_bob/template_{tb}.ocn`
  (`spice_failure_check.py:239-241`) — the real Maestro OCN templates.
* `ocean_utils_ctle.py` (`update_ocn_ctle`, `parse_ctle_csv`,
  `delete_maestro_history`) — an external module, re-implemented inline
  (`spice_failure_check.py:1201`, `:1220`, `:1332`).
* `ECE740` library / `tb_channel_CTLE_ML_{tb}` schematic — external GF22 design.
* `/home/annaik/workarea_GF22_FDX_EXT/` — the GF22 FDX EXT work area + PDK.
* `~/simulation/SPICE_failure` — the run work root (`spice_failure_check.py:244-246`).

### 3.3 What exactly does "Cadence-verified" refer to?

**The CTLE circuit — not the KNet.**

* The KNet fabric (`config.py`, `cell_library.py`, `topology.py`,
  `differential_stage.py`, `kirchhoff_net.py`) is a **PyTorch ODE simulator**
  with a normalised, unitless "voltage" state. `config.py:302` says so
  explicitly: *"Normalized physical limits (R7: not calibrated to real SI units)"*,
  and the paper says *"No quantity is tied to SI units or quantized to feasible
  device levels yet — C_eff, tspan, gm, conductances, and Isat are all
  continuous"* (`tmp/analysis/paper_raw.txt:768-773`).
* The only thing that ever enters Cadence is the **7-value CTLE knob vector**
  produced by the KNet (`spice_failure_check.py:1445-1446`: `var_map` →
  `desVar`), written into the external Maestro testbench and simulated by
  Spectre under the GF22 FDX PDK.
* Consequently the paper's "Cadence-verified failure rate" =
  *fraction of held-out specs for which the **Cadence-simulated CTLE** fails the
  degrade rule*, i.e. it validates the **controller's outputs**, not the analog
  fabric. Table II caption (`tmp/analysis/paper_raw.txt:793-794`):
  "surrogate failure rate over 2,000 held-out specs; Cadence failure rate over
  200 held-out specs".
* The paper's own conclusion lists what remains unverified:
  *"layout, noise, and PVT validation remain necessary before hardware
  implementation"* (`tmp/analysis/paper_raw.txt:987-988`).
* The Cadence evidence in the paper (Fig. 8/9: AC response, 56 Gb/s NRZ eye,
  28 GBaud PAM-4 eye) is **not reproducible from this repo**: the eye/AC assets
  exist only as pre-made SVGs in
  `Paper/DATE_conference_template_LaTeX/figures/eye_*.svg`, `eyes_ac_response.svg`,
  `eye_comp_final.svg`; there is no script here that regenerates them from a
  Spectre run.

Also note: the **KNet is never simulated in Spectre/ngspice anywhere** — the
only mention of doing so is as future work in the vendored upstream README
(`kirchhoffnet/README.md:41`).

---

## 4. Saved artifacts

### 4.1 What is present in *this* Windows checkout

**Model checkpoints (`.pt`) — the complete list:**

| Path | Size | What |
|---|---|---|
| `outputs/smoke_mlp_teacher/dagger_student_plain.pt` | 25,493 | smoke-test PlainMLP teacher (+ `teacher_config.json`, ZIG scalers) |
| `outputs/kn_bayes_opt/friedman2_knet_e1/trial_0000/model.pt` | 41,145 | synthetic KNet (friedman2, 1 epoch) |
| `outputs/kn_bayes_opt/friedman2_knet_e1/trial_0000/model_pruned.pt` | 27,985 | pruned variant |
| `output/_smoke_revised/smoke_k2.pt` | 20,753 | NARMA smoke |
| `outputs_alliance_can/narma_refresh2/fabric_refresh_k2_seed0.pt` | 59,682 | NARMA |

**→ There is NO CTLE KNet checkpoint and NO MoE/PlainMLP CTLE teacher checkpoint
in this checkout.** The `phase_a_best.pt` / `dagger_student_plain.pt` /
`dagger_student_moe.pt` files for the paper's CTLE runs live only on the Linux
cluster (`/home/annaik/Documents/kirchhoffnet_realistic/outputs/...`).

**Optuna study DBs — the complete list:**

| Path | Size | What |
|---|---|---|
| `outputs/kn_bayes_opt/friedman2_knet_e1/friedman2_knet_e1.db` | 114,688 | synthetic KNet BO, 1 trial |
| `outputs_alliance_can/study.db` | 135,168 | NARMA study |

The CTLE BO study DBs (`ctle_knet_e400`, `ctle_budget7000_e800`,
`phase_a_*`) are **not** present — only their summaries and logs were copied.

### 4.2 The BO summaries that *are* present

**Fixed-distillation BO round 1** — `outputs_alliance_can/kn_bayes_opt/fixed_knet_bo_round1/`
(`bo_summary.json` 22,478 B, `results.csv`, `results_live.csv`, `trial_log.jsonl`,
`trial_table.txt`, `dataset_prepare.log.txt`):

* 40 trials, 39 completed, 0 penalized.
* **Selected trial 15** (lexicographic: validation ZIG failure, then validation
  teacher-logit MSE).
* Config: `lr=4.7167711e-3`, `batch=256`, `hidden=10`, `stages=4`, `k=4`, `p=0.2`,
  `vca_rank=4`, `x_max=7.2935`, `solver_scale=0.876808`, `wd=3.6374911e-5`,
  boundary fan-out `{"0":[0,4],"1":[1,5],"2":[2,6],"3":[3,7]}`.
* **5,959 trainable params**; validation ZIG failure **6.81 %**, test **9.31 %**;
  validation logit MSE 5.03e-3.
* Fixed dataset: `/home/annaik/Documents/kirchhoffnet_realistic/outputs/fixed_teacher_dataset.npz` (not in the checkout).

**Fixed-distillation BO, MLP arm** — `outputs_alliance_can/mlp_bayes_opt/fixed_mlp_bo_plain_mlp_teacher/`
(`bo_summary.json`, `results_mlp_summary.csv`):

* 30 trials, 30 completed.
* Selected: `width=45, layers=3, silu, LayerNorm=True`, **4,957 params**,
  `lr=4.6627804e-3`, `batch=128`.
* Validation ZIG failure **4.84 %**, test **5.01 %**; validation logit MSE 3.53e-4.

> ⚠ **This contradicts the paper's direction.** In the *fixed* (clean, no-DAgger)
> benchmark stored in this repo, the KNet is **worse** than the MLP
> (9.31 % vs 5.01 % test failure), even though `compare-fixed-distillation-bo.py:40`
> requires the KNet to win on both metrics to print `SUCCESS`. The paper's
> claimed win (Table II) therefore comes from the **Phase-A** pipeline, not the
> fixed-distillation one.

**Phase-A BO (the paper's pipeline):**

* `outputs_alliance_can/kn_bayes_opt/phase_a_mlp_distill_ctle/` — 20 trial logs
  (`trial_0000.log.txt` … `trial_0019.log.txt`) + the parent SLURM log
  `run_train_knet_single_shot_distillation_2.sh-59014723.out`.
* `outputs_alliance_can/kn_bayes_opt/phase_a_mlp_distill_ctle_knet_original/` —
  36 trial directories (`trial_0000` … `trial_0036`, `trial_0022` absent).
* `outputs_alliance_can/mlp_bayes_opt/phase_a_mlp_distill_ctle/results.csv` —
  30 MoE trials with `phase_a_validation_failure_rate` / `phase_a_test_failure_rate`.
* `outputs_alliance_can/bo_descent_phase_a_ctle.csv` — the BO trajectory used for
  the paper's Fig. 8(a): **KNet best 7.65 % (trial 12)**, **MLP best 8.25 %
  (trial 28)**. (Both are validation failure rates; the paper's Table II quotes
  8.15 % / 8.70 %, presumably the corresponding *test*-split values of the frozen
  final models — those are not in this checkout.)

### 4.3 Which artifact corresponds to the paper's reported model

The paper's HR-KNet = **5,649 params** (`tmp/analysis/paper_raw.txt:805`;
`Paper/DATE_conference_template_LaTeX/template-A4.tex:749`).

`spice_failure_check.py:137-139` states the defaults mirror
`knet_model_files_manual_trial_0011/run_command.txt`
("verified against its .out log: 5,649 params, VCA on rank 3 core, shared-x2
readout, GLN boundary+readout"), and `docs/plans/spice-model-loaders.md:8` gives
the full configuration:

> stages **7**, hidden **11**, k **2**, p **0.2**, vca-rank **3**,
> MoE **3** experts / gate-rank **2** / top-k **1**, x-max **4.0**,
> gm/isat max (BO values), readout **shared-x2**, learnable-clip-sharpness,
> GLN B4 rank2 families=boundary,readout, **t-span 4.660583**, boundary fan-out
> `{"0":[0,4],"1":[1,5],"2":[2,6],"3":[3,7]}`; checkpoint `phase_a_best.pt`.

The CLI defaults in `spice_failure_check.py:140-203` encode exactly this
(`--kn-num-stages 7`, `--kn-num-hidden 11`, `--kn-small-world-k 2`,
`--kn-vca-rank 3`, `--kn-gm-max 4.241970`, `--kn-isat-max 2.868855e+01`,
`--kn-readout shared-x2`, `--kn-t-span 4.660583`, `--kn-gln-rails` on,
`--kn-gln-B 4 --kn-gln-rank 2 --kn-gln-families boundary,readout`).

**The closest artefact in this checkout** is
`outputs_alliance_can/kn_bayes_opt/phase_a_mlp_distill_ctle/trial_0011.log.txt`
(and `trial_0012.log.txt`), which has the *same* `t_span=4.660583`,
`gm_max=4.24197`, `isat_max=28.68855`, `k=2`, `rank=3`, `readout=shared-x2`,
learnable-clip + GLN, but **`num_stages=5`** (not 7) and **4,077 params**
(`trial_0011.log.txt:50`). Its command line names the cluster output dir
`outputs/phase_a_knet_bo_v2_gln_learnable_clip_no_moe_sampling/ctle_knet_e300/trial_0011`
(`trial_0011.log.txt:3`).

**The MoE baseline** = `mlp_bayes_opt/ctle_budget7000_e800` trial_0024
(`spice_failure_check.py:128`, `:582`; `--moe-trunk-width 34 --moe-trunk-layers 3
--moe-num-experts 2 --moe-activation silu`, `:129-132`).

**The live Cadence runs** referenced in the docs were
`knet_trial0011_test200` and `mlp_trial0024_test200`
(`docs/plans/spice-score-fix.md:8, :15`; `docs/specs/spice-score-spec.md:7`:
"the live knet_trial0011_test200 run (182/200 healthy sims)"). Their
`results.csv` / `summary.json` are **not** in this checkout.

### 4.4 How a checkpoint's parameters and topology are loaded

| Model | Save site | Load site |
|---|---|---|
| Generic KNet (synthetic) | `torch.save(raw_net.state_dict(), out_dir/"model.pt")` (`train_script.py:4573`); pruned variant `:5181-5185` | rebuild with `topology.build_net_from_config` + `cell_library.make_cell_library("tanh_free")` using the trial's `config_snapshot.txt` (`train_script.py:726-728`, example `outputs_alliance_can/kn_bayes_opt/friedman1_knet_e400/trial_0004_20260822_010645/config_snapshot.txt`) |
| CTLE Phase-A KNet | `phase_a_best.pt` / `phase_a_last.pt` (`ctle_dagger_common.py:2982-2983`) | `LocalKirchhoffStudentWrapper` (`dagger-nuance-distillation-kirchhoffnet.py:1692-1990`) or `spice_failure_check.py:load_knet_distill` (`:1084-1180`), which rebuilds via `build_net_from_config` with the `--kn-*` flags and `load_state_dict(strict=True)`, and takes `input_log_min/max` from the canonical npz |
| MoE KNet (CTLE, `ctle_kirchhoff_bo.py` variant) | `phase_a_best.pt` → copied to `best_model.pt` (`ctle_kirchhoff_bo.py:570`) | — |
| Fixed-distillation harness | `best_student.pt` (`fixed-mlp-distillation-kirchhoffnet.py:557`), `student_final.pt` + `student_bundle.pt` (`:588-591`) | same harness with `--resume`; the bundle carries `input_log_min/max` + `args` (`:589-591`) |
| DAgger student | `dagger_student_kirchhoff.pt` / `dagger_student_plain.pt` / `dagger_student_moe.pt` (`ctle_dagger_common.py:4091`) + `scaler_X.pkl`, `flow_scaler_C.pkl` (`:4092-4093`) | `load_plain_mlp_teacher` (`ctle_dagger_common.py:926-1063`) reads `teacher_config.json` beside the ckpt |
| Optuna studies | `<study>.db` (SQLite) | `optuna.load_study(storage="sqlite:///…")`; the resume fingerprint guard is `bo_param_sampling.py:68-100` |

---

## 5. What must change to retrain with physical units / realistic device values

### 5.1 Where the (non-physical) circuit parameters are defined

| Quantity | Current value | File:line | Nature |
|---|---|---|---|
| `C_eff` (node capacitance) | **1.0** | `config.py:305` | unitless |
| `x_max` (rail) | **3.0** (default), 4.0 in CTLE runs | `config.py:304`; `dagger-nuance-distillation-kirchhoffnet.py:632`; `fixed-mlp-distillation-kirchhoffnet.py:130` | unitless "volts" |
| `clip_current` | 0.05 | `config.py:306` | unitless current |
| `clip_softness` | 0.02 | `config.py:307` | unitless volts |
| `beta_softness` | 0.02 | `config.py:306` | — |
| `clip_sharpness_init/min/max` | 0.02 / 1e-3 / 0.2 | `config.py:315-317` | — |
| `gm_min`, `gm_max` | **0.01 / 10.0** | `config.py:270-271` | dimensionless |
| `isat_min`, `isat_max` | **0.01 / 10.0** | `config.py:272-273` | dimensionless |
| BO search windows for gm/isat | **(1.0, 50.0)** | `config.py:325-326` | dimensionless |
| `t_span` (integration horizon) | **7.0** | `config.py:532-533` | unitless "time" |
| `num_steps` | **70** (per-stage split, `dt = 0.1`) | `config.py:534`; `README.md:104-106` | — |
| leak (fixed) | **0.0486** = softplus(−3.0) | `README.md:46-48`, §9 | dimensionless conductance |
| edge params `a_raw`,`b_raw` | softplus(0) ≈ 0.693 | `README.md:75-76` | dimensionless width ratios |
| `g_resistive_raw` | softplus(−5) ≈ 0.0067 | `README.md:81` | dimensionless conductance |
| `VCA` gate range | (0, 2) | `config.py:603-630` | dimensionless ✔ (already physical in form) |
| `GLN` log-gm trim | `exp(Σ W·z)`, clamp to `[gm_min, gm_max]` | `README.md:346-350` | dimensionless factor ✔ |
| **CTLE output bounds** | `PARAM_LOG_BOUNDS` | `ctle_dagger_common.py:89-97` | **already physical** (F, A, H, Ω, V) |
| CTLE `VDD` range | **0.6 – 1.2 V** | `ctle_dagger_common.py:96` | **exceeds a 0.8 V GF22 core budget** |

The single most load-bearing line is `config.py:302`:
*"Normalized physical limits (R7: not calibrated to real SI units)"*.

### 5.2 Concrete change list for a GF22 (~0.8 V core) retrain

1. **Capacitance.** `PHYS["C_eff"]` must become a real farad value per node
   (e.g. 10–100 fF) — `config.py:305`. This changes the meaning of every
   current term: with `I` in A and `C` in F, `dx/dt` is V/s, so `t_span` must
   become seconds and `num_steps` must satisfy the Heun stability condition
   `dt ≲ 2·C·R_leak` (the paper's `t_span=7.0 / 70 steps / dt=0.1` is
   meaningless once C is physical).
2. **Rails.** `PHYS["x_max"]` → ~0.8 V (or 0.4 V differential for the
   differential-pair model of Fig. 2(d)); `clip_current` → an A value;
   `clip_softness` → a V value; `clip_sharpness_min/max` → V.
   The BO window `--x-max-min/max` (0.5–8.0 / 2.0–8.0) must be re-derived
   (`kn_bayes_opt.py:830-834`, `fixed-distillation-bayes-opt.py:362`).
3. **Device bounds.** `TANH_REALISTIC_GM_MIN/MAX` and
   `TANH_REALISTIC_ISAT_MIN/MAX` (`config.py:270-273`) must become GF22-derived
   ranges in **A/V** and **A**; `BO_GM_MAX_RANGE` / `BO_ISAT_MAX_RANGE`
   (`config.py:325-326`) and `--gm-max-min/max`, `--isat-max-min/max`
   (`kn_bayes_opt.py:836-848`) must be re-windowed.
   `softplus(a_raw)`/`softplus(b_raw)` must be re-interpreted as differential-pair
   width ratios (they already are, dimensionlessly).
4. **Leak.** The fixed 0.0486 must become a conductance in S (i.e. `1/R_leak`)
   or be re-expressed as a pole frequency; the `--leak non-programmable` mode
   used by every CTLE run (README §9) hard-codes it.
5. **Shunt `G`.** `softplus(g_raw) ≈ 0.0067` must become Ω⁻¹ with a GF22
   realistic range (`README.md:81`).
6. **Solver.** `SOLVER["t_span"] = 7.0` and `num_steps = 70`
   (`config.py:531-535`) must be re-derived in seconds; `solver_scale ∈ [0.5,2.0]`
   stays dimensionless.
7. **CTLE output bounds.** `PARAM_LOG_BOUNDS["VDD"] = (0.6, 1.2)` V
   (`ctle_dagger_common.py:96`) must be tightened to ≤ 0.8 V for a GF22 core
   (the paper's own task statement says "~0.8 V core"). This changes the
   teacher's output space and therefore invalidates the frozen teacher labels
   and the canonical dataset.
8. **Downstream regeneration.** Because the teacher's labels change, the
   following must be regenerated: the PlainMLP teacher (BO + DAgger), the
   canonical 20 k dataset (`ctle_dagger_common.py:1871-2062`), the ZIG surrogate
   (or reuse the existing GF22-fitted ZIG if the CTLE dataset is unchanged),
   and the failure-rate thresholds.
9. **Not yet implemented anywhere (paper's own future work list,
   `tmp/analysis/paper_raw.txt:439-442`):** DAC/code quantization of the 7
   outputs, mismatch injection, PVT/thermal corners, and on-chip
   SPICE-in-the-loop. The repo has *hooks* for PVT/mismatch
   (`config.py:652-...` `VARIATION`, `sim_context.py`) but they are **off by
   default** and were not used in the CTLE runs.
10. **A Verilog-A / Spectre model of the KNet fabric does not exist** and would
    have to be written from scratch (or the fabric exported as a netlist) —
    nothing in the repo does this today. The nearest existing artefact is the
    *paper figure* `Paper/DATE_conference_template_LaTeX/figures/edge_diagram.tex`
    and `param_explain_diff_pair.tex`, which are TikZ illustrations only.

### 5.3 Compute cost of a training run

Measured from the logs in this repo (all on 1× H100 80 GB, `nvidia_h100_80gb_hbm3_3g.40gb`):

| Run | Evidence | Wall time |
|---|---|---|
| Synthetic KNet BO trial (400 epochs, friedman1) | `outputs_alliance_can/kn_bayes_opt/friedman1_knet_e400/best_hyperparams.txt:27` (`subprocess_seconds: 489.1`), `.../trial_0004_20260822_010645/final_metrics.txt:9` (`elapsed_seconds: 361.32`) | **≈ 6–8 min / trial** → 50 trials ≈ **5–7 GPU-h** |
| Phase-A CTLE KNet trial (300 epochs) | `outputs_alliance_can/kn_bayes_opt/phase_a_mlp_distill_ctle/trial_0011.log.txt` epoch timestamps (19:38:12 → 19:41:03 → 19:44:15 → … = **19–21 s/epoch**) | **≈ 1.6–1.8 h / trial** → 30 trials ≈ **50–55 GPU-h** (SLURM request: 24 h, `run_phase_a_knet_bo.sh:6`) |
| Single "C-full" Phase-A KNet trial | `--time=03:00:00` (`run_train_knet_single_shot_distillation.sh:6`) | ≤ 3 h |
| Fixed-distillation BO, round 1 | `docs/specs/fixed-bo-round1.md:70-72` ("15 ep ≈ 5 min at 36 min/100 ep"; 20 trials, ~900 epoch-equivalents) | **≈ 5–6 h** |
| DAgger KNet (4 iterations × 100 epochs) | `dagger_distillation_trial_0002_recreation_no_vca_gate_shunt.out` — run start 12:27:31, last epoch 13:27:30 | **≈ 60 min** |
| PlainMLP DAgger | `dagger_distillation_mlp_plain.out:60-106` (350.7 s to build the flow dataset), `:919` (669.0 s training) | **≈ 17 min** total |
| Teacher BO (MLP, 20–30 trials × 800 epochs) | `mlp_bayes_opt.py:389` `--epochs 800` default | est. **10–20 GPU-h** |

Extra costs that are *not* training:
* **Cadence verification**: 200 specs per model, one Spectre/Maestro run per spec,
  8 testbenches in parallel, `--sim-timeout 600` s each
  (`spice_failure_check.py:249-252`). The live run reported 182/200 healthy sims
  (`docs/specs/spice-score-spec.md:7`) — i.e. **≈ 18 sim-hours wall, ≈ 2–3 h with 8-way
  parallelism**, plus the ~30-min PDK/Maestro startup overhead per worker.
* **Building the flow-teacher labels** for a 20 k canonical set with 5,000
  candidates/spec is the dominant one-off cost: ~350 s for 20 k plain-MLP DAgger
  samples (`dagger_distillation_mlp_plain.out:106`), but the Phase-A canonical
  build uses 5,000 candidates × 8 top-k per spec and is much heavier
  (`ctle_dagger_common.py:1877-1941`).

⚠ `outputs_alliance_can/mlp_bayes_opt/phase_a_mlp_distill_ctle/results.csv`
records `elapsed_seconds = 0.092` for every trial — a **CSV-parsing artifact**
(columns are misaligned; the value actually sitting in `elapsed_seconds` is the
failure rate). Do **not** use that file for timing; read the per-trial logs.

---

## 6. Inconsistencies, caveats and open risks (found while reading)

1. **Cadence vs surrogate degrade threshold differ** — 0.10
   (`spice_failure_check.py:262-263`) vs 0.20 (`ctle_dagger_common.py:122`).
   The paper describes a single fixed rule at 20 % (`tmp/analysis/paper_raw.txt:718-721`).
2. **Test leakage in the legacy BO objective** — `kn_bayes_opt.py:1701-1707`
   keeps `test_rate` as the Optuna value when `--ctle-objective test`; the
   default is now `validation` (`kn_bayes_opt.py:956`). The problem statement
   documents the historical leakage: `docs/ctle-knet-bo-problem-statement.md:75, 210`.
3. **Warm-start inflation** — `ctle_knet_e400_old_no_experts` numbers are
   invalid (7–8 pts optimistic); the corrected cold-start baseline is 14.2 % val
   / 11.0 % test (`docs/ctle-knet-bo-problem-statement.md:134-145, 274-284`).
4. **`parse_known_args` silently drops unsupported flags**, so several past
   "ablations" were no-ops (`docs/ctle-knet-bo-problem-statement.md:176`).
5. **The fixed-distillation BO result in this repo contradicts the paper**
   (KNet 9.31 % vs MLP 5.01 % test failure; §4.2).
6. **Three divergent `PARAM_LOG_BOUNDS`** (§1.2).
7. **A stray `run_knet_HiPPO.sh`** whose body runs a non-existent
   `test_memory_knet.py` (`run_knet_HiPPO.sh:36`).
8. **`ARCHITECTURE.md` is stale** relative to the CTLE pipeline: its File Map
   (`ARCHITECTURE.md:1332-1417`) predates `ctle_dagger_common.py`,
   `kn_bayes_opt.py`, `spice_failure_check.py`, the GLN/VCA work, and the whole
   Phase-A/fixed-distillation pipeline. `README.md` is much closer to current.
9. **No CTLE checkpoint / study DB in this checkout** — the paper's own model
   cannot be loaded or re-scored from here without pulling from the cluster.

---

## 7. Quick-reference file map (CTLE / Cadence relevant)

```
README.md                                  rebuild-grade description of the machine (§1–§15)
ARCHITECTURE.md                            older architecture doc (stale for CTLE)
config.py                                  SOLVER/PHYS/cell bounds/regularizers  <-- retrain knobs
cell_library.py                            FreeTanhLibrary (tanh_free cell): gm/Isat/G/A/B/s
topology.py                                small_world graph, build_net_from_config
differential_stage.py                      KCL scatter, Heun, VCA, freeze, shared-sense
kirchhoff_net.py                           KirchhoffNetWithIO (write/evolve/read)
io_mapper.py, stage_transfer.py, gln_rails.py
train.py / train_script.py                 generic trainer (synthetic benchmarks)
kn_bayes_opt.py                            15-dim TPE BO (synthetic + CTLE)         <-- search space
mlp_bayes_opt.py                           MLP / MoE BO (teacher + baseline)
bo_param_sampling.py                       joint feasible-arch sampler + fingerprint guard
ctle_dagger_common.py                      CTLE task infra: ZIG, flow teacher, MoE/PlainMLP,
                                           StudentEvaluator, Phase-A trainer, DAgger loop
dagger-nuance-distillation-kirchhoffnet.py KNet CTLE harness (Phase-A + DAgger)
dagger-mlp-distillation-kirchhoffnet.py    alternate KNet DAgger harness
fixed-distillation-bayes-opt.py            clean fixed-dataset BO (KNet vs MLP)
fixed-mlp-distillation-kirchhoffnet.py     fixed-distillation training harness
fixed-{knet,plainmlp}-bayes-opt.py         thin wrappers
fixed-mlp-distillation-plainmlp.py         thin wrapper
compare-fixed-distillation-bo.py           KNet-vs-MLP comparison gate
ctle_kirchhoff_bo.py                       older adjoint (dopri5) CTLE BO variant
train_ctle.py                              older 4-phase KD from RegimeAwareMoE (synthetic specs)
spice_failure_check.py                     *** the Cadence/Spectre harness ***
generate_report.py                         synthetic-benchmark HTML report only
run_phase_a_knet_bo.sh                    SLURM: Phase-A KNet BO (24 h)
run_train_knet_single_shot_distillation.sh SLURM: one C-full Phase-A KNet trial (3 h)
augmented-cvae-ctle/                       CTLE dataset (83,344 rows + raw Cadence logs)
docs/ctle-knet-bo-problem-statement.md     the canonical self-contained analysis of this task
docs/specs/spice-failure-spec.md           spec for the Cadence harness
docs/plans/spice-model-loaders.md          how the paper's trial_0011 KNet is loaded for Cadence
docs/specs/fixed-bo-round1.md              spec for the fixed-distillation BO
outputs_alliance_can/                      copied BO summaries + logs (no CTLE checkpoints)
```

**End of report.**
