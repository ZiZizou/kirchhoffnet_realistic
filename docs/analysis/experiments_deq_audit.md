# Experiment-artifact audit: `test_smooth2d_w_tanh`, `friedman2_normal`, `deq`

Audit date: 2026-11-21. Repo: `C:\Users\Atharva\Documents\kirchhoffnet_realistic`
(branch `simplified` @ `e11e2b0`; remotes `origin/main` @ `fefc3ef`, `origin/deq` @ `c463039`).
Artifacts: `C:\Users\Atharva\Downloads\trained_knet\{test_smooth2d_w_tanh,friedman2_normal,deq}`.
Notebooks: `C:\Users\Atharva\Downloads\trained_knet\realizable-kirchhoffnet_{f2_smooth2d,deq_run}.ipynb`.

**Audit-only:** no repository file was created, modified, or deleted. The only files written are this
report and scratch helpers under `tmp/analysis/scratch/` (`inspect_models.py`, `dump_notebooks.py`,
`match_versions.py`, `scan_lambdas.py`, `scan_snapshot_writer.py`, `match_state_dict.py`, `scan2.py`,
`scan3.py`, `scan_deq_branch.py`, `scan_deq_hdr.py`, `scan_mlp.py`, `param_counts.py`,
`history_stats.py`, `connectivity.py`). No training was run; only `torch.load` inspection and
read-only `git show` / `git log` inspection were performed.

---

## 0. Executive summary

| # | Artifact | What it actually is | Code version | Verdict |
|---|---|---|---|---|
| A | `test_smooth2d_w_tanh/` | **5-stage KNet** (Heun, FreeTanh/"tanh_free" cell family, boundary input OTAs, temporal readout, residual interstage transfer) on the **smooth2d / Franke** task | Post-`060fe36` (2026-09-07) `simplified` lineage, **with a locally modified `config.py` + a trimmed snapshot writer** (exact commit **not determined**) | Healthy training; val Huber loss 1.918e-3 @ epoch 599; no instabilities |
| B | `friedman2_normal/` | **Not a KNet and not Friedman-2**: a plain **MLP benchmark** (`mlp_benchmark.py`) on **smooth2d/Franke**, 40×3 ReLU, 1,801 params | `mlp_benchmark.py` at ≈ **`a0e70ef`** (2026-07-27) | Converged; best val MSE 2.899e-3; the directory name is misleading |
| C | `deq/` | **DEQ (deep-equilibrium) KNet**: 3-stage, 5×5 grid, legacy L/S/P/Z cell library, per-stage fixed point `rhs(x*)=0` | `origin/deq` lineage (`deq` branch of the old `kirchhoffnet` repo), i.e. **Jun 21 – Jul 4, 2026**, run on Kaggle via the DEQ notebook | **Did not converge**: forward solver hit max-iters (100) at *every* validation; multistability suspected; only the **last stage** received gradients |

Headline findings for the user's stated goals (Verilog-A/Cadence → GF22, then simplification):

1. The DEQ idea is the right hardware story (a KCL steady state is exactly a DC operating point) but
   **the DEQ artifact is not a usable trained model**: the fixed point never converged, the
   equilibrium was reported as *multistable*, and only stage 2 was actually trained (§5.4, §6).
2. **The DEQ code path is currently broken in the repo**: `deq_solver.py` / `deq_diagnostics.py`
   were deleted in `7b34b60` but `differential_stage.py:2218`, `train_script.py:237` and five call
   sites in `test_smoke.py` still import them → `--solver deq` raises `ModuleNotFoundError`
   at HEAD. `torchdeq` is also not installed in `venv/` (§6.5).
3. Two of the three artifacts are **not the paper's Friedman-2 HR-KNet** (Table I RMSE 1.2718):
   artifact A is Franke/smooth2d, artifact B is an MLP on Franke. The only Friedman-2 HR-KNet
   number in the repo is the BO trial referenced by
   `tmp/analysis/circuit_model_report.md:21` (no trained checkpoint in the tree) (§4.4, §7).

---

## 1. Provenance fingerprinting method

Three independent fingerprints were used, because each artifact was produced by a *different*
code state and two of them cannot be matched to any commit in this repository:

* **F1 — snapshot format.** `config_snapshot.txt` is written by
  `train_script.py:_save_config_snapshot` (HEAD: `train_script.py:726-800`) and by
  `mlp_benchmark.py` (HEAD: `mlp_benchmark.py:344-361`, `486-500`). The set of sections/keys in
  each artifact snapshot was compared against every commit that touches the writer.
* **F2 — `state_dict` key names.** The exact attribute names in `model.pt` were required to exist
  (and, for the DEQ artifact, the *old* names to be absent) in that commit's
  `differential_stage.py` / `io_mapper.py` / `cell_library.py` / `stage_transfer.py`.
* **F3 — config-value fingerprints.** `LAMBDAS`, `OPTIM`, `PRESETS[...]`, `SOLVER`, `DEQ` values
  were extracted by `exec`-ing each commit's `config.py` and compared to the snapshot text.

Scripts: `tmp/analysis/scratch/match_versions.py`, `scan_lambdas.py`, `scan3.py`,
`match_state_dict.py`, `scan2.py`, `scan_snapshot_writer.py`, `scan_deq_branch.py`,
`scan_deq_hdr.py`, `scan_mlp.py`.

---

## 2. Run A — `test_smooth2d_w_tanh/` (5-stage KNet, smooth2d/Franke, Heun)

### 2.1 Task and dataset

`problem: smooth2d` (`config_snapshot.txt:1`). The repo's smooth2d problem is the **Franke
function** on a 2-D grid with Gaussian noise (`mlp_benchmark.py:361` states
`dataset: smooth2d (Franke, 20k train / 4k val, sigma=0.01, normalized)`); the sibling MLP
artifact B uses the same generator. The KNet data generation itself is in `train_script.py`
(`make_data_smooth2d`-family); the noise level used for the KNet run is **not recorded** in the
snapshot (the snapshot's VARIATION block lists context/offset stds, not the target noise) →
**not determined**.

### 2.2 Architecture (from `config_snapshot.txt:2` + `model.pt` keys)

| Item | Value | Evidence |
|---|---|---|
| Stages | **5**, identical | `config_snapshot.txt:2` (5 stage dicts) |
| Hidden topology | `small_world` (Watts–Strogatz), `k=4, p=0.2, seed=100, bidirectional=False` | `config_snapshot.txt:2`; `model.pt` `core.stages.0.src/dst` = 32 edges over hidden nodes 0..7 |
| Hidden nodes | 8 per stage (`num_hidden=8`), `num_proj=0`, `num_outputs=0` | `config_snapshot.txt:2` |
| Edge repeats | `edge_repeats=2` → 16 unique WS edges × 2 = **32 core edges/stage** | `model.pt` `core.stages.N.z_logits` shape (32,) |
| Node state | 9 nodes/stage = 8 hidden + 1 readout accumulator | `model.pt` `core.stages.N.u_logits` shape (9,) |
| Cell library | **FreeTanh / "tanh_free"** family: per-edge `a_raw, b_raw, s_raw, gm_raw, isat_raw, theta_raw, g_resistive_raw` (7×32 = 224/stage) | `cell_library.py:224-436` (`FreeTanhLibrary`); `model.pt` `core.stages.N.cell_lib.*` |
| Boundary (input) OTAs | **4** boundary edges/stage: `boundary_src=[0,0,1,1]`, `boundary_dst=[0,1,4,5]` (input 0 → hidden 0,1; input 1 → hidden 4,5) | `model.pt` `core.stages.N.boundary_src/dst`; `differential_stage.py:1144-1160, 1549-1600` |
| Output (readout) ODE | 8 readout OTAs: `output_ode_src=[0..7]`, `output_ode_dst=[8]*8` (all hidden nodes → the single accumulator node 8) | `model.pt` `core.stages.N.output_ode_src/dst`; `differential_stage.py:1207-1215, 1636-1690` |
| Reference edges | **none** (`ref_dst` empty) | `model.pt` `core.stages.N.ref_dst` |
| Drive OTA | **none** (`_drive_idx` empty; `persistent_drive: False`) | `model.pt`; `config_snapshot.txt:50` |
| Interstage | **4 × StageTransfer**, per-transfer `residual_w1, residual_w2, residual_w3, residual_vth` (9 each) → "residual-relu-tanh" activation | `model.pt` `core.transfers.N.*`; `stage_transfer.py` |
| Readout | `OutputAffine`: `gain, bias, tanh_gain, relu_gain, relu_threshold` (1 each) → temporal readout `gain·x + bias + tg·tanh(x) + rg·relu(x−thr)` | `model.pt` `output_mapper.*`; `io_mapper.py:605-616` (per `circuit_model_report.md:66`) |
| Solver | **fixed-step Heun**, `t_span = 1.0`, `num_steps = 10` per stage | `config_snapshot.txt:2` |
| Loss | **Huber** | `config_snapshot.txt:2` |
| VCA | **off** (no `vca_W` / `vca_v*` keys) | `model.pt` |
| GLN | **off** (no `gln_rails.*` keys) | `model.pt` |
| Trainable params | **1,954** float values (5 × 361 + 4 × 36 + 5) | `tmp/analysis/scratch/param_counts.py` |

Per-stage parameter split (from `param_counts.py`): `cell_lib` 224 + `boundary_cell_lib` 28 +
`output_ode_cell_lib` 56 + `z_logits` 32 + `u_logits` 9 + `boundary_z_logits` 4 +
`output_ode_z_logits` 8 = 361; transfers 36 each; `output_mapper` 5.

### 2.3 Training settings (`config_snapshot.txt:26-36`)

AdamW-style: `weight_decay 1e-4`, `grad_clip_norm 5.0`, `batch_size 4096`, `lr 1.2e-3`
(= `3e-4 × 4096/1024`, the config's batch-scaled LR rule), `epochs 800` configured but the run
**stopped at 600** (loss history has epochs 0..599), `reg_warmup 100`, `reg_anneal 50`,
cosine scheduler `T_0=50, T_mult=1, eta_min=1e-5`.
Lambdas (`:15-24`): `sparsity 0, rail 0.1, edge_gate 0, node_gate 0, power 0, capacitance 0,
entropy 1e-4, tanh_sat 0.0, skip_linear_l2 1e-4` → **no structural regularisation at all**;
this run is a pure function-approximation fit (no sparsity/edge-gate/power pressure), i.e. the
"Phase A fit" of the schedule, not a pruned/BO'd model.
I/O mapping (`:46-62`): `input_mapper: NullInputMapper` (no net-level input parameters — the
write is done by the stage's boundary OTAs), `output_mapper: OutputAffine`, `write_idx: None`,
`read_idx: None`, `hid_count 8`, `proj_count 0`, `stage_lr_scale 1.0`, `mapper_lr_scale 1.0`,
`struct_lr_scale 4.0`, `dyn_lr_scale 1.0`, `retrain_stage_lr_scale 1.0`,
`retrain_mapper_lr_scale 0.1`, `freeze_mappers False`.

### 2.4 Final metrics and stability

* Loss (`loss_history.txt`, 600 rows, Huber): train 6.825e-1 → **2.629e-3**, val 6.465e-1 →
  **1.918e-3** (best at epoch 599). No NaN/Inf in any column
  (`tmp/analysis/scratch/history_stats.py`).
* Gradient norms (`grad_norms.txt`, 60 sampled epochs): all finite, max `out_mapper 6.56e-1`,
  `stage0_device_param 2.30e-1`, `stage4_z_logits 4.78e-3`. Transient spikes in the first ~100
  epochs (e.g. `stage4_z_logits` 3.7e-2 @ epoch 70) then monotone decay → **benign early
  transient, no instability**.
* Update norms (`update_norms.txt`): `mapper_param` ≈ 1.21, `struct_param` grows 3.0e-1 → 18.63
  (logistic logits growing, as expected), `dyn_param` **identically 0.0** (the dynamic-leak
  parameter group is empty/dead in this configuration), `other_param` ≈ 51.8 constant.
* `grad_norms.txt` header shows `stageN_device_param` (aggregate of the cell-library device
  parameters, `train_script.py:1890-1944`), `stageN_u_logits` (always `-`, i.e. the deprecated
  node-gate group receives no gradient — `differential_stage.py:2310-2314` documents node gates
  as deprecated), `stageN_z_logits`, `stage_transfer` (twice), and `lr0..lr2` (three param
  groups).

**Comparison with the paper's Table I:** *not applicable*. Table I
(`tmp/analysis/paper_raw.txt:825-915`) reports **Franke's Function**: HR-KNet 8,088 params →
RMSE 0.0018, MLP 9,937 → 0.0036, LightGBM 9,396 → 0.0059, CatBoost 10,080 → 0.0144, XGBoost
1,100 → 0.1020; models trained with Huber loss, 400 epochs. Artifact A is a Franke/smooth2d
KNet, so it is the right *task*, but: (i) it has 1,954 params, not 8,088; (ii) it reports a
**Huber loss (1.918e-3)**, not an RMSE, so a direct numeric comparison with 0.0018 RMSE is
**not determined** without re-evaluating the checkpoint with the paper's metric; (iii) it was
trained 600 epochs, not 400; (iv) it has **no** sparsity/power regularisation, so it is not the
budget-matched model the paper tabulates.

### 2.5 Which code version produced it?

**Most likely:** the `simplified` lineage **after `060fe36` (2026-09-07)** and at or before
`e11e2b0` (2026-09-19), i.e. the window in which `differential_stage.py` simultaneously contains
`g_resistive_raw`, `boundary_cell_lib`, `output_ode_cell_lib`, `z_logits`, `u_logits`,
`ref_dst`, `boundary_z_logits` **and** `io_mapper.py` contains `NullInputMapper`, `OutputAffine`
and `stage_transfer.py` contains `residual_w1/w3/vth`
(`tmp/analysis/scratch/match_state_dict.py` → 38 commits match, all dated 2026-09-07…09-19).

**But the exact commit is not determined**, because two fingerprints cannot be satisfied by any
commit in this repository:

1. The snapshot's `preset:` line is a **5-stage small_world smooth2d preset**
   (`t_span 1.0`, `num_steps 10`, `edge_repeats 2`, `write_mode/read_mode/leak_mode` keys). Every
   `config.py` in the repo defines `PRESETS['smooth2d']` as a **single-stage** `line`-family
   preset with `t_span 7.0 / num_steps 70` (`scan3.py` → `nSt=1` for all 265 commits), and
   `train_script.py` writes `f.write(f"preset: {PRESETS[problem]}\n\n")` in every commit from
   2026-07-10 to HEAD. ⇒ the run used a **locally edited `config.py`** that redefines the
   smooth2d preset (the same trick the notebooks use, `%%writefile .../config.py`).
2. The snapshot's `LAMBDAS (global)` block is `sparsity 0 … power 0`. In the repo, `sparsity 0`
   occurs **only at HEAD `e11e2b0`** (where `power = 1e-6`), and `power 0` occurs **nowhere**
   (`scan_lambdas.py`, `scan3.py`). ⇒ `config.py` was locally edited again.
3. The snapshot's I/O-MAPPING block has `persistent_drive` + `drive_mode` but **not**
   `struct_lr_scale` / `dyn_lr_scale`, whereas every repo commit from `1d1710a` (2026-07-16)
   onward writes all four (`scan_snapshot_writer.py`, `scan2.py`). ⇒ the snapshot writer was also
   locally modified (or the run used an unpublished state of the GitHub repo).

The **model semantics** (FreeTanh per-edge device params, boundary input OTAs, output-ODE
temporal readout, residual interstage transfer, `NullInputMapper` write path) pin it to the
**Jul–Sep 2026 "simplified" rework**, i.e. *after* the "simplified" rewrite (`7b34b60`,
2026-07-04) and *before* the HEAD state. Closest single commit by feature set: **`060fe36`
(2026-09-07, "Added new less costly parameter readout option")** or the commits immediately
after it — but with the caveats above, "exact commit: **not determined**".

---

## 3. Run B — `friedman2_normal/` (an MLP benchmark, *not* a KNet, *not* Friedman-2)

### 3.1 What the artifacts actually say

`config_snapshot.txt`:

```
model: MLPRegressor(2 -> 40 x 3 -> 1, activation=relu)
param_count: 1801
num_layers: 3
activation: relu
loss: mse
epochs: 600
lr: 0.0012
weight_decay: 0.0001
batch_size: 4096
grad_clip_norm: 5.0
patience: 500
min_delta: 0.0001
validate_every: 5
seed: 42
device: cuda
dataset: smooth2d (Franke, 20k train / 4k val, sigma=0.01, normalized)
```

`final_metrics.txt`:

```
param_count: 1801
loss: mse
best_val: 0.002899
best_epoch: 599
final_val: 0.002899
epochs_run: 600
elapsed_seconds: 117.47
```

**The directory name is wrong twice over**: the model is a *plain MLP* (no KirchhoffNet, no
analog ODE, no cells), and the dataset is *smooth2d/Franke*, not Friedman-2. The only
Friedman-2-related artefacts in the repo are `mlp_benchmark_friedman2.py` /
`kn_benchmark_friedman2.py` and the BO trials under `outputs/kn_bayes_opt/`
(`circuit_model_report.md:21`).

### 3.2 Architecture and parameters

`model.pt` is an `OrderedDict` with **aliased duplicates**: `layers.0/2/4.{weight,bias}` and
`fc1/fc2.{weight,bias}` are the *same tensors* (`fc1.weight` ≡ `layers.0.weight` numerically and
by identity). Unique parameters: `2→40` (120), `40→40` (1,640), `40→1` (41) = **1,801**
(matches `param_count`), ReLU activations, single output.
Value ranges (`tmp/analysis/scratch/inspect_models.py`): `layers.0.weight` ∈ [-0.947, 1.783],
`layers.2.weight` ∈ [-2.005, 1.517], `layers.4.weight` ∈ [-1.109, 0.894],
`layers.4.bias` = 0.0813 (i.e. essentially the Franke mean), `fc1.*` ≡ `layers.0.*`.

### 3.3 Training and results

600 epochs, batch 4096, lr 1.2e-3, wd 1e-4, grad-clip 5.0, patience 500, validate every 5,
seed 42, CUDA, 117.47 s wall-clock (≈5 batches/epoch → very fast).
`loss_history.txt` (601 lines): train 0.9776 → 0.00404, val 0.9359 → 0.002899. Smooth,
monotone, no spikes or non-finite values.

**Comparison with the paper:** Table I's *Franke* MLP row is **RMSE 0.0036** at 9,937 params
(`paper_raw.txt:898-914`), trained with **Huber** loss for 400 epochs. This artifact is an
MSE-trained, 1,801-param MLP for 600 epochs with **best val MSE 0.002899 → RMSE 0.0538 in the
normalised target units** — i.e. numerically *much worse* than the paper's MLP row if the two
RMSEs are on the same scale, but the scales are **not established** (this artifact predates the
`best_rmse_orig`/`final_rmse_orig` fields that HEAD's `mlp_benchmark.py:492-499` writes, so the
original-units RMSE was never recorded) ⇒ **direct comparison not determined**.
The paper's **Friedman-2** row (HR-KNet 1,069 params → RMSE **1.2718**; MLP 6,763 → 1.4310)
is **not addressed by this artifact at all**.

### 3.4 Which code version?

`mlp_benchmark.py` at commit **`a0e70ef` (2026-07-27, "changed smooth2d from mse to huber")** or
a state immediately adjacent to it:
* its `config_snapshot.txt` writer emits the **exact 16 lines** of this artifact, including
  `dataset: smooth2d (Franke, 20k train / 4k val, sigma=0.01, normalized)`;
* its `final_metrics.txt` writer emits exactly the **7 keys** of this artifact
  (`param_count, loss, best_val, best_epoch, final_val, epochs_run, elapsed_seconds`) — the
  `*_orig` fields were added in the *next* commit `ef5b38db` (2026-07-27);
* the artifacts also contain `loss_curve.png` and `output_fit.png`, matching the script's
  documented outputs (`mlp_benchmark.py:19`).

CLI equivalent: `mlp_benchmark.py --hidden-dim 40 --num-layers 3 --epochs 600 --loss mse
--seed 42 --patience 500 --validate-every 5 --activation relu --lr 0.0012
--weight-decay 0.0001 --batch-size 4096 --grad-clip 5.0`.

---

## 4. Run C — `deq/` (deep-equilibrium KNet)

### 4.1 Task, architecture

`config_snapshot.txt:1-2`: `problem: smooth2d_grid`; 3 stages, each
`num_inputs 2, num_hidden 25, num_proj 3, hidden_family 'grid' (5×5, kernel_size 3,
bidirectional False), edge_repeats 2, t_span 1.6666666666666667, num_steps 17`; write mode
`fan_out` with `write_fan_out {0:[0,10,20], 1:[4,14,24]}`; `read_idx [2,7,12,17,22,25,26,27]`;
`loss mse`; `schedule 'three_phase'`; preset lambdas `sparsity 1e-5, edge_gate 5e-6,
node_gate 0, power 1e-5, capacitance 0, rail 0.1`; `tau_anneal True`.

Those numbers are an **exact fingerprint of the DEQ notebook's embedded `config.py`**
(`make_smooth2d_grid_preset(5)`: `t_span = SOLVER.t_span/3 = 5.0/3 = 1.6667`,
`num_steps = round(50/3) = 17`, fan-out rows `[0,2,4]` → `{0:[0,10,20], 1:[4,14,24]}`,
`read_idx = centre column [2,7,12,17,22] + proj [25,26,27]`) — see
`tmp/analysis/scratch/nb_deq_run_code.txt:752-838`.

`model.pt` (64 entries, **4,269 float values**) confirms an **old-style** fabric:

| Item | Value | Evidence |
|---|---|---|
| Nodes/stage | 28 = 25 hidden + 3 proj | `core.stages.N.u_logits` (28,) |
| Core edges/stage | **219** | `core.stages.N.src/dst` (219,) |
| Cell library | **legacy L/S/P/Z (4 cells) shared, per-edge softmax mixing**: `logits` (219,4) | `model.pt` `core.stages.N.logits` (219,4) |
| Library constants | `gm [0.2,1,1,0]`, `isat [10,0.5,1,0]`, `gleak [0.01,0.01,0,0]`, `rho [1,1,1,0]`, `beta [1,1,0.1,1]`, `cell_type_code [0,0,1,0]`, `_is_rect [F,F,T,F]` | `model.pt` `core.stages.N.cell_lib.*` — **bit-identical to the notebook's `CELL_L/CELL_S/CELL_P/CELL_Z`** (`nb_deq_run_code.txt:79-121`) ⇒ *the notebook's `config.py` was the one in force* |
| Edge gate | `z_logits` (219,) | value −0.777848 at init for all edges/stages 0-1 |
| Edge multiplier | `raw_mult` (219,) | −0.667722 at init |
| Node leak | `raw_leak` (28,) | −2.99964 (init) for stages 0-1 |
| Input write | `input_mapper.gain/bias` (6,) + 3 × `drive_mappers.N.gain/bias` (6,) → persistent drive on the 6 fan-out nodes `[0,4,10,14,20,24]` | `model.pt`; `core.stages.N._drive_idx` (6,) |
| Readout | `output_mapper.proj.weight (1,8)`, `proj.bias (1,)` → linear read of the 8 `read_idx` nodes | `model.pt` |
| VCA / GLN | **off** (no such keys) | `model.pt` |
| Boundary / readout OTAs | **none** (no `boundary_cell_lib`, no `output_ode_cell_lib`) — pre-dates those features | `model.pt` |

### 4.2 Training settings

Command actually executed (DEQ notebook, last code cell,
`tmp/analysis/scratch/nb_deq_run_code.txt:947`):

```
!python3 ./kirchhoffnet/train_script.py --problem smooth2d_grid --solver deq --epochs 300 \
  --output /kaggle/working/test/ --persistent-drive --no-parallel --grad-log --grad-log-every 10 \
  --validate-every 10 --run-deq-diagnostics --deq-f-max-iter 100 --deq-f-tol 1e-5 --deq-step 0.05 \
  --leak-floor 0.05 --grid-size 5 --num-stages 3 --mapper-lr-scale 1 --stage-lr-scale 1 \
  --schedule legacy
```

i.e. **300 epochs**, batch 2048, lr 6e-4 (config default, `lr = 3e-4 × 2048/1024`), wd 1e-4,
grad-clip 5.0, tau annealing 1.0 → 0.10 over T_0 = 80 epochs, cosine LR to 1e-5, regulariser
warm-up from epoch 100 (`reg=0.02 … 1.00` in the notebook log), **legacy schedule** (phase column
is empty for all 300 epochs in `loss_history.txt`, i.e. **Phase A only — no compression/pruning
phase ever ran**). AMP + `torch.compile` enabled (notebook banner:
`amp=True compile=True parallel=False (2 GPUs)`).

### 4.3 Did it train successfully? **No.**

`deq_validation_history.txt` (31 validation rows, epochs 0…299):

| Quantity | Value | Reading |
|---|---|---|
| `nstep_mean`, `nstep_max` | **100.000000 at every single validation** | the forward fixed-point solver **never converged** — it exhausted `f_max_iter = 100` every time (tolerance 1e-5 never met) |
| `residual_mean` | 8.63e-4 → 1.06e-3 (rising) | `residual = deq_step · ‖rhs‖` (`train_script.py:1531`); with `deq_step = 0.05` ⇒ ‖rhs‖ ≈ 1.7e-2…2.1e-2 in model current units — **not a converged KCL solution** |
| `residual_max` | 3.13e-3 → 6.39e-3 | worst-node residual is ~2× the mean |
| `max_abs_state` | 0.796 → **1.502** (monotone rise) | the "equilibrium" magnitude drifted ~2× during training — a converged DC point should be stationary |
| train / val loss | 2.244 → 0.8561 / 2.165 → **0.8185** (best epoch 260) | loss **plateaus** at 0.818; ~430× worse than artifact A's Huber loss on the same family of task |
| `mapper_grad_norm` | 2.98 → 0.40 | decaying |
| `stage_logits_grad_norm` / `z_logits_grad_norm` | 1.6e-2 → 3.8e-3 / 7.7e-3 → 1.75e-3 | small but non-zero |

Notebook diagnostics printed at the end of the run
(`tmp/analysis/scratch/nb_deq_run_outputs.txt:419-431`):

```
[deq-diagnostics] gradient norm comparison (BPTT vs DEQ)
  z_logits_heun: 2.608e-02   z_logits_deq: 5.709e-03   z_logits_ratio: 2.189e-01
  logits_heun:   7.279e-02   logits_deq:   2.551e-02   logits_ratio:   3.505e-01
[deq-diagnostics] jacobian cond at equilibrium
  cond(J) = 2.188e+00 (well-conditioned)
[deq-diagnostics] multistart uniqueness
  max pairwise diff = 1.616e+00 (multistability suspected)
[train] restored best model from epoch 260 (val=0.8185)
```

So: **DEQ gradients were ~3–5× *smaller* than BPTT gradients** (the opposite of the
"2–4 orders of magnitude larger" hypothesis stated in `deq_diagnostics.py:11-13`), the Jacobian is
well-conditioned, and the **multistart probe flagged multistability** (4 different initial guesses
→ states differing by 1.62, i.e. the solver does not land on one common fixed point).

### 4.4 The decisive defect: only the **last stage** was trained

`tmp/analysis/scratch/param_counts.py` compares the three stages element-wise:

```
cell_lib.* (all)        stage0==stage1: True    stage0==stage2: True
logits                  stage0==stage1: True    stage0==stage2: False
raw_leak                stage0==stage1: True    stage0==stage2: False
raw_mult                stage0==stage1: True    stage0==stage2: False
z_logits                stage0==stage1: True    stage0==stage2: False
u_logits / raw_drive_g  stage0==stage1: True    stage0==stage2: True
```

* All three stages were **initialised identically** (stage 0 ≡ stage 1 bit-for-bit), and
  **stage 0 and stage 1 are still bit-identical to their initialisation** after 300 epochs
  (`raw_leak ≡ -2.99964 = INIT raw_leak_init = -3.0`; `z_logits ≡ -0.777848`; `logits` identical
  on every one of the 219 edges).
* **Only stage 2** (and `output_mapper.proj`) moved.
* `input_mapper.gain ≡ 1.0` and all three `drive_mappers.*.gain ≡ 0.999909` are also still at
  their initial values.
* `grad_norms.txt` in this run contains **only dashes** for every stage — the per-parameter
  gradient log was empty for all three stages (header lists `stage0_logits, stage0_raw_leak,
  stage0_raw_mult, stage0_u_logits, stage0_z_logits, …`).

**Mechanism (most likely, not proven in-repo):** the DEQ backward is torchdeq's
*phantom-gradient / implicit-function* backward. `deq_solver._solve_torchdeq` calls
`torchdeq.get_deq(...)` and returns the converged iterate; torchdeq re-evaluates `phi` at the
fixed point with the **initial guess detached**, so the gradient w.r.t. `x0` is injected as a
detached constant rather than propagated through autograd. Since `KirchhoffNet.forward` chains
stages as `x = stage(x0=x)` (`origin/deq:kirchhoff_net.py`, stage loop) and the stage itself
returns `x_star` without detaching (`origin/deq:differential_stage.py`, `solver == "deq"`
branch), the *only* way stage 0/1 parameters can receive gradient is through `x0` — which
torchdeq severs. Result: **the DEQ training signal reaches only the last stage**; the rest of
the fabric stays at initialisation. This exactly matches the saved checkpoint. (A second,
non-exclusive explanation — a detach inside the old stage/transfer code — cannot be excluded
without running the old tree; **the precise mechanism is not determined**.)

### 4.5 Which code version produced it?

**Fingerprints:** the old cell-library symbols (`raw_drive_g`, `cell_type_code`, `_is_rect`,
`raw_mult`, `raw_leak`) are present and the new ones (`boundary_cell_lib`,
`output_ode_cell_lib`, `g_resistive_raw`) are absent in exactly **52 commits**, spanning
**2026-06-18 (`3db50bc`) → 2026-07-04 (`c463039`)** — i.e. precisely the **`origin/deq` branch**
(`git log --all --oneline origin/deq`; `origin/deq` tip = `c463039` "Added friedman presets to
train_script.py", merge-base with `simplified` = `2dae9fd`).
Other matches: `deq_validation_history.txt` was introduced in `bf66c4c` (2026-06-22) and its
**14-column header is byte-identical** to the artifact's
(`epoch, phase, split, train, val, residual_mean, residual_max, nstep_mean, nstep_max,
max_abs_state, mapper_grad_norm, stage_logits_grad_norm, z_logits_grad_norm, probe_loss` —
`train_script.py:1634-1638`, `origin/deq:train_script.py:1664+`); the rows carry an **empty
phase field** and `split='val'`, consistent with `--schedule legacy`.
The DEQ notebook **overwrote `config.py` in the cloned repo** (cell 4:
`%%writefile /kaggle/working/kirchhoffnet/config.py`), and that file is the fingerprint that
matches the artifact exactly (legacy/v15/v2 libraries, no `tanh_free`, `DEQ` dict with
`f_max_iter 30, f_tol 1e-4, b_max_iter 20, anderson_m 5, deq_step 0.1, leak_floor 0.05`,
`SOLVER {t_span 5.0, num_steps 50}`, `make_smooth2d_grid_preset`).
⇒ **Origin: the DEQ notebook, executed on Kaggle, cloning `branch="deq"` of
`github.com/ZiZizou/kirchhoffnet` (the predecessor of `kirchhoffnet_realistic`), with the
notebook's own `config.py`; code state ≈ `origin/deq` (Jun 21 – Jul 4, 2026), most plausibly
`c463039`/`2dae9fd`.** The run itself is *not* reproducible from this repo alone because the
`deq` branch's `config.py` was replaced in-place by the notebook.

---

## 5. The DEQ formulation and its hardware meaning

### 5.1 What fixed point is solved

`DifferentialStage.forward_equilibrium` (`differential_stage.py:2188-2304`; identical logic in
`origin/deq:differential_stage.py`) solves, **per stage**:

```
x*  such that   rhs(x*) = 0
```

where `rhs` is the stage's node-capacitor KCL residual, `C_eff · dx/dt = I_KCL(x, u, params)`
(`differential_stage.py` `rhs`; `C_eff = 1.0` per node, `config.py:305`). **So the fixed point is
precisely the DC operating point of that stage's node network** (dV/dt = 0 at every node) — the
same algebraic system a Spectre `dcop` solves for the corresponding transistor sub-circuit. This
is the attractive part of the DEQ idea: *a DEQ fabric is a static analog network whose answer is
its DC solution*; no transient accuracy is required, only a well-posed DC solution. It also
matches the paper's framing of the fabric as a "physical" computation.

### 5.2 How it is solved

* Fixed-point map: `Φ(x) = x + dt·rhs(x)` with `dt = deq_step` (config default 0.1; the run
  used **0.05**) — `differential_stage.py:2278-2286`.
* Backend: `torchdeq.get_deq(f_solver='anderson', b_solver='anderson', f_max_iter, f_tol,
  b_max_iter, anderson_m)` — `deq_solver.py` (recoverable with `git show 0e3ae20:deq_solver.py`).
  The notebook installed `torchdeq==0.1.0` (cell 0), so the **torchdeq backend was used**.
* A **fallback** backend `fixed_point_iter` exists; it has **no implicit backward**
  (`deq_solver.py` docstring) — its loop leaves the autograd graph attached, i.e. it degenerates
  into an unrolled BPTT through up to `f_max_iter` iterations rather than an implicit
  gradient. (Discrepancy between the docstring and the code is noted; behaviour in the fallback
  is **not determined** because torchdeq was installed in the artifact run.)
* **Leak floor.** During the DEQ solve the per-node leak is forced to ≥ `leak_floor` (0.05) so
  that `Φ` is contractive (`differential_stage.py:2276`, restored in the `finally` block at
  `:2301-2304`). This is an **artificial conductance inserted into the physics**: the "DC point"
  the DEQ trains against is the DC point of a circuit with a different leak than the trained
  device has. (In this run `leak_constant = softplus(-3) = 0.0486 ≈ 0.05`, so the floor roughly
  matched the nominal leak — but stage 2's *learned* `raw_leak` fell to −3.59
  ⇒ softplus ≈ 0.027, i.e. **below the floor**, so the floor overrode the learned value during
  the solve.)
* Solve runs in fp32 (autocast disabled) — `deq_solver.py::_solve_torchdeq`.

### 5.3 How gradients are computed

**Implicit differentiation (IFT), not unrolling**, via torchdeq's `b_solver='anderson'` +
`b_max_iter`; `deq_solver.py` documents it as "implicit / **phantom-gradient** backward". The
Heun path, by contrast, backpropagates through the unrolled ODE steps.

*Evidence that the implicit backward did not propagate across stage boundaries:* §4.4 — stages 0
and 1 are bit-identical to their initialisation after 300 epochs while stage 2 moved.

### 5.4 Convergence / Jacobian / stability checking — what exists and what does not

| Check | Present? | Where | Outcome in this run |
|---|---|---|---|
| Iteration count vs. budget | yes | torchdeq `info['nstep']`, logged as `nstep_mean/nstep_max` (`train_script.py:1537-1542`) | **always 100 = `f_max_iter`** ⇒ never converged (`f_tol = 1e-5`) |
| Residual ‖rhs‖ | yes | `train_script.py:1525-1534` recomputes `dt·rhs` at the returned state | 8.6e-4 → 1.1e-3 (mean), up to 6.4e-3 (max) — **not** a solved KCL |
| Jacobian **conditioning** `cond(J)`, `J = ∂rhs/∂x` | yes | `deq_diagnostics.estimate_jacobian_cond` (SVD ratio, `git show 0e3ae20:deq_diagnostics.py`) | `cond(J) = 2.19` → "well-conditioned" |
| Jacobian **spectral radius** / eigenvalues | **NO** | grep `eigval|eigvals|eigen` over `differential_stage.py`, `train_script.py`, `kirchhoff_net.py`, `config.py` → **no matches** | not checked |
| Contractivity of `I − dt·J` (ρ < 1) | **NO** (only assumed via the leak floor) | — | not checked |
| Physical DC stability (Re λ(∂rhs/∂x) < 0) | **NO** | — | not checked |
| Uniqueness / multistart | yes | `deq_diagnostics.multistart_uniqueness` (starts −1, 0, +1, +5) | **max pairwise diff 1.616 → "multistability suspected"** |

(The repository *does* contain spectral-radius machinery — `narma_spectral_norm.py`,
`narma_linear_controls.py:856` `measure_spectral_radius`, `narma_phase7_leak_bands.py` — but it is
wired to the **NARMA probe**, not to the DEQ path.)

### 5.5 Would this match what Spectre actually does?

**Conceptually yes; empirically no — and three specific gaps exist.**

1. **Same algebra, different solver.** Spectre's `dcop` solves the *whole* netlist's KCL with
   Newton/gmin-stepping; the DEQ here solves each stage **sequentially** and passes `x*` to the
   next stage. That is only equivalent if the interstage coupling is strictly one-directional —
   which it is *by construction* here (stage *s+1* is driven by stage *s*'s settled output) but
   the interstage element itself is a **learned nonlinear re-write**
   (`w1·x + w2·tanh(x) + w3·relu(x−vth)`, `stage_transfer.py`), i.e. **not a physical DC
   element**. It requires the previous stage's *state* (a transient quantity) to be present at
   readout time. In silicon that is a sample-and-hold/latch, not a static wire — so the "DEQ
   fabric = static DC network" story only holds if the interstage transfer is also realised as
   memory (which the current model does not make).
2. **Uniqueness is exactly the issue Spectre has to resolve.** The multistart probe says the
   fixed point is **not unique** (spread 1.62 ≫ 1e-3 threshold). A circuit with several DC
   solutions is the classic "which branch does the simulator pick?" problem: Spectre picks one
   (typically via gmin stepping / homotopy) and the answer is **not guaranteed to be the branch
   the DEQ solver converged to**, which is exactly what a silicon-verified design must not
   depend on. ⇒ the DEQ-trained function would **not be reproducible in a real simulator or in
   silicon** unless uniqueness is enforced (e.g. by a monotone/large-leak design or by
   constraining the Jacobian). The artifact run gives **no evidence** either way beyond the
   failed multistart probe.
3. **The trained model is not the artifact that was measured.** Because only stage 2 was trained
   (§4.4) and the solver never converged (§4.3), this DEQ checkpoint is **not** a usable
   "DC-operating-point network"; it cannot support the hardware claim.

What the DEQ formulation *does* buy for the user's Verilog-A/GF22 goal, stated precisely:
a DEQ fabric can be written in Verilog-A as a **static nonlinear system** whose evaluation is a
DC solve (Spectre `dcop`/`tran` with a DC-stable settle) rather than a time integration of the
ODE — i.e. the Verilog-A block does **not** have to reproduce 10 Heun steps of a stiff ODE, and
the silicon only has to *settle*, not reproduce a trajectory. That is a genuine simplification
lever, but it is currently unsupported by any trained model.

### 5.6 Current state of the DEQ code in the repo (actionable)

* `deq_solver.py` and `deq_diagnostics.py` were **deleted** in `7b34b60` ("Removed cluster,
  legacy, v1.5, v2 cell families and deq", 2026-07-04) — but the *callers* survived:
  `differential_stage.py:2218` (`from deq_solver import solve_equilibrium`),
  `train_script.py:237` (`from deq_diagnostics import …`), and `test_smoke.py:6959, 6981, 7465,
  7507, 7533`.
  Verified in this session: `importlib.util.find_spec('deq_solver')` → **None**;
  `find_spec('deq_diagnostics')` → **None**. ⇒ **`--solver deq` and `--run-deq-diagnostics`
  raise `ModuleNotFoundError` at HEAD**, and the DEQ unit tests (`DEQ-1…DEQ-15`, `test_smoke.py:3168-3185`)
  cannot import.
* `torchdeq` is **not installed** in `venv/` (verified: `find_spec('torchdeq')` → None), so even
  after restoring the modules the torchdeq backend would fall back to `fixed_point_iter`, which
  does **not** implement the implicit backward the DEQ formulation depends on.
* The last DEQ-related commit on this branch is `e59db56` ("minor fix for deq path", 2026-09-16),
  which only adds a tied-tap gate to the DEQ `phi` closure
  (`differential_stage.py:2233-2237`) — it does **not** fix the missing modules.

---

## 6. The notebooks vs. the repository scripts

Both notebooks are **Kaggle drivers**, not self-contained re-implementations of the model:

### 6.1 `realizable-kirchhoffnet_deq_run.ipynb` (6 code cells)

* cell 0: `!pip install torchdeq numpy matplotlib joblib torch` → installs **torchdeq 0.1.0**
  (visible in the saved output).
* cell 2: clones **`branch = "deq"`** of `github.com/ZiZizou/kirchhoffnet` (the *predecessor*
  repo; `kirchhoffnet_realistic` is the current name) with a Kaggle secret token, and prepends
  `/kaggle/working/kirchhoffnet` to `sys.path`.
* cell 4: **`%%writefile /kaggle/working/kirchhoffnet/config.py`** — overwrites the cloned
  repo's `config.py` with a **complete, self-contained copy** of an older `config.py`
  (legacy/v15/v2 cell libraries, `PHYS`, `OPTIM` (batch 2048, lr 6e-4), `TAU`, `LAMBDAS`
  (7 keys), `PRUNE`, `SCHEDULE_THREE_PHASE`/`FOUR_PHASE`, `SOLVER {t_span 5.0, num_steps 50}`,
  `DEQ`, `DEGREE_BUDGET`, `DRIVE`, `INIT`, `VARIATION` (4 keys), `PRESETS` incl.
  `make_smooth2d_grid_preset`). **This is what produced the `deq/` artifact** (§4.5): the
  library constants in `model.pt` are bit-identical to this file's `CELL_L/CELL_S/CELL_P/CELL_Z`.
* cell 5: the `train_script.py --solver deq …` command quoted in §4.2.

**Differences from the repo's own scripts:** the notebook *pins* the configuration by
overwriting `config.py` (so the run is not reproducible from the repo alone); it uses the
**legacy** cell library and the **pre-rewrite** codebase (no boundary OTAs, no output-ODE
readout, no per-edge device parameters, no VCA/GLN, `t_span 5.0/num_steps 50`); it forces
`--schedule legacy` (Phase A only, no pruning); and it drives a *dynamic* `train_script.py`
rather than the repo's current CLI.

### 6.2 `realizable-kirchhoffnet_f2_smooth2d.ipynb` (44 cells, 42 code)

* cell 0: `!pip install torchdeq numpy matplotlib joblib torch torchdiffeq optuna`.
* cell 2: clones **`branch = "simplified"`** of `github.com/ZiZizou/kirchhoffnet_realistic`.
* cell 4: **`%%writefile /kaggle/working/kirchhoffnet_realistic/config.py`** — a **newer**
  copy of `config.py` than the DEQ notebook's: it *does* contain the `tanh_free` family
  (`:314`), `REF`, `VCA`, `DEGREE_BUDGET`, `OPTIM` with `batch_size 4096 / lr 1.2e-3`,
  `LAMBDAS` with `tanh_sat` and `skip_linear_l2` (**`sparsity 1e-6`, `power 1e-6`** — note:
  *not* the `sparsity 0 / power 0` of artifact A), `SOLVER {t_span 7.0, num_steps 70}`, the same
  `DEQ` block, and `make_smooth2d_grid_preset(grid_size=7)`.
* cells 6-8, 10+: the **only executed** cell is **cell 9**, which runs
  `dagger-mlp-distillation-kirchhoffnet.py` ("Fixed MLP → KNet distillation", 110 epochs,
  teacher = `dagger_student_plain.pt`, W=48 L=3 SiLU) and prints the per-component parameter
  breakdown (6,775 trainable: per stage `cell_lib 392, z_logits 56, u_logits 21,
  boundary_cell_lib 56, boundary_z_logits 8, output_ode_cell_lib 686, output_ode_z_logits 98,
  vca_W 16, vca_v 324`) and a **ZIG-isolation verdict: PASS** (teacher failure 6.89 % vs student
  9.04 %, gap 2.15 pp < 3.00 pp tolerance) with `val logit_mse 7.30e-4`.
* the remaining cells hold a **commented-out catalogue of candidate commands**, including
  `--problem smooth2d … --hidden-family small_world --num-hidden 8 --num-stages 5
  --interstage-activation residual-relu-tanh --small-world-k 6 --boundary-fan-out
  '{"0": [0,1], "1": [4,5]}' --temporal-readout --freeze-read --leak non-programmable`
  (line 1403) and the `--vca / --vca-core / --vca-separate-core-bus` variants
  (lines 1421-1436), plus the `--problem friedman2/friedman1/housing` sweeps and the
  `mlp_benchmark*.py` sweeps (lines 1442-1466).

**Key differences vs. the repo's scripts:** the notebook **overwrites `config.py`** (so all runs
inherit one pinned configuration), it clones a *branch* at run time (so the "code version" is
whatever GitHub served), and the smooth2d/Friedman sweeps it lists are **commented out** — the
executed cell is the *distillation* experiment. The KNet `smooth2d` sweep that produced
artifact A (`--small-world-k 4`, dense write/read, `write_mode/read_mode dense`) is **not**
among the executed cells, and the notebook's own `config.py` differs from artifact A's config
in `LAMBDAS` ⇒ **artifact A was not produced by this notebook** (it has a different, locally
edited config).

---

## 7. Cross-cutting observations for the planned simplification

1. **The three artifacts are three different code worlds** (2026-06-18→07-04 `deq` branch;
   2026-07-27 `mlp_benchmark`; 2026-09-07→09-19 `simplified` + local edits). Any "reproduce
   this" exercise must first pin which tree a given artifact belongs to; two of the three are
   **not reproducible from this repository** without the notebook's embedded `config.py`.
2. **The DEQ path is currently dead code** (§5.6): missing modules, missing dependency, and a
   phantom-gradient that (empirically) prevents gradient flow past the last stage. Either
   restore it properly (`git show 0e3ae20:deq_solver.py`, `git show 0e3ae20:deq_diagnostics.py`,
   add `torchdeq` to `venv/`) or delete the remaining callers — at HEAD both the CTLE path
   (`circuit_model_report.md:40`: "DEQ option exists but is not used for CTLE") and the DEQ
   tests are inconsistent with each other.
3. **For the Verilog-A/GF22 goal, the Heun path (artifact A) is the only trained, healthy
   multi-stage fabric in this set** — but it is a *transient* fabric: the Verilog-A block must
   reproduce 10 Heun steps per stage × 5 stages, not just a DC point. The DEQ formulation is the
   one that would map to a DC-only Verilog-A block; that is precisely why the DEQ issues above
   (convergence, uniqueness, cross-stage gradient flow) are the blocking items, not side notes.
4. **Naming hygiene:** `friedman2_normal/` contains an MLP on Franke. If the user's downstream
   tables are built by directory name, this will be silently mis-attributed. Recommend renaming
   (the audit did **not** rename anything).

---

## 8. Explicit "not determined" list

* Exact commit for artifact **A** (`test_smooth2d_w_tanh`) — the snapshot format and the
  `PRESETS['smooth2d']`/`LAMBDAS` values match **no** commit in this repo; it used a locally
  edited `config.py` and a modified snapshot writer.
* The target-noise level and the train/val split sizes used for artifact A's KNet run (not
  recorded in its snapshot).
* Whether artifact A's checkpoint can be re-scored in the paper's units (RMSE, original scale) —
  the artifact stores only Huber loss and no evaluation script was run.
* The unit/scale correspondence between artifact B's normalised-space MSE (2.899e-3) and the
  paper's Table I RMSE column.
* The **exact mechanism** of the stage-to-stage gradient break in the DEQ run (phantom-gradient
  detach in torchdeq is the best-supported explanation; a detach inside the old
  stage/transfer code cannot be excluded without executing the June/July tree).
* Whether the DEQ fixed point is physically unique/stable — the only probe available
  (multistart) *failed*, and no spectral-radius/eigenvalue analysis of `∂rhs/∂x` exists in the
  DEQ path.
* Whether Spectre would select the same DC branch as the DEQ solver (no Spectre model of the
  KNet exists anywhere in the repo — `pipeline_report.md:545`, `496`).
* The `fixed_point_iter` fallback's actual gradient semantics (code and docstring disagree;
  the artifact used the torchdeq backend).

---

### File index (audit evidence)

| Artifact | Files inspected |
|---|---|
| `trained_knet/test_smooth2d_w_tanh/` | `config_snapshot.txt` (62 l.), `loss_history.txt` (600 rows), `grad_norms.txt` (60 rows × 23 cols), `update_norms.txt` (60 rows × 14 cols), `model.pt` (190 tensors), PNGs (names only) |
| `trained_knet/friedman2_normal/` | `config_snapshot.txt` (16 l.), `final_metrics.txt` (7 l.), `loss_history.txt` (601 l.), `model.pt` (10 tensors), PNGs (names only) |
| `trained_knet/deq/` | `config_snapshot.txt` (53 l.), `deq_validation_history.txt` (31 rows × 14 cols), `grad_norms.txt` (all `-`), `loss_history.txt` (300 rows), `model.pt` (64 tensors), PNGs (names only) |
| Notebooks | `realizable-kirchhoffnet_deq_run.ipynb` (6 cells), `realizable-kirchhoffnet_f2_smooth2d.ipynb` (44 cells) — code + saved outputs dumped to `tmp/analysis/scratch/nb_*_{code,outputs}.txt` |
| Repo | `differential_stage.py` (2188-2304 DEQ path), `train_script.py` (206-305, 726-800, 1494-1660, 2639-2680), `config.py` (340-401, 530-556), `cell_library.py` (224-436), `io_mapper.py`, `stage_transfer.py`, `kirchhoff_net.py`; `git show 0e3ae20:deq_solver.py`, `git show 0e3ae20:deq_diagnostics.py`, `git show origin/deq:{train_script,kirchhoff_net,differential_stage}.py` |
| Scratch scripts (this audit) | `tmp/analysis/scratch/{inspect_models,dump_notebooks,match_versions,scan_lambdas,scan_snapshot_writer,match_state_dict,scan2,scan3,scan_deq_branch,scan_deq_hdr,scan_mlp,param_counts,history_stats,connectivity}.py` |
