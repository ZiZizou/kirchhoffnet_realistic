# Audit of the trained CTLE KNet artifacts

Scope: audit only. No existing file was modified and nothing was trained. Scratch scripts and raw outputs are in `tmp/analysis/scratch/`:

- `audit2.py` is the main forward/ablation script.
- `audit_out_thr0.20.txt` and `audit_out_thr0.10.txt` hold its raw output under the two failure gates.
- `run_thr0.20.log` and `run_thr0.10.log` are the console logs of those runs.

Evidence tags: **[log N]** is a line number in `run_train_knet_single_shot_distillation_2.sh-59014723.out`; **[code]** is a repo file and line; **[run]** is something I executed on CPU.

Two failure-rate gates appear throughout, so "failure rate" is ambiguous unless the gate is named:

- **20% gate.** A spec counts as degraded at ≥20% relative regression (higher power or jitter, lower eye height or width). A design fails if at least two of the four specs are degraded, or if the ZIG validity probability is below 0.5. This is the paper's stated rule (`paper_raw.txt` ~l.717).
- **10% gate.** Same rule with a 10% threshold.

---

## TL;DR

1. **The `.sh` and `59014723.out` match the paper.** The log gives 5,649 params and final failure of train 7.67% / val 8.05% / test **8.15%**, which is paper Table II. Two caveats:
   - This is not pure distillation. The loss includes `fwd_mse` and a validity NLL, both backpropagated through the ZIG forward model.
   - By my code reading at commit 5c6a4f4, the log's "failure" used a **10%** degrade gate, not the paper's stated 20%. I did not execute this, so treat it as a flag, not a fact (see 1.5).
2. **`fixed_mlp_knet` is a different, older model.**
   - It has 6,775 params, 4 stages, hidden 14, k=4, VCA rank 2, no GLN, no shared sense, no learnable clip. It is not the model in the `.out` logs.
   - It was produced by `fixed-mlp-distillation-kirchhoffnet.py` at commit `9fec95e` (2026-09-02), distilling a **W=48 L=3 plain MLP of ~5.3k params**, not the 96x3 `trial_0019` teacher.
3. **Its accuracy claim is correct.**
   - Test failure is 9.04% (student) vs 6.89% (teacher) on 1,117 rows under the 20% gate, a 2.15 pp gap against a 3.0 pp tolerance.
   - I rebuilt the model with the current HEAD code. State dict loads strictly with 0 missing and 0 unexpected keys. Logits reproduce to 3.8e-6, and the failure mask matches on 100% of rows.
   - At the 10% gate the same model fails 22.6%, and the teacher 17.4% (degradation only).
4. **It is not circuit-realistic as trained.** In order of impact:
   - It relies on `freeze_read` sample-and-hold: with live core OTAs, failure goes 9.0% → 26.7%.
   - It relies on per-edge VCA gains: with all gains forced to 1, failure goes 9.0% → 99.3%.
   - It relies on the per-edge resistive shunts: with G→0, failure goes 9.0% → 69.6%.
   - Stage-3 node voltages exceed the rail (max 5.35 vs x_max 4) on 165 of 256 rows.
   - All edge gates are fractional (0.39–0.83), so none can be pruned or binarized.
   - The OTAs run almost linearly (max |u| about 2, median under 1), with gm and Isat between 0.06 and 0.39.
5. **It does not exploit the integrator step size.** At 2x, 5x, 10x and 40x finer Heun steps, the max logit change is ≤2.3e-3 and the failure rate moves by at most +0.09 pp (1 row of 1,117).
6. **GLN ablation is not possible.** This checkpoint has no GLN (`stage.gln_rails is None`), and no checkpoint exists for the `.out` runs.

---

## 1. Provenance of the `.out` runs (59014723 authoritative, 58828787 for comparison)

### 1.1 Command and entry point

- Entry point: `dagger-nuance-distillation-kirchhoffnet.py` in Phase-A "canonical dataset gate" mode. Training ends with "exiting before DAgger body" [log 326].
- Repo version: `git_commit=5c6a4f4` [log 12], dated 2026-09-08 13:24 -0400. It is an ancestor of HEAD, 34 commits behind, on branch `simplified`. Both logs report the same commit.
- Full flag set is `.sh` lines 69–77 (seed 100, cuda). Key flags:
  - `--kn-num-stages 7 --kn-num-hidden 11 --kn-small-world-k 2 --kn-small-world-p 0.2 --kn-vca-rank 3`
  - `--kn-x-max 4 --kn-gm-max 4.24197 --kn-isat-max 28.68855`
  - `--kn-readout shared-x2 --kn-learnable-clip-sharpness`
  - `--kn-gln-rails --kn-gln-B 4 --kn-gln-rank 2 --kn-gln-families boundary,readout`
  - `--t-span 4.660583 --boundary-fan-out {"0":[0,4],"1":[1,5],"2":[2,6],"3":[3,7]}`
  - `--phase-a-epochs 300 --batch-size 512 --lr 7.069e-3 --weight-decay 6.087e-3 --kn-struct-lr-scale 4`
  - `--earlystop-eval-every 5`
- Loss weights (`.sh` l.49–51, log 289–291): MLP-teacher MSE 1.0, **forward-MSE 1.0**, power 0.0005, validity NLL 0.3 (ramp 10+30).
- Harmless `.sh` bug: lines 10–11 of the header comment wrap onto a bare word, so bash prints `canonical: command not found` [log 1]. It has no effect on the run.
- `--kn-moe-*` flags are passed (3 experts, rank 2, top-k 1). The student's parameter breakdown has no MoE, so they appear unused by the KNet student. I did not trace the code to confirm. Not determined.

### 1.2 Architecture, from the log and code

| Item | Value | Source |
|---|---|---|
| Stages | 7 | [log 56] |
| Hidden nodes per stage | 11 | [log 56] |
| Output-ODE nodes per stage | 7 | inferred from 18 nodes per stage |
| Small-world k / p | 2 / 0.2 | [log 56] |
| edge_repeats | 2 | [log 56] |
| Cell library | `tanh_free` | [log 56] |
| leak mode | non-programmable (0.0486) | [log 56] |
| Interstage activation | residual-relu-tanh | [log 56] |
| freeze_read | True | [log 56] |
| temporal_readout | True | [log 56] |
| Readout | `shared-x2` (two senses per node, crossbar 7x22), tail on | [log 55] |
| VCA | on, rank 3, core on, separate core bus, no bias, no gate-shunt | [log 54] |
| GLN | on, B=4, rank 2, families boundary,readout (100 params) | [log 56, 62] |
| Learnable clip | on, 1 param per stage | [log 56], `other: 1` per stage |
| x_max / input_rail | 4.0 / 4.0 | [log 56] |
| gm_max / isat_max | 4.24197 / 28.68855 | [log 56] |
| t_span | 4.660583 total, 0.6658 per stage | `.sh` l.77, code `dagger-nuance…py:869–871` |
| Steps | 70 total, 10 per stage, dt about 0.0666 | `config.py` SOLVER default 70 (inferred, not printed in the log); `ctle_dagger_common.py:1842–1843` |
| Solver | fixed-step Heun | `config.py` SOLVER |
| Cell library bounds | gm_min / isat_min = 0.01 (default) | `config.py` |
| Edges per stage | core 22, readout senses 22, boundary 8 | [log 64–72] |

Boundary edges feed each hidden node pair from the 4 inputs.

Parameter count is exactly 5,649 = 5,614 circuit + 35 projector (output affine) [log 56]. I closed the arithmetic myself, because the log breakdown omits the interstage transfers:

| Part | Params |
|---|---|
| Stage 0 | 770 (714 + 56 boundary_cell_lib) |
| Stages 1–6 | 6 x 714 = 4,284 |
| GLN | 100 |
| 6 interstage transfers | 6 x 4 x 18 = 432 |
| Post-readout transfer | 28 |
| Output affine | 35 |
| **Total** | **5,649** |

The 714 per stage is: cell_lib 154, z 22, u_logits 18, boundary_z 8, sense_z 22, sense cell_lib 154, crossbar 154, vref 1, vca_W 24, vca_v 156, clip 1. Note `u_logits` (18 per stage) is a "node gate" that `config.py` labels deprecated. Whether it is used in the forward pass was not verified.

### 1.3 Final metrics, 59014723

- Phase-A final failure [log 325]: **train 7.67%, val 8.05%, test 8.15%**. This matches paper Table II (HR-KNet 8.15%, 5,649 params).
- Best validation failure 8.05%, first reached at epoch 40 [log 297] and never improved in the remaining 260 epochs. At epoch 300 the validation failure is 10.75% [log 323].
  - So the final-line numbers come from the best-validation weights. This is my inference; the log does not say "restored".
- Validation failure swings between 8% and 13% from one evaluation to the next. Boundary-spec failure is 13–22% versus 3–5% for interior specs.
- Final training metrics [log 323]: `train_mse_mlp` 0.0295, `fwd_mse` 0.0344, `valid_nll` 0.0075, power 1.045x train mean.
- `val_huber` is flat at 1.97–1.98 from epoch 10 to 300, so it carries no information. I did not look up its definition.
- The log does **not** report a teacher failure rate.
- The 2,000-row test set has a standard error of about ±0.6 pp at 8%. The 8.70% (MoE) vs 8.15% (HR-KNet) gap in Table II is within noise.
- Run time was about 98 minutes. The log does not state the epoch count at which training stopped, but 300 epochs ran.

### 1.4 Comparison with 58828787

Both logs report `git_commit=5c6a4f4` and the same teacher, dataset, scalers and optimizer line.

| | 58828787 (2026-09-08) | 59014723 (2026-09-09) |
|---|---|---|
| Stages | **6** [log 56] | **7** |
| Params | **4,863** | **5,649** |
| Output dir in log | `trial_0008` | `trial_0011` |
| Train / val / test failure | 8.59 / 8.75 / **9.70%** [log 294] | 7.67 / 8.05 / **8.15%** |
| Final-epoch val failure | 11.25% | 10.75% |
| Final `mlp_mse` / `fwd_mse` | 0.0307 / 0.0335 | 0.0295 / 0.0344 |
| gm_max / isat_max / x_max / VCA rank / GLN | identical | identical |

- The log's own parameter breakdown shows one 6-stage and one 7-stage run, so stage count is a documented difference. The 58828787 flags are not echoed in the log, so any other flag difference is not determined.
- The `.sh` in Downloads holds the `trial_0011` flags. The repo's `run_train_knet_single_shot_distillation.sh` holds different flags (`trial_0005`: 6 stages, hidden 12, k=4, gm_max 1.35).
- The 6-stage run is slightly worse and 7.7% smaller. The difference of 1.55 pp on 2,000 rows is about 2 SE.

### 1.5 Flags worth confirming

**(a) Failure rate uses a 10% degrade gate in this code (not executed).**
- At 5c6a4f4, `ctle_dagger_common.py:2495–2502` makes an explicitly passed threshold set the degrade gate.
- Phase A passes `error_threshold = ERROR_THRESHOLD = 0.10` (`ctle_dagger_common.py:3234–3236`, `ctle_dagger_common.py:116`).
- The paper text says ≥20%.
- Evidence this matters: the fixed-MLP model (below) is 9.0% at a 20% gate but 22.6% at a 10% gate. If Table II came from the 10% gate, the paper rule description is wrong. If the paper's own number came from a 20% evaluation elsewhere, the `.out` number is not the same metric.
- The 10% gate also changed the earlier fixed-MLP evaluation path. Before commit `1482376` (2026-09-07) the threshold argument was ignored, which is why the fixed-MLP run used 20%.

**(b) Training uses the evaluator's forward model.** The Phase-A loss includes `fwd_mse` and validity NLL through ZIG. The paper's "never enters the gradients" statement (`paper_raw.txt` ~l.725) applies to the fixed isolation baseline, not to this script.

**(c) Teacher.**
- The log loads a PlainMLP, W=96 L=3, SiLU, LayerNorm, `knet` preprocessing [log 285–286], from `.../generative_plain_bayes_opt/plain_knet_budget25000_fproxy/trial_0019/dagger_student_plain.pt`.
- That file is **not** among the supplied files. A freshly built PlainMLP of that shape has 20,359 trainable params, which fits the paper's "25k budget".
- Teacher failure rate on the canonical set: not determined.

---

## 2. `fixed_mlp_knet`

### 2.1 Files

- `best_student.pt`, `student_final.pt` and `student_bundle["model"]` are bit-identical (181 tensors), and `last_checkpoint["model"]` equals them, with max parameter difference 0.0 [run]. The best epoch is therefore the last, 110 of 110 (history length 110).
- The bundle also carries `args` (47 keys) and the input log bounds.

### 2.2 Architecture, from `args` in `student_bundle.pt` / `metrics.json` and the state dict

| Item | Value |
|---|---|
| Stages / hidden / outputs | 4 stages, 14 hidden, 7 output-ODE nodes (21 nodes per stage) |
| Small-world k / p / seed | 4 / 0.2 / 1 |
| edge_repeats | 2 |
| Cell library | `tanh_free` |
| Edges per stage | core 56, boundary 8, readout mesh 98 (14 x 7) |
| Readout | temporal mesh (`output_ode_*` keys); no shared sense, no crossbar |
| freeze_read | True |
| freeze_boundary / freeze_temporal_read | False / False (boundary and readout OTAs are live) |
| VCA | on, rank 2, core on, separate core bus, no bias, no gate-shunt |
| GLN / learnable clip | none (no keys; `stage.gln_rails is None`) |
| Boundary fan-out | `{"0":[2,4],"1":[1,3],"2":[12,5],"3":[7,9]}` |
| x_max / input_rail | 4.0 / 4.0 |
| t_span / steps | 7.0 total, 1.75 per stage, 18 steps per stage (round of 17.5), dt 0.0972 |
| gm / Isat bounds | [0.01, 10] each (`config.py` default) |
| Interstage | residual-relu-tanh (3 transfers, 21 nodes) plus a post-readout transfer on 7 outputs |
| Output mapper | per-output gain, bias, tanh_gain, relu_gain, relu_threshold |
| Params | 6,775 (via `parameter_breakdown`) |

- Training: 110 epochs, batch 256, lr 1e-3, AdamW with weight decay 0, seed 100, split 0.8/0.1/0.1 of 11,161 rows. The run was on Kaggle (`/kaggle/working/fixed_mlp_knet`).
- The `boundary_cell_lib` is shared across all four stages. It is the same module object, and the 8-edge `a_raw` tensors are identical in the state dict across stages. Only `boundary_z_logits` and `vca_v_boundary` differ per stage.

### 2.3 Which code built it

- **Key-name match.** The state dict loads into the current HEAD `KirchhoffStudent` with `strict=False` and reports 0 missing and 0 unexpected keys (`tmp/analysis/scratch/load1.py`, [run]).
- **Args match.** The saved `args` key set equals the argparse key set of `fixed-mlp-distillation-kirchhoffnet.py` at commit `9fec95e` exactly, with no extras either way. Later commits add or remove keys: `d6f2cde` adds `bo_mode`, `fixed_dataset_path`, `student_*` and so on; HEAD adds `kn_solver_scale`.
- **Output-file layout matches `9fec95e`.** `zig_failure_comparison.npz` has unprefixed keys (`teacher_failure_mask`, `student_pred_specs`, …) and `metrics.json` holds a flat test-only summary. From `d6f2cde` onward the npz keys are prefixed `validation_` / `test_`.
- **Date match.** `9fec95e` is dated 2026-09-02 01:32 -0400, and `metrics.json` `created_at` is 2026-09-02 06:41 (Kaggle clock, likely UTC). The commit message is "Added evaluation with zig at the end".
- **Same solver and gm bounds then as now.** At that commit `config.py` had the same `SOLVER` (t_span 7.0, 70 steps) and gm/Isat bounds [0.01, 10]. That is why HEAD reproduces the saved logits.
- Notebook `realizable-kirchhoffnet.ipynb` is not present. The two sibling notebooks (`f2_smooth2d`, `deq_run`) are other experiments, so the origin notebook is not determined. Only the script is established.

### 2.4 Does it match the `.out` run?

No. The `.out` run has 7 stages, hidden 11, k=2, GLN, learnable clip, shared-x2 readout, VCA rank 3, and 5,649 params. This checkpoint has 4 stages, hidden 14, k=4, a temporal mesh readout, no GLN, VCA rank 2, and 6,775 params. It is also trained by plain logit-MSE distillation, with none of the HR-KNet's extra loss terms.

### 2.5 Which teacher

- `args`: `mlp_teacher_width 48`, `mlp_teacher_layers 3`, `use_layernorm False`, ckpt `/kaggle/.../improved-zig-nf-spline/pytorch/default/4/dagger_student_plain.pt`.
- `C:\Users\Atharva\Downloads\improved-zig-nf-spline-pytorch-default-v4\dagger_student_plain.pt` (hash F519E908…) is a W=48 L=3 plain MLP with 5,287 trainable params (5,301 floats including `log_lo` / `log_hi`).
- I ran it on the 11,161 saved specs. It reproduces `fixed_teacher_dataset.npz["teacher_logits"]` with max difference 3.8e-6 and mean 1.6e-7 [run]. **This is the teacher that labeled the fixed dataset.** It is a small teacher, not the paper's 25k plain MLP.

### 2.6 Accuracy

From `metrics.json`:
- Validation logit MSE 7.30e-4; test logit MSE 1.164e-3.
- log10 parameter MAE 0.0077 (val) and 0.0076 (test), about 1.8% in parameter value.
- Validation loss was still falling at epoch 110, so training had not converged.

ZIG failure comparison on the 1,117 test rows, 20% gate:

| | Failure rate | Invalid rate |
|---|---|---|
| Teacher | 6.893% | 3.671% |
| Student | 9.042% | 5.282% |
| Gap | 2.149 pp (tolerance 3.0, `isolation_pass` True) | |

- I recomputed both rates from `zig_failure_comparison.npz` and got 0.068935 and 0.090421, an exact match.
- Agreement: both fail on 5.19% of rows, student-only 3.85%, teacher-only 1.70%, disagreement 5.55%.
- `test_teacher_student_predictions.npz` matches the fixed-dataset test slice (`specs`, `teacher_logits`) exactly.
- Recomputation under both gates, from the saved `pred_specs` (degradation rule only) and from a full evaluator rerun on the rebuilt model:

| Gate | Student (npz pred_specs) | Teacher (npz pred_specs) | Student (full evaluator run) |
|---|---|---|---|
| 20% | 6.71% | 4.74% | 9.04%, identical mask on 100% of rows |
| 10% | 20.68% | 17.37% | 22.56% |

  The npz-only figures leave out the validity (`p_valid`) term; the full evaluator includes it.
- Sample size: n=1,117 gives a standard error of about ±0.9 pp at 9%, so the 2.15 pp gap is roughly 2 SE.
- Comparability: this test split is drawn from the 11,161 filtered measured rows, not from the 20,000-spec 50%-boundary canonical set used by the HR-KNet runs. The paper's 8.15% and this 9.04% are therefore **not directly comparable**, even apart from the gate, the loss, and the teacher.

### 2.7 Rebuild and agreement

- Rebuild with current HEAD code succeeds (`kn_solver_scale` defaults to 1.0; `student_kind` is set to "knet" by the script).
- Forward on all 1,117 test rows takes 0.6 s on CPU. Against saved student logits: max |Δ| 3.8e-6, mean 2.1e-7 (256-row batch), failure mask identical on 100% of rows.

---

## 3. Physical-realism statistics of the trained parameters

All numbers are from the 256-row batch and the state dict. Raw per-stage tables are in `tmp/analysis/scratch/audit_out_thr0.20.txt`. Units are the code's dimensionless ones; C_eff = 1.

### 3.1 Per-edge device parameters (min / median / max over edges)

Ranges below span all four stages:

| Quantity | Core | Boundary | Readout mesh |
|---|---|---|---|
| gm | 0.077 / 0.12–0.13 / 0.33 | 0.078 / 0.10 / 0.35 | 0.059 / 0.10–0.11 / 0.24 |
| Isat | 0.077 / 0.12–0.13 / 0.39 | 0.078 / 0.10 / 0.35 | 0.058 / 0.10–0.11 / 0.24 |
| A (softplus) | 0.49 / 0.9–1.0 / 2.35 (stage 0: all 0.693) | 0.69 / 0.85 / 1.43 | 0.25 / 0.67–0.79 / 2.65 |
| B | 0.36 / 0.83–0.93 / 1.64 (stage 0: all 0.693) | 0.28 / 0.86 / 1.11 | 0.43 / 0.82–0.85 / 1.29 |
| gm·A | 0.048 / 0.088–0.135 / 0.53 | 0.053 / 0.085 / 0.38 | 0.019 / 0.065–0.075 / 0.53 |
| Small-signal Isat·gm·A | 0.004 / 0.011–0.018 / 0.135 | 0.004 / 0.009 / 0.13 | 0.001 / 0.007 / 0.12 |
| theta | −0.73 / ±0.2 / +0.73 | −0.66 / −0.09 / +0.34 | −0.39 / −0.15 / +0.23 |
| Shunt G (softplus) | 0.0016 / 0.0035–0.0045 / 0.013 | 0.005 / 0.007 / 0.020 | 0.002 / 0.005 / 0.026 |
| Gate σ(z) | 0.39 / 0.58–0.64 / 0.83 | 0.47 / 0.52–0.53 / 0.82 | 0.40 / 0.57–0.60 / 0.77 |

- **No device is near its allowed ceiling.** Zero edges have gm or Isat above 9.9 (ceiling 10). Values sit close to init (gm_raw init −5 gives 0.077).
- **Sign:** `sign(s_raw)` = +1 on 50% of stage-0 core edges, only 7–12% of core edges in stages 1–3, and 18–20% of readout edges. Most edges are inverted, which in a circuit means swapping the differential pair terminals.
- **Shunt vs leak:** G is 0.002–0.026 against a fixed leak of 0.0486, so the shunts are small conductances. Over the 1.75-unit stage window nothing equilibrates.
- **Stage-0 core edges:** `a_raw == b_raw == 0` exactly. With `freeze_read=True` and x0 = 0, those edges see only theta, so they act as constant current sources: Isat·(tanh(gm·theta)+0.01·gm·theta)·gate, up to 0.0165. A and B never received gradient. Stage 0 core could be replaced by DC bias currents.
- **Gates are not binary.** No gate is below 0.1 or above 0.9. Gates scale both the tanh current and the shunt (`rhs` multiplies both by the edge mask), so they could be folded into Isat and G. But they cannot be pruned: hardening them to {0,1} gives 94.3% failure (logit MSE 2.73), and pruning "gate < 0.1" changes nothing because no edge qualifies.
- **Edge count:** 162 edges per stage (56 + 8 + 98) times 4 stages is 648 OTA edges, all active. The only sparsity is the small-world graph.

### 3.2 Interstage and post-readout transfers (`w1·x + w2·tanh(x) + w3·relu(x − vth)`)

| Transfer | w1 | w2 | w3 | vth |
|---|---|---|---|---|
| 0 | 0.62 / 1.28 / 1.98 | −0.38 / 0.28 / 0.98 | −0.37 / −0.16 / 0.61 | −0.63 / −0.20 / 0.03 |
| 1 | 0.98 / 1.28 / 2.07 | −0.01 / 0.28 / 1.05 | −0.35 / −0.12 / 0.66 | −0.56 / −0.18 / −0.06 |
| 2 | 0.72 / 1.34 / 1.77 | −0.27 / 0.32 / 0.86 | −0.29 / −0.12 / 0.76 | −0.37 / −0.14 / 0.36 |
| post-readout | 1.11 / 1.30 / 1.45 | 0.06 / 0.24 / 0.34 | −0.63 / −0.01 / 0.25 | −0.11 / 0 / 0.16 |

- The transfers are active-gain stages: w1 up to 2.07, with a tanh and a thresholded-ReLU branch per node. They are not passive wiring.
- Output affine: gain 1.10–1.54, bias −0.21 to +0.06, tanh_gain 0.07–0.37, relu_gain −0.38 to +0.35, relu threshold −0.23 to +0.02.

### 3.3 Node voltages (x_max = 4.0)

Scaled input to the net is within ±0.91 on every dimension (rail 4).

| Stage | Max abs V, hidden | Output-ODE nodes | x0 max abs V | Rows with any \|V\| > 4 |
|---|---|---|---|---|
| 0 | 0.353 | 0.162 | 0 | 0 / 256 |
| 1 | 1.260 | 0.536 | 0.661 | 0 / 256 |
| 2 | 3.373 | 1.311 | 2.654 | 0 / 256 |
| 3 | **5.348** (p99 4.27) | 3.156 | **4.940** | **165 / 256** |

- **Stage 3 leaves the rail.** Its initial state (the output of interstage transfer 2) is already 4.94, above x_max. During the stage, 2.1% of core OTA terminal samples, 1.0% of readout, and 0.45% of boundary terminal samples exceed 4. Beyond the rail, the compliance gate `sigmoid((x_max − |V|)/0.02)` shuts the tanh current off.
- Post-readout transfer: max input 5.18, max output 7.65 (the logits). These are analog output voltages before the digital sigmoid-to-parameter map, and they are larger than the rail.
- **OTA operating point** (|u| is the tanh argument): max |u| 0.13 (stage 0 core), 0.39, 1.27, 1.98 (stage 3 core); readout 0.11–0.96; boundary (stage 3) 0.69. The fraction of samples with |u| > 2 is 0.000 everywhere. The network is nearly linear. Only stages 2–3 reach into the tanh bend.

### 3.4 VCA gains over the 256 rows (`2·σ((u@W)@vᵀ)`)

| Stage | Family | Gain min / max | Per-edge swing over inputs (median / max) | Edges outside [0.5, 1.5] somewhere |
|---|---|---|---|---|
| 0 | core | 0.27 / 1.93 | 1.10 / 1.66 | 46/56 |
| 3 | core | 0.002 / 2.0 | 1.57 / 2.00 | 44/56 |
| 1 | readout | 0.018 / 1.96 | 0.61 / 1.50 | 49/98 |
| 3 | readout | 0.063 / 2.0 | 0.60 / 0.91 | 66/98 |
| all | boundary | 0.28–0.63 / 1.80–1.98 | 0.36–0.52 / 0.9–1.65 | 3–4 of 8 |

- Gains reach the 0 and 2 extremes on several edges, so those VCAs behave like switches.
- `vca_W` entries reach |1.25|. There are no VCA bias terms.
- Physically, each of the 648 edges has its own multiplier driven by a rank-2 mix of the 4 inputs.
- `vca_W_core` and `vca_W` are the shared 4x2 projections, one per stage.

### 3.5 GLN

Not present in this checkpoint (no GLN keys, `gln_rails` is None), so no GLN gain ranges exist to report. The `.out` run does have GLN (100 params). No checkpoint for it was supplied, so its GLN gains cannot be inspected.

---

## 4. Solver-fidelity test

Same total t_span (1.75 per stage) and the same trained parameters. Steps per stage: baseline 18. The run compares full-row output against the 1x result (n = 1,117 unless noted; 20% gate):

| Steps per stage | Max abs logit change | Mean abs change | Max log10-param change | Failure rate | Mask flips |
|---|---|---|---|---|---|
| 9 (0.5x) | 7.2e-3 | n/a | n/a | 9.13% | n/a |
| 18 (1x) | 0 | 0 | 0 | 9.04% | 0 |
| 36 (2x) | 1.7e-3 | 3.7e-5 | 0.0018 | 9.04% | 0 |
| 90 (5x) | 2.2e-3 | 4.5e-5 | 0.0022 | 9.13% | 1 |
| 180 (10x) | 2.3e-3 | 4.6e-5 | 0.0023 | 9.13% | 1 |
| 720 (40x, n=256) | 2.3e-3 | 5.4e-5 | 0.0019 | not run | n/a |

At the 10% gate the failure rates are 22.56%, 22.56%, 22.47%, 22.47% for 1x, 2x, 5x, 10x and 22.20% for 0.5x.

**Interpretation.**
- The solution has converged by 2x refinement. A 10x refinement moves logits by at most 2.3e-3, versus a student–teacher RMS of about 0.034 (√1.16e-3). The model does **not** exploit the Heun step size.
- That result holds only with `freeze_read=True`. The step-size test does not touch the sample-and-hold, which is a bigger discretization-like assumption (see section 5). With `freeze_read=False` and 10x steps, the logit MSE against the saved student is 1.32e-2, the same as at 1x (1.33e-2), so that gap is structural, not numerical.

---

## 5. Ablations (no retraining; n=1,117 test rows; 20% gate unless noted)

Baseline: logit MSE 1.16e-3 vs teacher, failure 9.04% (teacher 6.89%).

| Ablation | Logit MSE vs saved student | Max abs Δlogit | log10-param MAE vs saved | Failure (20% gate) | Failure (10% gate) |
|---|---|---|---|---|---|
| VCA boundary gains = 1 | 2.34e-2 | 2.14 | 0.016 | 14.59% | 32.41% |
| VCA readout gains = 1 | 1.59e-1 | 3.74 | 0.068 | 59.71% | 68.93% |
| VCA core gains = 1 | 9.71e-2 | 2.61 | 0.068 | 85.86% | 88.99% |
| **VCA all gains = 1** | 3.17e-1 | 5.02 | 0.101 | **99.28%** | 99.64% |
| `freeze_read` = False (live core OTAs) | 1.33e-2 | 0.66 | 0.030 | **26.68%** | 38.76% |
| All resistive shunts G → 0 | 3.59e-1 | 1.86 | 0.136 | 69.56% | 77.80% |
| Prune edges with gate < 0.1 | 1.5e-13 | 3.8e-6 | 0.000 | 9.04% | 22.56% |
| Harden all gates to {0,1} (σ > 0.5 → 1) | 2.73 | 5.06 | 0.312 | 94.27% | 94.36% |
| **GLN gains = 1** | not possible, no GLN in this checkpoint | | | | |

VCA ablations were done by zeroing `vca_W_core` (core) or `vca_v_boundary` / `vca_v_readout` (the others), which sets gain to 2·σ(0) = 1.

**What this says for the Cadence goal.**
- The model is **highly dependent on the VCA**. All gains forced to 1 leaves it failing on essentially every spec. The input-to-output mapping flows mostly through the per-edge gain control, not through the 8 boundary OTAs (removing only boundary VCA costs about 5.5 pp).
- The model is **dependent on the sample-and-hold of the core currents**. In a continuous-time circuit the core OTAs would see evolving node voltages, which is the `freeze_read=False` row (26.7%). To realize this model faithfully, each stage needs a sample-and-hold of the core OTA inputs at stage entry, with stages run in sequence. That means 4 sequential phases, held currents on 56 edges per stage, plus the transfer blocks between them.
- The small shunt conductances carry real weight; removing them breaks the model.
- Because gates are fractional, the full 648-edge count has to be built.
- Total signal and device ranges are small compared with the allowed ranges, so the model would be a good candidate for retraining under tighter, more circuit-friendly constraints (rail-limited voltages, live core OTAs, bounded VCA gain).

---

## 6. Teacher and forward-model directory (`Y:\Documents\ASPDAC_2026\kirchhoff_redesign\mlp_distillation_target`)

| File | What it is |
|---|---|
| `ctle_conditional_flow.pt` | Conditional spline flow (1,727,154 params, 156 tensors). Upstream flow teacher; config in `flow_config.json` (7 inputs, 4 conditions, 10 couplings, hidden 256, 8 bins). |
| `hybrid_hurdle_ctle_model.pt` + scalers + `hybrid_hurdle_config.pkl` | ZIG forward model (52,934 params): device params → height / width / jitter / power. The failure evaluator uses it. Config: eye scales 47.10 / 87.53 / 46.72, `per_target_hurdle` False. |
| `dagger_student_moe.pt` | MoE student: trunk 160x3, 3 experts, gate and regime classifier, 56,403 floats. **Not a plain MLP and not 25k.** It also does not match Table II's 6,820-param MoE baseline. |
| `dagger_student_plain.pt` | **Absent from this directory.** |

- All nine files in this directory are byte-identical (hash) to the same names in `C:\Users\Atharva\Downloads\improved-zig-nf-spline-pytorch-default-v4`. That folder also holds `dagger_student_plain.pt` (the W=48 L=3 teacher above) and `scaler_X.pkl`.
- The `.sh` points at `.../improved-zig-nf-spline-pytorch-default-v2` for flow/ZIG artifacts (`.sh` l.62) and at `trial_0019/dagger_student_plain.pt` for the teacher (`.sh` l.36). Numbers in the log match this directory's artifacts: scaler_X mean (−5.22107447, −2.65785535, −8.41856601), scaler_y_p mean −2.0538 / scale 0.7406, eye scales 47.10 / 87.53 / 46.72 [log 17–23]. Whether v2 is byte-identical to v4 is not determined, but the numbers agree.
- So the paper's "25k PlainMLP teacher" is consistent with the log (W=96 L=3 LN, about 20.4k params, under 25k), but the actual file is not available here. The Y: directory's teacher-like file is the MoE, which is not the production teacher.

---

## 7. Open items / not determined

- Whether the Table II 8.15% used the 10% or 20% degrade gate (section 1.5a). A quick check is to rerun the evaluation on the `trial_0011` checkpoint at both gates. That checkpoint is not in the supplied files.
- The 58828787 flags beyond what the log prints.
- The `trial_0019` teacher's failure rate and weights (file absent).
- Whether `--kn-moe-*` flags have any effect on the KNet student.
- The origin notebook `realizable-kirchhoffnet.ipynb`.
- Whether `u_logits` (18 or 21 per stage, labeled deprecated) is used in the forward pass.
- freeze_boundary / freeze_temporal_read settings for the `.out` runs (the log does not print them).
- HR-KNet GLN behavior, since no checkpoint exists.
