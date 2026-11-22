# Hardware-constrained retraining (SLURM)

Scripts for the HR-KNet hardware refactor, Phase 3b: retrain the paper
network under the Phase-3a hardware constraints and under the Phase-3b
time-multiplexed-fabric model, and compare every rung against the paper
baseline at the **unified 10 % failure rule**.

| File | What it does |
|---|---|
| `retrain_hw_phase_a.sh` | one Phase-A trial with the **fixed paper topology** (7 stages x 11 hidden, trial_0011 hyper-parameters). One SLURM job = one rung of the ladder. |
| `retrain_hw_bo.sh` | the same objective under a **30-trial Bayesian optimisation** (`kn_bayes_opt.py`, Phase-A mode). One study directory per `HW_PRESET`. |

Reference: `kn_hwdocs/phase3b_fabric_impl_retrain.md` sections 3.1-3.5.

---

## 1. Prerequisites

| Input | Default in the scripts | Notes |
|---|---|---|
| Teacher (trial_0019 PlainMLP) | `$SLURM_SUBMIT_DIR/assets/teachers/trial_0019/dagger_student_plain.pt` | repository copy; passed as `--phase-a-mlp-teacher-ckpt`. Override with `TEACHER_CKPT`. |
| Canonical dataset (schema 2, 20k specs, seed 100) | `$SLURM_SUBMIT_DIR/outputs/canonical_ctle_phase_a_v2.npz` | copy it from ssr0 (`~/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/models/canonical_ctle_phase_a_v2.npz`). **Do not** reuse the schema-1 `canonical_ctle_phase_a.npz`. Override with `CANONICAL`. |
| ZIG surrogate | `/home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2` | must exist at the same path on the cluster. Override with `ZIG_DIR`. |
| CTLE training data | `/home/annaik/Documents/augmented-cvae-ctle` | override with `DATA_DIR`. |
| Virtualenv | `$SLURM_SUBMIT_DIR/ENV` | the 2026 run used `kirchhoffnet_realistic/ENV`; `ENV/` is **not** in the repository checkout. The scripts source it when it exists and otherwise fall back to the module python. Override with `ENV_DIR`. |

`sbatch` is **not** installed on ssr0 (`which sbatch` fails there). Submit from
the Alliance login node, from the repository root, so that `$SLURM_SUBMIT_DIR`
is the repository root:

```bash
cd <repo-root>
sbatch --export=ALL,HW_PRESET=phase3a_full,SEED=100 scripts/slurm/retrain_hw_phase_a.sh
```

## 2. The ladder (Phase 3b section 3.4)

Every rung uses the same seed and the same 10 % rule, so the rungs are directly
comparable with R0 and with the paper run (train 7.67 %, val 8.05 %, test
8.15 % - all already 10 %-rule numbers, nothing has to be re-measured).

| Rung | `HW_PRESET` (extra flags) | Adds | Question it answers |
|---|---|---|---|
| R0 | `none` | nothing; the unified 10 % rule | what does the paper network score at 10 %? (reproduction check) |
| R1 | `phase3a_core` | ratio A/B, slope bound, IC coupling, no alpha | does the unbounded slope matter? (init `gm*A` of 2.9+ against the 0.87 hardware max) |
| R2 | `phase3a_full` | + compliance rail, `c_node` physical, GLN tail, Vref/crossbar quantisation | does the network survive a hard rail and real caps? |
| R3 | `phase3a_full` + `HW_EXTRA="--hw-mismatch-sigma isat=0.05,g=0.10,a=0.01,theta=0.1 --hw-power-lambda 1e-3"` | + mismatch, power penalty | robustness and power |
| R4 | `phase3b_tm` + `HW_EXTRA="--hw-tm --hw-tm-c-hold-ff 100 --hw-tm-xfer-mode shared_vth --hw-tm-param-bits 6"` | + S&H, injection, droop, kT/C, quantised parameters, simplified transfer | does one time-multiplexed fabric hold the 10 % rule? |

Example submissions (the rungs do not depend on each other, only the
comparison does, so they can run in parallel):

```bash
sbatch --export=ALL,HW_PRESET=none,SEED=100                scripts/slurm/retrain_hw_phase_a.sh
sbatch --export=ALL,HW_PRESET=phase3a_core,SEED=100        scripts/slurm/retrain_hw_phase_a.sh
sbatch --export=ALL,HW_PRESET=phase3a_full,SEED=100        scripts/slurm/retrain_hw_phase_a.sh
sbatch --export=ALL,HW_PRESET=phase3b_tm,SEED=100,HW_EXTRA="--hw-tm --hw-tm-c-hold-ff 100 --hw-tm-xfer-mode shared_vth --hw-tm-param-bits 6" scripts/slurm/retrain_hw_phase_a.sh
# final rung: two more seeds
for s in 101 102; do
  sbatch --export=ALL,HW_PRESET=phase3b_tm,SEED=$s,HW_EXTRA="--hw-tm --hw-tm-c-hold-ff 100 --hw-tm-xfer-mode shared_vth --hw-tm-param-bits 6" scripts/slurm/retrain_hw_phase_a.sh
done
```

`--export` with spaces in a value needs quoting; in practice put such a rung
into a JSON file and pass it as `HW_CONFIG=hw/presets/r3.json`.

Option B (30-trial BO) chains three 24 h jobs:

```bash
j1=$(sbatch --parsable scripts/slurm/retrain_hw_bo.sh)
j2=$(sbatch --parsable --dependency=afterany:$j1 scripts/slurm/retrain_hw_bo.sh)
sbatch --dependency=afterany:$j2 scripts/slurm/retrain_hw_bo.sh
```

The study directory is `$SLURM_SUBMIT_DIR/outputs/hw_retrain_bo/${HW_PRESET}`
and is the same on every resubmission (`kn_bayes_opt.py` resumes `study.db`).

## 3. Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `HW_PRESET` | `phase3a_full` (`none` = baseline) | named preset from `hw/presets/*.json` |
| `HW_CONFIG` | empty | JSON file overriding the preset keys |
| `HW_EXTRA` | empty | extra `--hw-*` flags, split on whitespace |
| `DEGRADE_THR` | empty | **optional** failure-threshold override. Empty = the unified `DEGRADE_REL_THRESHOLD = 0.10`; when set, `--zig-degrade-thr` is passed (that flag only exists if W4 adds it) |
| `SEED` | `100` | random seed |
| `RUN_TAG` | `${HW_PRESET}_s${SEED}` | output sub-directory (Phase-A script) |
| `N_TRIALS` | `30` | BO trials (BO script) |
| `TEACHER_CKPT`, `CANONICAL`, `ZIG_DIR`, `DATA_DIR`, `ENV_DIR` | see section 1 | path overrides |

## 4. Cost estimates (Phase 3b section 3.4)

* one Phase-A trial: 19-21 s/epoch x 300 epochs = **1.6-1.75 h** on one
  `3g.40gb` H100 slice;
* assumed hardware overhead factors 1.0, 1.1, 1.3, 1.3, 1.4 for R0-R4
  **[ASSUMPTION - not measured]**;
* the ladder: 1.7 h x 6.1 = **~10 GPU-h**, plus two extra seeds on R4
  (2 x 1.7 x 1.4 = 4.8 h) = **~15 GPU-h**, plus ~30 % for failed or repeated
  runs = **~20 GPU-h**;
* a rung at 1.4x overhead takes 2.4 h and is close to the `03:00:00` limit of
  the Phase-A script; above ~1.7x it will time out - raise `--time` if the
  first hardware run measures more than 1.6x the baseline epoch time;
* Option B (30-trial BO): **~50-55 GPU-h** at baseline cost, **~60-75 GPU-h**
  with the 1.2-1.4x hardware overhead; 3 chained 24 h jobs at baseline cost,
  3-4 with overhead.

| | Option A (fixed ladder) | Option B (30-trial BO) |
|---|---|---|
| Trials | 5 rungs + 2 seeds = 7 (+ retries) | 30 |
| GPU-h | ~10-15 (~20 with retries) | ~50-55 (~60-75 with overhead) |
| Wall time | one 3 h job per rung, rungs in parallel | 24 h per job, 3-4 chained jobs |
| Output | a clean per-constraint attribution | the best hardware-aware topology, but confounded |
| Risk | the paper topology may be a poor fit | stages change the number of S&H boundaries, `t_span` changes `Tu`/`Iu`, study resume with the new fingerprint is untested |

Run **Option A first** (R0 and R1 give an answer within a day), then Option B
with the preset that passed, restricted to at most 7 stages.

## 5. CLI flags used by the scripts

### 5.1 `retrain_hw_phase_a.sh` -> `dagger-nuance-distillation-kirchhoffnet.py`

All of these exist today (verified by grepping the argparse block of the
script, `dagger-nuance-distillation-kirchhoffnet.py:682-797`):

`--canonical-dataset`, `--phase-a-epochs`, `--phase-a-n-samples`,
`--phase-a-boundary-ratio`, `--phase-a-validity-weight`,
`--phase-a-validity-ramp-start`, `--phase-a-validity-ramp-epochs`,
`--phase-a-mlp-weight`, `--phase-a-fwd-weight`, `--phase-a-power-weight`,
`--phase-a-power-norm`, `--phase-a-mlp-teacher-ckpt`, `--prepare-canonical-dataset`,
`--kn-num-stages`, `--kn-num-hidden`, `--kn-small-world-k`, `--kn-small-world-p`,
`--kn-vca-rank`, `--kn-x-max`, `--kn-gm-max`, `--kn-isat-max`,
`--kn-mapper-lr-scale`, `--kn-struct-lr-scale`, `--kn-dyn-lr-scale`,
`--kn-readout`, `--kn-learnable-clip-sharpness`, `--kn-gln-rails`, `--kn-gln-B`,
`--kn-gln-rank`, `--kn-gln-families`, `--t-span`, `--boundary-fan-out`,
`--teacher-dir`, `--data-dir`, `--device`, `--seed`, `--output`, `--lr`,
`--weight-decay`, `--batch-size`, `--earlystop-eval-every`.

### 5.2 `retrain_hw_bo.sh` -> `kn_bayes_opt.py`

All of these exist today (verified with `python kn_bayes_opt.py --help`):
`--dataset`, `--ctle-phase-a`, `--ctle-canonical-dataset`, `--ctle-objective`,
`--ctle-phase-a-mlp-teacher-ckpt`, `--ctle-phase-a-mlp-weight`,
`--ctle-phase-a-fwd-weight`, `--ctle-phase-a-power-weight`,
`--ctle-phase-a-validity-weight`, `--ctle-phase-a-validity-ramp-start`,
`--ctle-phase-a-validity-ramp-epochs`, `--ctle-phase-a-power-norm`,
`--readout`, `--learnable-clip-sharpness`, `--gln-rails`, `--gln-B`,
`--gln-rank`, `--gln-families`, `--param-budget`, `--param-tolerance`,
`--epochs`, `--n-trials`, `--seed`, `--device`, `--no-seed-trial`, `--output`.

### 5.3 Flags that do **not** exist yet - requires W4 CLI plumbing

| Flag | Where | Status |
|---|---|---|
| `--hw-preset NAME` | both scripts | **does not exist** - to be added by the W4 CLI plumbing (`parse_hw_args` in the dagger script, passthrough in `kn_bayes_opt.py`). Presets are ready in `hw/presets/*.json`. |
| `--hw-config PATH` | both scripts | **does not exist** - same W4 work. |
| `--hw-*` individual flags (`--hw-tm`, `--hw-tm-xfer-mode`, `--hw-tm-param-bits`, `--hw-tm-c-hold-ff`, `--hw-mismatch-sigma`, `--hw-power-lambda`, ...) | both scripts (through `HW_EXTRA`) | **do not exist** - same W4 work. |
| `--zig-degrade-thr T` | both scripts, only when `DEGRADE_THR` is set | **does not exist** - optional override of the unified `DEGRADE_REL_THRESHOLD = 0.10`; the scripts do **not** pass it by default, so the ladder works without it. |

Until W4 lands, the hardware scripts will fail at start-up with
`unrecognized arguments`. The `--hw-*` flags and the `--hw-preset` /
`--hw-config` flags are the only reason the ladder cannot be submitted today.

### 5.4 Deliberately absent

* `--kn-moe-num-experts`, `--kn-moe-gate-rank`, `--kn-moe-top-k`: **removed** by
  the surgery; they must not be passed.
* `--freeze-read`: no flag needed, `KN_FREEZE_READ = True` is the harness
  default.
* A failure-threshold flag: not needed, the rule is one constant now.

## 6. Notes

* The BO script deliberately does **not** rebuild the canonical dataset (it
  fails fast instead): every resubmission must use the same `.npz`.
* `--kn-isat-max 16` replaces 28.69 when the hardware flags are on;
  `--kn-gm-max` stays at the paper value for traceability (with
  `hw_slope_bound` it no longer sets the tanh slope, it only bounds the GLN
  `slope` mode).
* `kn_bayes_opt.py` still needs, for the hardware runs (W4): the
  `--hw-config`/`--hw-preset` passthrough into the trial command, the
  `HWConfig.hash()` in the study fingerprint, no `gm_max`/`isat_max` sampling
  when `hw_slope_bound` is on (`isat_max = 16`), `HWUnits` built per trial from
  the sampled `ctle_t_span`, and the hardware-aware parameter budget.
* The post-train audit (`python -m hw.audit ...`) is guarded by
  `[ -f hw/audit.py ]` and is therefore a no-op until `hw/audit.py` exists.
