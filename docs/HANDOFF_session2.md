# Handoff: HR-KNet → Cadence refactor, session 2 (continue Phase 3 follow-ups + Kaggle notebook)

Written 2026-11-22 (America/Toronto) at the end of session 1. Phases 0–3 of the original handoff (`handoff-kirchhoffnet-refactor.md`) are **done and merged**. This file covers only what is left, plus the facts you need to do it.

**Start the new session with the remote workspace `annaik@ssr0.eng.uwaterloo.ca:/home/annaik/Documents/kirchhoffnet_refactor`.**

A copy of this file is in the repo at `docs/HANDOFF_session2.md` (untracked). Commit it, or delete it once you've read it.

## 0. Why session 1 stopped
- The DSH file sandbox (`rw_write_file`/`rw_edit`) was tied to the session-1 workspace, `~/Documents/tron_eq`. That is an UNRELATED repo: the user's tron-eq project.
- Session-1 worktrees and helper scripts ended up under `tron_eq`. All of them have been removed or relocated:
  - The user's tracked `tron_eq/conftest.py` had been overwritten. It was restored from git.
  - The leftovers are in `~/Documents/kirchhoffnet_refactor/.agent_scratch/tron_eq_leftovers/`.
- **Never write anything under `~/Documents/tron_eq`.**
- `rw_pick_workspace` does NOT move the sandbox. A new session started in `kirchhoffnet_refactor` does.

## 1. Repo state (ssr0)
- **Repo:** `~/Documents/kirchhoffnet_refactor`, branch **`hw-refactor` @ `0b19ba5`** (Task B merged), clean.
  - `venv/bin/python -m pytest -q` gives **694 passed, 57 deselected (slow)** in about 3 min. `pytest.ini` deselects `-m slow`, which is the CPU learning harness.
- **Other checkouts:** `~/Documents/Narma_experiments` (`narma-experiments`) and `~/Documents/DEQ_experiments` (`deq-experiments`), both clean.
  - Archive branch: `archive/full-snapshot-2026-11`. Tag `pre-surgery` = `2568705`.
- **Prepared empty worktrees** (all at `d46f3bc`, no changes; reuse them or delete them):
  - `.worktrees/ab` (`hw-ab-real`)
  - `.worktrees/shunt` (`fix-shunt`)
  - `.worktrees/kaggle` (`kaggle-nb`)
  - `.worktrees/` and `.agent_scratch/` are listed in `.git/info/exclude`.
- **venv:** Python 3.12, torch 2.15, CPU only. ssr0 has no GPU and 48 shared cores.
  - Installed: pytest, optuna, sklearn, scipy, matplotlib, joblib, torchdiffeq.
  - Missing: torchdeq, nbformat, nbconvert.
- **External data on ssr0:**
  - `~/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/models/canonical_ctle_phase_a_v2.npz` (1.4 MB).
  - `~/Documents/improved-zig-nf-spline-pytorch-default-v2/` (the ZIG forward surrogate, 7.2 MB).
  - `~/Documents/augmented-cvae-ctle`: a symlink to `ASPDAC_2026/kirchhoff_redesign/augmented-cvae-ctle` (the CTLE CSVs).
  - The paper checkpoint is in the repo, at `assets/baselines/paper_trial_0011/`.

## 2. What exists
See `README.md`, `docs/hw/*` and `tests/golden/README.md` for details.
- **Surgery is done.**
  - Removed: NARMA, DEQ, dynamic_leak, noise/SimContext, the legacy libs, budget/drive/ref/skip-linear/node gates, the 4-phase schedule, and the MoE flags.
  - Kept: VCA (`gating/vca.py`), GLN (`gating/gln.py`), freeze_read, shared-sense, the interstage transfer, and the learnable clip.
  - The 10% rule is a single constant: `ctle_metrics_config.DEGRADE_REL_THRESHOLD = 0.10`.
- **Golden suite:** 18 cases in `tests/golden/`, including `paper_trial_0011`, the real paper checkpoint. `scripts/eval_paper_checkpoint.py` reproduces **7.67 / 8.05 / 8.15 %** exactly in about 10 s.
- **hw package:**
  - `hw/units.py`: HWUnits with Vu=0.07 V, C=50 fF, Tu=75 ns, x_hard=5.714.
  - `hw/constraints.py`: the HWConstraints flags, all default OFF.
  - `hw/timemux.py`, and `hw/audit.py`.
  - `hw/cli.py`: unknown `--hw-*` flags raise an error.
  - `hw/presets/{phase3a_core,phase3a_full,phase3b_tm}.json`.
  - Every entry script accepts `--hw-preset/--hw-config/--hw-<flag>`.
  - `hw_slope_g_min` is configurable, default 1/16.
- **SLURM:** `scripts/slurm/retrain_hw_phase_a.sh`, `retrain_hw_bo.sh` and `README.md`. The README includes a local CPU pre-flight, about 50 s/epoch.
- **Docs:**
  - `docs/hw/phase3a_units_and_constraints.md`
  - `docs/hw/phase3b_fabric_impl_retrain.md`
  - `docs/hw/cpu_learning_validation.md`: the fair-lr table, plus a lead caveat that the 3-seed ablation is too noisy to rank constraints.

## 3. User decisions made at the end of session 1 (binding)
1. **Make the OTA input weights A/B as physically realistic as possible** (task A).
2. **Fix the shunt double count** (task B), and deliberately regenerate the affected golden data.
3. **Do NOT run any local retrains on ssr0.** All training goes into a **Kaggle notebook** that the user runs (task C).
   - Tiny smoke checks are acceptable for validating the notebook: a few minutes at most, 1 epoch on a tiny sample.
4. **Phase 4** (Verilog-A modules + PyTorch→Spectre exporter + validation) is now **requested**: see **Task D**. Do it after Task A (the A/B mode decides the OTA input stage). It does not need the Kaggle results; it starts from the paper checkpoint.

## 4. Tasks

### Task A — realistic A/B input attenuation (branch `hw-ab-real`)
**Physics.**
- A and B are gains on the two inputs of the OTA's differential pair, in `cell_library.FreeTanhLibrary._tanh_core`:
  `I = σ(z)·Isat·tanh(gm·(s·(A·x_src − B·x_dst) + θ))`
- They are NOT transistor W/L. W/L and the tail current set gm and Isat.
- In a real single-ended 22FDX diff pair, both gates share gm, so A = B natively.
- A ≠ B needs a passive attenuator on each input: a capacitive divider, gain ≤ 1, switch-reset. That fits the time-multiplexed fabric.
- Asymmetric W/L gives offset, not a gain ratio, so reject that option.

**Implement `hw_ab_mode ∈ {None, "ratio", "differential", "cap_divider"}`.**
- `hw_ratio_ab=True` must stay equivalent to `"ratio"` (backward compatibility).
- `"differential"`: A = B = 1 with no per-edge A/B params. The gain comes from the slope attenuation g.
- `"cap_divider"`: Â and B̂ ∈ [1/r_max, 1], STE-quantised to an N-bit cap code.
  - Defaults: `hw_ab_bits = 3`, `hw_ab_rmax = 2`.
  - Justify them by cap-ratio matching, area, and the extra node parasitic C. Tie that C into the `hw_c_node` model if it's easy.
- Both modes require `hw_slope_bound`.
- Freeze the dead legacy `a_raw`/`b_raw`, as `hw_ratio_ab` already does.
- Keep the BO param counter (`bo_param_sampling.knet_param_count`, which builds with hw=) correct.

**Also:**
- `hw/audit.py`: report the A/B check for the active mode.
- Presets: switch `phase3a_core`, `phase3a_full` and `phase3b_tm` to `cap_divider` (bits 3, rmax 2). Add `phase3a_core_diff` (differential).
- Tests:
  - the mode invariants and the STE gradient;
  - the slope bound still holds;
  - no dead params under each preset (`tests/test_hw_no_dead_params.py`);
  - flags-off golden unchanged;
  - a partial prune keeps the new params (`tests/test_hw_prune_specialised_libs.py`).
- Docs: add a subsection to `docs/hw/phase3a…` §3.1. Note that the paper checkpoint has A/B up to about 11–13, so clipping to rmax = 2 needs a retrain.

### Task B — shunt double count — **DONE, merged into hw-refactor @ 0b19ba5** (commits 7cda290..0b19ba5; only golden `paper_freeze_read_off` refreshed via new `make_golden.py --refresh`; see tests/golden/README.md Regenerations; new tests/test_core_shunt_once.py). Kept below for reference only; remove the empty .worktrees/shunt.
**Bug.** With `freeze_read=False`, the shunt is counted twice:
- `DifferentialStage.rhs` (`differential_stage.py`, around L1256–1261) calls `self.cell_lib(...)`, which is `FreeTanhLibrary.forward`.
- That already returns `i_tanh + i_res` (`cell_library.py`, around L461–462).
- The `if self._has_resistive:` block (around L1308–1326) then adds the shunt AGAIN.
- The copy inside forward is also VCA-gated, even when `vca_gate_shunt=False`.
- Measured on TOY_NET: the shunt contribution is exactly 2.000×.

These paths already count it once: the frozen path (freeze_read=True, which the paper model uses), the boundary/sense/mesh paths, and the hw_compliance path.

**Fix.**
- In the dynamic core path, use the tanh-only current and keep the explicit resistive block.
- freeze_read=True must stay bit-identical: the golden files and the paper checkpoint are unchanged.

**Golden data.**
- Regenerate only the affected golden files. Expected: `paper_freeze_read_off`, but check every case.
- Use `tests/golden/make_golden.py --only <name>`.
- First verify that old − new equals exactly one shunt copy.
- Log the regeneration under a "Regenerations" section in `tests/golden/README.md`.

**Also:** add an analytic regression test, and update the double-count notes in `tests/test_hw_compliance_coverage.py` and `docs/hw/*`.

### Task C — Kaggle notebook (branch `kaggle-nb`)
Do this after A and B are merged, or reference the presets by name.

**Bundle.** `scripts/kaggle/make_kaggle_bundle.sh <git-ref> <out.zip>` builds one zip to upload as a Kaggle Dataset. It contains:
- `git archive <ref>` of the repo, including assets;
- the canonical npz;
- the ZIG v2 dir;
- **the CTLE CSV dir.** This is needed: `dagger-nuance-distillation-kirchhoffnet.py` concatenates 14 named CSVs from DATA_DIR at top level (around L2047–2070), even in canonical mode. Follow the symlink when packing;
- `MANIFEST.txt` with the git sha and file hashes.

Keep the zip at about 100 MB or less.

**Notebook.** `notebooks/kaggle_hw_retrain.ipynb`, nbformat v4, committed with outputs cleared.
- **Parameters cell:** RUNG, EPOCHS, SEED, LR, STEPS_PER_STAGE, DATA_ROOT, OUT_DIR, RESUME, SMOKE, RUN_TOY_ABLATION.
- **Environment:**
  - GPU check and pip installs.
  - Locate the bundle in `/kaggle/input/*` and unpack it to `/kaggle/working/repo`.
  - Override the hardcoded `/home/annaik/...` paths via `--teacher-dir/--data-dir/--canonical-dataset`. Check the exact flags in the script and in `scripts/slurm/retrain_hw_phase_a.sh`.
- **Sanity:** a fast pytest subset, plus `scripts/eval_paper_checkpoint.py` with an assert on 8.05 / 8.15 %.
- **Ladder,** one rung per Kaggle session:

  | Rung | Config |
  |---|---|
  | R0 | paper config, no hw, from scratch (should reproduce about 8.15 %) |
  | R1 | `phase3a_core` |
  | R1d | `phase3a_core_diff` |
  | R2 | `phase3a_full` |
  | R3 | `phase3b_tm` |

  - The paper config and its flags are in `assets/baselines/paper_trial_0011/run_command.txt`. Drop the removed `--kn-moe-*` flags.
  - The hw rungs follow `docs/hw/cpu_learning_validation.md`:
    - Learning rate: start at 0.03 and bracket with 0.1 and 0.01.
    - Use more Heun steps per stage; the rail checks need about 4× the steps.
    - A per-stage `stage_num_steps` exists around L1760 of the dagger script, but there is no CLI flag yet. If needed, add one (e.g. `--kn-steps-per-stage`) as a small code change with a test.
- **Running:**
  - Stream the logs. Save checkpoints and history to OUT_DIR.
  - Check whether the script can resume. If it can't, document how to continue across the 12 h session limit, or save a checkpoint at each early-stop eval.
- **Evaluation:**
  - train/val/test failure at the 10 % rule;
  - `python -m hw.audit` on the best checkpoint;
  - plots;
  - an append-only results CSV.
- **Optional toy ablation:** 10 or more seeds (3 seeds was too noisy). Report means and 95 % CIs.
- **Final cell:** zip OUT_DIR.

**Docs.** `docs/kaggle.md` covers:
- the upload steps;
- the notebook settings (GPU, internet on);
- the run order;
- time estimates: an H100 takes about 1.6–1.8 h per 300 epochs, and Kaggle gives about 12 h per session and about 30 GPU-h per week. Estimate T4/P100 times;
- how to bring results back.

**Validate on ssr0 (SMOKE only).**
1. Install nbformat, nbconvert and ipykernel into the venv (`~/.local/bin/uv pip install --python venv/bin/python …`).
2. Unzip the bundle to `/tmp/kaggle_sim/input/...`.
3. Execute the notebook headlessly for R0 and R1 with a tiny sample count, 5 min each at most.
4. Clean up.

**When done:** rebuild the bundle from the merged `hw-refactor`, and give the user the notebook, the zip path and `docs/kaggle.md`.

### Task D — Phase 4: Verilog-A modules, Spectre exporter, validation (branch e.g. `phase4-veriloga`)
Build behavioural Verilog-A for every block, using the decisions from Phase 3 and earlier, plus an exporter from a checkpoint to a Spectre netlist, then validate it against the PyTorch forward. Transistor-level 22FDX replacement (OTA first, SLVT/LVT) comes AFTER this and only when the user asks.

**Source of truth for every equation and number:**
- `docs/analysis/circuit_model_report.md`: §3 has the per-component spec C1–C13 with Verilog-A notes; §7 is the suggested component list.
- `docs/hw/phase3a_units_and_constraints.md`:
  - §2.4–2.5: units;
  - §3.1–3.10: constraint formulas and their Verilog-A mappings;
  - the Task A cap-divider subsection, once written.
- `docs/hw/phase3b_fabric_impl_retrain.md` §1: S&H, phase sequence, timing budget, parameter banks.
- `hw/units.py`, `hw/constraints.py`, `hw/timemux.py`: the code is authoritative where the docs and the code differ.

**Fixed decisions to encode:**
- **Supply and nodes:** 0.8 V core, **single-ended** nodes around **Vcm = 0.4 V**, and `x = (V − Vcm)/Vu`.
- **Units** (`HWUnits` defaults): Vu = 0.07 V/unit, C_u = 50 fF per unit of C_eff, Tu = 350 ns / 4.660583 ≈ 75.1 ns per time unit, Iu = C_u·Vu/Tu ≈ 46.6 nA, G_u = C_u/Tu ≈ 0.666 µS, x_hard = VDD/(2·Vu) = 5.714.
- **One fabric reused in time** across the 7 stages, with per-stage parameter banks.
  - Physical stage copies are the documented fallback; the exporter must support both (`--fabric timemux|copies`).
- **Single 10% failure rule** (`ctle_metrics_config`).
- **The shunt is counted once.** The fix is merged at `7cda290`.

**Two fidelity modes** (a per-module parameter or a netlist switch):
1. `legacy` replicates the current dimensionless model exactly, to validate the exporter on `assets/baselines/paper_trial_0011`. It uses:
   - softplus A/B;
   - `gm·(s·(A·x_s − B·x_d) + θ)` plus the 0.01·u gradient-floor term;
   - the input compliance gate σ((x_max−|V|)/0.02);
   - the soft clip (two steep sigmoid currents, 0.05 units);
   - conservative two-terminal edge current;
   - instantaneous interstage rewrite.
2. `hw` implements the enabled `HWConstraints`:
   - the slope `slope = g·f_IC/(2·n·$vt)` with g ∈ [g_min, 1];
   - the A/B input attenuator from Task A (`cap_divider`: V(pin) = Vcm + Â·(V(src)−Vcm), Â on an N-bit code ≤ 1; or `differential`);
   - θ bound;
   - **leg-wise output compliance** `I(d) <+ Ip·(1−exp(−vtop/$vt)) − In·(1−exp(−vbot/$vt))` with λv output conductance, replacing the soft clip and the input gate;
   - transfer clamp ±x_hard; per-node C including parasitics; GLN "tail" mode (Isat multiplier); VCA as a tail-current multiplier; Isat_eff ceiling;
   - quantised Vref and crossbar codes;
   - S&H non-idealities.

**Modules** (`veriloga/*.va` plus a shared `veriloga/hrknet_constants.vams`; use `disciplines.vams`/`constants.vams`, electrical, `$vt`, `limexp`, no hard discontinuities):

| Module | Function |
|---|---|
| `ota_edge` | tanh OTA. Parameters A/B or Â/B̂, s (switch bit), θ, gm or g, Isat. σ(z) is folded into Isat_eff/G_eff at export, which is exact when the VCA gate is 1; otherwise keep σ(z) as a parameter. Inputs for the VCA gain and the GLN gm/Isat trim as control voltages or per-inference real params. Mode legacy or hw. |
| `shunt` | G_eff = G·σ(z), VCA-gated only if `vca_gate_shunt`. |
| `node_cap_leak` | C_node: c_eff·C_u, or the physical per-node vector. Fixed leak 0.0486·G_u ≈ 32 nS to Vcm (an active gm leak in transistors). Optional legacy soft clip. |
| `rail_clip` | legacy only; in hw mode the rails come from the OTA leg compliance. |
| `sample_hold` | switch plus hold cap (default 100 fF, `timemux.c_hold_ff`). Charge-injection offset, droop, optional kT/C (transient noise). Clocked by the phase controller. With freeze_read, the core OTA inputs read the held stage-entry voltages, while the shunt/sense/boundary/leak read the live node. |
| `boundary_ota` | same equation as `ota_edge`. Its source is an ideal input voltage V = Vcm + Vu·u, where u is the model's scaled input. The log10 + min-max + clamp preprocessing stays off-chip, in the exporter/testbench. Boundary fan-out `{0:[0,4],1:[1,5],2:[2,6],3:[3,7]}` for the paper config. |
| `sense_ota` + `crossbar` + `accumulator` | shared-sense readout (2 senses per node in the paper config), with Vref per stage. The signed crossbar is offset-binary around Vcm with a replica column (3a §3.7). 7 accumulator nodes. |
| `interstage_transfer` | per node x' = w1·x + w2·tanh(x) + w3·ReLU(x − Vth), clamped to ±x_hard in hw mode. Implement it as an event at the stage boundary that rewrites the node capacitor, e.g. `idt(..., ic, reset)` or a switched voltage source during the transfer phase. |
| `vca_bias`, `gln_bias` | `analog function`s of the 4 input voltages that give the per-edge multipliers. They are static for one inference (they depend only on u). |
| `phase_ctrl` | clock/phase sequencer per stage boundary: hold → transfer → bank switch → evolve, with durations from the 3b §1.3/§1.5 budget. Outputs the stage index that selects the parameter banks. |

- The OutputAffine and the post-readout transfer are digital, so apply them off-chip in Python to the final accumulator voltages.
- Parameter banks: Verilog-A parameter arrays indexed by stage (`parameter real isat[0:6]`), or netlist-level per-stage instances in `copies` mode.

**Exporter.** `export/spectre_export.py`, CLI e.g. `python -m export.spectre_export --checkpoint ... [--hw-config ...] --mode legacy|hw --fabric timemux|copies --specs N --out dir/`.
- It reads the checkpoint (use `checkpoints.load_paper_checkpoint` for the paper model), the hw config, and the topology buffers persisted in the state_dict (`src/dst/boundary_*/readout_sense_src`, ...).
- It converts everything to physical units through `HWUnits`.
- It emits a `.scs` netlist with `ahdl_include`, one instance per edge, per-stage parameter banks, ideal input sources for N test specs, and a `tran` covering t_span·Tu plus the boundary overhead.
- **Strict mapping.** Every state_dict key must map to a Verilog-A parameter, or appear in an explicit, commented list of digital/off-chip keys. An unmapped key fails the export; this is the "no feature the exporter can't express" test. Also emit a JSON parameter table.

**Validation:**
1. **Without Spectre (required, CI-able).** Write a small torch "netlist interpreter" that simulates the exported parameter table with the module equations, and check that it reproduces `model.forward`:
   - legacy mode on the paper checkpoint: exact to float tolerance, after unit conversion;
   - hw mode on a hw-flagged TOY_NET.
   Add pytest tests (`tests/test_spectre_export.py`) and keep the default suite all-pass.
2. **Verilog-A syntax/compile check, and Spectre transient if available.**
   - Spectre is NOT on the PATH in non-interactive shells on ssr0. Source the Cadence/PDK setup in tcsh, the way `spice_failure_check.py` does (around L254 and L1294; `~/workarea_GF22_FDX_EXT/kit.gf22fdx_ext.v1.0_4.1_stack19_oa.csh`), then check `which spectre` and the licence.
   - If it is available, run the transient on 16–64 canonical test specs. Compare the per-stage node trajectories and the output logits against PyTorch, and report the max error.
3. **CTLE closure.** Feed the predicted 7 CTLE knobs from the Spectre-simulated network to `spice_failure_check.py --pred-params-csv` (Ocean templates in `~/Documents/train_ctle/ocn_template_ideal_bob`, PDK path hardcoded in that script). Report the failure rate at the 10% rule against the PyTorch-predicted knobs.
4. **Checkpoints:**
   - the paper checkpoint (legacy mode) first;
   - then the Kaggle-retrained hw checkpoints (hw mode), when the user brings them back;
   - run `python -m hw.audit` on each hw checkpoint before export.

**Docs.** `docs/hw/phase4_veriloga.md` covers: the module list with port and parameter tables, the unit conversions, the fidelity modes, how to run the exporter and Spectre, the validation results, and known gaps.
### Task E — make `~/Documents/Narma_experiments` and `~/Documents/DEQ_experiments` self-sufficient
The user wants each clone to contain everything it needs, with no reliance on other workspaces: `kirchhoffnet_refactor` and its venv, `tron_eq`, `ASPDAC_2026`, `augmented-cvae-ctle`, `improved-zig-*`, or Windows paths. The folder name is `Narma_experiments`; there is no uppercase `NARMA_experiments`. Ask the user before renaming it.

**Read-only audit (session 1, 2026-11-22):**
- **Git.** Both clones have full history and are not shallow: Narma `narma-experiments` @ `03849d7` (269 commits), DEQ `deq-experiments` @ `ef76c73` (271 commits). Tracked files are clean.
  - Remotes: `origin` (GitHub) and `local` (→ ~/Documents/kirchhoffnet_refactor).
  - Both come from the pre-surgery snapshot, so they hold all scripts, docs, tests and notebooks.
- **No venv inside either clone.** The only requirements files are `requirements_local.txt` and `requirements_all_can.txt`, which are large platform-specific environment snapshots rather than minimal manifests.
  - With the refactor venv as interpreter, `train_script`, `narma_experiment`, `narma_revised_plan` and `narma_linear_controls` import fine.
  - On the Narma clone, `deq_solver`/`deq_diagnostics` are missing, because they were deleted in 7b34b60 before the snapshot. On the DEQ clone they import fine (they were restored).
- **pytest.** `--collect-only` collects 313 items in both clones, then errors: `test_friedman2_small_world.py:5` raises `KeyError 'hidden_family'`, because the module runs a diagnostic print at import time.
- **DEQ artifacts.** `artifacts/deq/` (~1.8 MB: `model.pt` 43 KB, the run notebook, diagnostics and plots) is on disk but **gitignored**, so a fresh clone won't have it.
- **NARMA data is synthetic.** `narma_experiment` generates its own streams and needs no data files.
- **Hardcoded external paths in live code:**
  - `ctle_dagger_common.py:74-76` and `dagger-mlp-distillation-kirchhoffnet.py:48,51,766-767` point to `/home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2` (`ctle_conditional_flow.pt` 6.96 MB, `dagger_student_moe.pt` 232 KB), `augmented-cvae-ctle` (a symlink into ASPDAC_2026) and `dagger_output`. The same applies to `dagger-nuance-distillation-kirchhoffnet.py` and the generative-distillation scripts.
  - The NARMA/post-twin `*_batch.sh` scripts load cluster modules and `source /home/annaik/Documents/tron_eq/venv/bin/activate`.
  - `spice_failure_check.py:31,240,254,1294` uses `~/Documents/train_ctle` (Ocean templates) and `~/workarea_GF22_FDX_EXT` (Cadence/PDK). These are external EDA tools; leaving them external is acceptable, but make them configurable.
  - Windows paths: `generate_report.py`, `_count_params.py`, `test_bo_guard.py`.
  - `test_training.py` points at the ASPDAC_2026 venv and repo.
  - Historical docs and scratch files contain `C:\Users`, `Y:\Documents\ASPDAC_2026` and `kirchhoffnet_realistic` paths. These are archival.
- **Not finished:** the full AST import-gap inventory and a tiny DEQ forward pass. Do them first.

**Fix plan, for each clone, committed on its own branch inside its own folder:**
1. **Own environment.** Create `venv/` inside the clone (`~/.local/bin/uv venv --python 3.12 venv`, gitignored). Add a minimal `requirements.txt` covering every third-party import found by an AST scan: torch, numpy, pandas, scipy, scikit-learn, matplotlib, joblib, networkx, optuna, pytest, torchdiffeq, and torchdeq for DEQ. Install it and verify every module imports in that venv.
2. **Missing local modules.** On the Narma clone, either restore `deq_solver.py`/`deq_diagnostics.py` from `7b34b60^` (cheap; makes the imports resolve) or guard the imports. List any other local module that doesn't exist, e.g. `utils.model` in `kn_benchmark_friedman2.py`, and fix it or mark it dead.
3. **Data and assets inside the clone.** Copy, don't symlink, the CTLE data that the dagger/distillation/CTLE scripts need into the clone:
   - `assets/zig/`: the ZIG v2 dir;
   - `assets/teachers/`: the MLP teachers `trial_0019` and the v4 `dagger_student_plain.pt`, if any script uses them;
   - `data/augmented-cvae-ctle/`: the CSVs, ~18 MB;
   - the canonical npz, if it's used.
   Commit the small binaries via a `!assets/**` gitignore exception. Keep the larger CSVs ignored but present on disk, or commit them if the user prefers. On the DEQ clone, **commit `artifacts/deq/`** (1.8 MB) so a fresh clone gets it.
4. **Paths.** Rewrite live absolute paths to repo-relative defaults with environment-variable overrides, e.g. `KNET_DATA_DIR`, `KNET_TEACHER_DIR`, `KNET_OUTPUT_DIR`, `CADENCE_SETUP`, `OCN_TEMPLATE_DIR`. Change the batch scripts to activate `"$REPO/venv"` instead of `tron_eq/venv`. Leave historical docs and scratch files as they are, but add a note that their paths are archival.
5. **Tests.** Fix the import-time `KeyError` in `test_friedman2_small_world.py` so that `pytest --collect-only` is clean, then run a fast subset in the clone's own venv and record the pass/fail count. Pre-surgery failures are expected; document them rather than fixing them all.
6. **Remove the `local` remote**, which points at kirchhoffnet_refactor. Keep `origin`.
7. **Per-clone `README_BRANCH.md`:** what the branch is for, venv setup, the data and assets layout, how to run the main NARMA experiment or the DEQ notebook and script, and the known failing tests.
8. **Verify isolation.**
   - `grep` shows zero live references to external workspaces.
   - Imports, `--help` of the main scripts, one tiny NARMA step and one tiny DEQ forward all run with the clone's venv, with cwd = clone and outputs going to `/tmp`.
   - A fresh `git clone <clone> /tmp/x` plus a copy of the ignored data dirs also works.
   - No real training. Tiny smoke steps only, as with the Kaggle rule.
## 5. Other findings and open questions (tell the user, or keep in mind)
- **A/B strictness:** the user asked for maximal realism (task A). Ask before choosing anything less strict than `cap_divider` with 3 bits and rmax 2.
- **CPU toy validation:** mean held-out MSE relative to the unconstrained model, each at its own best lr, 3 seeds.

  | Preset | Friedman-2 | Toy |
  |---|---|---|
  | `phase3a_core` | 0.79× | 2.1× |
  | `phase3a_full` | 0.82× | 3.9× |
  | `phase3b_tm` | 3.6× | 4.4× |

  The unconstrained seed spread is itself up to 11×, so read 0.79× as "no measurable cost", not "better".
- **Stiffness:**
  - The trajectory-Jacobian dt·ρ (2–5) and `hw.constraints.stiffness_report` (0.1–0.25) disagree.
  - Fixed-weight step refinement changes the outputs by less than 1e-5.
  - A rail overshoot (6.70) resolves at ×4 steps.
  - So use more steps for the hw rungs, and audit both metrics.
- **Paper checkpoint vs hw limits** (`hw/audit.py`): 13 FAIL / 2 WARN / 11 PASS.
  - The interstage-transfer output reaches 10.4 units, against x_hard 5.7.
  - Isat is below the 0.6 floor on about 80 % of core cells.
  - A/B reaches about 11.
- **Fixed in session 1:**
  - surgery regressions: the `solver=` kwarg at 6 call sites, and the `--solver` BO flag;
  - the partial-pruning library loss;
  - three three_phase crashes;
  - the missing compliance scatters.
- **Not pushed:** the GitHub origin (`ZiZizou/kirchhoffnet_realistic`) has NOT been pushed with these branches. The bundle zip is how the code gets to Kaggle.

## 6. Working conventions on ssr0
- **Shell:** the remote login shell is tcsh, so always use `bash -c '...'`. Heredocs inside `bash -c` break; write scripts to files in `.agent_scratch/` instead.
- **Long jobs:** rw_exec calls die after about 20 s. For anything long, use `nohup … > /tmp/x.log 2>&1 &` and poll.
- **CRLF:** `kn_bayes_opt.py` uses CRLF line endings; keep them. A helper is at `.agent_scratch/tron_eq_leftovers/patch_crlf.py`.
- **CPU threads:** ssr0 is shared. Many tiny processes with 2 threads each are faster than a few with 12 threads.
- **Subagent routes (session 1):** kiro/claude-sonnet-5.5 and minimax often died silently. openai-codex/gpt-6-luna and the default route were reliable. Running subagents could not be messaged, so have them commit after each step.
- **Golden files:** never regenerate them except deliberately (task B), and log every regeneration.




