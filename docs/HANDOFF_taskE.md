# Task E — self-sufficiency audit and handoff plan

Read-only audit on ssr0. Clone files were not modified. Audit scripts/results are in `kirchhoffnet_refactor/.agent_scratch/taskE/`. Import checks used the shared refactor venv; torchdeq is absent.

**Pre-audit ignored status:** Narma: `!! .pytest_cache/`, `!! __pycache__/`; DEQ: same plus `!! artifacts/`. Tracked status was clean. Recheck exactly at the end.

## Narma (`/home/annaik/Documents/Narma_experiments`)

- Tracked `.py` files AST-scanned: 113; top-level import checks: 92 (max 4 parallel, 20 s timeout, cwd=clone).
- Minimal third-party PyPI requirements; importing files:
  - `catboost` — `tree_benchmark.py`
  - `joblib` — `ctle_dagger_common.py`, `dagger-mlp-distillation-kirchhoffnet.py`, `dagger-nuance-distillation-kirchhoffnet.py`, `docs/analysis/scratch/audit.py`, `docs/analysis/scratch/audit2.py`, `docs/analysis/scratch/dbg.py`, `docs/analysis/scratch/teacher.py`, `fixed-mlp-distillation-kirchhoffnet.py`, `generative-distillation-improved-dagger-nuance-mlp.py`, `mlp_benchmark_ctle.py`, `train_ctle.py`
  - `lightgbm` — `tree_benchmark.py`
  - `matplotlib` — `ctle_dagger_common.py`, `dagger-mlp-distillation-kirchhoffnet.py`, `dagger-nuance-distillation-kirchhoffnet.py`, `generative-distillation-improved-dagger-nuance-mlp.py`, `generative-distillation-plain-mlp-bayes-opt.py`, `kn_bayes_opt.py`, `kn_benchmark_friedman2.py`, `mlp_bayes_opt.py`, `mlp_benchmark.py`, `mlp_benchmark_ctle.py`, `mlp_benchmark_friedman1.py`, `mlp_benchmark_friedman2.py`, `mlp_benchmark_friedman3.py`, `mlp_benchmark_housing.py`, `plot_bo_descent.py`, `plot_bo_descent_4pane.py`, `plot_phase_a_ctle_descent.py`, `train_ctle.py`, `train_script.py`, `tree_benchmark.py`, `visualize.py`, `xgb_benchmark_housing.py`
  - `networkx` — `visualize.py`
  - `numpy` — `check_optuna_versions.py`, `ctle_dagger_common.py`, `ctle_kirchhoff_bo.py`, `dagger-mlp-distillation-kirchhoffnet.py`, `dagger-nuance-distillation-kirchhoffnet.py`, `docs/analysis/scratch/audit.py`, `docs/analysis/scratch/audit2.py`, `docs/analysis/scratch/dbg.py`, `docs/analysis/scratch/dbg2.py`, `docs/analysis/scratch/final.py`, `docs/analysis/scratch/flags.py`, `docs/analysis/scratch/fwd1.py`, `docs/analysis/scratch/load1.py`, `docs/analysis/scratch/npz.py`, `docs/analysis/scratch/npz2.py`, `docs/analysis/scratch/same.py`, `docs/analysis/scratch/teachchk.py`, `dry_run_knet_input_clipping.py`, `fixed-mlp-distillation-kirchhoffnet.py`, `generative-distillation-improved-dagger-nuance-mlp.py`, `generative-distillation-plain-mlp.py`, `kn_bayes_opt.py`, `knet_gating_audit1.py`, `knet_multistage_audit.py`, `mlp_bayes_opt.py`, `mlp_benchmark_ctle.py`, `plot_bo_descent.py`, `plot_bo_descent_4pane.py`, `plot_phase_a_ctle_descent.py`, `spice_failure_check.py`, `test_smoke.py`, `train_ctle.py`, `tree_benchmark.py`, `xgb_benchmark_housing.py`
  - `optuna` — `bo_param_sampling.py`, `check_optuna_versions.py`, `ctle_kirchhoff_bo.py`, `fixed-distillation-bayes-opt.py`, `generative-distillation-plain-mlp-bayes-opt.py`, `kn_bayes_opt.py`, `mlp_bayes_opt.py`, `probe_study_db.py`, `summarize_fixed_bo.py`, `test_bo_guard.py`
  - `pandas` — `ctle_dagger_common.py`, `dagger-mlp-distillation-kirchhoffnet.py`, `dagger-nuance-distillation-kirchhoffnet.py`, `dry_run_knet_input_clipping.py`, `fixed-mlp-distillation-kirchhoffnet.py`, `generative-distillation-improved-dagger-nuance-mlp.py`
  - `pytest` — `test_f1_f2.py`
  - `scikit-learn` — `ctle_dagger_common.py`, `dagger-nuance-distillation-kirchhoffnet.py`, `gen_network_images.py`, `generative-distillation-improved-dagger-nuance-mlp.py`, `test_smoke.py`, `train_script.py`, `tree_benchmark.py`
  - `scipy` — `train_script.py`
  - `torch` — `_count_params.py`, `_test_step.py`, `analog_noise.py`, `audit_aux_core.py`, `audit_post_twin_gates.py`, `audit_twin_standalone.py`, `aux_core_probe.py`, `cell_library.py`, `config.py`, `ctle_dagger_common.py`, `ctle_kirchhoff_bo.py`, `dagger-mlp-distillation-kirchhoffnet.py`, `dagger-nuance-distillation-kirchhoffnet.py`, `differential_stage.py`, `docs/analysis/scratch/audit.py`, `docs/analysis/scratch/audit2.py`, `docs/analysis/scratch/connectivity.py`, `docs/analysis/scratch/dbg.py`, `docs/analysis/scratch/final.py`, `docs/analysis/scratch/flags.py`, `docs/analysis/scratch/fwd1.py`, `docs/analysis/scratch/insp.py`, `docs/analysis/scratch/inspect_models.py`, `docs/analysis/scratch/load1.py`, `docs/analysis/scratch/param_counts.py`, `docs/analysis/scratch/plain.py`, `docs/analysis/scratch/same.py`, `docs/analysis/scratch/teachchk.py`, `docs/analysis/scratch/teacher.py`, `fixed-mlp-distillation-kirchhoffnet.py`, `gen_network_images.py`, `generate_report.py`, `generative-distillation-improved-dagger-nuance-mlp.py`, `generative-distillation-plain-mlp-bayes-opt.py`, `generative-distillation-plain-mlp.py`, `gln_rails.py`, `io_mapper.py`, `kirchhoff_net.py`, `kirchhoff_noise.py`, `kn_bayes_opt.py`, `kn_benchmark_friedman2.py`, `knet_gating_audit1.py`, `knet_multistage_audit.py`, `mlp_bayes_opt.py`, `mlp_benchmark.py`, `mlp_benchmark_ctle.py`, `mlp_benchmark_friedman1.py`, `mlp_benchmark_friedman2.py`, `mlp_benchmark_friedman3.py`, `mlp_benchmark_housing.py`, `narma_advisor_probes.py`, `narma_experiment.py`, `narma_linear_controls.py`, `narma_phase7_leak_bands.py`, `narma_phase8_tied_gate_probe.py`, `narma_phase8_tied_gates_audit.py`, `narma_revised_plan.py`, `narma_spectral_norm.py`, `post_twin_gates.py`, `probe_audit1.py`, `probe_audit2.py`, `probe_audit_nodeact.py`, `probe_audit_revised.py`, `probe_audit_split.py`, `probe_audit_transplant.py`, `sim_context.py`, `smoke_audit3.py`, `spice_failure_check.py`, `stage_transfer.py`, `tap_rails.py`, `test_f1_f2.py`, `test_smoke.py`, `test_training.py`, `topology.py`, `train.py`, `train_ctle.py`, `train_script.py`, `tree_benchmark.py`, `vca_sanity.py`, `visualize.py`, `xgb_benchmark_housing.py`
  - `torchdiffeq` — `ctle_kirchhoff_bo.py`
  - `tqdm` — `narma_experiment.py`, `train_script.py`
  - `xgboost` — `tree_benchmark.py`, `xgb_benchmark_housing.py`

- Local-MISSING modules (every listed occurrence is unguarded by try/except):
  - `deq_solver` — `differential_stage.py:2218; test_smoke.py:6959/6981/7465`
  - `deq_diagnostics` — `test_smoke.py:7507/7533; train_script.py:237`
  - `narma_gain_norm` — `_test_step.py:2`
  - `utils.model` — `kn_benchmark_friedman2.py:70,560`
  - `utils.topology` — `kn_benchmark_friedman2.py:71,561`
  - `sparse_solver_data` — `test_smoke.py:648,666,681,698,756,773`
  - `sparse_solver_topology` — `test_smoke.py:682,699`
  - `sparse_solver_baseline` — `test_smoke.py:736,757`
  - `sparse_solver_track` — `test_smoke.py:774`

- Import check table:
  - PASS 81: `__future__`, `analog_noise`, `argparse`, `aux_core_probe`, `bo_param_sampling`, `cell_library`, `collections`, `config`, `copy`, `csv`, `ctle_dagger_common`, `dataclasses`, `differential_stage`, `functools`, `gc`, `gln_rails`, `glob`, `hashlib`, `html`, `importlib`, `inspect`, `io`, `io_mapper`, `itertools`, `joblib`, `json`, `kirchhoff_net`, `kirchhoff_noise`, `logging`, `math`, `matplotlib`, `mlp_benchmark`, `mlp_benchmark_friedman1`, `mlp_benchmark_friedman2`, `mlp_benchmark_friedman3`, `mlp_benchmark_housing`, `multiprocessing`, `narma_advisor_probes`, `narma_experiment`, `narma_linear_controls`, `narma_phase7_leak_bands`, `narma_revised_plan`, `narma_spectral_norm`, `networkx`, `numpy`, `optuna`, `os`, `pandas`, `pathlib`, `pickle`, `post_twin_gates`, `pytest`, `random`, `re`, `runpy`, `scipy`, `shlex`, `shutil`, `sim_context`, `sklearn`, `socket`, `sqlite3`, `stage_transfer`, `statistics`, `subprocess`, `sys`, `tap_rails`, `tempfile`, `time`, `topology`, `torch`, `torchdiffeq`, `tqdm`, `train`, `train_ctle`, `train_script`, `types`, `typing`, `visualize`, `warnings`, `zipfile`.
  - FAIL `catboost`: FAIL: ModuleNotFoundError: No module named 'catboost'.
  - FAIL `deq_diagnostics`: FAIL: ModuleNotFoundError: No module named 'deq_diagnostics'.
  - FAIL `deq_solver`: FAIL: ModuleNotFoundError: No module named 'deq_solver'.
  - FAIL `lightgbm`: FAIL: ModuleNotFoundError: No module named 'lightgbm'.
  - FAIL `narma_gain_norm`: FAIL: ModuleNotFoundError: No module named 'narma_gain_norm'.
  - FAIL `sparse_solver_baseline`: FAIL: ModuleNotFoundError: No module named 'sparse_solver_baseline'.
  - FAIL `sparse_solver_data`: FAIL: ModuleNotFoundError: No module named 'sparse_solver_data'.
  - FAIL `sparse_solver_topology`: FAIL: ModuleNotFoundError: No module named 'sparse_solver_topology'.
  - FAIL `sparse_solver_track`: FAIL: ModuleNotFoundError: No module named 'sparse_solver_track'.
  - FAIL `utils`: FAIL: ModuleNotFoundError: No module named 'utils'.
  - FAIL `xgboost`: FAIL: ModuleNotFoundError: No module named 'xgboost'.

- Absolute-path grep (tracked live `.py`/`.sh`/yaml/toml/json; excludes `docs/`, archives, scratch; tracked `.ipynb` scanned as text too).
  - `_count_params.py:4` — target `C:` (Windows path; absent on ssr0): `root = Path(r'C:\Users\Atharva\Downloads\results_many_models_realizable_knet_nb')`
  - `_test_step.py:1` — target `C:` (Windows path; absent on ssr0): `import sys; sys.path.insert(0, r'C:\Users\Atharva\Documents\kirchhoffnet_realistic')`
  - `ctle_dagger_common.py:74` — target `improved-zig-nf-spline-pytorch-default-v2` (exists): `DEFAULT_TEACHER_DIR = '/home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2/'`
  - `ctle_dagger_common.py:75` — target `augmented-cvae-ctle` (exists (symlink)): `DEFAULT_DATA_DIR = '/home/annaik/Documents/augmented-cvae-ctle/'`
  - `ctle_dagger_common.py:76` — target `other path` (unverified): `DEFAULT_OUTPUT_DIR = '/home/annaik/Documents/dagger_output'`
  - `dagger-mlp-distillation-kirchhoffnet.py:48` — target `augmented-cvae-ctle` (exists (symlink)): `DATA_DIR = '/home/annaik/Documents/augmented-cvae-ctle/'`
  - `dagger-mlp-distillation-kirchhoffnet.py:51` — target `other path` (unverified): `OUTPUT_DIR = '/home/annaik/Documents/dagger_output_mlp_knet'`
  - `dagger-mlp-distillation-kirchhoffnet.py:766` — target `improved-zig-nf-spline-pytorch-default-v2` (exists): `        '/home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2/',`
  - `dagger-mlp-distillation-kirchhoffnet.py:767` — target `augmented-cvae-ctle` (exists (symlink)): `        '/home/annaik/Documents/augmented-cvae-ctle/',`
  - `dagger-mlp-distillation-kirchhoffnet.py:1055` — target `augmented-cvae-ctle` (exists (symlink)): `            "Pass --dataset-dir /kaggle/input/datasets/awekill/augmented-cvae-ctle "`
  - `dagger-nuance-distillation-kirchhoffnet.ipynb:61` — target `/home/annaik/Documents/train_ctle` (exists): `    "# os.chdir('/home/annaik/Documents/train_ctle')\n",`
  - `dagger-nuance-distillation-kirchhoffnet.ipynb:63` — target `other path` (unverified): `    "TEACHER_DIR = '/kaggle/input/models/awekill/improved-zig-nf-spline/pytorch/default/2'\n",`
  - `dagger-nuance-distillation-kirchhoffnet.ipynb:64` — target `augmented-cvae-ctle` (exists (symlink)): `    "DATA_DIR = '/kaggle/input/datasets/awekill/augmented-cvae-ctle'\n",`
  - `dagger-nuance-distillation-kirchhoffnet.ipynb:117` — target `kirchhoffnet_realistic` (missing on ssr0): `    "    \"/kaggle/working/kirchhoffnet_realistic\",\n",`
  - `dagger-nuance-distillation-kirchhoffnet.ipynb:1459` — target `augmented-cvae-ctle` (exists (symlink)): `    "history_dir = '/kaggle/input/datasets/awekill/augmented-cvae-ctle'\n",`
  - `dagger-nuance-distillation-kirchhoffnet.py:64` — target `improved-zig-nf-spline-pytorch-default-v2` (exists): `TEACHER_DIR = '/home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2/'`
  - `dagger-nuance-distillation-kirchhoffnet.py:65` — target `augmented-cvae-ctle` (exists (symlink)): `DATA_DIR = '/home/annaik/Documents/augmented-cvae-ctle/'`
  - `dagger-nuance-distillation-kirchhoffnet.py:66` — target `other path` (unverified): `OUTPUT_DIR = '/home/annaik/Documents/dagger_output'`
  - `dagger-nuance-distillation-kirchhoffnet.py:801` — target `other path` (unverified): `    # checkpoint (e.g. /home/annaik/.../dagger_checkpoint.pt from a prior`
  - `dry_run_knet_input_clipping.py:73` — target `augmented-cvae-ctle` (exists (symlink)): `                        default=Path("/home/annaik/Documents/augmented-cvae-ctle"))`
  - `gen_network_images.py:3` — target `ASPDAC_2026` (exists): `Saves to /home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/network_visualization/`
  - `gen_network_images.py:6` — target `ASPDAC_2026` (exists): `sys.path.insert(0, '/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal')`
  - `gen_network_images.py:23` — target `ASPDAC_2026` (exists): `OUT_DIR = '/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/network_visualization'`
  - `generate_report.py:12` — target `C:` (Windows path; absent on ssr0): `RESULTS_ROOT = Path(r"C:\Users\Atharva\Downloads\results_many_models_realizable_knet_nb")`
  - `generate_report.py:13` — target `C:` (Windows path; absent on ssr0): `EXTRA_EPOCHS_ROOT = Path(r"C:\Users\Atharva\Downloads\results_mlp_benchmarks_extra_epochs")`
  - `generate_report.py:14` — target `C:` (Windows path; absent on ssr0): `OUTPUT_HTML = Path(r"C:\Users\Atharva\Documents\kirchhoffnet_realistic\comparison_report.html")`
  - `generate_report.py:17` — target `kirchhoffnet_realistic` (missing on ssr0): `SKIP_DIRS = {"kirchhoffnet", "kirchhoffnet_realistic"}`
  - `generative-distillation-improved-dagger-nuance-mlp.py:40` — target `/home/annaik/Documents/train_ctle` (exists): `# os.chdir('/home/annaik/Documents/train_ctle')`
  - `generative-distillation-improved-dagger-nuance-mlp.py:42` — target `improved-zig-nf-spline-pytorch-default-v2` (exists): `TEACHER_DIR = '/home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2/'`
  - `generative-distillation-improved-dagger-nuance-mlp.py:43` — target `augmented-cvae-ctle` (exists (symlink)): `DATA_DIR = '/home/annaik/Documents/augmented-cvae-ctle/'`
  - `generative-distillation-improved-dagger-nuance-mlp.py:44` — target `other path` (unverified): `OUTPUT_DIR = '/home/annaik/Documents/dagger_output'`
  - `generative-distillation-improved-dagger-nuance-mlp.py:47` — target `augmented-cvae-ctle` (exists (symlink)): `    _kaggle_fallback = '/kaggle/input/datasets/awekill/augmented-cvae-ctle'`
  - `generative-distillation-improved-dagger-nuance.ipynb:64` — target `/home/annaik/Documents/train_ctle` (exists): `    "# os.chdir('/home/annaik/Documents/train_ctle')\n",`
  - `generative-distillation-improved-dagger-nuance.ipynb:66` — target `other path` (unverified): `    "TEACHER_DIR = '/kaggle/input/models/awekill/improved-zig-nf-spline/pytorch/default/1'\n",`
  - `generative-distillation-improved-dagger-nuance.ipynb:67` — target `augmented-cvae-ctle` (exists (symlink)): `    "DATA_DIR = '/kaggle/input/datasets/awekill/augmented-cvae-ctle'\n",`
  - `generative-distillation-improved-dagger-nuance.ipynb:1601` — target `augmented-cvae-ctle` (exists (symlink)): `    "history_dir = '/kaggle/input/datasets/awekill/augmented-cvae-ctle'\n",`
  - `narma_advisor_probes_batch.sh:47` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_advisor_probes_p2_batch.sh:42` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_diag_batch.sh:58` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_diag_batch.sh:65` — target `tron_eq` (exists): `  || { echo "ERROR: torch/triton import failed in /home/annaik/Documents/tron_eq/venv. Aborting before any GPU work." >&2; exit 3; }`
  - `narma_gating_depth_batch.sh:42` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_gating_e0_batch.sh:40` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_gating_e0_extend_batch.sh:47` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_gating_rung1_batch.sh:42` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_gating_spike_batch.sh:42` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_linear_controls_batch.sh:44` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_revised_plan_batch.sh:63` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_round2_gated_batch.sh:97` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_split_batch.sh:43` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_transplant_batch.sh:46` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_twin_batch.sh:38` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_twin_fine_batch.sh:42` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_vca_sparse_batch.sh:86` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `post_twin_gates_batch.sh:49` — target `tron_eq` (exists): `VENV="${VENV:-/home/annaik/Documents/tron_eq/venv/bin/activate}"`
  - `run_phase_a_knet_bo.sh:44` — target `improved-zig-nf-spline-pytorch-default-v2` (exists): `    --teacher-dir /home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2 \`
  - `run_phase_a_knet_bo.sh:45` — target `augmented-cvae-ctle` (exists (symlink)): `    --data-dir /home/annaik/Documents/augmented-cvae-ctle \`
  - `run_train_knet_single_shot_distillation.sh:62` — target `improved-zig-nf-spline-pytorch-default-v2` (exists): `    --teacher-dir /home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2 \`
  - `run_train_knet_single_shot_distillation.sh:63` — target `augmented-cvae-ctle` (exists (symlink)): `    --data-dir /home/annaik/Documents/augmented-cvae-ctle \`
  - `run_train_knet_single_shot_distillation.sh:69` — target `kirchhoffnet_realistic` (missing on ssr0): `/home/annaik/Documents/kirchhoffnet_realistic/ENV/bin/python /home/annaik/Documents/kirchhoffnet_realistic/dagger-nuance-distillation-kirchhoffnet.py --canonical-dataset "$CANONICAL" \`
  - `run_train_knet_single_shot_distillation.sh:74` — target `kirchhoffnet_realistic` (missing on ssr0): `  --phase-a-power-norm train_mean --kn-readout shared-x2 --output /home/annaik/Documents/kirchhoffnet_realistic/outputs/manual_knet_distillation_mix_trials/trial_0005 \`
  - `spice_failure_check.py:31` — target `/home/annaik/Documents/train_ctle` (exists): `        --template-dir /home/annaik/Documents/train_ctle/ocn_template_ideal_bob \`
  - `spice_failure_check.py:240` — target `/home/annaik/Documents/train_ctle` (exists): `                   default=Path("/home/annaik/Documents/train_ctle/ocn_template_ideal_bob"),`
  - `spice_failure_check.py:254` — target `/home/annaik/workarea_GF22_FDX_EXT` (exists): `                   default="source /home/annaik/workarea_GF22_FDX_EXT/kit.gf22fdx_ext.v1.0_4.1_stack19_oa.csh",`
  - `spice_failure_check.py:453` — target `kirchhoffnet_realistic` (missing on ssr0): `                "Run from the kirchhoffnet_realistic directory or add it to PYTHONPATH.")`
  - `spice_failure_check.py:1097` — target `kirchhoffnet_realistic` (missing on ssr0): `            "Run from the kirchhoffnet_realistic directory.")`
  - `spice_failure_check.py:1294` — target `/home/annaik/workarea_GF22_FDX_EXT` (exists): `        "cd /home/annaik/workarea_GF22_FDX_EXT\n"`
  - `test_audit.py:3` — target `ASPDAC_2026` (exists): `os.chdir('/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal')`
  - `test_bo_guard.py:153` — target `C:` (Windows path; absent on ssr0): `        "fdbo_under_test", r"C:\Users\Atharva\Documents\kirchhoffnet_realistic\fixed-distillation-bayes-opt.py")`
  - `test_bo_guard.py:329` — target `C:` (Windows path; absent on ssr0): `        "fdbo_budget_test", r"C:\Users\Atharva\Documents\kirchhoffnet_realistic\fixed-distillation-bayes-opt.py")`
  - `test_bo_guard.py:334` — target `C:` (Windows path; absent on ssr0): `    _sys.path.insert(0, r"C:\Users\Atharva\Documents\kirchhoffnet_realistic")`
  - `test_bo_guard.py:400` — target `C:` (Windows path; absent on ssr0): `        "fdbo_prior_test", r"C:\Users\Atharva\Documents\kirchhoffnet_realistic\fixed-distillation-bayes-opt.py")`
  - `test_bo_guard.py:402` — target `C:` (Windows path; absent on ssr0): `    _sys.path.insert(0, r"C:\Users\Atharva\Documents\kirchhoffnet_realistic")`
  - `test_bo_guard.py:433` — target `C:` (Windows path; absent on ssr0): `        "fdbo_fanout_test", r"C:\Users\Atharva\Documents\kirchhoffnet_realistic\fixed-distillation-bayes-opt.py")`
  - `test_bo_guard.py:435` — target `C:` (Windows path; absent on ssr0): `    _sys.path.insert(0, r"C:\Users\Atharva\Documents\kirchhoffnet_realistic")`
  - `test_bo_guard.py:448` — target `C:` (Windows path; absent on ssr0): `    src = (_P(r"C:\Users\Atharva\Documents\kirchhoffnet_realistic")`
  - `test_bo_guard.py:458` — target `C:` (Windows path; absent on ssr0): `    fdbo = (_P(r"C:\Users\Atharva\Documents\kirchhoffnet_realistic")`
  - `test_smoke.py:17` — target `ASPDAC_2026` (exists): `    ~/Documents/ASPDAC_2026/venv/bin/python kirchhoff_redesign/ideal/test_smoke.py`
  - `test_smoke.py:2198` — target `ASPDAC_2026` (exists): `    venv_py = os.path.expanduser("~/Documents/ASPDAC_2026/venv/bin/python")`
  - `test_smoke.py:2409` — target `ASPDAC_2026` (exists): `    venv_py = os.path.expanduser("~/Documents/ASPDAC_2026/venv/bin/python")`
  - `test_training.py:8` — target `ASPDAC_2026` (exists): `    ~/Documents/ASPDAC_2026/venv/bin/python kirchhoff_redesign/ideal/test_training.py`
  - `test_training.py:170` — target `ASPDAC_2026` (exists): `    venv_py = "/home/annaik/Documents/ASPDAC_2026/venv/bin/python"`
  - `test_training.py:173` — target `ASPDAC_2026` (exists): `         "/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/mlp_benchmark.py",`
  - `test_training.py:179` — target `ASPDAC_2026` (exists): `         "/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/mlp_benchmark_housing.py",`
  - `test_training.py:274` — target `ASPDAC_2026` (exists): `             "/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/mlp_benchmark.py",`
  - `test_training.py:281` — target `ASPDAC_2026` (exists): `            cwd="/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal",`

- Tiny smoke results:
  - WORKS: built `KirchhoffNetWithIO` using `make_narma_preset(10)` and `make_cell_library("tanh_free")`, generated 2 synthetic samples, ran a 2-item forward with `initial_state=zeros(2,25)`, output shape `(2,1)`. No training. Default topology config had read/write overlap; smoke selected write `[0]`, read `[12]`.

- Assets and actual references: ZIG v2 `/home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2/` = 7.2 MiB; live dagger/generative scripts reference it. Teacher `trial_0019/dagger_student_plain.pt` = 87,210 B at `/home/annaik/Documents/kirchhoffnet_refactor/assets/teachers/trial_0019/dagger_student_plain.pt`; absent in clone but teacher code expects this named checkpoint. `augmented-cvae-ctle` symlink resolves to `/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/augmented-cvae-ctle`, 8.3 MiB/18 files; live CTLE/dagger references. Canonical NPZ `/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/models/canonical_ctle_phase_a_v2.npz`, 1.5 MiB (du rounded), referenced by training scripts.
  - Narma has no `artifacts/deq/`; code inventory contains DEQ imports but no artifact folder.

- Ordered fix checklist, following Task E steps 1–8 in `docs/HANDOFF_session2.md:276-293`:
  1. Clone-local Python 3.12 `venv/`; create minimal `requirements.txt` from third-party list above (DEQ also `torchdeq`); install/verify imports.
  2. Resolve missing imports: Narma restore `deq_solver.py`/`deq_diagnostics.py` from pre-deletion parent `7b34b60^` or guard dead paths; resolve `narma_gain_norm`, sparse_solver data/topology/baseline/track, and `utils.model`/`utils.topology`. DEQ resolve the latter six. Fix or mark dead; current missing imports unguarded.
  3. Copy, never symlink: `assets/zig/` 7.2 MiB; `assets/teachers/trial_0019/dagger_student_plain.pt` 87,210 B (and v4 plain checkpoint if used); `data/augmented-cvae-ctle/` 8.3 MiB; canonical NPZ 1.5 MiB if used. Small binaries may be committed with `.gitignore` exception; CSVs may remain ignored/present. DEQ also commits `artifacts/deq/` 1.8 MiB.
  4. Path edits: `ctle_dagger_common.py:74-76`; `dagger-mlp-distillation-kirchhoffnet.py:48,51,766-767`; `dagger-nuance-distillation-kirchhoffnet.py:64-66`; `generative-distillation-improved-dagger-nuance-mlp.py:42-47`; `dry_run_knet_input_clipping.py:73`; `run_phase_a_knet_bo.sh:44-45`; `run_train_knet_single_shot_distillation*.sh:62-74`; batch scripts sourcing `tron_eq/venv`; `spice_failure_check.py:31,240,254,1294`. Use repo-relative defaults/environment overrides; leave archive/docs/scratch unchanged.
  5. Fix `test_friedman2_small_world.py:5` import-time `KeyError hidden_family`; then `PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider --collect-only` and fast subset. Document remaining pre-surgery failures.
  6. Remove `local` remote (kirchhoffnet_refactor); retain `origin`.
  7. Add `README_BRANCH.md` covering branch purpose, venv, asset layout, main experiment/notebook run, known failing tests.
  8. Verify zero live external refs, imports/`--help`, tiny NARMA and DEQ forward using clone venv/cwd and outputs `/tmp`, plus fresh clone and copied ignored assets. No training.
- Environment-variable names: `KNET_DATA_DIR`, `KNET_TEACHER_DIR`, `KNET_OUTPUT_DIR`, `CADENCE_SETUP`, `OCN_TEMPLATE_DIR`.

## DEQ (`/home/annaik/Documents/DEQ_experiments`)

- Tracked `.py` files AST-scanned: 115; top-level import checks: 93 (max 4 parallel, 20 s timeout, cwd=clone).
- Minimal third-party PyPI requirements; importing files:
  - `catboost` — `tree_benchmark.py`
  - `joblib` — `ctle_dagger_common.py`, `dagger-mlp-distillation-kirchhoffnet.py`, `dagger-nuance-distillation-kirchhoffnet.py`, `docs/analysis/scratch/audit.py`, `docs/analysis/scratch/audit2.py`, `docs/analysis/scratch/dbg.py`, `docs/analysis/scratch/teacher.py`, `fixed-mlp-distillation-kirchhoffnet.py`, `generative-distillation-improved-dagger-nuance-mlp.py`, `mlp_benchmark_ctle.py`, `train_ctle.py`
  - `lightgbm` — `tree_benchmark.py`
  - `matplotlib` — `ctle_dagger_common.py`, `dagger-mlp-distillation-kirchhoffnet.py`, `dagger-nuance-distillation-kirchhoffnet.py`, `generative-distillation-improved-dagger-nuance-mlp.py`, `generative-distillation-plain-mlp-bayes-opt.py`, `kn_bayes_opt.py`, `kn_benchmark_friedman2.py`, `mlp_bayes_opt.py`, `mlp_benchmark.py`, `mlp_benchmark_ctle.py`, `mlp_benchmark_friedman1.py`, `mlp_benchmark_friedman2.py`, `mlp_benchmark_friedman3.py`, `mlp_benchmark_housing.py`, `plot_bo_descent.py`, `plot_bo_descent_4pane.py`, `plot_phase_a_ctle_descent.py`, `train_ctle.py`, `train_script.py`, `tree_benchmark.py`, `visualize.py`, `xgb_benchmark_housing.py`
  - `networkx` — `visualize.py`
  - `numpy` — `check_optuna_versions.py`, `ctle_dagger_common.py`, `ctle_kirchhoff_bo.py`, `dagger-mlp-distillation-kirchhoffnet.py`, `dagger-nuance-distillation-kirchhoffnet.py`, `docs/analysis/scratch/audit.py`, `docs/analysis/scratch/audit2.py`, `docs/analysis/scratch/dbg.py`, `docs/analysis/scratch/dbg2.py`, `docs/analysis/scratch/final.py`, `docs/analysis/scratch/flags.py`, `docs/analysis/scratch/fwd1.py`, `docs/analysis/scratch/load1.py`, `docs/analysis/scratch/npz.py`, `docs/analysis/scratch/npz2.py`, `docs/analysis/scratch/same.py`, `docs/analysis/scratch/teachchk.py`, `dry_run_knet_input_clipping.py`, `fixed-mlp-distillation-kirchhoffnet.py`, `generative-distillation-improved-dagger-nuance-mlp.py`, `generative-distillation-plain-mlp.py`, `kn_bayes_opt.py`, `knet_gating_audit1.py`, `knet_multistage_audit.py`, `mlp_bayes_opt.py`, `mlp_benchmark_ctle.py`, `plot_bo_descent.py`, `plot_bo_descent_4pane.py`, `plot_phase_a_ctle_descent.py`, `spice_failure_check.py`, `test_smoke.py`, `train_ctle.py`, `tree_benchmark.py`, `xgb_benchmark_housing.py`
  - `optuna` — `bo_param_sampling.py`, `check_optuna_versions.py`, `ctle_kirchhoff_bo.py`, `fixed-distillation-bayes-opt.py`, `generative-distillation-plain-mlp-bayes-opt.py`, `kn_bayes_opt.py`, `mlp_bayes_opt.py`, `probe_study_db.py`, `summarize_fixed_bo.py`, `test_bo_guard.py`
  - `pandas` — `ctle_dagger_common.py`, `dagger-mlp-distillation-kirchhoffnet.py`, `dagger-nuance-distillation-kirchhoffnet.py`, `dry_run_knet_input_clipping.py`, `fixed-mlp-distillation-kirchhoffnet.py`, `generative-distillation-improved-dagger-nuance-mlp.py`
  - `pytest` — `test_f1_f2.py`
  - `scikit-learn` — `ctle_dagger_common.py`, `dagger-nuance-distillation-kirchhoffnet.py`, `gen_network_images.py`, `generative-distillation-improved-dagger-nuance-mlp.py`, `test_smoke.py`, `train_script.py`, `tree_benchmark.py`
  - `scipy` — `train_script.py`
  - `torch` — `_count_params.py`, `_test_step.py`, `analog_noise.py`, `audit_aux_core.py`, `audit_post_twin_gates.py`, `audit_twin_standalone.py`, `aux_core_probe.py`, `cell_library.py`, `config.py`, `ctle_dagger_common.py`, `ctle_kirchhoff_bo.py`, `dagger-mlp-distillation-kirchhoffnet.py`, `dagger-nuance-distillation-kirchhoffnet.py`, `deq_diagnostics.py`, `deq_solver.py`, `differential_stage.py`, `docs/analysis/scratch/audit.py`, `docs/analysis/scratch/audit2.py`, `docs/analysis/scratch/connectivity.py`, `docs/analysis/scratch/dbg.py`, `docs/analysis/scratch/final.py`, `docs/analysis/scratch/flags.py`, `docs/analysis/scratch/fwd1.py`, `docs/analysis/scratch/insp.py`, `docs/analysis/scratch/inspect_models.py`, `docs/analysis/scratch/load1.py`, `docs/analysis/scratch/param_counts.py`, `docs/analysis/scratch/plain.py`, `docs/analysis/scratch/same.py`, `docs/analysis/scratch/teachchk.py`, `docs/analysis/scratch/teacher.py`, `fixed-mlp-distillation-kirchhoffnet.py`, `gen_network_images.py`, `generate_report.py`, `generative-distillation-improved-dagger-nuance-mlp.py`, `generative-distillation-plain-mlp-bayes-opt.py`, `generative-distillation-plain-mlp.py`, `gln_rails.py`, `io_mapper.py`, `kirchhoff_net.py`, `kirchhoff_noise.py`, `kn_bayes_opt.py`, `kn_benchmark_friedman2.py`, `knet_gating_audit1.py`, `knet_multistage_audit.py`, `mlp_bayes_opt.py`, `mlp_benchmark.py`, `mlp_benchmark_ctle.py`, `mlp_benchmark_friedman1.py`, `mlp_benchmark_friedman2.py`, `mlp_benchmark_friedman3.py`, `mlp_benchmark_housing.py`, `narma_advisor_probes.py`, `narma_experiment.py`, `narma_linear_controls.py`, `narma_phase7_leak_bands.py`, `narma_phase8_tied_gate_probe.py`, `narma_phase8_tied_gates_audit.py`, `narma_revised_plan.py`, `narma_spectral_norm.py`, `post_twin_gates.py`, `probe_audit1.py`, `probe_audit2.py`, `probe_audit_nodeact.py`, `probe_audit_revised.py`, `probe_audit_split.py`, `probe_audit_transplant.py`, `sim_context.py`, `smoke_audit3.py`, `spice_failure_check.py`, `stage_transfer.py`, `tap_rails.py`, `test_f1_f2.py`, `test_smoke.py`, `test_training.py`, `topology.py`, `train.py`, `train_ctle.py`, `train_script.py`, `tree_benchmark.py`, `vca_sanity.py`, `visualize.py`, `xgb_benchmark_housing.py`
  - `torchdeq` — `deq_solver.py`
  - `torchdiffeq` — `ctle_kirchhoff_bo.py`
  - `tqdm` — `narma_experiment.py`, `train_script.py`
  - `xgboost` — `tree_benchmark.py`, `xgb_benchmark_housing.py`

- Local-MISSING modules (every listed occurrence is unguarded by try/except):
  - `narma_gain_norm` — `_test_step.py:2`
  - `utils.model` — `kn_benchmark_friedman2.py:70,560`
  - `utils.topology` — `kn_benchmark_friedman2.py:71,561`
  - `sparse_solver_data` — `test_smoke.py:648,666,681,698,756,773`
  - `sparse_solver_topology` — `test_smoke.py:682,699`
  - `sparse_solver_baseline` — `test_smoke.py:736,757`
  - `sparse_solver_track` — `test_smoke.py:774`

- Import check table:
  - PASS 83: `__future__`, `analog_noise`, `argparse`, `aux_core_probe`, `bo_param_sampling`, `cell_library`, `collections`, `config`, `copy`, `csv`, `ctle_dagger_common`, `dataclasses`, `deq_diagnostics`, `deq_solver`, `differential_stage`, `functools`, `gc`, `gln_rails`, `glob`, `hashlib`, `html`, `importlib`, `inspect`, `io`, `io_mapper`, `itertools`, `joblib`, `json`, `kirchhoff_net`, `kirchhoff_noise`, `logging`, `math`, `matplotlib`, `mlp_benchmark`, `mlp_benchmark_friedman1`, `mlp_benchmark_friedman2`, `mlp_benchmark_friedman3`, `mlp_benchmark_housing`, `multiprocessing`, `narma_advisor_probes`, `narma_experiment`, `narma_linear_controls`, `narma_phase7_leak_bands`, `narma_revised_plan`, `narma_spectral_norm`, `networkx`, `numpy`, `optuna`, `os`, `pandas`, `pathlib`, `pickle`, `post_twin_gates`, `pytest`, `random`, `re`, `runpy`, `scipy`, `shlex`, `shutil`, `sim_context`, `sklearn`, `socket`, `sqlite3`, `stage_transfer`, `statistics`, `subprocess`, `sys`, `tap_rails`, `tempfile`, `time`, `topology`, `torch`, `torchdiffeq`, `tqdm`, `train`, `train_ctle`, `train_script`, `types`, `typing`, `visualize`, `warnings`, `zipfile`.
  - FAIL `catboost`: FAIL: ModuleNotFoundError: No module named 'catboost'.
  - FAIL `lightgbm`: FAIL: ModuleNotFoundError: No module named 'lightgbm'.
  - FAIL `narma_gain_norm`: FAIL: ModuleNotFoundError: No module named 'narma_gain_norm'.
  - FAIL `sparse_solver_baseline`: FAIL: ModuleNotFoundError: No module named 'sparse_solver_baseline'.
  - FAIL `sparse_solver_data`: FAIL: ModuleNotFoundError: No module named 'sparse_solver_data'.
  - FAIL `sparse_solver_topology`: FAIL: ModuleNotFoundError: No module named 'sparse_solver_topology'.
  - FAIL `sparse_solver_track`: FAIL: ModuleNotFoundError: No module named 'sparse_solver_track'.
  - FAIL `torchdeq`: FAIL: ModuleNotFoundError: No module named 'torchdeq'.
  - FAIL `utils`: FAIL: ModuleNotFoundError: No module named 'utils'.
  - FAIL `xgboost`: FAIL: ModuleNotFoundError: No module named 'xgboost'.

- Absolute-path grep (tracked live `.py`/`.sh`/yaml/toml/json; excludes `docs/`, archives, scratch; tracked `.ipynb` scanned as text too).
  - `_count_params.py:4` — target `C:` (Windows path; absent on ssr0): `root = Path(r'C:\Users\Atharva\Downloads\results_many_models_realizable_knet_nb')`
  - `_test_step.py:1` — target `C:` (Windows path; absent on ssr0): `import sys; sys.path.insert(0, r'C:\Users\Atharva\Documents\kirchhoffnet_realistic')`
  - `ctle_dagger_common.py:74` — target `improved-zig-nf-spline-pytorch-default-v2` (exists): `DEFAULT_TEACHER_DIR = '/home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2/'`
  - `ctle_dagger_common.py:75` — target `augmented-cvae-ctle` (exists (symlink)): `DEFAULT_DATA_DIR = '/home/annaik/Documents/augmented-cvae-ctle/'`
  - `ctle_dagger_common.py:76` — target `other path` (unverified): `DEFAULT_OUTPUT_DIR = '/home/annaik/Documents/dagger_output'`
  - `dagger-mlp-distillation-kirchhoffnet.py:48` — target `augmented-cvae-ctle` (exists (symlink)): `DATA_DIR = '/home/annaik/Documents/augmented-cvae-ctle/'`
  - `dagger-mlp-distillation-kirchhoffnet.py:51` — target `other path` (unverified): `OUTPUT_DIR = '/home/annaik/Documents/dagger_output_mlp_knet'`
  - `dagger-mlp-distillation-kirchhoffnet.py:766` — target `improved-zig-nf-spline-pytorch-default-v2` (exists): `        '/home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2/',`
  - `dagger-mlp-distillation-kirchhoffnet.py:767` — target `augmented-cvae-ctle` (exists (symlink)): `        '/home/annaik/Documents/augmented-cvae-ctle/',`
  - `dagger-mlp-distillation-kirchhoffnet.py:1055` — target `augmented-cvae-ctle` (exists (symlink)): `            "Pass --dataset-dir /kaggle/input/datasets/awekill/augmented-cvae-ctle "`
  - `dagger-nuance-distillation-kirchhoffnet.ipynb:61` — target `/home/annaik/Documents/train_ctle` (exists): `    "# os.chdir('/home/annaik/Documents/train_ctle')\n",`
  - `dagger-nuance-distillation-kirchhoffnet.ipynb:63` — target `other path` (unverified): `    "TEACHER_DIR = '/kaggle/input/models/awekill/improved-zig-nf-spline/pytorch/default/2'\n",`
  - `dagger-nuance-distillation-kirchhoffnet.ipynb:64` — target `augmented-cvae-ctle` (exists (symlink)): `    "DATA_DIR = '/kaggle/input/datasets/awekill/augmented-cvae-ctle'\n",`
  - `dagger-nuance-distillation-kirchhoffnet.ipynb:117` — target `kirchhoffnet_realistic` (missing on ssr0): `    "    \"/kaggle/working/kirchhoffnet_realistic\",\n",`
  - `dagger-nuance-distillation-kirchhoffnet.ipynb:1459` — target `augmented-cvae-ctle` (exists (symlink)): `    "history_dir = '/kaggle/input/datasets/awekill/augmented-cvae-ctle'\n",`
  - `dagger-nuance-distillation-kirchhoffnet.py:64` — target `improved-zig-nf-spline-pytorch-default-v2` (exists): `TEACHER_DIR = '/home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2/'`
  - `dagger-nuance-distillation-kirchhoffnet.py:65` — target `augmented-cvae-ctle` (exists (symlink)): `DATA_DIR = '/home/annaik/Documents/augmented-cvae-ctle/'`
  - `dagger-nuance-distillation-kirchhoffnet.py:66` — target `other path` (unverified): `OUTPUT_DIR = '/home/annaik/Documents/dagger_output'`
  - `dagger-nuance-distillation-kirchhoffnet.py:801` — target `other path` (unverified): `    # checkpoint (e.g. /home/annaik/.../dagger_checkpoint.pt from a prior`
  - `dry_run_knet_input_clipping.py:73` — target `augmented-cvae-ctle` (exists (symlink)): `                        default=Path("/home/annaik/Documents/augmented-cvae-ctle"))`
  - `gen_network_images.py:3` — target `ASPDAC_2026` (exists): `Saves to /home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/network_visualization/`
  - `gen_network_images.py:6` — target `ASPDAC_2026` (exists): `sys.path.insert(0, '/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal')`
  - `gen_network_images.py:23` — target `ASPDAC_2026` (exists): `OUT_DIR = '/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/network_visualization'`
  - `generate_report.py:12` — target `C:` (Windows path; absent on ssr0): `RESULTS_ROOT = Path(r"C:\Users\Atharva\Downloads\results_many_models_realizable_knet_nb")`
  - `generate_report.py:13` — target `C:` (Windows path; absent on ssr0): `EXTRA_EPOCHS_ROOT = Path(r"C:\Users\Atharva\Downloads\results_mlp_benchmarks_extra_epochs")`
  - `generate_report.py:14` — target `C:` (Windows path; absent on ssr0): `OUTPUT_HTML = Path(r"C:\Users\Atharva\Documents\kirchhoffnet_realistic\comparison_report.html")`
  - `generate_report.py:17` — target `kirchhoffnet_realistic` (missing on ssr0): `SKIP_DIRS = {"kirchhoffnet", "kirchhoffnet_realistic"}`
  - `generative-distillation-improved-dagger-nuance-mlp.py:40` — target `/home/annaik/Documents/train_ctle` (exists): `# os.chdir('/home/annaik/Documents/train_ctle')`
  - `generative-distillation-improved-dagger-nuance-mlp.py:42` — target `improved-zig-nf-spline-pytorch-default-v2` (exists): `TEACHER_DIR = '/home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2/'`
  - `generative-distillation-improved-dagger-nuance-mlp.py:43` — target `augmented-cvae-ctle` (exists (symlink)): `DATA_DIR = '/home/annaik/Documents/augmented-cvae-ctle/'`
  - `generative-distillation-improved-dagger-nuance-mlp.py:44` — target `other path` (unverified): `OUTPUT_DIR = '/home/annaik/Documents/dagger_output'`
  - `generative-distillation-improved-dagger-nuance-mlp.py:47` — target `augmented-cvae-ctle` (exists (symlink)): `    _kaggle_fallback = '/kaggle/input/datasets/awekill/augmented-cvae-ctle'`
  - `generative-distillation-improved-dagger-nuance.ipynb:64` — target `/home/annaik/Documents/train_ctle` (exists): `    "# os.chdir('/home/annaik/Documents/train_ctle')\n",`
  - `generative-distillation-improved-dagger-nuance.ipynb:66` — target `other path` (unverified): `    "TEACHER_DIR = '/kaggle/input/models/awekill/improved-zig-nf-spline/pytorch/default/1'\n",`
  - `generative-distillation-improved-dagger-nuance.ipynb:67` — target `augmented-cvae-ctle` (exists (symlink)): `    "DATA_DIR = '/kaggle/input/datasets/awekill/augmented-cvae-ctle'\n",`
  - `generative-distillation-improved-dagger-nuance.ipynb:1601` — target `augmented-cvae-ctle` (exists (symlink)): `    "history_dir = '/kaggle/input/datasets/awekill/augmented-cvae-ctle'\n",`
  - `narma_advisor_probes_batch.sh:47` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_advisor_probes_p2_batch.sh:42` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_diag_batch.sh:58` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_diag_batch.sh:65` — target `tron_eq` (exists): `  || { echo "ERROR: torch/triton import failed in /home/annaik/Documents/tron_eq/venv. Aborting before any GPU work." >&2; exit 3; }`
  - `narma_gating_depth_batch.sh:42` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_gating_e0_batch.sh:40` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_gating_e0_extend_batch.sh:47` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_gating_rung1_batch.sh:42` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_gating_spike_batch.sh:42` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_linear_controls_batch.sh:44` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_revised_plan_batch.sh:63` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_round2_gated_batch.sh:97` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_split_batch.sh:43` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_transplant_batch.sh:46` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_twin_batch.sh:38` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_twin_fine_batch.sh:42` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `narma_vca_sparse_batch.sh:86` — target `tron_eq` (exists): `source "/home/annaik/Documents/tron_eq/venv/bin/activate"`
  - `post_twin_gates_batch.sh:49` — target `tron_eq` (exists): `VENV="${VENV:-/home/annaik/Documents/tron_eq/venv/bin/activate}"`
  - `run_phase_a_knet_bo.sh:44` — target `improved-zig-nf-spline-pytorch-default-v2` (exists): `    --teacher-dir /home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2 \`
  - `run_phase_a_knet_bo.sh:45` — target `augmented-cvae-ctle` (exists (symlink)): `    --data-dir /home/annaik/Documents/augmented-cvae-ctle \`
  - `run_train_knet_single_shot_distillation.sh:62` — target `improved-zig-nf-spline-pytorch-default-v2` (exists): `    --teacher-dir /home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2 \`
  - `run_train_knet_single_shot_distillation.sh:63` — target `augmented-cvae-ctle` (exists (symlink)): `    --data-dir /home/annaik/Documents/augmented-cvae-ctle \`
  - `run_train_knet_single_shot_distillation.sh:69` — target `kirchhoffnet_realistic` (missing on ssr0): `/home/annaik/Documents/kirchhoffnet_realistic/ENV/bin/python /home/annaik/Documents/kirchhoffnet_realistic/dagger-nuance-distillation-kirchhoffnet.py --canonical-dataset "$CANONICAL" \`
  - `run_train_knet_single_shot_distillation.sh:74` — target `kirchhoffnet_realistic` (missing on ssr0): `  --phase-a-power-norm train_mean --kn-readout shared-x2 --output /home/annaik/Documents/kirchhoffnet_realistic/outputs/manual_knet_distillation_mix_trials/trial_0005 \`
  - `spice_failure_check.py:31` — target `/home/annaik/Documents/train_ctle` (exists): `        --template-dir /home/annaik/Documents/train_ctle/ocn_template_ideal_bob \`
  - `spice_failure_check.py:240` — target `/home/annaik/Documents/train_ctle` (exists): `                   default=Path("/home/annaik/Documents/train_ctle/ocn_template_ideal_bob"),`
  - `spice_failure_check.py:254` — target `/home/annaik/workarea_GF22_FDX_EXT` (exists): `                   default="source /home/annaik/workarea_GF22_FDX_EXT/kit.gf22fdx_ext.v1.0_4.1_stack19_oa.csh",`
  - `spice_failure_check.py:453` — target `kirchhoffnet_realistic` (missing on ssr0): `                "Run from the kirchhoffnet_realistic directory or add it to PYTHONPATH.")`
  - `spice_failure_check.py:1097` — target `kirchhoffnet_realistic` (missing on ssr0): `            "Run from the kirchhoffnet_realistic directory.")`
  - `spice_failure_check.py:1294` — target `/home/annaik/workarea_GF22_FDX_EXT` (exists): `        "cd /home/annaik/workarea_GF22_FDX_EXT\n"`
  - `test_audit.py:3` — target `ASPDAC_2026` (exists): `os.chdir('/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal')`
  - `test_bo_guard.py:153` — target `C:` (Windows path; absent on ssr0): `        "fdbo_under_test", r"C:\Users\Atharva\Documents\kirchhoffnet_realistic\fixed-distillation-bayes-opt.py")`
  - `test_bo_guard.py:329` — target `C:` (Windows path; absent on ssr0): `        "fdbo_budget_test", r"C:\Users\Atharva\Documents\kirchhoffnet_realistic\fixed-distillation-bayes-opt.py")`
  - `test_bo_guard.py:334` — target `C:` (Windows path; absent on ssr0): `    _sys.path.insert(0, r"C:\Users\Atharva\Documents\kirchhoffnet_realistic")`
  - `test_bo_guard.py:400` — target `C:` (Windows path; absent on ssr0): `        "fdbo_prior_test", r"C:\Users\Atharva\Documents\kirchhoffnet_realistic\fixed-distillation-bayes-opt.py")`
  - `test_bo_guard.py:402` — target `C:` (Windows path; absent on ssr0): `    _sys.path.insert(0, r"C:\Users\Atharva\Documents\kirchhoffnet_realistic")`
  - `test_bo_guard.py:433` — target `C:` (Windows path; absent on ssr0): `        "fdbo_fanout_test", r"C:\Users\Atharva\Documents\kirchhoffnet_realistic\fixed-distillation-bayes-opt.py")`
  - `test_bo_guard.py:435` — target `C:` (Windows path; absent on ssr0): `    _sys.path.insert(0, r"C:\Users\Atharva\Documents\kirchhoffnet_realistic")`
  - `test_bo_guard.py:448` — target `C:` (Windows path; absent on ssr0): `    src = (_P(r"C:\Users\Atharva\Documents\kirchhoffnet_realistic")`
  - `test_bo_guard.py:458` — target `C:` (Windows path; absent on ssr0): `    fdbo = (_P(r"C:\Users\Atharva\Documents\kirchhoffnet_realistic")`
  - `test_smoke.py:17` — target `ASPDAC_2026` (exists): `    ~/Documents/ASPDAC_2026/venv/bin/python kirchhoff_redesign/ideal/test_smoke.py`
  - `test_smoke.py:2198` — target `ASPDAC_2026` (exists): `    venv_py = os.path.expanduser("~/Documents/ASPDAC_2026/venv/bin/python")`
  - `test_smoke.py:2409` — target `ASPDAC_2026` (exists): `    venv_py = os.path.expanduser("~/Documents/ASPDAC_2026/venv/bin/python")`
  - `test_training.py:8` — target `ASPDAC_2026` (exists): `    ~/Documents/ASPDAC_2026/venv/bin/python kirchhoff_redesign/ideal/test_training.py`
  - `test_training.py:170` — target `ASPDAC_2026` (exists): `    venv_py = "/home/annaik/Documents/ASPDAC_2026/venv/bin/python"`
  - `test_training.py:173` — target `ASPDAC_2026` (exists): `         "/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/mlp_benchmark.py",`
  - `test_training.py:179` — target `ASPDAC_2026` (exists): `         "/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/mlp_benchmark_housing.py",`
  - `test_training.py:274` — target `ASPDAC_2026` (exists): `             "/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/mlp_benchmark.py",`
  - `test_training.py:281` — target `ASPDAC_2026` (exists): `            cwd="/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal",`

- Tiny smoke results:
  - WORKS: `deq_solver.solve_equilibrium` on `phi(z)=0.5*z+1`, tiny x0 `(1,2)`; output `(1,2)`. `torchdeq` is NOT installed/required for this fallback smoke; backend is `fixed_point_iter` only. `artifacts/deq/model.pt` loads as a state_dict, not a serialized module; compatibility/reconstruction not demonstrated.

- Assets and actual references: ZIG v2 `/home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2/` = 7.2 MiB; live dagger/generative scripts reference it. Teacher `trial_0019/dagger_student_plain.pt` = 87,210 B at `/home/annaik/Documents/kirchhoffnet_refactor/assets/teachers/trial_0019/dagger_student_plain.pt`; absent in clone but teacher code expects this named checkpoint. `augmented-cvae-ctle` symlink resolves to `/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/augmented-cvae-ctle`, 8.3 MiB/18 files; live CTLE/dagger references. Canonical NPZ `/home/annaik/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/models/canonical_ctle_phase_a_v2.npz`, 1.5 MiB (du rounded), referenced by training scripts.
  - DEQ-specific `artifacts/deq/` = 1.8 MiB / 20 files (`model.pt` 43,569 B, notebook, diagnostics, plots); ignored and absent from `git ls-files`; live DEQ workflow should include it.

- Ordered fix checklist, following Task E steps 1–8 in `docs/HANDOFF_session2.md:276-293`:
  1. Clone-local Python 3.12 `venv/`; create minimal `requirements.txt` from third-party list above (DEQ also `torchdeq`); install/verify imports.
  2. Resolve missing imports: Narma restore `deq_solver.py`/`deq_diagnostics.py` from pre-deletion parent `7b34b60^` or guard dead paths; resolve `narma_gain_norm`, sparse_solver data/topology/baseline/track, and `utils.model`/`utils.topology`. DEQ resolve the latter six. Fix or mark dead; current missing imports unguarded.
  3. Copy, never symlink: `assets/zig/` 7.2 MiB; `assets/teachers/trial_0019/dagger_student_plain.pt` 87,210 B (and v4 plain checkpoint if used); `data/augmented-cvae-ctle/` 8.3 MiB; canonical NPZ 1.5 MiB if used. Small binaries may be committed with `.gitignore` exception; CSVs may remain ignored/present. DEQ also commits `artifacts/deq/` 1.8 MiB.
  4. Path edits: `ctle_dagger_common.py:74-76`; `dagger-mlp-distillation-kirchhoffnet.py:48,51,766-767`; `dagger-nuance-distillation-kirchhoffnet.py:64-66`; `generative-distillation-improved-dagger-nuance-mlp.py:42-47`; `dry_run_knet_input_clipping.py:73`; `run_phase_a_knet_bo.sh:44-45`; `run_train_knet_single_shot_distillation*.sh:62-74`; batch scripts sourcing `tron_eq/venv`; `spice_failure_check.py:31,240,254,1294`. Use repo-relative defaults/environment overrides; leave archive/docs/scratch unchanged.
  5. Fix `test_friedman2_small_world.py:5` import-time `KeyError hidden_family`; then `PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider --collect-only` and fast subset. Document remaining pre-surgery failures.
  6. Remove `local` remote (kirchhoffnet_refactor); retain `origin`.
  7. Add `README_BRANCH.md` covering branch purpose, venv, asset layout, main experiment/notebook run, known failing tests.
  8. Verify zero live external refs, imports/`--help`, tiny NARMA and DEQ forward using clone venv/cwd and outputs `/tmp`, plus fresh clone and copied ignored assets. No training.
- Environment-variable names: `KNET_DATA_DIR`, `KNET_TEACHER_DIR`, `KNET_OUTPUT_DIR`, `CADENCE_SETUP`, `OCN_TEMPLATE_DIR`.

