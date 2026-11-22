# Kaggle retraining guide

## Build and upload the input dataset

From the repository root on ssr0, build the bundle (the script follows the configured external-data paths and includes the repository archive, canonical dataset, ZIG v2 files, CSVs, and `MANIFEST.txt`):

```bash
scripts/kaggle/make_kaggle_bundle.sh HEAD /tmp/kirchhoffnet-kaggle.zip
```

Check the reported size (the builder enforces 100 MB), then create a Kaggle Dataset, upload that ZIP, and attach the Dataset to a new notebook. Rebuild from the desired merged commit when refreshing the dataset. To regenerate the committed notebook after changing its generator, run `python scripts/kaggle/build_notebook.py`.

## Notebook settings and run order

Use **GPU**, turn **Internet on** (environment setup installs missing Python packages), and enable notebook persistence. Run one rung per session in this sequence:

1. R0 — paper configuration, unconstrained;
2. R1 — `phase3a_core`;
3. R1d — `phase3a_core_diff`;
4. R2 — `phase3a_full`;
5. R3 — `phase3b_tm`.

The notebook does not pass removed `--kn-moe-*` flags. It uses `--teacher-dir`, `--data-dir`, and `--canonical-dataset` to override host-specific paths. Outputs are separated under `OUT_DIR/<rung>_seed<seed>/`.

Notebook parameters can be changed in the first cell or headlessly via `KNB_RUNG`, `KNB_EPOCHS`, `KNB_SEED`, `KNB_LR`, `KNB_STEPS_PER_STAGE`, `KNB_DATA_ROOT`, `KNB_OUT_DIR`, `KNB_RESUME`, `KNB_SMOKE`, and `KNB_RUN_TOY_ABLATION`. `SMOKE=1` runs one epoch and sets `--phase-a-sample-limit 32`; it is for plumbing only, not an accuracy result. For the hardware rungs, begin at LR 0.03 and compare 0.1/0.01 as validation-driven alternatives. The paper solver uses 70 Heun steps over 7 stages, or 10 per stage (`round(70 / 7)`). Hardware rungs default to 40 steps per stage (4× the paper setting); R0 omits the override and retains the paper solver default. Use the documented stiffness/trajectory diagnostics to validate this integration grid.

## Runtime and Kaggle limits

A 3g H100 run is reported at 1.6–1.8 hours for 300 paper-config epochs. This implies about 19–22 seconds/epoch. **Planning assumptions, not measured Kaggle benchmarks:** estimate a T4 at 0.35× H100 throughput and a P100 at 0.25×. Thus 300 R0 epochs take approximately 4.6–5.1 h on T4 and 6.4–7.2 h on P100. Within a 12-hour session, those estimates allow about 700–780 T4 or 500–560 P100 R0 epochs before overhead. Leave at least 1 hour for evaluation and saving.

The hardware rungs default to 40 steps per stage versus 10 for the paper solver (4×), so conservatively budget four times the epoch cost under an approximately linear-in-step-count scaling assumption: a 300-epoch rung is about 18–20 h on T4 or 26–29 h on P100, before preset overhead. A 12-hour session therefore fits about 150–180 T4 epochs or 110–125 P100 epochs at that resolution (less with hardware overhead). Kaggle's approximate 30 GPU-hour weekly quota is also a constraint: prioritize one rung at a time and expect to spread a hardware rung over sessions/weeks. These are planning estimates—not measured 40-step Kaggle benchmarks—so measure the first epoch and adjust.

To continue after a session limit, persist/download the output directory and set `KNB_RESUME` to its `phase_a_last.pt`. Resume restores model and optimizer, scheduler, epoch, history, best-so-far model/loss, and early-stop patience. Keep the same rung, seed, dataset, and configuration when resuming; use the same total `KNB_EPOCHS` target. The best evaluated model is `phase_a_best.pt`.

## Results and artifact retrieval

The evaluation cell checks train/val/test against the shared 10% failure rule, runs `hw.audit` for hardware presets, saves training plots and audit JSON, and appends (never replaces) a row in `OUT_DIR/results.csv`. The final cell creates a ZIP of `OUT_DIR`. Download the notebook output ZIP or use Kaggle's persisted `/kaggle/working/out`; bring the ZIP back to ssr0 and extract it outside the source checkout for comparison. Keep `train.log`, `phase_a_history.json`, checkpoint files, `results.csv`, audit JSON, and plots together. Do not compare a smoke run as a trained result.

The optional toy ablation setting is off by default. The existing CPU-learning report warns that its three-seed ablation is too noisy; any follow-up should use at least ten seeds and report the mean and 95% confidence interval.
