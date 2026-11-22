# Spec: spice-failure-spec

Scope: feature

# SPICE failure-check harness (feature spec)

## Goal
True-SPICE failure rate for ONE trained checkpoint (KNet DAgger student OR PlainMLP BO winner) on N target specs, using the 7-param ideal-Bob Cadence flow on ssr0.

## Non-goals
- No retraining, no BO, no ZIG-proxy-only reporting. No 16-var `ctle_maestro.ocn` path.

## Inputs (CLI)
- `--model-kind {knet,mlp}` (required), `--ckpt <.pt>` (required).
- KNet rebuild: `--kn-config-json` (trial dir config, preferred) OR explicit `--kn-num-stages/--kn-num-hidden/--kn-small-world-k/--kn-small-world-p/--kn-vca-rank/--kn-x-max/--kn-gm-max/--kn-isat-max/--kn-moe-* /--kn-readout/--t-span/--boundary-fan-out/--kn-gln-*` flags mirroring the trial_0005 CLI.
- MLP: auto-read `teacher_config.json` beside ckpt (fallback: `--mlp-width/--mlp-layers/--mlp-activation/--mlp-layernorm/--input-preprocessing knet`).
- Specs: `--specs-npz $CANONICAL` XOR `--specs-csv`; `--n-specs`, `--spec-offset`, `--seed`; column order `[power, stage_2_jitter, stage_2_eye_max_height, stage_2_eye_max_width]`.
- Ocean: `--template-dir` (default `<train_ctle>/ocn_template_ideal_bob`), `--tb-ids` (default `1,2,3,4,5,6,7,9`; template_8 absent upstream), `--work-root` (default `~/simulation/SPICE_failure`), `--sim-timeout` (default 600), `--jobs` (default = len(tb-ids)), `--key-map` override, `--degrade-thr` (default 0.10), `--run-tag`.
- Modes: `--dry-run`, `--smoke` (forces N=2, 1 TB), `--offline-csv <results.csv>` (re-score only), `--resume` (default on), `--dump-keys` (print one CSV's keys and exit).

## Stage 1 — inference
- CPU/GPU batch forward, no-grad; KNet via trial preprocessing (knet scaler), MLP via `load_plain_mlp_teacher` semantics.
- Output physical params in `PARAM_COLS=[fW,current,ind,Rd,Cs,Rs,VDD]` order; clip to `PARAM_LOG_BOUNDS`.
- Write `<work-root>/<run-tag>/pred_params.csv`, `target_specs.csv`.

## Stage 2 — Ocean (ssr0 runtime only)
- Per spec dir `spec_<idx:05d}/` with `params.json`, `target.json`, `spec.ocn`, `result.csv`, `parsed.json`.
- OCN via `update_ocn_ctle(base=template_{tb}.ocn, use_new_templates=True, var_map={7 params}, csv_path=<abs result.csv>, tb_idx=<pinned tb>)`; absolute csv path required.
- Launch: csh PDK wrapper (`source .../kit.gf22fdx_ext.v1.0_4.1_stack19_oa.csh` + `ocean -nograph`), `delete_maestro_history(tb)` first, poll `result.csv` until timeout, `parse_ctle_csv`.
- Pool: `min(jobs, len(tb_ids))`, each worker pinned to one TB id (no two sims share a TB concurrently).

## Stage 3 — scoring (ZIG degrade rule on SPICE)
- Key map (defaults, overridable): power=`Power`, jitter=`eye_p2pJitterAverage_norm stage 2`, height=`eye_maxHeight_norm Vout_2 56G`, width=`eye_maxWidth_norm Vout_2 56G`.
- `compute_forward_errors` semantics: one-sided degrades (power/jitter: (pred-target)/target; height/width: (target-pred)/target), `degraded_dims>=2` → failure at `degrade_thr`. Missing keys / `eval err` / `sim err` / NaN → failure with `error=1`.
- Write per-spec `results.csv` (target×4, measured×4, err×4, degraded_dims, failure, error) + `summary.json` (`n, failures, failure_rate, dim_degrade_rates[4], error_rate, ckpt, ckpt_sha256, template_sha, degrade_thr, key_map, hostname, date`).

## Resume / manifest
- Skip specs with valid `parsed.json` when `--resume`; `manifest.json` records full CLI, ckpt sha256, template dir sha, key map, thresholds, hostname.

## Local verification (no Cadence)
- `python -m py_compile spice_failure_check.py`; `--smoke --dry-run` writes OCN/params; mocked `parse_ctle_csv` dict exercises scoring math.