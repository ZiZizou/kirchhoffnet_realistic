---
plan name: spice-failure-check
plan description: True-SPICE failure validation harness
plan status: active
---

## Idea
New script in kirchhoffnet_realistic (e.g. spice_failure_check.py) that loads ONE trained checkpoint — either a KNet DAgger student (dagger-nuance-distillation-kirchhoffnet.py trial dir) or a PlainMLP BO winner (mlp_bayes_opt.py dagger_student_plain.pt + teacher_config.json) — runs inference over N target specs from the canonical Phase-A .npz (or a specs CSV), writes predicted 7-dim physical params [fW,current,ind,Rd,Cs,Rs,VDD] to CSV, then drives true Cadence Spectre sims on ssr0 using the 7-param ideal-Bob flow (ocn_template_ideal_bob/template_{tb}.ocn + update_ocn_ctle with use_new_templates=True, ocean -nograph via csh PDK wrapper, parse_ctle_csv), maps measured outputs to [power, jitter, height, width] 56G stage-2 keys, recomputes the ZIG-proxy degrade failure_mask (compute_forward_errors, >=2 of 4 dims degraded) per spec against the original target, and reports true-SPICE failure rate + per-dim degrade rates + sim-error rate, with resume/skip-existing, parallelism over 9 TBs, and dry-run/offline-scoring modes. Coded locally in realistic repo (torch + numpy only, no Cadence import at module scope), pulled to ssr0 for execution where ~/simulation + ECE740_local TBs + GF22 PDK exist.

## Implementation
- Add spice_failure_check.py in kirchhoffnet_realistic with CLI: --model-kind {knet,mlp}, --ckpt, --model-config overrides for KNet rebuild, --specs-npz/--specs-csv, --n-specs/--spec-offset/--seed, --template-dir ocn_template_ideal_bob path, --tb-ids, --work-root ~/simulation/SPICE_failure, --sim-timeout, --jobs, --resume, --dry-run, --offline-csv scoring mode
- Implement Stage-1 inference: KNet loader (rebuild net from trial CLI/config incl. stages/hidden/k/p/t-span/readout/G LN + load state dict, kbit preprocessing) and MLP loader (load_plain_mlp_teacher path with teacher_config.json fallback), batch forward specs->bounded 10^logit physical params in PARAM_COLS order, save pred_params.csv + target_specs.csv
- Implement Stage-2 Ocean driver (ssr0-only at runtime): per-spec update_ocn_ctle(base template_N, use_new_templates=True, var_map 7 params, csv_path per-spec result.csv, tb_idx round-robin over 1..9), run_simulation csh ocean wrapper with PDK source, delete_maestro_history before each run, wait/poll result.csv with timeout, parse_ctle_csv; multiprocessing pool size=min(jobs, len(tb-ids)) with per-worker tb pinning
- Implement Stage-3 true-failure scoring: map parsed keys to [power, eye_p2pJitterAverage_norm stage 2, eye_maxHeight_norm Vout_2 56G, eye_maxWidth_norm Vout_2 56G] (configurable key map + --key-map override), reuse compute_forward_errors degrade logic (>=2 dims degraded at ERROR_THRESHOLD default 0.10, --degrade-thr override) vs target spec; sim-error/NaN counts as failure; write per-spec results.csv + summary.json (failure_rate, dim degrade rates, error rate, N, ckpt id)
- Add resume + determinism: per-spec dirs <work-root>/<run-tag>/spec_<idx>/ (params.json, target.json, result.csv, parsed.json); skip specs with valid parsed.json on --resume; manifest.json records ckpt sha, CLI, template sha, key-map, thresholds, hostname; seeded spec subsample from canonical npz
- Add safety/UX: --dry-run writes OCN files + params without launching ocean; --offline-csv re-scores existing results.csv without sims; --smoke (N=2, 1 tb) path; py_compile + 2-spec mocked-parse unit check runnable locally without Cadence/torch bootstrap guard

## Required Specs
<!-- SPECS_START -->
- spice-failure-spec
<!-- SPECS_END -->