---
plan name: spice-score-fix
plan description: Fix power key, add run-dir scoring
plan status: active
---

## Idea
Fix the silent 100%-failure corruption in spice_failure_check.py exposed by the live knet_trial0011_test200 run (182/200 sims healthy, all mis-scored): (1) DEFAULT_KEY_MAP power Power->power with comment citing template ?name power and Bob output_dict.get(power); (2) new --score-run-dir mode that walks spec_*/parsed.json + spec_*/target.json, scores finished specs only (pending excluded from rate per user decision), writes results.csv + summary.json with n_done/n_pending marker, zero Ocean calls, refuses _mocked entries unless --force; (3) regression audit with fixture shaped like the real spec_00099 parsed.json (lowercase power plus Nominal/Region junk keys) asserting all 4 dims resolve; (4) verify py_compile + mocked E2E + real-shape fixture through score_spec and --score-run-dir on a synthetic run dir.

## Implementation
- Change DEFAULT_KEY_MAP power to lowercase power with provenance comment
- Add --score-run-dir mode: walk spec dirs, score finished only, write results.csv and summary.json with n_done and n_pending, refuse _mocked without --force
- Add regression fixture mirroring real parsed.json junk keys and assert 4-dim resolution
- Verify with py_compile plus mocked E2E plus synthetic run-dir scoring
- Hand over ssr0 pull plus rescore commands for knet_trial0011_test200

## Required Specs
<!-- SPECS_START -->
- spice-score-spec
<!-- SPECS_END -->