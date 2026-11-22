# Spec: spice-score-spec

Scope: feature

# SPICE score fix: power key + run-dir scoring (feature spec)

## Bug (confirmed on live run knet_trial0011_test200, 182/200 healthy sims)
- `DEFAULT_KEY_MAP["power"]` is `"Power"`, but the ideal-Bob template emits
  `?name "power"` (lowercase) and Bob's reference flow reads
  `output_dict.get("power")`. Every spec raised KeyError on the power lookup
  and scored `failure=True` — a silent ~100% failure rate with 0% errors.
- Real `parsed.json` files additionally carry junk keys the parser keeps
  (`"Parameter": "Nominal"`, `"Output": "Nominal"`,
  `"stage 1/2 Region": 2.0`); scoring must ignore everything but the 4
  mapped dims.

## Fix 1 — key map
- `DEFAULT_KEY_MAP["power"] = "power"`, with a comment citing the template
  `?name "power"` line and Bob's `output_dict.get("power")`.
- No migration of on-disk data needed: `parsed.json` files are unaffected.

## Fix 2 — `--score-run-dir <work>/<run-tag>` mode (zero Ocean calls)
- Walk `spec_*/parsed.json` + `spec_*/target.json`.
- Specs missing either file are **pending**: excluded from numerator AND
  denominator (user decision: rate over finished only).
- Write `results.csv` (same schema as Stage 3) + `summary.json` with extra
  `n_done`, `n_pending` fields; re-runnable at any time (overwrite).
- Refuse `_mocked` entries (dry-run artifacts) unless `--force` is passed,
  so mocked numbers can never pose as SPICE.
- `--key-map`, `--degrade-thr`, `--min-degraded-dims` honored as usual.

## Regression test
- Fixture shaped exactly like the live `spec_00099/parsed.json` (lowercase
  `power` + Nominal/Region junk keys) asserting `score_spec` resolves all 4
  dims, ignores the rest, and does not mark failure on an exact-match spec.
- End-to-end: synthetic run dir (2 finished + 1 pending + 1 mocked) through
  `--score-run-dir` asserting `n_done=2`, `n_pending=1`, mocked refusal
  without `--force`, acceptance with `--force`.

## Verification (local, no Cadence)
- `py_compile`; existing mocked E2E still green; new fixture + synthetic
  run-dir audit green.