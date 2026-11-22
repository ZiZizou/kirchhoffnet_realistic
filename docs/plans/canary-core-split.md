---
plan name: canary-core-split
plan description: Decompose bimodal core, gate jitter fix
plan status: active
---

## Idea
Follow up the canary verdict (CORE-USEFUL-NO-PASS, output/canary/canary.json) with the advisor's decomposition sequence. Phase 0: add a probe-side-only per-node diagnostics instrument (per-node max|x|/x_max histogram + shunt-G sums + sign balance + boundary current + correlations) and run it on the exact priority corner (gm-3.5/cell-Isat-2.0/raw:-0.5/drive-0.2) on Alliance, reproducing the 4-hot/5-dead bimodality. Phase 1: three-arm Experiment 1 on the same corner (A shunt-only kill via g_resistive_raw->-20, B tanh-only kill via gm/isat->-20, C gate-zero control reproducing 0.8122/0.86) with the pre-registered reading table. Phase 2 (gated on Phase 0/1 verdicts): ~10-line per-edge log-uniform jitter surgery in apply_gain_override (or balanced-sign init) + probe_audit checks. Phase 3 (gated): 12-corner circulation grid (jitter x retention x swing, gates MC>=4/rails<=5%/ridge<0.6); on pass, the 54-corner transplant grid gains a jitter axis and a short k=2 train follows. Step 6 forget-gate stays parked; x_max sweep and hetero-C rhs surgery deferred with revival conditions. Local work limited to py_compile + thin CPU smokes + audits; all grids run on Alliance batch.

## Implementation
- Implement node_diagnostics() in narma_advisor_probes.py (per-node max|x|/xmax, shunt-G sums, sign balance, boundary current, correlations) + JSON writer, probe-side only
- Add probe_audit checks: diagnostics determinism, legacy-path byte-identity, s_raw-seed identity across rebuilds
- Write Alliance batch script canary_split_batch.sh: preflight (audit + anchor-repro vs 0.7611/0.8122) -> Phase 0 inspection -> Phase 1 arms A/B/C -> summary
- Local: py_compile + thin CPU smoke (canary corner at reduced samples/streams) + audit run
- Phase 2 (gated): jitter/balanced-sign surgery in apply_gain_override + audit; Phase 3 (gated): 12-corner circulation grid + verdict

## Required Specs
<!-- SPECS_START -->
- per-node-diagnostics
- advisor-probe-suite
- narma-e0-gate
- core-decomp-legs
<!-- SPECS_END -->