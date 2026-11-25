#!/bin/tcsh -f
# Manual long-running closure; invoke under nohup from any directory.
set REPO = /home/annaik/Documents/kirchhoffnet_refactor
set N = 64
set FABRIC = copies
if ( $#argv >= 1 ) set N = "$argv[1]"
if ( $#argv >= 2 ) set FABRIC = "$argv[2]"
if ( "$FABRIC" != "copies" && "$FABRIC" != "timemux" ) then
    echo "FABRIC must be copies or timemux"
    exit 2
endif
set STAMP = `date +%Y%m%d_%H%M%S`
set OUT = "$REPO/.agent_scratch/ctle_closure/$STAMP"
mkdir -p "$OUT"
set LOG = "$OUT/ctle_closure.log"
echo "[$STAMP] Estimate: 64 copies specs: Spectre 4x16 decks (~560 s each at reltol=1e-8) plus two Ocean scoring passes; expected about 1 hour, highly host/license dependent. timemux is somewhat longer." >> "$LOG"
echo "Output: $OUT" >> "$LOG"
set PY = "$REPO/venv/bin/python"
set PRED = "$REPO/scripts/ctle_closure/predict_knobs.py"
set CMP = "$REPO/scripts/ctle_closure/compare_failures.py"
set CKPT = "$REPO/assets/baselines/paper_trial_0011/phase_a_best.pt"
set DATA = "$HOME/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/models/canonical_ctle_phase_a_v2.npz"
set PRESET = "$REPO/hw/presets/phase3a_core.json"
if ( "$FABRIC" == "copies" ) then
    set EST = `echo "$N" | awk '{printf "%.0f", (($1+15)/16)*560}'`
    echo "Copies Spectre estimate: about ${EST}s for $N specs at reltol=1e-8; add two Ocean passes (runtime not statically verified)." >> "$LOG"
else
    echo "Timemux Spectre estimate: somewhat over the copies estimate; Ocean runtime not statically verified." >> "$LOG"
endif
set CMD = ( "$PY" "$PRED" --checkpoint "$CKPT" --fabric "$FABRIC" --mode legacy --split test --limit "$N" --out "$OUT/prediction" )
echo "[`date '+%Y-%m-%d %H:%M:%S'`] START predict_knobs rc=running" >> "$LOG"
$CMD --run-spectre >>& "$LOG"
set RC = $status
if ( $RC == 0 ) then
    set STATE = DONE
else
    set STATE = FAILED
endif
echo "[`date '+%Y-%m-%d %H:%M:%S'`] $STATE predict_knobs exit_code=$RC" >> "$LOG"
if ( $RC != 0 ) exit $RC
foreach KIND ( spectre torch )
    echo "[`date '+%Y-%m-%d %H:%M:%S'`] START spice_failure_check_$KIND rc=running" >> "$LOG"
    $PY "$REPO/spice_failure_check.py" --model-kind mlp --ckpt "$CKPT" --specs-npz "$DATA" --specs-idx-key test_idx --n-specs "$N" --seed 100 --pred-params-csv "$OUT/prediction/pred_${KIND}.csv" --enable-ocean --work-root "$OUT/spice" --run-tag "$KIND" >>& "$LOG"
    set RC = $status
    if ( $RC == 0 ) then
        set STATE = DONE
    else
        set STATE = FAILED
    endif
    echo "[`date '+%Y-%m-%d %H:%M:%S'`] $STATE spice_failure_check_$KIND exit_code=$RC" >> "$LOG"
    if ( $RC != 0 ) exit $RC
end
set SRES = "$OUT/spice/spectre/results.csv"
set TRES = "$OUT/spice/torch/results.csv"
echo "[`date '+%Y-%m-%d %H:%M:%S'`] START compare_failures rc=running" >> "$LOG"
$PY "$CMP" --spectre-results "$SRES" --torch-results "$TRES" --spectre-knobs "$OUT/prediction/pred_spectre.csv" --torch-knobs "$OUT/prediction/pred_torch.csv" --spec-ids "$OUT/prediction/spec_ids.npy" --out "$OUT/summary" >>& "$LOG"
set RC = $status
if ( $RC == 0 ) then
    set STATE = DONE
else
    set STATE = FAILED
endif
echo "[`date '+%Y-%m-%d %H:%M:%S'`] $STATE compare_failures exit_code=$RC" >> "$LOG"
if ( $RC != 0 ) exit $RC
echo "Closure complete: $OUT/summary/summary.md" >> "$LOG"
exit 0
