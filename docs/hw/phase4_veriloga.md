# Phase 4 — Verilog-A behavioral implementation (D2 status)

## Unit contract

Single-ended 0.8 V supply, `Vcm = 0.4 V`, and normalized state `x=(V−Vcm)/Vu`. The common macros use the `HWUnits` defaults: `Vu=0.07 V`, `Cu=50 fF`, `Tu=350 ns/4.660583 = 75.097 ns`, `Iu=Cu·Vu/Tu ≈ 46.60 nA`, and `Gu=Cu/Tu ≈ 0.666 µS`. Current parameters are model units unless a module's table says otherwise; capacitors and resistors are SI values. The macros live in `veriloga/hrknet_constants.vams`.

## Module interfaces

Ports are electrical. Unless otherwise noted, model-unit parameters are converted at the cell boundary using `Iu` or `Gu`.

| Module | Ports (direction) | Exporter-facing parameters |
|---|---|---|
| `ota_edge` | `src,dst` (inout) | `mode,ab_mode,A,B,Ahat,Bhat,cap_code_a,cap_code_b,cap_bits,cap_rmax,s,theta,gm,Isat,z_gate,sigma_z,vca,gln,g_hw,g_min,f_ic,lambda_v` |
| `shunt` | `src,dst` (inout) | `G,z_gate,vca,vca_gate_shunt` |
| `node_cap_leak` | `node` (inout) | `c_eff,c_phys,g_leak` |
| `rail_clip` | `node` (inout) | `x_max,sharp,Iclip` |
| `sample_hold` | `in,clk` (input), `out` (output) | `C_hold,R_on,R_off,inject_v,droop_tau,Vlogic` |
| `boundary_sense_ota` | `src,dst` (inout) | `gm,Isat,A,B,theta,s,z_gate,alpha,x_max,gate_width,sigma_z` |
| `crossbar_accumulator` | `s0p,s0n,…,s3p,s3n` (sense-current branch ports), `acc0..acc2` (inout; 4×3 primitive) | `W[0:11],hw_mode,hw_bits,hw_bound,acc_compliance_en,Wmax,lambda_v,x_hard,Vcm,Vu,Ut,k_soft` |
| `interstage_transfer` | `in,enable` (input), `out` (output) | `w1,w2,w3,vth,x_hard,Ron,Roff,smooth_width,enable_width` |
| `vca_bias` | `u0..u3` (input), `gain` (output) | `w0..w3,bias,rank,zeta_max` |
| `gln_bias` | `u0..u3` (input), `trim` (output) | `w0..w3,bias,delta_max` |
| `phase_ctrl` | `clk` (input), `phase,stage` (output) | `stages,t_hold,t_transfer,t_switch,t_evolve` |

All module files include `hrknet_constants.vams`. The fast structural suite checks file presence, the shared include, and these parameter names. `python -m pytest tests/test_veriloga_modules.py -q` currently passes (3 tests).

## Fidelity modes and analog behavior

`ota_edge.mode=0` is the legacy path: exporter-resolved softplus A/B, `u=gm·(s·(A·xs−B·xd)+theta)`, `Isat·(tanh(u)+0.01u)`, and endpoint compliance gates. The edge is conservative and scaled by `Iu`. Hardware mode (`mode=1`) supports `ab_mode=0` legacy A/B, `1` ratio A/B, `2` differential (`A=B=1`), and `3` cap-divider. For mode 3, integer codes produce `Ahat/Bhat = 1/rmax +(1−1/rmax)·k/(2^N−1)`. The hardware OTA uses split source/sink legs, rail factors, and the `lambda_v` channel-length term; unlike legacy, each node's rail legs can draw current from/to the supplies. `g_hw`, `f_ic`, GLN and VCA multipliers are explicit real parameters. The maximum-current ceiling and fully banked time-multiplexed parameters still have to be supplied by the exporter.

The primitive blocks model shunt and leak conductances, node capacitance, legacy soft rail clipping, S&H acquisition/injection/hold droop, crossbar accumulation, VCA gain (`2·sigmoid(z)`), GLN tail trim, and a transfer rewrite conductance. The transfer rewrite is a discrete stage-boundary operation, not part of the continuous ODE: its target uses exact `max(x-vth,0)` and hard `min(max(y,-x_hard),x_hard)`. Spectre's branch current is continuous in the output voltage; the target is evaluated algebraically from the input. `phase_ctrl` sequences hold → transfer/write → bank switch → evolve with explicit per-phase durations after a rising `clk`; its timing is not yet characterized.

## Exporter interface and resolved table

The Python behavioral exporter is available as `python -m export.spectre_export --checkpoint ... [--hw-config ... | --hw-preset NAME] --mode legacy|hw --fabric timemux|copies --specs N --out DIR`; library callers can use `export_model(model, ...)` for an already-built KirchhoffNetWithIO. `params.json` contains `resolved` (schema `kirchhoffnet-resolved-params-v1`) with input scaling, topology, stage settings/cells/gates/readout/VCA/GLN, transfers and output mapping. Dimensionless values are the interpreter's source of truth. In hw mode each stage additionally has a `physical` view (V, A, S, F, s) and cap-divider integer codes. `HWUnits` provides the conversions: x→V uses Vcm+Vu*x, currents use Iu, conductances use Gu, capacitances use Cu, and time uses Tu.

The state-dict map classifies serialized keys as topology, mapped parameter, or explicitly documented digital/off-chip exceptions (`DIGITAL_OFFCHIP_KEYS`). During resolved-table construction, a `TorchDispatchMode` tracker records which original state-dict tensors participate in tensor operations by data pointer. Export accepts parameters only when the builder consumed them, and accepts topology only when the builder read it as well; unmatched keys fail with `ValueError("unmapped state_dict key: ...")`. Input log scaling and output affine/post-transfer are always digital/off-chip. In cap-divider A/B mode, the retained legacy `a_raw`/`b_raw` checkpoint tensors are unused; their six narrowly scoped cell-library patterns are documented compatibility exceptions. The known-name regression injects `net.core.stages.0.cell_lib.unused.a_raw`, which has an allowed leaf and owner prefix but is rejected because the table builder never reads it. The paper checkpoint accounts for 317 keys: 255 behavioral parameter, 49 topology, and 13 digital/off-chip. TOY_NET under `phase3b_tm` accounts for 114 keys: 75 behavioral parameter, 14 topology, and 25 digital/off-chip (including the 12 retained legacy A/B tensors).

### Python interpreter support matrix

The standalone interpreter resolves and reproduces the table for symmetric/6-bit Vref; bounded/5-bit signed crossbar; transfer rail clamp; physical node capacitance; GLN tail modulation (including `delta_max`); and the supported shared-sense/core compliance paths. They are applied in the same transfer/RHS location as the model and tested on perturbed `TOY_NET` parameters. The existing core and core-differential toy preset round trips also remain exact.

| Configuration | Result on toy round-trip | Notes |
|---|---:|---|
| `phase3a_core` | 0 output / 0 state | exact |
| `phase3a_core_diff` | 0 output / 0 state | exact |
| core + symmetric 6-bit Vref | 0 / 0 | exact |
| core + bounded 5-bit signed crossbar | 0 / 0 | exact |
| core + transfer clamp | 0 / 0 | exact |
| core + physical node capacitance | 0 / 0 | exact |
| core + GLN tail, `delta_max=ln(4)` | 0 / 0 | exact for current TOY_NET GLN configuration |
| `phase3a_full` with `hw_compliance=false` | 0 output / 0 state | combined non-compliance flags compose exactly |
| `phase3a_core` with `hw_compliance=true` | ≤ 1e-9 output / state | leg-wise core, boundary, shared-sense and accumulator compliance; frozen core legs |
| `phase3a_full` | ≤ 1e-9 output / state | full preset composed with compliance |
| `phase3a_core` + timemux, `ktc=false`, 4-bit node quantisation | 0 output / 0 state | injection, gain error, hold-only droop, boundary quantisation and full transfer |
| `phase3b_tm` + recorded kT/C draws | 2.98e-11 output / 2.76e-10 state | full phase3a preset plus S&H; 3 samples, tested at 1e-9 |
| temporal output mesh + compliance | rejected | mesh destination compliance is not yet mirrored |

### Supported vs refused forward-affecting flags

Supported hardware flags are the round-trip groups in the table above, including active `timemux` injection, gain error, held-branch droop, `xfer_mode`, and node quantisation. The interpreter's `simulate(table, x, noise=...)` consumes one exact standard-normal tensor per active boundary; scale is `sqrt(k_B*T/C_u)/V_u`. Without explicit draws it follows the default eval behavior, where kT/C is inactive. `param_bits` is applied to transfer coefficients when `xfer_mode='quant'`; node quantisation is tested at 4 bits. The configured `c_hold_ff`, `c_int_ff`, `isat_floor`, and `settle` currently have no effect on this repository's forward pass: `c_hold_ff` is not used by capture, while `settle_core` / `effective_isat` are utilities not called from `KirchhoffNet` or `DifferentialStage`. They are retained in the exported config without inventing behavior.

Refused hardware flags remain: temporal output mesh + `hw_compliance` (mesh destination compliance is not mirrored); `hw_isat_eff_clamp` (per-family post-gate/VCA Isat clamping not resolved); `hw_vca_bits` (quantized VCA logits not resolved); stochastic mismatch (model-shared per-slot draws are not exported); and `hw_power_lambda` (training-only objective, not inference behavior). Validate flags using `HWConfig` before export.

### Current-mode crossbar design

`crossbar_accumulator` accepts four sense-OTA currents through four ideal zero-volt series branches (`s0p/s0n` through `s3p/s3n`) and injects the weighted current into three accumulator ports (`acc0..acc2`). The branch current direction is p→n; connect each sense OTA output/current source in series with its pair. `W[0:11]` is a flattened row-major 3×4 parameter array (`W[j*4+k]`). There is no sense-voltage conductance, local capacitor, or Vcm bleed: accumulator-node capacitance/leak is owned by `node_cap_leak`.

For `hw_mode=1`, optional `hw_bound` first applies `Wmax*tanh(W/Wmax)`; `hw_bits` quantizes an offset-binary magnitude code over `[0,2*Wmax]`. The effective signed contribution is the unipolar code current minus a fixed replica-column contribution `Wmax*I_sense[k]`, exactly matching `hw.constraints.xbar_map(..., mode="offset_binary")`. With `acc_compliance_en`, each net signed output current is selected as the top leg when positive or bottom leg when negative, then multiplied by normalized `leg_factors` using `k_soft=2000`, `lambda_v`, `Ut`, `Vu`, and `x_hard`, matching `hw.constraints.acc_compliance`.

## Spectre compilation

Run all generated inputs, Cadence setup files, PSF databases, and logs under the git-excluded run root. The kit is sourced only after entering the run directory, because sourcing it writes `cds.lib`, `display.drf`, `pdk.cdsinit`, and `tech.lib` into the current directory.

Exact per-netlist invocation (the wrapper keeps the short-lived remote shell alive):

```sh
bash -c 'nohup tcsh -c "cd <rundir>; source ~/workarea_GF22_FDX_EXT/kit.gf22fdx_ext.v1.0_4.1_stack19_oa.csh >& /dev/null; spectre -format psfascii <net>.scs +log <net>.log" > <rundir>/run.out 2>&1 &'
```

Run the compile matrix from the phase4b worktree with:

```sh
bash -c 'cd /home/annaik/Documents/kirchhoffnet_refactor/.worktrees/phase4b; mkdir -p /home/annaik/Documents/kirchhoffnet_refactor/.agent_scratch/taskD2/runs/compile_session; nohup /home/annaik/Documents/kirchhoffnet_refactor/venv/bin/python scripts/spectre/compile_modules.py > /home/annaik/Documents/kirchhoffnet_refactor/.agent_scratch/taskD2/runs/compile_session/helper.out 2>&1 &'
```

The helper emits an independent tiny `ahdl_include` + DC deck per `.va` module and records the complete Spectre output. Results from Spectre 23.1.0.063:

| Module | Compile | Spectre errors / warnings |
|---|---|---:|
| `ota_edge` | PASS | 0 / 0 |
| `shunt` | PASS | 0 / 0 |
| `node_cap_leak` | PASS | 0 / 0 |
| `rail_clip` | PASS | 0 / 0 |
| `sample_hold` | PASS | 0 / 0 |
| `boundary_sense_ota` | PASS | 0 / 0 |
| `crossbar_accumulator` | PASS | 0 / 0 |
| `interstage_transfer` | PASS | 0 / 0 |
| `vca_bias` | PASS | 0 / 0 |
| `gln_bias` | PASS | 0 / 0 |
| `phase_ctrl` | PASS | 0 / 0 |

The baseline compile matrix needed no source changes; after the current-mode redesign, `crossbar_accumulator` was recompiled with 0 errors and 0 warnings. Spectre emitted routine notices (e.g. output filter relaxation on compile-only DC decks), but no warnings.

## Spectre-versus-reference characterization

`/home/annaik/Documents/kirchhoffnet_refactor/venv/bin/python scripts/spectre/characterize.py` generates PSFASCII DC and transient analyses with 70 netlist cases (82 measured result rows, including per-output crossbar measurements). OTA/shunt/rail-clip/boundary-sense/crossbar currents are converted to `Iu`; transfer voltage error and node-cap transient error are in x units. Node-cap leakage is compared against `x0·exp(−leak·t_model/c_eff)` over five RC constants. Rail clipping is evaluated via `DifferentialStage.soft_clip` in Torch at DC points spanning both rails. Boundary/sense uses the actual `FreeTanhLibrary.forward_tanh` path, with boundary `(u,node)` and sense `(node,Vref)` ordering from `DifferentialStage`. Transfer is compared with `StageTransfer` from `stage_transfer.py` (including the configured hardware clamp); VCA uses `gating.vca.vca_edge_gain`. Core OTA, shunt, and GLN values still use copied equations rather than callable framework forward paths. In this checkout, `hw.constraints.cap_divider_gain` is absent, so cap-divider code resolution remains a copied equation rather than a call to that helper.

| Module / mode | Cases | Maximum absolute error | Maximum relative error |
|---|---:|---:|---:|
| `ota_edge` legacy | 16 | 8.52e-16 Iu | 8.36e-15 |
| `ota_edge` hw (differential / cap-divider) | 6 | 5.68e-16 Iu | 8.64e-16 |
| `shunt` DC | 3 | 2.84e-16 Iu | 1.87e-16 |
| `node_cap_leak` transient | 4 | 8.71e-6 x | 1.45e-6 |
| `rail_clip` DC (Torch `DifferentialStage.soft_clip`) | 11 | 1.39e-17 Iu | 2.78e-16 |
| `boundary_sense_ota` boundary + sense (Torch `FreeTanhLibrary.forward_tanh`) | 16 | 7.84e-15 Iu | 3.24e-14 |
| `crossbar_accumulator` (3 ideal W + hw offset-binary; Torch `xbar_map`/`acc_compliance`) | 18 outputs | 1.07e-6 Iu | 1.26e-6 |
| `sample_hold` deterministic capture | 1 transient | 1.81e-5 x | 1.33e-5; real `StageBoundary.capture` (kT/C excluded), injection/gain and 50 ns droop interval |
| `interstage_transfer` HW clamp vs `StageTransfer` | 4 | 5.72e-9 x | 5.33e-10 |
| `vca_bias` | 2 | 4.44e-16 gain | 4.18e-16 |
| `gln_bias` tail trim | 2 | 0 trim | 0 |
| `phase_ctrl` | 2 boundaries (8 phase edges + 2 stage edges) | 2.78e-8 ns (3.70e-10 Tu) | 5.56e-9 versus longest configured phase duration |

The characterization result JSON and all raw runs are under `.agent_scratch/taskD2/runs/characterize_<timestamp>/`; no Cadence-generated files belong in the worktree. The Spectre-backed test is slow-marked and explicitly run with `python -m pytest tests/test_veriloga_modules.py -q -m slow -k spectre`; it skips if `tcsh`, the kit setup, or Spectre is unavailable.

## Known gaps / release status

- Characterization is partial: S&H deterministic capture/droop and two phase-controller boundaries are now measured; the phase output durations are set by the Verilog-A parameters because `hw/timemux.py` has no phase-duration API. Its `HWUnits.tu` supplies the normalized timing unit. Crossbar DC mixing/compliance, node-cap RC transient, and rail clipping have characterized coverage, but this is not full module equivalence validation.
- Boundary/sense OTA uses the real `FreeTanhLibrary.forward_tanh` Torch path and matches 16 Spectre points to 7.84e-15 Iu maximum absolute and 3.24e-14 maximum relative error. Core OTA/shunt/GLN references are copied formulas rather than callable paths. Transfer uses the exact piecewise stage rewrite and matches callable `StageTransfer` to 5.72e-9 x maximum absolute error; cap-divider helper resolution remains unavailable.
- The canonical A/B modes compile, but per-stage parameter arrays/indexing and complete timemux bank switching are still missing. The exporter and parameter table must drive actual instances consistently.
- HW OTA compliance is a behavioral leg approximation and still needs comparison against `hw.constraints.leg_factors`/`compliance_legs` (including normalization and 85 °C thermal voltage). The specified Isat ceiling is not enforced inside this primitive.
- S&H deterministic capture characterization is compared with `StageBoundary.capture` using kT/C disabled; stochastic kT/C remains out of scope. The Torch boundary has no C_int charge-sharing term (C_int only sets bias-settling tau), so no charge-sharing effect is claimed. Droop is the model's fixed-sign hold offset. The VA includes `inject_gain_err` and `droop_v` to represent that deterministic formula.
- `phase_ctrl` was checked over two boundaries; `interstage_transfer` still models an algebraic rewrite, not physical event timing. Crossbar current polarity, offset-binary replica subtraction, and accumulator compliance match the selected Torch cases only; do not claim full model equivalence or use for tapeout.


## Network netlist (legacy copies)

`export.spectre_export.export(..., mode="legacy", fabric="copies")` writes a resolved `params.json`, a deck with one copy per stage, and a sized crossbar VA beside it. Scalar cell parameters, gains, gates, weights, node capacitance/leak, and transfer coefficients come from the resolved table; positive physical input specs are log-scaled off-chip as in `netlist_interp._spec_inputs`. Node voltage encodes `x=(V−Vcm)/Vu`.

Stage sequencing uses `veriloga/stage_gate.va`. Stage 0 starts at Vcm; each node in stage k>0 is clamped through a strong conductance to the previous stage's nodewise transferred state until cumulative `sum(t_span[0:k])*Tu`, then the clamp transitions to zero. The preceding copy evolves for its full stage duration before transfer and may continue evolving afterward because no later stage reads it. The crossbar's internal zero-volt branch is the sole sense branch; Vref drives its return node directly (no duplicate `Vsense` or 1 mΩ `Rref`). For `freeze_read`, `ota_edge_sampled` evaluates from the held S&H outputs while applying current to live stage nodes.

Run all decks and compile jobs only under `/home/annaik/Documents/kirchhoffnet_refactor/.agent_scratch/taskD3/runs/<subdir>`. Enter that directory before sourcing the GF22 setup, then invoke Spectre 23.1 with `-format psfascii`. `compile_modules.py --module stage_gate` uses the same run root. Convert node traces to x for interpreter comparison; output mapping and post-transfer are off-chip.

| Comparison (fixed Spectre deck + Verilog-A) | Spectre | Interpreter-10 | Interpreter-1000 | Result |
|---|---:|---:|---:|---|
| Sixteen canonical test specs; maximum absolute logit error | mapped final-stage voltages | 6.5025e-3 | 9.1004e-6 | row 12 vs interp-10 is coarse-reference error |
| Sixteen canonical test specs; maximum trajectory error per stage vs interp-1000 | 1.22e-6 / 2.32e-6 / 4.12e-6 / 6.65e-6 / 1.23e-5 / 1.43e-5 / 3.4098e-5 x | — | — | all stage maxima are <3.5e-5 x |
| Row 12; max trajectory error per stage vs interp-1000 | 1.20e-6 / 2.32e-6 / 4.12e-6 / 6.65e-6 / 9.71e-6 / 5.83e-6 / 3.4098e-5 x | — | — | stage-6 error reduced by 490x |

Spectre initially ignored `reltol`, `vabstol`, and `iabstol` because they were specified on the `tran` line (`SFE-106`). The exporter now puts them in `simulatorOptions options` and uses `maxstep=dt/100`. This corrected the settings but did not resolve row 12: with effective global tolerances, its stage-6 error was still 1.6702e-2 x. Comparing endpoint derivatives then exposed the circuit mismatch: `boundary_sense_ota.va` multiplied the parallel `G*(xs-xd)` conductance by the endpoint compliance gates, whereas interpreter `_cell` applies those gates only to the nonlinear OTA term; the conductance remains outside them but is still multiplied by the external stage gate/VCA. When row 12's stage-6 source nodes crossed the ±4 rails, Spectre therefore suppressed conductance current that the reference retained, accumulating the late readout error. The Verilog-A current equation now matches `_cell`: `Isat*(tanh(u)+alpha*u)*gate + G*(xs-xd)*z_gate*sigma_z`. With that fix and the corrected settings, Spectre reran all 16 specs in 562 s with 0 errors, 11 warnings, and 11 notices. Row 12's stage-6 trajectory error is now 3.4098e-5 x and the max logit error vs interp-1000 is 9.1004e-6. Updated per-spec logits and per-stage errors are in [phase4_netlist_validation.csv](phase4_netlist_validation.csv).

### Row-12 convergence, trajectory, and metric diagnosis

`netlist_interp.simulate` was run in float64 with 10, 40, 100, 1,000, 4,000, and 16,000 Heun steps per stage. The complete stage-6 final state vectors (node indices 0–17) and logits are:

```text
10:    state=[-0.133883522262, 4.590987811260, -6.114375386397, -0.722000840652, 2.608916578604, 0.104092453749, -0.103092224566, -0.141144637405, -0.033857110841, -1.849677633809, 0.652376872079, 0.210626919592, -0.851271983077, -0.161133169146, -0.276002383712, -0.190079311144, -0.037856576702, -0.307082948323]
       logits=[-1.874828212722, -3.965724387378, -1.126746910465, -0.503237709524, -0.065756922141, 0.409679218579, -0.895641984088]
40:    state=[-0.133901490125, 4.594344465549, -6.114427468717, -0.721995106477, 2.608960401916, 0.104092123170, -0.103099020392, -0.141264690909, -0.033850681287, -1.849680208480, 0.652430962633, 0.211828682642, -0.851340367309, -0.159894555445, -0.278089051774, -0.190030774063, -0.039274208169, -0.303292994200]
       logits=[-1.873547833742, -3.965828371398, -1.125738371091, -0.506388445762, -0.065739904440, 0.407450832662, -0.889493065929]
100:   state=[-0.133902503365, 4.594412429375, -6.114403992553, -0.721994717022, 2.608963078860, 0.104092094813, -0.103099381994, -0.141271492334, -0.033850315577, -1.849680304758, 0.652434018775, 0.211878240983, -0.851342589175, -0.159815189411, -0.278196004004, -0.190012922504, -0.039340519642, -0.303079979960]
       logits=[-1.873495034399, -3.965831749929, -1.125673747912, -0.506549913715, -0.065733645493, 0.407346595416, -0.889147352503]
1000:  state=[-0.133902677624, 4.594416359833, -6.114404600888, -0.721994658602, 2.608963495458, 0.104092095746, -0.103099443042, -0.141272762669, -0.033850248446, -1.849680360835, 0.652434565833, 0.211878970432, -0.851343010673, -0.159815893753, -0.278196047633, -0.190013738356, -0.039340928842, -0.303080755894]
       logits=[-1.873494257246, -3.965832390853, -1.125674321418, -0.506549979582, -0.065733931539, 0.407345952179, -0.889148611834]
4000:  state=[-0.133902674714, 4.594416384781, -6.114404605396, -0.721994658107, 2.608963492981, 0.104092095892, -0.103099443589, -0.141272772267, -0.033850248861, -1.849680362402, 0.652434565404, 0.211878976827, -0.851343015527, -0.159815892192, -0.278196053797, -0.190013741330, -0.039340934270, -0.303080744057]
       logits=[-1.873494250433, -3.965832398234, -1.125674320146, -0.506549988888, -0.065733932581, 0.407345943648, -0.889148592624]
16000: state=[-0.133902674342, 4.594416383060, -6.114404606479, -0.721994658102, 2.608963496212, 0.104092095945, -0.103099443476, -0.141272772870, -0.033850248850, -1.849680361573, 0.652434565480, 0.211878977178, -0.851343016210, -0.159815891919, -0.278196053885, -0.190013741367, -0.039340934329, -0.303080745448]
       logits=[-1.873494250059, -3.965832399272, -1.125674319924, -0.506549989021, -0.065733932595, 0.407345943555, -0.889148594881]
```

The interpreter is converged for this case: 1,000 vs 16,000 steps differ by at most 2.33e-8 x in the stage-6 state and 1.70e-8 in logits. Row 12's trajectory maxima by stage are `0.140774, 0.255239, 0.349876, 1.243356, 1.672368, 1.326032, 6.114405` x; the maximum among the other 15 specs at those stages is `0.135522, 0.536532, 1.050808, 2.200013, 2.762890, 3.692515, 3.782321` x. Thus only row 12 makes a large stage-6 excursion beyond both rails (`x_max=4`), reaching +4.594 and −6.114; stage-6 clip sharpness is 0.03353 and the cell endpoint-gate width is 0.02. The legacy run has `hw_compliance` disabled, so these are the soft clip and endpoint gates, not hardware compliance legs.

Row 12 crosses the transfer ReLU `Vth` during the trajectories of 3, 1, 1, 3, 1, and 1 nodes at transfer boundaries 0–5 (10 nodes total). The closest sampled trajectory distance to `Vth` per boundary is 0, 2.21e-5, 4.35e-6, 9.30e-6, 6.99e-5, and 1.08e-4 x; endpoint minima are 7.71e-4, 1.91e-3, 6.91e-3, 6.03e-3, 1.57e-2, and 2.26e-2 x. This makes the trajectory numerically demanding but is not chaotic in the tested perturbation: adding 1e-9 x to all 18 stage-6 initial nodes changed the final stage-6 state by at most 1.31e-9 x.

Using `StudentEvaluator.identify_failures` with the ZIG forward surrogate, `DEGRADE_REL_THRESHOLD=0.10`, and the same validity gate as `scripts/eval_paper_checkpoint.py`, the failure flags are identical for corrected Spectre, interp-10, and interp-1000: rows 0–15 are `P,P,P,F,P,P,P,P,P,P,P,P,P,P,F,P` (only rows 3 and 14 fail). Thus row 12's numerical discrepancy does not change the 10% pass/fail metric. After the circuit fix, row 12's maximum stage-6 trajectory error is 3.4098e-5 x; over all 16 specs, maximum stage-6 state error is also 3.4098e-5 x and maximum logit error vs interp-1000 is 9.1004e-6.

The mismatch that initially prevented stage-0 validation was missing readout conductance; it was fixed by adding `G*(src-vref)` to the high-impedance `boundary_sense_ota` destination-current equation rather than a loading shunt. A separate late-stage bug remained: the Verilog-A expression put that G term inside the source/destination endpoint compliance factors, unlike `_cell`, which gates only the nonlinear OTA term. It is now `-(Isat*(tanh(u)+alpha*u)*gate + G*(xs-xd)*z_gate*sigma_z)*Iu`, preserving the source's high impedance while matching the interpreter's gate ordering. The earlier stage-6 check at 0.333 Tu (all 22 `I(dst)` values within 1e-8 Iu, W parameters within 4.5e-16, and accumulator derivatives within 3.6e-5 x/Tu) did not sample the later high-rail excursion. With corrected tolerance placement and conductance gating, the 16-spec run reduces row 12's stage-6 error from 1.6702e-2 x to 3.4098e-5 x.


Stage-gate standalone characterization (Spectre 23.1, 50 fF node, 1 S clamp, 10 ns release): after the initial 1 ns settling interval, maximum pre-release error was 2.27e-9 x; measured series-probe current after release (>11 ns) was 0 A. `compile_modules.py --module stage_gate` passed with 0 errors / 0 warnings.


## HW-mode netlist

`export.spectre_netlist.render(..., mode="hw")` now emits a copied-fabric behavioral deck for `phase3a_core` and `phase3a_core_diff`, and the exact `phase3a_full` feature settings. The exporter rejects other active forward-affecting flags or altered values instead of silently selecting legacy equations. Core OTA instances use `mode=1`, cap-divider codes or differential `ab_mode`, the table-resolved hardware slope/current-coupling factor, and zero floor term. Frozen-read cores use the four-terminal sampled OTA so held values drive the nonlinearity while live nodes receive current. The physical capacitance path uses table `C_node_F`; tail GLN scales resolved boundary/readout Isat; resolved symmetric Vref and signed crossbar parameters are emitted; transfer hard clamp is flag gated. Boundary/sense and accumulator compliance parameters are wired when enabled.

### Measured HW-mode status

The phase3a_core seeded/perturbed TOY_NET case was run with Spectre 23.1 (`-format psfascii`) across both stages and compared at Spectre sample times to `netlist_interp.simulate(..., steps_per_stage=1000)`: max trajectory errors were `8.40117279e-9 x` and `7.40322308e-9 x`. `phase3a_core_diff` measured `9.26453781e-9 x` and `8.47375777e-9 x`. The complete phase3a_full deck compiles with zero Spectre errors and now meets the ≤1e-4 x trajectory target: maximum errors are `1.08354029e-8 x` and `1.04585804e-8 x` per stage. A diagnostic run first exposed a polarity inversion in the sampled OTA's explicit node-current legs; negating the branch contributions fixed the full compliance result. With compliance disabled, the other full-preset groups together matched at `8.40e-9 x` maximum. Full preset compliance is now covered by a real Spectre trajectory, but remains behavioral Verilog-A and is not a tapeout qualification. Per-run details are in [phase4_hw_validation.csv](phase4_hw_validation.csv); reproducible M1/M2/M3 logs are in `.agent_scratch/hwnet/M1.log`, `M2.log`, `M3.log`, and `M3NC.log`.

`tests/test_hw_spectre_netlist.py` adds fast structural coverage and a slow Spectre/interpreter test which skips when the GF22 kit or Spectre is unavailable. The M1/M2 measured runs used one seeded spec. The explicit-checkpoint M4 run on the paper checkpoint, `phase3a_core_diff`, and 16 canonical test specs passed with 0 Spectre errors; per-stage maximum errors were `9.99e-7, 1.86e-6, 3.31e-6, 6.67e-6, 8.23e-6, 8.40e-6, 1.65e-5 x`. These measurements are included in the CSV.


### Reproduction runner

`/home/annaik/Documents/kirchhoffnet_refactor/.agent_scratch/hwnet/RUN_ALL_hwnet.csh` runs the default pytest suite when invoked without arguments, then the phase3a_core_diff audit, 16-spec export, Spectre PSFASCII transient, and trajectory comparison. The runner uses core_diff so the checked-in paper checkpoint loads without cap-divider-only tensors. Passing a checkpoint path skips the default suite and performs that audit/export/16-spec Spectre/compare sequence. It writes the complete log to `.agent_scratch/hwnet/RUN_ALL_hwnet.log` and keeps all generated Cadence files beneath `.agent_scratch/hwnet/RUN_ALL_run/`. Launch long runs with `bash -c 'cd /home/annaik/Documents/kirchhoffnet_refactor/.agent_scratch/hwnet && nohup tcsh RUN_ALL_hwnet.csh > /dev/null 2>&1 &'`; source the kit only after entering the run directory.

### Recurring window clamp characterization

`window_clamp(node,target,win)` uses a smooth logistic of `V(win)` and implements `g_clamp*(V(node)-V(target))*w`; it can be opened repeatedly, unlike `stage_gate`'s one-shot absolute-time release. The scratch transient case is `/.agent_scratch/tmnet/window_clamp_characterize.scs` (ran with Spectre 23.1, 0 errors / 0 warnings): `V(win)` is low on [0,1 ns), high on [1.1,3 ns), and low thereafter, with a 0.1 V initial node-target error. The measured maximum open-window node-target error (after a 0.1 ns opening ramp) is `1.00e-10 V`; the closed-window clamp-current upper bound from measured node/control samples is `1.93e-23 A` (logistic multiplier at 0 V). Run script/log: `.agent_scratch/tmnet/compile_window_clamp.csh` and `.agent_scratch/tmnet/compile_window_clamp.log`.


### Time-multiplexed fabric

`fabric="timemux"` uses one shared 18-node state with seven coefficient banks. Each evolve window remains exactly `t_span[k] * Tu`; the capture, transfer, and settle intervals are outside the ODE-active span. Boundary samples are captured before transfer starts, and the freeze-read samples are taken from the settled stage-entry state. The boundary/read clocks and stage-specific readout references use near-vertical PWL edges: plain PWL interpolation had produced long ramps between pulses, causing the sample switches to remain enabled and the readout references to drift across stages. Closing the boundary sample 5 ps before the transfer clamp opens avoids a coincident sample/transfer edge.

For one canonical test spec, after the timing corrections the maximum node error against `netlist_interp.simulate(..., steps_per_stage=1000)` was `8.9e-8 x` in stage 0 and `3.02e-5 x` in stage 1 (stage-1 entry transfer mismatch `2.56e-5 x`). The seven-stage transient was also run. Across the 16 canonical test specs, maximum trajectory errors by stage were `4.87e-7 / 5.04e-5 / 1.54e-4 / 6.77e-4 / 1.08e-3 / 2.19e-3 / 3.46e-3 x`; maximum post-readout logit error was `1.58e-3`. The 10% ZIG pass/fail flags remain `P,P,P,F,P,P,P,P,P,P,P,P,P,P,F,P`. Per-spec stage errors and flags are recorded in [phase4_timemux_validation.csv](phase4_timemux_validation.csv).

The slow Spectre regression `tests/test_timemux_spectre.py` checks stage 0–1 against interp-1000 and skips when tcsh, the GF22 kit, Spectre, or the canonical dataset is unavailable. Run it with `pytest -m 'slow and spectre' tests/test_timemux_spectre.py`. The one-spec debug transient is under `/home/annaik/Documents/kirchhoffnet_refactor/.agent_scratch/tmnet/timemux_try`; the 16-spec validation log is `/home/annaik/Documents/kirchhoffnet_refactor/.agent_scratch/tmnet/compile_timemux_16.log`.
