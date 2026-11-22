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
