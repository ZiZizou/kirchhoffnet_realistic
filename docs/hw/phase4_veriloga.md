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
| `crossbar_accumulator` | `sense` (input), `acc` (inout) | `weight,Vref,Gunit,Cacc,Vcm` |
| `interstage_transfer` | `in,enable` (input), `out` (output) | `w1,w2,w3,vth,x_hard,Ron,Roff,smooth_width,enable_width` |
| `vca_bias` | `u0..u3` (input), `gain` (output) | `w0..w3,bias,rank,zeta_max` |
| `gln_bias` | `u0..u3` (input), `trim` (output) | `w0..w3,bias,delta_max` |
| `phase_ctrl` | `clk` (input), `phase,stage` (output) | `stages,t_hold,t_transfer,t_switch,t_evolve` |

All module files include `hrknet_constants.vams`. The fast structural suite checks file presence, the shared include, and these parameter names. `python -m pytest tests/test_veriloga_modules.py -q` currently passes (3 tests).

## Fidelity modes and analog behavior

`ota_edge.mode=0` is the legacy path: exporter-resolved softplus A/B, `u=gm·(s·(A·xs−B·xd)+theta)`, `Isat·(tanh(u)+0.01u)`, and endpoint compliance gates. The edge is conservative and scaled by `Iu`. Hardware mode (`mode=1`) supports `ab_mode=0` legacy A/B, `1` ratio A/B, `2` differential (`A=B=1`), and `3` cap-divider. For mode 3, integer codes produce `Ahat/Bhat = 1/rmax +(1−1/rmax)·k/(2^N−1)`. The hardware OTA uses split source/sink legs, rail factors, and the `lambda_v` channel-length term; unlike legacy, each node's rail legs can draw current from/to the supplies. `g_hw`, `f_ic`, GLN and VCA multipliers are explicit real parameters. The maximum-current ceiling and fully banked time-multiplexed parameters still have to be supplied by the exporter.

The primitive blocks model shunt and leak conductances, node capacitance, legacy soft rail clipping, S&H acquisition/injection/hold droop, crossbar accumulation, VCA gain (`2·sigmoid(z)`), GLN tail trim, and a transfer rewrite conductance. The transfer's ReLU and rail clamp use narrow smooth approximations to avoid hard discontinuities. `phase_ctrl` sequences hold → transfer/write → bank switch → evolve with explicit per-phase durations after a rising `clk`; it remains unvalidated against `hw/timemux.py`.

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

No Verilog-A source changes were needed to compile these modules. Spectre emitted routine notices (e.g. output filter relaxation on the compile-only DC decks), but no warnings.

## Spectre-versus-reference characterization

`/home/annaik/Documents/kirchhoffnet_refactor/venv/bin/python scripts/spectre/characterize.py` generates one PSFASCII DC deck with 33 parallel test cases and compares the operating point to the exact formulas implemented by the Verilog-A cells. OTA/shunt currents are converted to `Iu`; transfer voltage error is converted back to x units. The legacy OTA set covers both signs, nonzero theta, saturation, the 0.01·u floor, and the compliance boundary; hardware cases include differential and two cap-divider code pairs. VCA and GLN use rank-1 input vectors matching `gating/vca.py`'s `2*sigmoid(z)` and `gating/gln.py`'s clamped exponential tail equation. Transfer uses exact ReLU and hard ±x_hard clamp as the reference; the smooth clamp implemented in Spectre is therefore measured as a residual. These are exact copied formulas, not full framework/model forward calls.

| Module / mode | Cases | Maximum absolute error | Maximum relative error |
|---|---:|---:|---:|
| `ota_edge` legacy | 16 | 8.52e-16 Iu | 8.36e-15 |
| `ota_edge` hw (differential / cap-divider) | 6 | 5.68e-16 Iu | 8.64e-16 |
| `shunt` DC | 3 | 2.84e-16 Iu | 1.87e-16 |
| `node_cap_leak` | — | not measured | not measured |
| `rail_clip` | — | not measured | not measured |
| `sample_hold` | — | not measured | not measured |
| `boundary_sense_ota` | — | not measured | not measured |
| `crossbar_accumulator` | — | not measured | not measured |
| `interstage_transfer` HW clamp | 4 | 1.11 x | 9.68e-2 |
| `vca_bias` | 2 | 4.44e-16 gain | 4.18e-16 |
| `gln_bias` tail trim | 2 | 0 trim | 0 |
| `phase_ctrl` | — | not measured | not measured |

The characterization result JSON and all raw runs are under `.agent_scratch/taskD2/runs/characterize_<timestamp>/`; no Cadence-generated files belong in the worktree. The Spectre-backed test is slow-marked and explicitly run with `python -m pytest tests/test_veriloga_modules.py -q -m slow -k spectre`; it skips if `tcsh`, the kit setup, or Spectre is unavailable.

## Known gaps / release status

- Characterization is partial: node-cap RC transient, sample/hold capture and droop, crossbar accumulation, boundary OTA, rail clip, and phase duration edges against `hw/timemux.py` remain to be simulated and compared. Thus this is not full module equivalence validation.
- OTA/shunt, VCA, and GLN formulas match Spectre at sampled points to floating-point precision. The transfer's smooth tanh rail clamp differs from the exact hard-clamp reference by up to 1.11 x in the four-point sweep; decide whether that approximation is acceptable or change the implementation. This is not full Torch/PyTorch coverage.
- The canonical A/B modes compile, but per-stage parameter arrays/indexing and complete timemux bank switching are still missing. The exporter and parameter table must drive actual instances consistently.
- HW OTA compliance is a behavioral leg approximation and still needs comparison against `hw.constraints.leg_factors`/`compliance_legs` (including normalization and 85 °C thermal voltage). The specified Isat ceiling is not enforced inside this primitive.
- S&H `kT/C` noise and exact acquisition/droop characterization are absent; deterministic injection/droop remain approximate circuit abstractions.
- Transfer event timing, crossbar polarity/reference, and phase sequencing still require validation. Do not claim full model equivalence or use for tapeout.
