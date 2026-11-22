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

The primitive blocks model shunt and leak conductances, node capacitance, legacy soft rail clipping, S&H acquisition/injection/hold droop, crossbar accumulation, VCA gain (`2·sigmoid(z)`), GLN tail trim, and a transfer rewrite conductance. The transfer's ReLU and rail clamp use narrow smooth approximations to avoid hard discontinuities. `phase_ctrl` is a behavioral phase/stage sequencer; its timing and bank-switch semantics have not yet been validated against `hw/timemux.py`.

## Spectre invocation

Use a scratch run directory outside the worktree and source the kit before calling Spectre. The exact verified shell pattern is:

```sh
tcsh -c "source ~/workarea_GF22_FDX_EXT/kit.gf22fdx_ext.v1.0_4.1_stack19_oa.csh >& /dev/null; cd /tmp/taskD2_runs/<module>; spectre -format psfascii <module>.scs"
```

The per-module elaboration helper is `scripts/spectre/compile_modules.py`; it writes decks/logs only under `/tmp/taskD2_runs`. Run it with `/home/annaik/Documents/kirchhoffnet_refactor/venv/bin/python scripts/spectre/compile_modules.py`. It has **not** been executed in this delegated session: the remote sandbox denied creating `/tmp/taskD2_runs`, and Cadence setup files must not be generated in the worktree.

## Spectre-versus-reference results

| Module / mode | Maximum absolute error | Maximum relative error | Status |
|---|---:|---:|---|
| `ota_edge` legacy / hw | not measured | not measured | Spectre blocked; no compile result |
| `shunt` | not measured | not measured | Spectre blocked; no characterization deck |
| `node_cap_leak` | not measured | not measured | Spectre blocked; no transient run |
| `rail_clip` | not measured | not measured | Spectre blocked |
| `sample_hold` | not measured | not measured | Spectre blocked; hold/droop not characterized |
| `boundary_sense_ota` | not measured | not measured | Spectre blocked |
| `crossbar_accumulator` | not measured | not measured | Spectre blocked |
| `interstage_transfer` | not measured | not measured | Spectre blocked; event/timing not characterized |
| `vca_bias` | not measured | not measured | Spectre blocked; torch comparison not run |
| `gln_bias` | not measured | not measured | Spectre blocked; torch comparison not run |
| `phase_ctrl` | not measured | not measured | Spectre blocked; no timing comparison |

## Known gaps / release status

- No Spectre compile has been run and no per-module PSF characterization/error table exists yet. The helper currently provides compile decks only; it is not the requested set of torch-correlated testbenches.
- The canonical A/B code modes exist, but per-stage parameter arrays/indexing and complete timemux bank switching are still missing. The exporter and parameter table must drive the actual instances consistently.
- HW OTA compliance is a behavioral leg approximation and needs comparison against `hw.constraints.leg_factors`/`compliance_legs` (including normalization and 85 °C thermal voltage). The specified Isat ceiling is not enforced inside this primitive.
- S&H `kT/C` noise and exact acquisition/droop characterization are absent; deterministic injection/droop are only approximate circuit abstractions.
- Transfer smoothening, event timing, VCA/GLN bias transforms, crossbar polarity/reference, and phase sequencing still require cell-level torch comparisons. Do not claim model equivalence or use for tapeout.
