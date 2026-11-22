# Phase 4 — Verilog-A behavioral implementation

## Status

This is an initial behavioral primitive library, not yet a validated checkpoint-to-Spectre flow. The canonical constants and block-level modules are committed on `phase4-veriloga`. Task A wiring is pending: `ota_edge.ab_mode` reserves 0=legacy, 1=ratio, 2=differential, 3=cap-divider; mode 2 forces A=B=1 and mode 3 consumes Ahat/Bhat. The circuit-level scaling/validation still needs review against the model before use for signoff.

## Unit contract

Single-ended core supply is 0.8 V, Vcm=0.4 V, Vu=0.07 V/unit, C_u=50 fF, T_u=75.1 ns/model-time-unit, I_u=C_u Vu/T_u, G_u=C_u/T_u. Model state is x=(V−Vcm)/Vu; currents in model units are scaled by I_u and conductances by G_u. `HWUnits` is authoritative for configurable conversions.

## Behavioral modules

| File / module | Ports | Principal parameters |
|---|---|---|
| `ota_edge.va` / `ota_edge` | src,dst electrical | mode,A,B,Ahat,Bhat,ab_mode,s,theta,gm,Isat,z_gate,vca,gln |
| `shunt.va` / `shunt` | src,dst electrical | G,z_gate,vca,vca_gate_shunt |
| `node_cap_leak.va` / `node_cap_leak` | node electrical | c_eff,c_phys,g_leak,Vcm |
| `rail_clip.va` / `rail_clip` | node electrical | x_max,sharp,Iclip,Vcm,Vu |
| `sample_hold.va` / `sample_hold` | in,out,clk electrical | C_hold,R_on,R_off,inject_v,droop_tau |
| `boundary_sense_ota.va` / `boundary_sense_ota` | src,dst electrical | gm,Isat,A,B,theta,s,z_gate |
| `crossbar_accumulator.va` / `crossbar_accumulator` | sense,acc electrical | weight,Vref,Gunit,Cacc |
| `interstage_transfer.va` / `interstage_transfer` | in,out,enable electrical | w1,w2,w3,vth,x_hard |
| `vca_bias.va` / `vca_bias` | u0..u3,gain electrical | weights,bias |
| `gln_bias.va` / `gln_bias` | u0..u3,trim electrical | weights,bias,delta_max |
| `phase_ctrl.va` / `phase_ctrl` | clk,phase,stage electrical | stages and phase durations |

## Fidelity and gaps

`ota_edge` contains legacy and hardware parameter branches and the A/B extension hook. Legacy implementation is intended to represent softplus A/B values, signed differential tanh, edge/VCA multiplier, compliance gating and the linear floor term. Hardware branch exposes slope attenuation and A/B normalized inputs. These are primitive implementations; exact end-to-end legacy reproduction has not been established. Hardware output-leg compliance, parameter-bank scheduling, exact shared-sense wiring, and nonideal S&H parameter conversion remain to be completed/verified. OutputAffine and input log/min-max scaling are digital/off-chip by design.

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

## Validation

The interpreter has no model imports. The paper-checkpoint legacy path and per-stage dynamics, unit conversions, and hardware toy round trips are tested. Compliance round trips are checked at a 1e-9 maximum absolute tolerance; other listed groups retain their exact round-trip checks. The full suite is not an analog signoff: Spectre/ahdlcmi availability and Verilog-A simulation are environment dependent. Temporal-output-mesh compliance and active timemux dynamics remain explicitly rejected.
