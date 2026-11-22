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

The state-dict map classifies serialized keys as topology, mapped parameter, or explicit digital/off-chip (`DIGITAL_OFFCHIP_KEYS`; input log scaling and output affine/post-transfer are off-chip). Unknown or unrecognized-owner keys fail export. The mapper uses owner/module-and-leaf restrictions; the known-name regression case `net.core.stages.0.unused.z_logits` fails rather than being accepted merely because its leaf is familiar. On the paper checkpoint the current export accounts for 317 keys: 255 behavioral parameter keys, 49 topology keys, and 13 digital/off-chip keys. This remains whitelist-based rather than instrumentation of each table-builder tensor read, so extending the builder requires maintaining the owner/leaf map.

### Python interpreter support matrix

The standalone interpreter resolves and reproduces the table for the following forward-affecting hardware groups: symmetric/6-bit Vref; bounded/5-bit signed crossbar; transfer rail clamp; physical node capacitance; and GLN tail modulation (including `delta_max`). They are applied in the same transfer/RHS location as the model and are tested independently on perturbed `TOY_NET` parameters. The existing core and core-differential toy preset round trips also remain exact.

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
| `phase3a_full` | rejected | `hw_compliance` remains unsupported (see below) |
| `phase3b_tm` | rejected | compliance plus active S&H/timemux dynamics remain unsupported |

Remaining refused flags: `hw_compliance` requires all of the model's leg-wise current locations (core, boundary terminals, shared-sense, crossbar accumulator and temporal output mesh), replacing the input soft gate and soft clip while preserving the distinction between live node voltages and frozen leg currents; a single signed edge current plus global soft clip cannot represent these equations. `hw_isat_eff_clamp` needs per-family post-gate/VCA Isat clamping. `hw_vca_bits` needs quantized VCA logits. Stochastic mismatch needs the model's shared per-slot draws. `hw_power_lambda` is a training objective, not an inference behavior. Active timemux requires sample/hold injection, hold-only droop, optional kT/C draws, internal-capacitance settling, and transfer-mode handling; it remains rejected, including `phase3b_tm`. The current interpreter does not accept a `noise=` draw interface, so kT/C is not being approximated as deterministic. Validate flags using `HWConfig` before export.

## Validation

The interpreter has no model imports. The paper-checkpoint legacy path and per-stage dynamics, unit conversions, and hardware toy round trips are tested. The listed supported single groups each have an individual round-trip case with maximum output and final-state error 0. The full suite is not an analog signoff: Spectre/ahdlcmi availability and Verilog-A simulation are environment dependent. `phase3a_full` and `phase3b_tm` are intentionally not reported as supported until their remaining dynamics are represented.
