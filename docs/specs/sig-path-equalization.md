# Spec: sig-path-equalization

Scope: feature

# Signal-Path Equalization Experiment Protocol

## Contingency
**This experiment is contingent on NARMA results.** Only proceed if the evolving core (freeze_read=False) clearly beats the frozen core (freeze_read=True) on NARMA-20 (expect NRMSE ~0.1–0.2 vs ~0.3+). If freeze_read ≈ no-freeze, the evolving core is dead and we commit fully to the control-path paper.

## Architecture
The fabric replaces the CTLE directly in the signal path:
- RX waveform (post fixed VGA) drives boundary terminals continuously
- Fabric state runs free across symbols (NO resets — same state-carryover as NARMA)
- Output node(s) drive the slicer directly

## Fabric Configuration

### Physical Sections, Not Time-Multiplexing
- 3–5 cascaded stages with independent parameters
- Each stage = one physical CTLE section
- Independent per-stage t_span, gm, Isat, leak

### Time Constant Rescaling
- Rescale so dominant tau ~ 0.5–5 UI
- Sanity-check implied gm/C against target process node
- Report honest baud rate (2–20 GBd, not 100 GBd)
- Use the rescale to adjust SOLVER config, cell library gm/isat ranges, leak, drive params

### State Carryover (same as NARMA)
- No state resets between symbols
- Fabric state carries continuously across symbols
- This memory IS the equalization — analogous to CTLE's continuous-time filtering

## Link Simulation
Reuse the differentiable link pipeline from the control-path experiment (link_sim.py).

## Metrics
- BER vs SNR curves for:
  - (a) No equalizer
  - (b) Optimized fixed CTLE (golden per-channel)
  - (c) Fabric as CTLE replacement
- Per-channel adaptability: can the fabric adapt to different channels without recalibration?

## Baselines
- No equalizer (the floor)
- Optimized fixed CTLE (the ceiling for this experiment)
- ADC+DSP alternative power estimate via Murmann FoM at the claimed baud rate

## Power Analysis
- Compute fabric power estimate from cell library parameters (gm, Isat per edge, number of edges, stages)
- Compute ADC+DSP power at claimed baud rate using Murmann FoM data
- Power comparison table

## Target Node
- Mature CMOS node (e.g., 28 nm or 16 nm)
- Claim 2–20 GBd operation (not 100 GBd)
- This is still a publishable regime for a first fabric demonstration

## Win Condition
- Beat fixed CTLE on per-channel adaptability (fabric adjusts automatically to channel variations without recalibration)
- Match or exceed CTLE margin at comparable or lower power
- Even matching CTLE margin with adaptability is a result (CTLE is a fixed filter; fabric can adapt)

## Decision Gate
- If fabric does not beat fixed CTLE on per-channel adaptability, the signal-path branch is dead even if NARMA was positive
- Report honestly

## Files to Modify/Create
- `link_sim.py`: reuse from control-path experiment
- Signal-path configuration in config.py (time constant rescaling)
- State-carryover code (shared with NARMA)
- `sig_path_experiment.py`: evaluation script