---
plan name: control-path-equalization
plan description: SerDes control-path equalizer demo
plan status: active
---

## Idea
Build a differentiable wireline link simulation in PyTorch where a conventional analog front-end (VGA, CTLE, optional 1-tap DFE) stays in the data path, and the frozen-core KirchhoffNet replaces the adaptation engine that sets the control voltages. The fabric runs at adaptation bandwidth (kHz-MHz), not baud rate, so the frozen core is sufficient. Analog feature extractors (total received power, HF power content, edge rate, slicer error magnitude) drive boundary terminals. Fabric outputs control voltages g1..g4 (VGA gain, CTLE peaking, CTLE DC gain, DFE tap). Train across an ensemble of channels (50-200 with randomized IL characteristics: 10-35 dB loss at Nyquist, plus temperature corners). Loss is differentiable BER surrogate or negative eye margin. Baselines: fixed nominal settings (floor), golden per-channel exhaustive calibration (ceiling), digital SAR ADC + FSM (incumbent), fabric (frozen core). Win condition: match golden-calibration margin at adaptation-engine power orders below the ADC+FSM+DAC stack, with no-ADC-at-all story.

## Implementation
- Design the differentiable link simulation in PyTorch: TX (NRZ/PAM4, sps=32, Gaussian bandlimit 0.5-0.75x baud, optional jitter), channel (frequency-domain loss model with IL(f)=a*sqrt(f)+b*f dB, apply via FFT), RX front-end with parametric VGA (gain g1 in [0.5,2]), CTLE (two-zero/two-pole with peaking g2 and DC-gain g3 knobs), optional 1-tap DFE (subtract g4*d_{n-1}), and slicer.
- Build the channel ensemble: sample 50-200 channels with randomized a,b parameters giving 10-35 dB loss at Nyquist, plus temperature corners scaling the loss. This ensemble is the whole point — adaptation earns nothing on one fixed channel.
- Design analog feature extractors (4-8 features, all ADC-free, continuous-time): (1) total received power — rectify + RC lowpass, (2) high-frequency power content — signal minus its own RC lowpass, then rectify + lowpass (the classic analog channel-loss sensor), (3) edge rate / transition density from CDR bang-bang phase detector, (4) slicer error magnitude — analog integrator accumulating |d - d_hat| over a window.
- Build the frozen-core fabric network: boundary-fan-out from feature extractors to hidden nodes, freeze_read=True (only boundary and temporal-readout OTA edges learn), outputs g1..g4 control voltages (rail-limited by the fabric's output compliance).
- Implement the training loop: loss = differentiable BER surrogate (softplus of signed distance to threshold), or negative eye margin. Backprop through the entire link into the fabric. Train across channel ensemble (random channel per batch element). Structurally identical to existing Friedman pipeline.
- Implement baselines: (a) fixed nominal settings (no adaptation) — the floor, (b) golden per-channel exhaustive calibration (grid search over knobs offline) — the ceiling, (c) digital 8-bit 1 MS/s SAR ADC + LUT/FSM + DACs — the incumbent.
- Implement PVT robustness evaluation: inject 1-5% mismatch into the fabric (via SimContext), recalibrate readout only (frozen-core calibration story), show margin retention across corners.
- Metrics: eye margin / BER across the channel ensemble, power overhead of adaptation engine itself (the USP story), convergence/robustness across corners.
- Plot eye diagrams: show (a) no equalizer, (b) fixed nominal settings, (c) golden calibration, (d) fabric adaptation, (e) digital adaptation — illustrating the fabric matches (c)/(d) within small delta.
- Analyze and compare power: fabric vs ADC+FSM+DAC stack. The win condition: match golden-calibration margin at adaptation-engine power orders below the incumbent, with no-ADC-at-all story. Even a tie on margin with cleaner power/PVT story is a paper.

## Required Specs
<!-- SPECS_START -->
- ctrl-path-sim
<!-- SPECS_END -->