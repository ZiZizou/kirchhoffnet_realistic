---
plan name: signal-path-equalization
plan description: Signal-path CTLE replacement fabric
plan status: active
---

## Idea
If NARMA shows the evolving core is computationally valuable, replace the CTLE directly in the signal path. The fabric runs free across symbols (state carryover), forming a continuous-time adaptive equalizer. 3-5 cascaded physical stages with independent parameters, each with tau ~0.5-5 UI. The honest baud-rate claim is 2-20 GBd (not 100 GBd). The CTLE is the baseline that matters. Metric is BER vs SNR curves against: (a) no equalizer, (b) optimized fixed CTLE, (c) fabric. Power sketch using Murmann FoM for ADC+DSP alternative at claimed baud rate.

## Implementation
- Rescale fabric time constants so dominant tau ~ 0.5-5 UI: sanity-check implied gm/C against target process node, report honest baud rate claim (2-20 GBd, not 100 GBd). Core code change: rescale t_span, num_steps, leak, and gm/isat/drive parameters.
- Build the signal-path link simulation (reuse control-path link_sim.py): RX waveform (post fixed VGA) drives boundary terminals continuously; fabric state runs free across symbols (NO resets — same state-carryover code as NARMA).
- Configure fabric as cascaded physical sections: 3-5 stages with independent parameters (mirrors real multi-stage CTLEs). Not time-multiplexing — each stage is a physical section.
- Implement BER vs SNR evaluation: sweep SNR, measure BER for (a) no equalizer, (b) optimized fixed CTLE, (c) fabric as CTLE replacement. Generate BER vs SNR curves.
- Compute power sketch: use Murmann FoM numbers for the ADC+DSP alternative at the claimed baud rate. Compare against fabric power estimate from cell library gm/Isat parameters.
- Write results report: BER vs SNR curves, power comparison table, baud-rate claim with justification. The CTLE is the baseline that matters — beating it (margin at fixed power, or per-channel adaptability it lacks) is the result.
- Decision gate: if fabric does not beat fixed CTLE on per-channel adaptability, the signal-path branch is dead even if NARMA was positive. Report honestly.

## Required Specs
<!-- SPECS_START -->
- sig-path-equalization
<!-- SPECS_END -->