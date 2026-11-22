# Citation audit: DATE KirchhoffNet manuscript

Scope: externally verifiable, high-level claims in
`Paper/DATE_conference_template_LaTeX/template-A4.tex`, checked against original
research papers and publisher/proceedings pages.  This is a recommendation
list, not a change to the manuscript or bibliography.

## Highest-value, low-cost changes

### 1. Anchor the small-world terminology

**Where.** Line 379 (and, optionally, the contribution at line 204):
`Watts--Strogatz small-world graph ... while keeping high clustering with short paths.`

**Suggested change.** Add `~\cite{watts98smallworld}` immediately after the
first use of “Watts--Strogatz small-world graph.”  The paper's own edge-count
calculation should remain uncited; it follows directly from the selected
degree (k).

**Why this is a good fit.** Watts and Strogatz introduce the construction and
identify its characteristic combination of high clustering and short path
length.  It supports exactly the conceptual claim, without asserting that the
topology is automatically easy to lay out in a particular CMOS process.

**Proposed bibliography entry.**

```bibtex
@article{watts98smallworld,
  author  = {D. J. Watts and S. H. Strogatz},
  title   = {Collective dynamics of 'small-world' networks},
  journal = {Nature},
  volume  = {393},
  number  = {6684},
  pages   = {440--442},
  year    = {1998},
  doi     = {10.1038/30918},
  url     = {https://doi.org/10.1038/30918}
}
```

Primary source: [Watts and Strogatz, *Nature* (1998)](https://doi.org/10.1038/30918).

### 2. Strengthen the SerDes ADC/DSP motivation using an already-present reference

**Where.** Lines 151--154: `At 56--112 Gb/s PAM-4, ADC-plus-DSP receivers are
common, but the converters, clocking, and datapath impose substantial area and
power costs ...`

**Suggested change.** Change the existing citation to
`\cite{murmann10,upadhyaya20,krupnik21}`.  `krupnik21` is already in
`refs.bib` but is not cited in the source.

**Why this is a good fit.** Upadhyaya *et al.* report a 56-Gb/s ADC-based
wireline receiver datapath and explicitly attribute excessive power to the
heavy DSP arithmetic.  Krupnik *et al.* is a direct 112-Gb/s PAM-4 ADC-based
SerDes example, so together they cover both stated operating-rate regimes.
Do **not** claim that the ADC alone dominates a complete SerDes unless the
paper adds a measured breakdown for the chosen architecture.

Primary sources: [Upadhyaya *et al.*, ISSCC (2019), DOI 10.1109/ISSCC.2019.8662505](https://doi.org/10.1109/ISSCC.2019.8662505),
[Krupnik *et al.*, JSSC (2021), DOI 10.1109/JSSC.2020.3024261](https://doi.org/10.1109/JSSC.2020.3024261).

### 3. Cite a CTLE receiver precedent at the start of the CTLE case study (optional)

**Where.** First sentence of Sec. `CTLE Inverse Design` (line 737), or the
caption of Fig. `ctle-schematic` (line 573).

**Suggested wording.** “CTLE remains a practical front-end equalization
element in high-rate PAM-4 receivers~\cite{liu24ctle112}.”  This is deliberately
narrow: it supports use of CTLE in the demonstration, not the novelty or
performance of the proposed CTLE.

**Proposed bibliography entry.**

```bibtex
@inproceedings{liu24ctle112,
  author    = {Y. Liu and others},
  title     = {A 112-Gb/s PAM-4 Receiver with Ultra-Fine Gain-Adjustment {CTLE}
               and Novel Sample-and-Reset Slicer in 28-nm {CMOS}},
  booktitle = {IEEE Asian Solid-State Circuits Conference (A-SSCC)},
  year      = {2024},
  doi       = {10.1109/A-SSCC59632.2024.10719519},
  url       = {https://doi.org/10.1109/A-SSCC59632.2024.10719519}
}
```

Before adding, export the IEEE citation from the publisher page to verify the
author list and proceedings title; the landing page exposes the title and DOI
but not reliably all metadata without JavaScript.

Primary source: [IEEE Xplore record](https://ieeexplore.ieee.org/document/10719519/).

### 4. Support the "low-dimensional analog control" characterization (optional)

**Where.** Lines 156--159: `Programmable-CTLE tuning is naturally a
low-dimensional analog control problem ...`

**Suggested change.** Rephrase to the less interpretive, externally supported
claim “Programmable CTLEs expose a small set of gain and bias controls,” then
cite `\cite{choi14adaptivectle}`.  The cited fabricated receiver adaptively
adjusts CTLE gain and power; it therefore supports the premise that CTLE
adaptation is a circuit-control task, but not the precise dimensionality of
this paper's seven-target parameterization.

```bibtex
@inproceedings{choi14adaptivectle,
  author    = {K.-Y. Choi and others},
  title     = {A 5--20 Gb/s Power Scalable Adaptive Continuous-Time Linear
               Equalizer Using Edge Counting},
  booktitle = {IEEE Asian Solid-State Circuits Conference (A-SSCC)},
  year      = {2014},
  doi       = {10.1109/ASSCC.2014.7008913},
  url       = {https://doi.org/10.1109/ASSCC.2014.7008913}
}
```

Verify the full author list from the IEEE export before inserting the record.
Primary source: [IEEE DOI record](https://doi.org/10.1109/ASSCC.2014.7008913).

## Useful support, but only if the surrounding sentence is broadened

### 5. Converter overhead in mixed-signal analog accelerators

**Where.** Lines 161--164, only if the sentence is recast from a SerDes-only
statement into a general analog-accelerator observation.

**Safe insertion.** After “on-chip data converters add a further tax,” add
`This interface overhead is also a documented limitation of mixed-signal
analog in-memory accelerators~\cite{rasch24aimc}.`

The source says that many analog in-memory designs convert accumulated analog
currents/charges back to digital with highly parallel ADCs.  It is relevant
context, but it does **not** measure this paper's SerDes control loop; avoid
using it to quantify the present design.

```bibtex
@article{rasch24aimc,
  author  = {M. J. Rasch and F. Carta and O. Fagbohungbe and others},
  title   = {Fast and robust analog in-memory deep neural network training},
  journal = {Nature Communications},
  volume  = {15},
  pages   = {7133},
  year    = {2024},
  doi     = {10.1038/s41467-024-51221-z},
  url     = {https://doi.org/10.1038/s41467-024-51221-z}
}
```

Primary source: [Rasch *et al.*, *Nature Communications* (2024)](https://doi.org/10.1038/s41467-024-51221-z).

### 6. General analog-memory precedent (optional; do not use for SerDes claims)

If the introduction adds one compact general sentence about analog neural
hardware, Ambrogio *et al.* provide an original experimental precedent for
parallel analog MAC at the weight location.  It supports analog-memory
acceleration, not a converter-less wireline receiver, so it should not be
attached to the SerDes-specific motivation.

```bibtex
@article{ambrogio18analogmemory,
  author  = {S. Ambrogio and P. Narayanan and H. Tsai and others},
  title   = {Equivalent-accuracy accelerated neural-network training using
             analogue memory},
  journal = {Nature},
  volume  = {558},
  number  = {7708},
  pages   = {60--67},
  year    = {2018},
  doi     = {10.1038/s41586-018-0180-5},
  url     = {https://doi.org/10.1038/s41586-018-0180-5}
}
```

Primary source: [Ambrogio *et al.*, *Nature* (2018)](https://doi.org/10.1038/s41586-018-0180-5).

## Existing citations that are already well matched

- Lines 169--170: `dillavou22decentral` directly supports decentralized,
  physics-driven learning in resistor networks.  No additional citation is
  needed.
- Lines 172--174: `kirchhoffnet_end2end`, `kendall20eqprop`, and
  `kendall21gradest` are appropriate precedents for the carefully limited
  “end-to-end analog ODE acceleration” statement.
- Lines 183--185: `kirchhoffnet` and `neuralode` directly support the KCL/ODE
  description and the Neural-ODE relationship.  The original KirchhoffNet
  preprint explicitly describes voltage-node evolution governed by ODEs and
  learnable edge parameters.

  Primary sources: [Gao *et al.*, KirchhoffNet](https://arxiv.org/abs/2310.15872),
  [Chen *et al.*, NeurIPS (2018)](https://proceedings.neurips.cc/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html).

## Claims to soften rather than cite

- Fig. 1's caption says the shaded ADC/DAC blocks “dominate” area and power.
  The cited receiver papers establish meaningful converter/DSP cost, but do
  not establish dominance for every depicted chain.  Safer caption: “highlight
  the conversion interfaces and their associated area/power cost.”
- “Routability comes from” a Watts--Strogatz graph is an engineering inference,
  not a result of the Watts--Strogatz paper.  Keep the concrete sparse edge
  count, but phrase the motivation as “targets a lower interconnect count” or
  qualify it as a design objective unless post-layout routing evidence is
  available.
- The conclusion's “effective ADC-less SerDes controllers” is supported by the
  manuscript's own Cadence experiment, not an external reference.  Retain the
  result citation-free and avoid extending it to a general taped-out hardware
  claim.

## Recommendation

For a page-limited paper, add only items 1 and 2.  They require just two
additional citation tokens in the text (one new bibliography entry), directly
shore up the main topology and converter-tax motivations, and do not add
background prose.  Item 3 is useful only if reviewers are likely to challenge
CTLE as a current 112-Gb/s receiver technique.
