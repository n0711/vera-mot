# VERA-MOT research manuscript (working draft v0.1)

**Title:** VERA-MOT: Reliability-Gated Appearance Association and Protected Feature Memory — A Reproducible Implementation Note for Multi-Object Tracking.

**Author:** Charalampos Nadiotis. **Dated:** 3 November 2026.

**Status:** Preliminary implementation note, not peer-reviewed, and not a report of improvements on real annotated vehicles. The paper describes deterministic quality gating and protection of BoT-SORT appearance-memory updates, with synthetic tests only. It does not establish improvements in IDF1, HOTA, AssA, identity switches, or inference latency.

The editable XeLaTeX source is [VERA_MOT_Implementation_Note.tex](VERA_MOT_Implementation_Note.tex). When the GitHub Actions paper build has completed on main, its PDF is available at [VERA_MOT_Implementation_Note.pdf](VERA_MOT_Implementation_Note.pdf). The workflow builds the PDF, verifies the file exists, and commits the binary to main. The workflow artifact also retains a copy for 30 days.

Local build:

```bash
cd docs/papers
xelatex -interaction=nonstopmode -halt-on-error VERA_MOT_Implementation_Note.tex
xelatex -interaction=nonstopmode -halt-on-error VERA_MOT_Implementation_Note.tex
```

Required XeLaTeX packages: fontspec, geometry, amsmath, mathtools, booktabs, tabularx, enumitem, fancyhdr, xcolor, hyperref and TikZ; fonts: Liberation Serif, Liberation Sans, DejaVu Sans Mono. A4 portrait, 11-point.

Original software and source are located in this repository. Related original upstream methods (BoT-SORT, Deep SORT, ByteTrack and HOTA) are credited in the manuscript. Dataset and model weight rights are separate from manuscript publishing rights.

**ResearchGate:** not submitted. The current document is a public engineering working paper, not a validated empirical study. Quantitative identity-annotated comparisons and a substantive revision should precede an academic preprint upload.
