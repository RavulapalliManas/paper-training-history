# Training History and the Fate of a Learned Capability: Supervision Supply, and the Structure of Early Data

Code, committed data, and paper source for the paper. Manas Venkata Sai Ravulapalli, 2026.

Every macro-backed number in the paper is a LaTeX macro generated from a committed results
file. No headline number in the prose is typed by hand. This repository contains the full
chain: the retention, revival, and futures cells (`results/devaxis_analyses.json`, written
by `analysis/devaxis_analyses.py`), the pre-registered gap-law experiment
(`results/x14_gap_law.json`, `analysis/x14_gap_law.py`), the raw-record priming
re-derivation (`results/h100_reanalysis.json`, `analysis/h100_reanalysis.py`), the
pre-registered dose-250 replication (`results/d250_verdict.json`,
`analysis/d250_verdict.py`), the token-denominated conversion (`results/token_costs.json`,
derived by `analysis/token_costs.py` from the raw run records), the number pipeline
(`analysis/make_numbers.py` writes `numbers.tex`), the figure generator
(`analysis/make_p456_figs.py`, house style in `figs/pubstyle.py`), and the paper source
(`training_history.tex`).

## Reproduce the paper from the committed data

Requirements: Python 3.10+, `numpy`, `scipy`, `matplotlib`, and a LaTeX distribution with
`pdflatex`.

```bash
# 1. Regenerate every numeric macro from the committed results
python analysis/make_numbers.py          # rewrites numbers.tex byte-identically

# 2. Regenerate the figures (the generator audits itself for text/data collisions)
python analysis/make_p456_figs.py

# 3. Compile the paper
pdflatex training_history.tex && pdflatex training_history.tex
```

The three steps are independent: the committed `numbers.tex` and `figs_p456/*.pdf` already
match the committed results, so step 3 alone rebuilds the paper as released.

The raw run records behind the committed result files (the Trainium and H100 training runs
reported in the paper) live in the two training codebases and are available on request; the
analysis scripts above state exactly how each committed file was derived from them.
Campaign-document numbers that are not macro-backed carry their provenance rows in the
paper's appendix.
