# Jev-Semantic MIDAS

[中文说明](README_ZH.md)

How do you fold central-bank text signals into a mixed-frequency inflation state-space model without look-ahead leakage or over-weighting noisy narrative scores?

This repository implements the core state-space and calibration modules from *Jev-Semantic MIDAS: Loss-Calibrated FOMC Signals for Inflation Nowcasting* (IEEE BigData 2026, accepted). The code enforces strict as-of timestamp alignment on FOMC releases, projects text signals onto a semantic basis, filters them inside an HF-MIDAS state-space update, and calibrates signal intensity directly against historical forecast loss instead of standalone sentiment accuracy.

- `src/semantic_midas/asof.py`: strict release-date alignment preventing look-ahead bias.
- `src/semantic_midas/basis.py` and `semantics.py`: semantic signal projection and basis construction.
- `src/semantic_midas/filter.py`: mixed-frequency state-space filtering step.
- `src/semantic_midas/calibration.py` and `theory.py`: loss-calibrated signal scaling and theoretical error decomposition.

## Quick start

```bash
pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

`run_demo.py` constructs synthetic quarterly inflation targets and release-aligned semantic shocks, runs the HF-MIDAS filter, and prints calibrated vs. uncalibrated nowcast errors.

## Scope

This is the public MVP with synthetic inputs. Full FOMC text extraction and 60-quarter inflation replication scripts are available upon request.
