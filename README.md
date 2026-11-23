# Jev-Semantic MIDAS

Residual-loss calibration and gating for mapping directional FOMC text signals onto a fixed numerical inflation-nowcasting backbone (IEEE BigData 2026 LLM-Finance Workshop, accepted, S42203).

Using an LLM to output numeric CPI forecasts directly is noisy, while uncalibrated text adjustments can hurt a numerical MIDAS baseline. Here language supplies direction and disjoint historical residuals determine scale.

- `src/semantic_midas/calibration.py`: non-negative residual scale fitting (`fit_scale`), Clopper-Pearson directional lower bounds (`cp_lower`), bootstrap gating (`calibrate`), and multi-arm comparison (`forecast_arms`).
- `src/semantic_midas/theory.py`: closed-form expected gain and robust quadratic entry checks.
- `src/semantic_midas/asof.py`, `basis.py`, `filter.py`, `semantics.py`: vintage lag alignment and MIDAS basis utilities.

## Quick start

```bash
pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

`run_demo.py` simulates multi-episode residuals with noisy directional signals, fits frozen calibration scales on disjoint fit/gate splits, and compares synthetic test MSE across `ZERO`, `CALIBRATED`, `GATE`, `P_SHRINK`, `SHUFFLED`, `OPPOSITE`, and `ORACLE`.

## Scope

Recovered calibration and gating core from S42203; the original event-time constructor and hosted Jev extractor outputs are not included.
