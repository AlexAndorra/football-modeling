# SFMMO — World Cup edition

A Bayesian model that forecasts international football match outcomes (home / draw / away) — World Cup finals and qualifiers — together with a small, tested toolkit to score those forecasts properly and compare them against bookmaker odds.

## What's here

- `sfmmo_fit.py` — fits the model (PyMC / nutpie) on past fixtures and writes out-of-sample match forecasts.
- `sfmmo_eval.py`, `compare_improved.py`, `compare_strict.py` — score forecasts (log-loss, RPS, Brier, accuracy) with Bayesian-bootstrap uncertainty and calibration (randomized PIT), and compare against de-vigged consensus bookmaker odds.
- `prior_predictive_check.py`, `model_criticism.py` — prior-predictive, convergence, sensitivity, and identifiability checks.
- `foresight_scoring/` — vendored, unit-tested proper-scoring primitives.
- `tests/` — unit tests for the scoring.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync && uv run pytest -q                  # scoring deps + tests
uv sync --group model                        # add PyMC / nutpie to fit

uv run python 00_code/sfmmo_fit.py           # fit -> out-of-sample forecasts
uv run python 00_code/compare_improved.py    # score the forecasts vs the bookmaker consensus
```

Reports and figures are written to `artifacts/` when you run the scripts. Data lives in `../10_data/`. The model is the SFMMO World-Cup model (`SFMMOwm__dev_EW.ipynb`); the scoring is vendored from the `foresight` proper-scoring package.
