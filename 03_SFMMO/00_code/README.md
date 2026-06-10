# SFMMO World-Cup — honest, properly-scored evaluation

A reproducible re-evaluation of the SFMMO World-Cup model against the bookmakers, plus the
prior-predictive check the model spec was missing. Built to be auditable end-to-end: every
number regenerates from the committed prediction pickles + odds with a fixed seed, and the
proper-scoring code is unit-tested.

Scoring / calibration / uncertainty are **vendored from the [foresight](https://github.com/AlexAndorra/foresight)
package** (byte-for-byte, see `foresight_scoring/`) so this reproduces standalone without
that repo. Swap for `import foresight` once it is published.

## What it found

1. **On proper scores, the consensus bookmakers beat the model on the qualifiers.** Read
   correctly (lower is better for log-loss / RPS / Brier), the de-vigged consensus line
   ("Avg") beats every model variant A–K on the two large folds (WMQ2018, WMQ2026); the
   model only "wins" the 64-game WM2018 finals, which is noise. (The emailed tables used
   the best-price "Max" line, which is not a coherent probability — its implied odds sum
   below 1 on ~39% of matches — and even against it the books win the big folds.)

2. **Two legitimate fixes roughly halve the gap — and the model becomes perfectly calibrated.**
   - *Outcome renormalization.* The W/D/L probabilities are a Poisson goal-PMF collapse
     truncated at `k_max=5`; on extreme-λ (blowout / cold-start) fixtures they lose up to
     ~80% of their mass (~13% of qualifier matches sum below 0.9), which inflates the
     model's log-loss exactly where it trails. Renormalizing recovers most of it.
   - *Leave-one-fold-out temperature calibration.* The notebook computed this but left it
     off (`T_star = None`).
   - Net: pooled (n=1709) the log-loss gap to the consensus falls **0.113 → 0.055** and the
     normalized-RPS gap **~0.022 → 0.015** (books still ahead, 95% CIs exclude zero), and the
     model's randomized-PIT is **flat (KS p = 0.97** vs the book's 0.42) — i.e. essentially
     perfectly calibrated.

3. **The prior-predictive check (the spec's one model ask) shows the priors are still too
   wild.** Tightening `beta` and adding the intercept was not enough: the team attack/defence
   *scale* priors (`sigma ~ Gamma(2,4)` × `ZeroSumNormal(1)`) still imply absurd goals
   (q99.9 ≈ 113, max ≈ 9000). A fixed-scale `ZeroSumNormal(σ=0.30)` team prior is sane
   (q99.9 ≈ 16) — and, since the wide priors are what produce the extreme-λ blowups behind
   the `k_max` truncation, this is the *root-cause* fix for finding (2) as well.

4. **The A–K ladder is a dead end.** All eleven variants are within ~0.001–0.01 on every
   fold: the momentum / elo-encoding choices barely move anything. The gap that matters is
   model-vs-market.

See `artifacts/honest_report.md`, `artifacts/honest_calibration.png`, `artifacts/prior_predictive.png`.

## Layout

```
foresight_scoring/        # vendored, tested proper scoring / uncertainty / calibration
honest_eval.py            # load pickles + odds -> proper scores + bootstrap P(model beats book) + PIT + report
prior_predictive_check.py # rebuilds the priors; shows shipped=wild, proposed=sane
tests/                    # unit tests for the vendored scoring (ported from foresight)
artifacts/                # report.md + figures (regenerated)
```

## Run

```bash
uv sync                              # light eval deps
uv run pytest -q                     # 22 tests
uv run python honest_eval.py         # -> artifacts/honest_report.md + honest_calibration.png

uv sync --group model                # adds pymc/numpyro for the prior check
uv run python prior_predictive_check.py
```

## Recommended fixes for the re-fit (`fit_sfmmo.py`, TODO)

The above are post-hoc on the committed predictions. A proper re-fit should: (a) adopt the
tightened team-effect prior; (b) raise `k_max` (≈15) so the goal-PMF is not truncated; (c)
set a seed and use ≥4 chains with committed R-hat/ESS/divergences; (d) carry the per-fold
temperature calibration. Expectation: this narrows the qualifier gap further and removes the
truncation artifact at the source — though the consensus market, which prices live team news
the model does not ingest, is likely to stay ahead until that information is added.
