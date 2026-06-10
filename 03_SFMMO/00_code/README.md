# SFMMO World-Cup — reproducible re-fit + honest, properly-scored evaluation (PR #1)

A reproducible re-fit of the SFMMO World-Cup model and an honest, proper-scoring evaluation
against the bookmakers. Every number regenerates from version control with a fixed seed; the
scoring code is unit-tested; the model fit, the prior-predictive check, the calibration and
the model criticism all follow the Bayesian workflow (nutpie, `compute_log_prior`,
`arviz_stats.diagnose`, power-scaling sensitivity).

Scoring / calibration / uncertainty are **vendored from the
[foresight](https://github.com/AlexAndorra/foresight) package** (byte-for-byte, see
`foresight_scoring/`) so this reproduces standalone. Swap for `import foresight` once published.

> **Scope.** This is **PR #1**: reproducibility + calibration + the `k_max` truncation fix +
> a tightened *team* prior + the prior-predictive check + the honest eval. The hardcoded
> per-confederation Elo weighting is **deliberately left for PR #2** (a principled hierarchical
> confederation prior); an LLM news/injury adjuster is **PR #3**.

## Headline — the consensus market beats the model on the qualifiers (robustly)

The re-fit (`sfmmo_fit.py`, nutpie, all folds R-hat ≤ 1.003, 0 divergences, 4 chains) is scored
against the de-vigged consensus odds ("Avg") with a paired Bayesian bootstrap, under **both**
forecasting semantics (see *Forecasting semantics* below). Lower log-loss / RPS is better;
**model − book**, so a negative ΔlogLik / positive ΔRPS means the book is better:

| semantics | n | ΔlogLik (model − book) | ΔRPS | P(model better) | calibrated (PIT) |
| --- | --: | --: | --: | --: | --: |
| **Sequential** (info to kickoff — what the closing odds use) | 1709 | **−0.059** [−0.076, −0.041] | +0.016 | 0.00 | p = 0.44 ✓ |
| **Strict-holdout** (pre-tournament, Elo frozen at train cut) | 1709 | **−0.110** [−0.131, −0.090] | +0.031 | 0.00 | p = 0.34 ✓ |

The books win decisively in both (95% CIs exclude zero), and the gap roughly **doubles** under
strict-holdout. The model's only edge anywhere is the 64-match WM2018 finals fold, whose CI
includes zero (noise). The model is calibrated under both semantics (randomized-PIT KS
p > 0.05, seed-robust).

**The re-fit is not a predictive improvement over a post-hoc patch.** Paired on the same
fixtures, the re-fit (tightened priors + `k_max=15`) scores ≈ Max's original predictions with
post-hoc renormalization + calibration (ΔlogLik **−0.0034** [−0.0053, −0.0015] — a hair
*worse*). The prior tightening buys **principledness** (a sane prior-predictive), not accuracy;
the `k_max` fix and renormalization are equivalent. The market's edge is **information** (live
team news the model does not ingest), not a modelling detail — which is why no re-parameterization
closes it, and why PR #3 (an LLM adjuster) is the path that could.

## What PR #1 changed (vs Max's committed DevK)

`sfmmo_fit.py` is a **verbatim** extraction of Max's Colab notebook (auditable diff) plus:

1. **`k_max` 5 → 15** in the goal-PMF collapse. The W/D/L probabilities are a truncated
   Poisson collapse; on extreme-λ (blowout / cold-start) fixtures the worst cases lost most of
   their mass (minimum row-sum ≈ **0.20**; ~13% of qualifier matches summed below 0.9; median
   0.98), which inflated the model's log-loss exactly where it trails. At `k_max=15` the W/D/L
   sum to ≈ 1 (min ≈ 0.96).
2. **Tightened team prior**: `sigma ~ Gamma(2,4) × ZeroSumNormal(1)` → fixed
   `ZeroSumNormal(σ=0.30)`. The shipped priors implied absurd goals (prior-predictive q99.9 ≈
   113, max ≈ 9000); σ=0.30 is sane (q99.9 ≈ 16) and is in fact near the **loosest** defensible
   value (σ=0.5 → q99.9 ≈ 27, σ=1.0 → 170), not a tight cherry-pick. See `prior_predictive_check.py`.
3. **Reproducibility**: nutpie sampler, no hardcoded chains (let nutpie pick the multi-chain
   default), descriptive seed, `compute_log_likelihood` + `compute_log_prior`, committed per-fold
   R-hat / divergence diagnostics.
4. **Post-hoc calibration**: leave-one-fold-out temperature scaling (the notebook computed it
   but left it off, `T_star = None`).

The A–K momentum/elo-encoding ladder is a dead end (all within ~0.001–0.01 on every fold).

## Model criticism (`model_criticism.py`)

- **Convergence** clean on every fold (R-hat ≤ 1.003, 0 divergences, E-BFMI satisfactory).
- **Identifiability** clean: max posterior correlation among the intercept `mu`, league effects
  `kappa` and mean home advantage is ≈ −0.45 (well separated, not collinear).
- **Prior sensitivity** is moderate and load-bearing (power-scaling: `beta`/`alpha`/`delta`
  ≈ 0.30–0.45). This is expected for deliberately regularizing priors — but it means individual
  **team strengths are partly prior-shrunk** under ZSN(0.30); the OOS verdict rests partly on
  that shrinkage. A prior-scale sweep is a sensible PR #2 refinement.

## Forecasting semantics (disclosed)

The per-league Elo updates on prior match results. **Sequential** (the primary) lets a prior
held-out qualifier's result inform a later qualifier's pre-game Elo — the same information set
the bookmaker *closing* odds are formed on, so it is the fair "vs the closing line" comparison.
**Strict-holdout** (`SFMMO_STRICT_HOLDOUT=1`) freezes Elo at the train cut (a pre-tournament
forecast). The books win under both; we report both. (Cumulative-goals features come
pre-computed from the data CSV and are identical across variants.)

An independent adversarial review reproduced the headline ΔlogLik/ΔRPS, verified the de-vig,
the W/D/L ordering, the matched set, the paired bootstrap and the faithfulness of the port, and
could not break the directional verdict.

## Layout

```
sfmmo_fit.py              # the re-fit (verbatim DevK + the 4 changes); SFMMO_STRICT_HOLDOUT=1 for the robustness run
sfmmo_eval.py             # post-hoc eval of the ORIGINAL predictions: proper scores + bootstrap + PIT
compare_improved.py       # re-fit vs books AND vs original (paired)
compare_strict.py         # sequential vs strict-holdout, both vs books (robustness)
prior_predictive_check.py # shipped priors are wild; ZeroSumNormal(0.30) is sane
model_criticism.py        # prior sensitivity (psense) + identifiability on the per-fold idata
foresight_scoring/        # vendored, unit-tested proper scoring / uncertainty / calibration
tests/                    # 22 tests for the vendored scoring
artifacts/                # reports + figures (regenerated)
```

## Run

```bash
uv sync && uv run pytest -q            # light eval deps + 22 tests
uv run python compare_improved.py      # re-fit vs books vs original (needs the __improved.pkl)

uv sync --group model                  # pymc + nutpie + arviz-stats for fitting/criticism
uv run python 00_code/sfmmo_fit.py                         # sequential re-fit -> __improved.pkl
SFMMO_STRICT_HOLDOUT=1 uv run python 00_code/sfmmo_fit.py  # strict-holdout    -> __improved_strict.pkl
uv run python 00_code/compare_strict.py                   # dual verdict
uv run python 00_code/prior_predictive_check.py
uv run python 00_code/model_criticism.py
```

## Roadmap

- **PR #2** — replace the hardcoded per-confederation Elo multipliers (0.8/0.8/0.75/0.6) with a
  principled hierarchical confederation-strength prior; sweep the team-prior scale.
- **PR #3** — an LLM news/injury adjuster (+ RL on the `log_score`) to ingest the live
  information the market prices and the model currently lacks.

## Provenance

The model is Max Göbel's DevK SFMMO World-Cup model (`SFMMOwm__dev_EW.ipynb`), extracted
verbatim. Scoring/calibration/uncertainty are vendored from
[AlexAndorra/foresight](https://github.com/AlexAndorra/foresight). MIT.
