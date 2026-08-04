# Vendored from the foresight package (github.com/AlexAndorra/foresight, MIT). The functions
# are unchanged from upstream; the docstrings are genericized for this repo.
"""Bayesian uncertainty for forecast scores.

Two orthogonal sources, both Bayesian:

- **Bayesian bootstrap** (Rubin) over the held-out matches — model-agnostic, so it
  works for every forecaster (including those with no posterior). Dirichlet
  weights over the per-match scores give a posterior over the mean, and over pairwise
  differences -> P(A beats B).
- **Posterior propagation** — push a Bayesian forecaster's own posterior draws
  (e.g. the SFM's ``goals_scored_probs``) through a metric for a free posterior over
  its score. The two capture different things; an honest SFM bar folds in both.
"""

import numpy as np


def bayesian_bootstrap(values, n_draws=4000, rng=None):
    """Posterior over the mean of ``values`` via the Bayesian bootstrap.

    Draws ``n_draws`` Dirichlet(1, ..., 1) weight vectors over the N items and
    returns the weighted means -> ``(n_draws,)`` posterior samples, centered on the
    sample mean with spread that shrinks as N grows.
    """
    rng = np.random.default_rng() if rng is None else rng
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    weights = rng.dirichlet(np.ones(n), size=n_draws)  # (n_draws, n)
    return weights @ values


def score_difference_posterior(scores_a, scores_b, rng=None, n_draws=4000):
    """Posterior over the mean *paired* score difference ``A - B``.

    Forecasters are scored on the same held-out matches, so differences are paired
    per match. ``mean(draws > 0)`` is P(A's mean score exceeds B's) — read it with the
    metric's direction in mind (higher-is-better for log score; lower for RPS/Brier).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    return bayesian_bootstrap(a - b, n_draws=n_draws, rng=rng)


def posterior_score(prob_draws, outcomes, metric_fn):
    """Propagate a Bayesian forecaster's posterior through a metric.

    ``prob_draws`` is ``(S, N, K)`` (S posterior draws of the N forecasts); for each
    draw the metric is computed over the N matches, giving ``(S,)`` posterior values
    of the score. ``metric_fn`` is any batch metric, e.g. :func:`scoring.mean_log_loss`.
    """
    prob_draws = np.asarray(prob_draws, dtype=float)
    return np.array([metric_fn(prob_draws[s], outcomes) for s in range(prob_draws.shape[0])])
