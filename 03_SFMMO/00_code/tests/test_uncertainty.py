"""Tests for foresight_scoring.uncertainty — Bayesian uncertainty on scores (vendored).

Two orthogonal sources: a model-agnostic Bayesian bootstrap over the held-out
matches (works for any forecaster, including the bookmakers) and posterior
propagation of a Bayesian forecaster's own draws (free, for the SFM).
"""

import numpy as np
import pytest

from foresight_scoring.scoring import mean_log_loss
from foresight_scoring.uncertainty import (
    bayesian_bootstrap,
    posterior_score,
    score_difference_posterior,
)


def test_bayesian_bootstrap_centers_on_the_sample_mean():
    rng = np.random.default_rng(0)
    scores = rng.normal(loc=2.0, scale=1.0, size=2000)
    draws = bayesian_bootstrap(scores, n_draws=4000, rng=rng)
    assert draws.shape == (4000,)
    assert draws.mean() == pytest.approx(scores.mean(), abs=0.05)


def test_bayesian_bootstrap_uncertainty_shrinks_with_more_data():
    rng = np.random.default_rng(1)
    small = bayesian_bootstrap(rng.normal(size=50), n_draws=3000, rng=rng)
    large = bayesian_bootstrap(rng.normal(size=5000), n_draws=3000, rng=rng)
    assert large.std() < small.std()


def test_score_difference_posterior_recovers_gap_and_prob_a_beats_b():
    rng = np.random.default_rng(2)
    a = rng.normal(loc=1.0, scale=0.5, size=2000)
    b = rng.normal(loc=0.0, scale=0.5, size=2000)  # A is ~1 higher per match
    diff = score_difference_posterior(a, b, rng=rng, n_draws=4000)
    assert diff.mean() == pytest.approx((a - b).mean(), abs=0.05)
    assert np.mean(diff > 0) > 0.99  # P(A beats B) ~ 1


def test_posterior_score_propagates_draws_through_a_metric():
    # 3 posterior draws of probs over 2 matches -> 3 posterior values of the metric
    prob_draws = np.array(
        [
            [[0.7, 0.3], [0.4, 0.6]],
            [[0.6, 0.4], [0.5, 0.5]],
            [[0.8, 0.2], [0.3, 0.7]],
        ]
    )
    outcomes = np.array([0, 1])
    post = posterior_score(prob_draws, outcomes, mean_log_loss)
    assert post.shape == (3,)
    assert post[0] == pytest.approx(mean_log_loss(prob_draws[0], outcomes))
