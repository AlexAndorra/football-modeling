"""Tests for foresight_scoring.scoring — proper scoring rules (vendored from foresight).

The log score is the load-bearing scoring primitive, so it gets pinned hard.
"""

import numpy as np
import pytest

from foresight_scoring.scoring import (
    brier_score,
    log_score,
    mean_log_loss,
    predictive_entropy,
    ranked_probability_score,
)


@pytest.mark.parametrize(
    "probs, outcome, expected",
    [
        # a confident, correct prediction scores 0 — the maximum of the log score
        ([1.0, 0.0, 0.0], 0, 0.0),
        # in general it is the log of the probability assigned to the realized outcome
        ([0.7, 0.2, 0.1], 0, np.log(0.7)),
        # a uniform forecast over K=4 scores log(1/4) — the max-entropy reference
        ([0.25, 0.25, 0.25, 0.25], 2, np.log(0.25)),
    ],
)
def test_log_score_is_log_prob_of_realized_outcome(probs, outcome, expected):
    assert log_score(np.asarray(probs), outcome) == pytest.approx(expected)


def test_log_score_vectorizes_over_a_batch():
    # scoring the whole held-out set at once: (N, K) probs + (N,) outcomes -> (N,) scores
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    outcomes = np.array([0, 2])
    np.testing.assert_allclose(log_score(probs, outcomes), [np.log(0.7), np.log(0.3)])


def test_log_score_floors_zero_probability_to_stay_finite():
    # a forecast that put 0 on the realized outcome must score very negative but
    # FINITE -- a -inf score would be unusable downstream.
    score = log_score(np.array([1.0, 0.0]), 1)
    assert np.isfinite(score)
    assert score < -20.0


# --- mean_log_loss -----------------------------------------------------------


def test_mean_log_loss_is_negative_mean_log_score():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    outcomes = np.array([0, 2])
    assert mean_log_loss(probs, outcomes) == pytest.approx(-np.mean([np.log(0.7), np.log(0.3)]))


# --- ranked_probability_score (ordered) --------------------------------------


def test_rps_matches_hand_computed_value():
    # K=3, outcome=1: pred CDF [0.2,0.7,1.0] vs obs CDF [0,1,1];
    # sum sq diff = 0.04+0.09+0 = 0.13; normalized by (K-1)=2 -> 0.065
    assert ranked_probability_score(np.array([0.2, 0.5, 0.3]), 1) == pytest.approx(0.065)


def test_rps_is_zero_for_confident_correct_forecast():
    assert ranked_probability_score(np.array([0.0, 0.0, 1.0, 0.0]), 2) == pytest.approx(0.0)


def test_rps_rewards_ordinally_closer_forecasts():
    # both put 0.6 on a wrong class; "near" is adjacent to the truth, "far" is 3 away
    truth = 0
    near = np.array([0.4, 0.6, 0.0, 0.0])
    far = np.array([0.4, 0.0, 0.0, 0.6])
    assert ranked_probability_score(near, truth) < ranked_probability_score(far, truth)


def test_rps_vectorizes_to_mean_over_batch():
    probs = np.array([[0.2, 0.5, 0.3], [0.0, 0.0, 1.0]])
    outcomes = np.array([1, 2])
    assert ranked_probability_score(probs, outcomes) == pytest.approx(np.mean([0.065, 0.0]))


# --- brier_score (multiclass) ------------------------------------------------


def test_brier_score_matches_hand_computed_value():
    # outcome=0, one-hot [1,0,0]; probs [0.7,0.2,0.1]
    # (0.7-1)^2 + 0.2^2 + 0.1^2 = 0.09+0.04+0.01 = 0.14
    assert brier_score(np.array([0.7, 0.2, 0.1]), 0) == pytest.approx(0.14)


def test_brier_score_is_zero_for_confident_correct():
    assert brier_score(np.array([0.0, 1.0]), 1) == pytest.approx(0.0)


# --- predictive_entropy (sharpness) ------------------------------------------


def test_predictive_entropy_uniform_is_log_k():
    assert predictive_entropy(np.array([0.25, 0.25, 0.25, 0.25])) == pytest.approx(np.log(4))


def test_predictive_entropy_confident_is_zero():
    assert predictive_entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)


def test_predictive_entropy_vectorizes_over_batch():
    probs = np.array([[0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(predictive_entropy(probs), [np.log(4), 0.0])
