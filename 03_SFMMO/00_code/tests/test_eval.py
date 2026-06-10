"""Tests for the load-bearing evaluation logic.

These pin the correctness an adversarial review verified by hand: the bookmaker de-vig is a
coherent simplex with the right home/draw/away ordering, renormalization fixes the k_max
truncation, the proper-score direction is what the head-to-head relies on, and folds split
into qualifiers vs World-Cup finals correctly (the cut that separates the two populations).
"""

import numpy as np
import pandas as pd
import pytest

from foresight_scoring import log_score
from sfmmo_eval import book_probs, competition, point_metrics, renorm


# --- bookmaker de-vig ---------------------------------------------------------


def test_book_probs_devig_is_a_coherent_simplex():
    odds = pd.DataFrame(
        {"id_match": ["m1", "m2"], "AvgH": [1.5, 3.0], "AvgD": [4.0, 3.4], "AvgA": [6.0, 2.3]}
    )
    probs, overround = book_probs(odds, "Avg")
    np.testing.assert_allclose(probs[["away", "draw", "home"]].sum(axis=1).to_numpy(), 1.0)
    assert (overround > 1.0).all()  # a real bookmaker margin (probabilities over-sum before de-vig)


def test_book_probs_orders_home_draw_away_from_HDA_odds():
    # home is the favourite (lowest decimal odds) -> highest 'home' probability
    odds = pd.DataFrame({"id_match": ["m1"], "AvgH": [1.4], "AvgD": [4.5], "AvgA": [7.0]})
    row = book_probs(odds, "Avg")[0].iloc[0]
    assert row["home"] > row["draw"] > row["away"]


# --- renormalization (the k_max truncation fix) -------------------------------


def test_renorm_makes_truncated_rows_sum_to_one():
    P = np.array([[0.1, 0.05, 0.05], [0.3, 0.3, 0.3]])  # rows sum < 1 (truncated)
    np.testing.assert_allclose(renorm(P).sum(axis=1), 1.0)


# --- metrics + proper-score direction ----------------------------------------


def test_point_metrics_match_hand_computed():
    P = np.array([[0.1, 0.2, 0.7]])  # one match, realized outcome = home (2)
    m = point_metrics(P, np.array([2]))
    assert m["ACC"] == 1.0
    assert m["logLik"] == pytest.approx(-np.log(0.7))


def test_a_sharper_correct_forecaster_scores_a_higher_log_score():
    # direction the head-to-head depends on: higher log score == better forecaster
    y = np.array([2, 0, 1])
    good = np.array([[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    bad = np.array([[0.4, 0.4, 0.2], [0.2, 0.4, 0.4], [0.4, 0.2, 0.4]])
    assert log_score(good, y).mean() > log_score(bad, y).mean()


# --- competition split (qualifiers vs World-Cup finals) -----------------------


@pytest.mark.parametrize(
    "fold, expected",
    [
        ("WMQ2018", "qualifiers"),
        ("WMQ2026", "qualifiers"),
        ("WM2018", "finals"),
        ("WM2022", "finals"),
    ],
)
def test_competition_classifies_qualifiers_vs_finals(fold, expected):
    assert competition(fold) == expected
