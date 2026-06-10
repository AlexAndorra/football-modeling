"""Tests for foresight_scoring.calibration — reliability + the discrete randomized PIT.

The randomized PIT is the calibration tool the spec singles out: on integer
support a plain (continuous) PIT clumps at the category boundaries, so we use the
randomized correction PIT = F(y-1) + U*p(y), which is Uniform(0,1) iff calibrated.
"""

import numpy as np
from scipy.stats import kstest

from foresight_scoring.calibration import randomized_pit, reliability_table


def test_reliability_table_recovers_per_bin_observed_frequency():
    # class_k=1: two forecasts at p1=0.1 (both realized class 0) and two at p1=0.9
    # (both realized class 1) -> low bin observes freq 0, high bin observes freq 1.
    probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])
    outcomes = np.array([0, 0, 1, 1])
    mean_pred, obs_freq, count = reliability_table(probs, outcomes, class_k=1, n_bins=2)
    assert count.tolist() == [2, 2]
    np.testing.assert_allclose(obs_freq, [0.0, 1.0])
    np.testing.assert_allclose(mean_pred, [0.1, 0.9])


def test_randomized_pit_lies_in_the_cell_of_the_realized_class():
    # K=2, p=[0.5,0.5]: realized class 0 -> PIT in [0, 0.5]; class 1 -> PIT in [0.5, 1].
    # The uniform draw only moves PIT *within* the realized class's probability cell.
    rng = np.random.default_rng(0)
    probs = np.tile([0.5, 0.5], (1000, 1))
    pit0 = randomized_pit(probs, np.zeros(1000, dtype=int), rng)
    pit1 = randomized_pit(probs, np.ones(1000, dtype=int), rng)
    assert np.all((pit0 >= 0.0) & (pit0 <= 0.5))
    assert np.all((pit1 >= 0.5) & (pit1 <= 1.0))


def test_randomized_pit_is_uniform_when_calibrated():
    # the defining property: if outcomes are drawn FROM the forecasts, the randomized
    # PIT is Uniform(0,1). (A non-randomized PIT would fail this on integer support.)
    rng = np.random.default_rng(42)
    n = 5000
    forecasts = rng.dirichlet(np.ones(4), size=n)
    outcomes = np.array([rng.choice(4, p=f / f.sum()) for f in forecasts])
    pit = randomized_pit(forecasts, outcomes, rng)
    assert kstest(pit, "uniform").pvalue > 0.05
