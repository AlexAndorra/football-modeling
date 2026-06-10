# Vendored verbatim from the foresight package (github.com/AlexAndorra/foresight, MIT).
# Do not edit here — keep in sync with the upstream module. See foresight_scoring/__init__.py.
"""Calibration diagnostics for ordered categorical forecasts.

Two views: per-class **reliability** (predicted vs observed frequency, binned) and
the discrete **randomized PIT**. On integer support a plain PIT clumps at the
category boundaries; the randomized correction ``F(y-1) + U*p(y)`` is Uniform(0,1)
if and only if the forecaster is calibrated, so its histogram should be flat.
"""

import numpy as np


def reliability_table(probs, outcomes, class_k, n_bins=10):
    """Per-class reliability for class ``class_k``.

    Bins the predicted probability of ``class_k`` into ``n_bins`` equal-width bins
    over [0, 1] and, per bin, returns the mean predicted probability, the observed
    frequency of ``class_k``, and the count. Returns three length-``n_bins`` arrays
    ``(mean_pred, obs_freq, count)``; empty bins carry ``nan`` mean/freq and 0 count.
    A perfectly-calibrated forecaster lies on the diagonal (obs_freq == mean_pred).
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes)
    p_k = probs[:, class_k]
    hit = (outcomes == class_k).astype(float)
    # interior edges only -> digitize returns bin index in [0, n_bins-1]
    interior = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    idx = np.clip(np.digitize(p_k, interior), 0, n_bins - 1)

    mean_pred = np.full(n_bins, np.nan)
    obs_freq = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        in_bin = idx == b
        c = int(in_bin.sum())
        count[b] = c
        if c:
            mean_pred[b] = p_k[in_bin].mean()
            obs_freq[b] = hit[in_bin].mean()
    return mean_pred, obs_freq, count


def randomized_pit(probs, outcomes, rng):
    """Randomized Probability Integral Transform for a discrete/ordered target.

    ``PIT_i = F(y_i - 1) + U_i * p(y_i)`` with ``U_i ~ Uniform(0, 1)``. Under
    calibration the PIT is Uniform(0, 1). ``probs`` is ``(N, K)``, ``outcomes`` is
    ``(N,)``; returns ``(N,)`` PIT values in [0, 1]. ``rng`` is a ``np.random.Generator``.
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes)
    n = probs.shape[0]
    rows = np.arange(n)
    cdf = np.cumsum(probs, axis=1)
    p_y = probs[rows, outcomes]
    cdf_below = cdf[rows, outcomes] - p_y  # F(y-1): mass strictly below realized class
    u = rng.uniform(size=n)
    return cdf_below + u * p_y
