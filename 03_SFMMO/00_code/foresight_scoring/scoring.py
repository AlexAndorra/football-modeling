# Vendored from the foresight package (github.com/AlexAndorra/foresight, MIT). The scoring
# functions are unchanged from upstream; the docstrings are genericized for this repo.
"""Proper scoring rules and sharpness for categorical forecasts.

`log_score` is the load-bearing scoring primitive, kept dependency-free (numpy only)
and free of any model coupling. Every function
accepts a single forecast (``probs`` shape ``(K,)``) or a batch (``probs`` shape
``(N, K)`` with ``outcome``/``outcomes`` shape ``(N,)``).
"""

import numpy as np

# Floor on the realized-outcome probability so log_score stays finite when a
# forecast assigns ~0 to what actually happened.
_PROB_FLOOR = 1e-12


def log_score(probs, outcome):
    """Log of the probability the forecast assigned to the realized outcome.

    Single forecast -> ``float``; batch -> ``(N,)`` array. Always <= 0, with 0
    (its maximum) for a confident, correct call. The realized probability is
    floored at ``1e-12`` so the result is never ``-inf``.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        p = float(probs[outcome])
        return float(np.log(max(p, _PROB_FLOOR)))
    rows = np.arange(probs.shape[0])
    p = probs[rows, np.asarray(outcome)]
    return np.log(np.maximum(p, _PROB_FLOOR))


def mean_log_loss(probs, outcomes):
    """Mean negative log score over a batch (lower is better)."""
    return float(-np.mean(log_score(np.asarray(probs, dtype=float), np.asarray(outcomes))))


def ranked_probability_score(probs, outcomes, normalize=True, reduce="mean"):
    """Ranked Probability Score for *ordered* categories (lower is better).

    Per event: ``sum_k (CDF_pred_k - CDF_obs_k)^2``, normalized by ``K-1`` when
    ``normalize`` (puts it on [0, 1]). ``reduce="mean"`` returns the batch mean;
    ``reduce="none"`` returns the per-event vector (feeds the Bayesian bootstrap).
    A single forecast is treated as a batch of one.
    """
    probs = np.asarray(probs, dtype=float)
    single = probs.ndim == 1
    p = probs[None, :] if single else probs
    o = np.atleast_1d(np.asarray(outcomes))
    n_classes = p.shape[1]
    cdf_pred = np.cumsum(p, axis=1)
    cdf_obs = (np.arange(n_classes)[None, :] >= o[:, None]).astype(float)
    per_event = np.sum((cdf_pred - cdf_obs) ** 2, axis=1)
    if normalize:
        per_event = per_event / (n_classes - 1)
    if reduce == "mean":
        return float(np.mean(per_event))
    return float(per_event[0]) if single else per_event


def brier_score(probs, outcomes, reduce="mean"):
    """Multiclass Brier score (lower is better): summed squared error against the
    one-hot outcome, in [0, 2]. ``reduce`` as in :func:`ranked_probability_score`.
    """
    probs = np.asarray(probs, dtype=float)
    single = probs.ndim == 1
    p = probs[None, :] if single else probs
    o = np.atleast_1d(np.asarray(outcomes))
    n_classes = p.shape[1]
    onehot = (np.arange(n_classes)[None, :] == o[:, None]).astype(float)
    per_event = np.sum((p - onehot) ** 2, axis=1)
    if reduce == "mean":
        return float(np.mean(per_event))
    return float(per_event[0]) if single else per_event


def predictive_entropy(probs):
    """Shannon entropy (nats) of each forecast — the sharpness axis.

    Single forecast -> ``float``; batch -> ``(N,)`` array. 0 for a point mass,
    ``log K`` for the uniform forecast.
    """
    probs = np.asarray(probs, dtype=float)
    # Replace zeros with 1.0 before the log so log(0) is never evaluated
    # (0 * log(1) = 0 contributes nothing) — avoids a transient nan warning.
    safe = np.where(probs > 0, probs, 1.0)
    ent = -np.sum(probs * np.log(safe), axis=-1)
    return float(ent) if probs.ndim == 1 else ent
