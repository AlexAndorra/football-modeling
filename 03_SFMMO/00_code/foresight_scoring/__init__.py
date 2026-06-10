"""Vendored proper-scoring / calibration / uncertainty from the foresight package.

Kept byte-for-byte in sync with github.com/AlexAndorra/foresight (MIT). Vendored
(not depended-on) so this SFMMO evaluation reproduces standalone for reviewers
without the foresight repo. Swap for ``import foresight`` once it is published.
"""

from .calibration import randomized_pit, reliability_table
from .scoring import (
    brier_score,
    log_score,
    mean_log_loss,
    predictive_entropy,
    ranked_probability_score,
)
from .uncertainty import (
    bayesian_bootstrap,
    posterior_score,
    score_difference_posterior,
)

__all__ = [
    "log_score",
    "mean_log_loss",
    "ranked_probability_score",
    "brier_score",
    "predictive_entropy",
    "bayesian_bootstrap",
    "score_difference_posterior",
    "posterior_score",
    "reliability_table",
    "randomized_pit",
]
