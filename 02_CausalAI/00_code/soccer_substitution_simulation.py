"""soccer_substitution_simulation.py

Generate a synthetic dataset of soccer match scenarios to study the causal
impact of player substitutions on team performance in the final 30 minutes
of a match.

The simulation follows the causal structure described in the project design:
    - Team offensive/defensive strength
    - Current score differential at 60'
    - Substitution decision (treatment)
    - Post‑substitution strengths
    - Match‑event outcomes (shots, goals, passes, tackles, etc.)

Author: Alexandre Andorra (2025‑07‑09)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple

__all__ = [
    "simulate_matches",
]

# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_matches(
    n_matches: int = 10_000,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Simulate *n_matches* independent match situations.

    Each row corresponds to a single team’s final‑30‑minutes scenario.

    Parameters
    ----------
    n_matches : int, default 10_000
        Number of simulated match instances.
    seed : int | None, optional
        Seed for the NumPy random generator. ``None`` uses global RNG.

    Returns
    -------
    pd.DataFrame
        A tidy data frame with covariates, treatment indicator, and outcomes.
    """
    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------------
    # 1. Baseline team skill ratings (offense & defense)
    # ---------------------------------------------------------------------
    off_old = rng.normal(loc=55.0, scale=10.0, size=n_matches)
    # Correlate defensive strength with offense (ρ≈0.5) for realism
    def_old = 0.5 * off_old + 0.5 * rng.normal(loc=55.0, scale=10.0, size=n_matches)

    # ---------------------------------------------------------------------
    # 2. Score differential at 60' (confounder)
    #    Positive => leading; negative => trailing
    # ---------------------------------------------------------------------
    score_for = off_old * 0.02 + rng.normal(0.0, 0.5, n_matches)
    score_against = (100 - def_old) * 0.02 + rng.normal(0.0, 0.5, n_matches)
    score_diff = np.round(score_for - score_against).astype(int)
    score_diff = np.clip(score_diff, -3, 3)

    # ---------------------------------------------------------------------
    # 3. Substitution decision (treatment indicator T)
    #    Logistic model with higher sub probability when trailing
    # ---------------------------------------------------------------------
    logit = -1.8 * score_diff + 0.4
    p_sub = 1.0 / (1.0 + np.exp(-logit))
    T = (rng.random(n_matches) < p_sub).astype(int)

    # ---------------------------------------------------------------------
    # 4. Post‑substitution strengths (Off_new, Def_new)
    # ---------------------------------------------------------------------
    off_new = off_old.copy()
    def_new = def_old.copy()

    # Attacking subs when losing/tied (score_diff ≤ 0)
    att_idx = (T == 1) & (score_diff <= 0)
    off_new[att_idx] += np.abs(rng.normal(5.0, 2.0, att_idx.sum()))
    def_new[att_idx] -= np.abs(rng.normal(5.0, 2.0, att_idx.sum()))

    # Defensive subs when winning (score_diff > 0)
    def_idx = (T == 1) & (score_diff > 0)
    off_new[def_idx] -= np.abs(rng.normal(5.0, 2.0, def_idx.sum()))
    def_new[def_idx] += np.abs(rng.normal(5.0, 2.0, def_idx.sum()))

    # ---------------------------------------------------------------------
    # 5. Outcome generation (final‑30‑min match events)
    # ---------------------------------------------------------------------
    # Shots
    lam_shots = 0.10 * off_new + 2.0 * np.maximum(0, -score_diff)
    shots = rng.poisson(lam_shots)

    # Goals (binomial on shots with 15% conversion)
    goals = rng.binomial(shots, 0.15)

    # Passes
    lam_pass = 0.50 * (off_new + def_new) + 5.0 * score_diff
    lam_pass = np.clip(lam_pass, 0, None)
    passes = rng.poisson(lam_pass)

    # Tackles (defensive actions scale with def strength & protecting lead)
    lam_tackles = 0.10 * def_new + np.maximum(score_diff, 0)
    tackles = rng.poisson(lam_tackles)

    # Clearances
    lam_clear = 0.08 * def_new + 1.5 * np.maximum(score_diff, 0)
    clearances = rng.poisson(lam_clear)

    # Blocks
    lam_blocks = 0.05 * def_new + np.maximum(score_diff, 0)
    blocks = rng.poisson(lam_blocks)

    # Pressures (higher when trailing)
    lam_press = 0.12 * def_new + 1.2 * np.maximum(0, -score_diff)
    pressures = rng.poisson(lam_press)

    # Dribbles/Carries
    lam_drib = 0.06 * off_new + np.maximum(0, -score_diff)
    dribbles = rng.poisson(lam_drib)

    # Saves by goalkeeper (facing more shots when ahead)
    lam_save = 0.07 * def_new + 1.2 * np.maximum(score_diff, 0)
    saves = rng.poisson(lam_save)

    # Fouls (scale with pressures)
    lam_fouls = 0.04 * pressures + 0.5
    fouls = rng.poisson(lam_fouls)

    # Cards (10% of fouls)
    cards = rng.binomial(fouls, 0.10)

    # ---------------------------------------------------------------------
    # Assemble tidy DataFrame
    # ---------------------------------------------------------------------
    data = pd.DataFrame({
        "off_old": off_old,
        "def_old": def_old,
        "score_diff": score_diff,
        "sub": T,
        "off_new": off_new,
        "def_new": def_new,
        "shots": shots,
        "goals": goals,
        "passes": passes,
        "tackles": tackles,
        "clearances": clearances,
        "blocks": blocks,
        "pressures": pressures,
        "dribbles": dribbles,
        "saves": saves,
        "fouls": fouls,
        "cards": cards,
    })

    return data


# ---------------------------------------------------------------------------
# CLI for quick testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = simulate_matches()
    print(df.head())
