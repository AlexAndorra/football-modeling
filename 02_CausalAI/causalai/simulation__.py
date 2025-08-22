"""
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

__all__ = [
    "simulate_matches",
    "get_treatment_effects",
]


def simulate_matches(
    n_matches: int = 10_000,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Simulate *n_matches* independent match situations.

    Each row corresponds to a single team's final‑30‑minutes scenario.

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
    # 1.A Quartiles of Baseline Skill Differentials between offense & defense
    # ---------------------------------------------------------------------
    strengthDiff_p = np.quantile(off_old-def_old, q=(0.25,0.5,0.75))
    print(strengthDiff_p)
    strength_quartile = np.where(off_old-def_old <= strengthDiff_p[0],"Q1",
                                 np.where((off_old-def_old > strengthDiff_p[0]) & (off_old-def_old <= strengthDiff_p[1]),"Q2",
                                          np.where((off_old-def_old > strengthDiff_p[1]) & (off_old-def_old <= strengthDiff_p[2]),"Q3",
                                                   "Q4")))

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
    # Initial chemistry disruption, then boost
    # ---------------------------------------------------------------------
    off_new = off_old.copy()
    def_new = def_old.copy()

    # Attacking subs when losing/tied (score_diff ≤ 0)
    att_idx = (T == 1) & (score_diff <= 0)
    if att_idx.sum() > 0:
        # Initial disruption: temporary decrease in coordination
        disruption = np.abs(rng.normal(2.0, 1.0, att_idx.sum()))
        off_new[att_idx] -= disruption * 0.5
        def_new[att_idx] -= disruption * 0.5

        # Eventual boost: larger than disruption
        off_boost = np.abs(rng.normal(7.0, 2.0, att_idx.sum()))
        def_change = rng.normal(-2.0, 1.5, att_idx.sum())  # Can be positive or negative
        off_new[att_idx] += off_boost
        def_new[att_idx] += def_change

    # Defensive subs when winning (score_diff > 0)
    def_idx = (T == 1) & (score_diff > 0)
    if def_idx.sum() > 0:
        # Initial disruption
        disruption = np.abs(rng.normal(2.0, 1.0, def_idx.sum()))
        off_new[def_idx] -= disruption * 0.5
        def_new[def_idx] -= disruption * 0.5

        # Eventual boost
        def_boost = np.abs(rng.normal(7.0, 2.0, def_idx.sum()))
        off_change = rng.normal(-2.0, 1.5, def_idx.sum())  # Can be positive or negative
        def_new[def_idx] += def_boost
        off_new[def_idx] += off_change

    # Ensure strengths stay reasonable
    off_new = np.clip(off_new, 20, 90)
    def_new = np.clip(def_new, 20, 90)

    # ---------------------------------------------------------------------
    # 5. Direct substitution effects (not mediated by strength changes)
    # ---------------------------------------------------------------------
    # Fresh legs boost for specific actions
    fresh_legs_boost = np.where(T == 1, rng.normal(1.2, 0.3, n_matches), 1.0)
    fresh_legs_boost = np.clip(fresh_legs_boost, 0.8, 1.8)

    # Tactical mismatch factor
    tactical_factor = np.where(T == 1, rng.normal(1.1, 0.2, n_matches), 1.0)
    tactical_factor = np.clip(tactical_factor, 0.9, 1.4)

    # ---------------------------------------------------------------------
    # 6. Outcome generation (final‑30‑min match events)
    # ---------------------------------------------------------------------
    # Shots
    lam_shots = (0.10 * off_new + 2.0 * np.maximum(0, -score_diff)) * fresh_legs_boost
    shots = rng.poisson(lam_shots)

    # Goals (binomial on shots with 15% conversion, tactical boost)
    goal_prob = 0.15 * tactical_factor
    goal_prob = np.clip(goal_prob, 0.05, 0.30)
    goals = rng.binomial(shots, goal_prob)

    # Passes (will be calculated using potential outcomes framework below)
    base_passes = 0.35 * (off_new + def_new) + 5.0 * score_diff

    # ---------------------------------------------------------------------
    # Calculate TRUE treatment effects (ground truth for model validation)
    # ---------------------------------------------------------------------
    # compute both potential outcomes for everyone
    # Y(0): outcome without treatment (fresh_legs_boost = 1.0)
    lam_pass_y0 = base_passes * 1.0  # No fresh legs boost
    lam_pass_y0 = np.clip(lam_pass_y0, 0, None)
    y0_passes = rng.poisson(lam_pass_y0)

    # Y(1): Outcome with treatment (fresh_legs_boost from treatment)
    # Need to simulate fresh legs boost for everyone (not just treated)
    fresh_legs_boost_if_treated = rng.normal(1.2, 0.3, n_matches)
    fresh_legs_boost_if_treated = np.clip(fresh_legs_boost_if_treated, 0.8, 1.8)
    lam_pass_y1 = base_passes * fresh_legs_boost_if_treated
    lam_pass_y1 = np.clip(lam_pass_y1, 0, None)
    y1_passes = rng.poisson(lam_pass_y1)


    # Observed outcomes = Y(T) for each unit
    passes = np.where(T == 1, y1_passes, y0_passes)  # Observed outcome based on actual treatment

    # Counterfactual outcomes (the unobserved potential outcome)
    passes_counterfactual = np.where(
        T == 1,
        y0_passes,  # Treated units: counterfactual is Y(0) (without treatment)
        y1_passes,  # Control units: counterfactual is Y(1) (with treatment)
    )

    # Individual Treatment Effects (ITE) = Y_i(T=1) - Y_i(T=0) for everyone --- expected:
    ite_passes_expected = base_passes * (fresh_legs_boost_if_treated - 1.0)

    # Individual Treatment Effects (ITE) = Y_i(T=1) - Y_i(T=0) --- observed:
    ite_passes_observed = y1_passes - y0_passes  # True ITE using both potential outcomes

    # Population Average Treatment Effect (ATE) - average over entire population
    ate_passes_expected = np.mean(ite_passes_expected)



    return pd.DataFrame(
        {
            "off_old": off_old,
            "def_old": def_old,
            'strength_diff': off_old - def_old,
            "strength_quartile": strength_quartile,
            "score_diff": score_diff,
            "sub": T,
            "off_new": off_new,
            "def_new": def_new,
            "fresh_legs_boost": fresh_legs_boost,
            "tactical_factor": tactical_factor,
            "shots": shots,
            "goals": goals,
            "passes": passes,
            # Ground truth for causal inference validation
            "passes_counterfactual": passes_counterfactual,
            "ite_passes_expected": ite_passes_expected,
            "ite_passes_observed": ite_passes_observed,
            "ate_passes_expected": np.full(n_matches, ate_passes_expected),
        }
    )


def get_treatment_effects(df: pd.DataFrame) -> dict:
    """Extract ground truth treatment effects from simulation data.

    Parameters
    ----------
    df : pd.DataFrame
        Output from simulate_matches() containing ground truth treatment effects.

    Returns
    -------
    dict
        Dictionary containing various treatment effect measures:
        - 'ate_expected': Population Average Treatment Effect (expected value)
        - 'ate_observed': Population ATE from observed outcomes
        - 'ite_expected': Individual Treatment Effects (expected values)
        - 'ite_observed': Individual Treatment Effects (observed outcomes)
        - 'effect_heterogeneity': Standard deviation of ITEs
        - 'effect_by_score': Conditional ATEs by score differential
    """
    treated_mask = df["sub"] == 1

    # Population Average Treatment Effects
    ate_expected = df.loc[treated_mask, "ite_passes_expected"].mean()
    ate_observed = df.loc[treated_mask, "ite_passes_observed"].mean()

    # Individual Treatment Effects
    ite_expected = df["ite_passes_expected"].values
    ite_observed = df["ite_passes_observed"].values

    # Effect heterogeneity
    effect_heterogeneity = df.loc[treated_mask, "ite_passes_expected"].std()

    # Conditional treatment effects by score differential
    effect_by_score = {}
    for score in sorted(df["score_diff"].unique()):
        mask = (df["score_diff"] == score) & treated_mask
        if mask.sum() > 0:
            effect_by_score[score] = df.loc[mask, "ite_passes_expected"].mean()

    return {
        "ate_expected": ate_expected,
        "ate_observed": ate_observed,
        "ite_expected": ite_expected,
        "ite_observed": ite_observed,
        "effect_heterogeneity": effect_heterogeneity,
        "effect_by_score": effect_by_score,
        "n_treated": treated_mask.sum(),
        "n_control": (~treated_mask).sum(),
    }
