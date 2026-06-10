"""prior_predictive_check.py — the prior predictive check the review asked for.

The technical review's one model-spec ask was a prior predictive check: before seeing
data, what goal distribution do the priors imply? The updated notebook tightened the
*factor* coefficient (``beta ~ N(0, 0.1)``) and added an explicit intercept
(``mu ~ N(log 1.4, 0.3)``) — but never *demonstrated* the result.

This script does, by rebuilding the exact prior structure on a synthetic standardized
design (factors are ~N(0,1) after the notebook's cross-sectional standardization, so the
priors' implications don't need the real data pipeline) and sampling the prior predictive
of per-team goals.

FINDING: the *shipped* priors are still not sane — the team attack/defence SCALE priors
(``sigma_alpha, sigma_delta ~ Gamma(2,4)`` times ``ZeroSumNormal(1)``) let ``exp(eta)``
explode (q99.9 ~ 100+ goals, max in the thousands). Tightening *beta* did not touch this.
A tightened team-effect prior fixes it — and, because those same wide priors produce the
extreme-lambda blowouts that overflow the k_max=5 goal-PMF truncation, this is also the
root-cause fix for the probability-mass loss that `honest_eval.py` flags on the qualifiers.

Needs the heavy Bayesian stack:  uv sync --group model
Run:                              uv run python prior_predictive_check.py
"""

import os

import matplotlib
import numpy as np
import pymc as pm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "artifacts")
SEED = sum(map(ord, "sfmmo"))

# DevK's global factors (home_pitch enters separately as X_home): points_diff,
# goalsscored_cum_{team,opp}, goalsconceded_cum_{team,opp}, elo_team, elo_opp.
N_FACTORS = 7


def build_model(beta_sigma, team_kind, team_scale, n_obs=4000, n_teams=48, n_leagues=6):
    """Rebuild the SFMMO prior structure on a synthetic standardized design.

    ``team_kind="hier"``: the shipped non-centered prior, ``sigma ~ Gamma(2, team_scale)``
    times ``ZeroSumNormal(1)``. ``team_kind="fixed"``: a tightened prior, team effects
    drawn directly from ``ZeroSumNormal(sigma=team_scale)`` (no heavy-tailed scale).
    """
    rng = np.random.default_rng(SEED)
    X_gf = rng.normal(size=(n_obs, N_FACTORS))          # standardized factors ~ N(0,1)
    X_home = rng.integers(0, 2, size=n_obs)
    it = rng.integers(0, n_teams, size=n_obs)
    io = (it + rng.integers(1, n_teams, size=n_obs)) % n_teams   # opponent != team
    il = rng.integers(0, n_leagues, size=n_obs)
    coords = {
        "factor_g": [f"f{i}" for i in range(N_FACTORS)],
        "teams": [f"t{i}" for i in range(n_teams)],
        "leagues": [f"l{i}" for i in range(n_leagues)],
        "obs_id": np.arange(n_obs),
    }
    with pm.Model(coords=coords) as m:
        mu = pm.Normal("mu", mu=np.log(1.4), sigma=0.3)

        if team_kind == "hier":   # shipped: Gamma scale x ZeroSumNormal(1)
            sigma_alpha = pm.Gamma("sigma_alpha", alpha=2, beta=team_scale)
            alpha = pm.Deterministic(
                "alpha", pm.ZeroSumNormal("alpha_raw", sigma=1, dims="teams") * sigma_alpha,
                dims="teams")
            sigma_delta = pm.Gamma("sigma_delta", alpha=2, beta=team_scale)
            delta = pm.Deterministic(
                "delta", pm.ZeroSumNormal("delta_raw", sigma=1, dims="teams") * sigma_delta,
                dims="teams")
        else:                     # tightened proposal: fixed-scale ZeroSumNormal
            alpha = pm.ZeroSumNormal("alpha", sigma=team_scale, dims="teams")
            delta = pm.ZeroSumNormal("delta", sigma=team_scale, dims="teams")

        mu_gamma = pm.Normal("mu_gamma", mu=0.30, sigma=0.20)
        sigma_gamma = pm.Gamma("sigma_gamma", alpha=2, beta=20)
        gamma_raw = pm.Normal("gamma_raw", 0, 1, dims="teams")
        beta_home = pm.Deterministic("beta_home", mu_gamma + gamma_raw * sigma_gamma, dims="teams")

        kappa = pm.ZeroSumNormal("kappa", sigma=0.3, dims="leagues")
        beta = pm.Normal("beta", mu=0, sigma=beta_sigma, dims="factor_g")

        eta = (
            mu + alpha[it] - delta[io] + kappa[il]
            + X_home * beta_home[it] + pm.math.dot(X_gf, beta)
        )
        pm.Poisson("match_outcome", mu=pm.math.exp(eta),
                   observed=np.zeros(n_obs, dtype=int), dims="obs_id")
    return m


def prior_goals(beta_sigma, team_kind, team_scale, draws=2000):
    with build_model(beta_sigma, team_kind, team_scale):
        idata = pm.sample_prior_predictive(draws=draws, random_seed=SEED)
    return idata.prior_predictive["match_outcome"].values.ravel()


def summarize(tag, goals):
    q = np.quantile(goals, [0.5, 0.9, 0.99, 0.999])
    frac = {k: float(np.mean(goals > k)) for k in (6, 10, 15, 30)}
    print(f"\n[{tag}]\n   mean={goals.mean():.2f}  median={q[0]:.0f}  q90={q[1]:.0f}  "
          f"q99={q[2]:.0f}  q99.9={q[3]:.0f}  max={goals.max()}")
    print("   P(>6)={:.4f}  P(>10)={:.5f}  P(>15)={:.6f}  P(>30)={:.7f}".format(
        frac[6], frac[10], frac[15], frac[30]))
    return frac


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Prior predictive check — per-team goals implied by the SFMMO priors")
    print("=" * 70)
    # Shipped priors (what the committed notebook samples from):
    g_ship = prior_goals(0.10, "hier", 4)
    f_ship = summarize("SHIPPED: beta~N(0,0.1), team sigma~Gamma(2,4) x ZSN(1)", g_ship)
    # Tightened proposal — fixed-scale team effects:
    g_fix = prior_goals(0.10, "fixed", 0.30)
    f_fix = summarize("PROPOSED: beta~N(0,0.1), team effects ~ ZSN(sigma=0.30)", g_fix)

    # --- figure ---
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    bins = np.arange(0, 16) - 0.5
    ax.hist(np.clip(g_ship, 0, 15), bins=bins, density=True, alpha=0.55,
            label=f"shipped (max {g_ship.max()})", color="C3")
    ax.hist(np.clip(g_fix, 0, 15), bins=bins, density=True, histtype="step", lw=2.2,
            label=f"proposed (max {g_fix.max()})", color="C0")
    ax.set_xlabel("goals scored by a team (prior predictive; clipped at 15 for display)")
    ax.set_ylabel("density")
    ax.set_title("SFMMO prior predictive — goals per team (shipped vs tightened)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "prior_predictive.png"), dpi=120)
    plt.close(fig)

    print(f"\nSHIPPED priors imply implausible goals: P(>15)={f_ship[15]:.4f}, "
          f"P(>30)={f_ship[30]:.5f}, max={g_ship.max()} -> the team-effect scale priors "
          "are still too wild (tightening `beta` did not fix this).")
    # The PROPOSED prior must be sane — this doubles as a regression test for the fix.
    assert f_fix[15] < 0.01, f"proposed P(goals>15) = {f_fix[15]:.4f} — still too wild"
    assert f_fix[30] < 1e-4, f"proposed P(goals>30) = {f_fix[30]:.6f} — still too wild"
    print(f"PROPOSED prior is sane: P(>15)={f_fix[15]:.4f}, P(>30)={f_fix[30]:.6f}, "
          f"max={g_fix.max()}. Recommend adopting it in the re-fit (`fit_sfmmo.py`).")
    print(f"wrote {OUT}/prior_predictive.png")


if __name__ == "__main__":
    main()
