"""
A2/A3/A4 — Local, faithful re-run of SFMMO (devVersion 'A') for review diagnostics.

Ported from 03_SFMMO/00_code/SFMMOwm__dev_EW.ipynb (Cells 7, 9, 12). Changes vs the
Colab original, all documented:
  - sampler nutpie (CPU) instead of numpyro/GPU              (bayesian-workflow default)
  - ELO write vectorized (identical values, no O(n^2) .loc)  (speed only)
  - adds a PRIOR PREDICTIVE check (absent upstream)          (workflow requirement)
  - descriptive seed instead of 42                            (workflow requirement)
  - draw-calibration computed on a thinned posterior          (avoids ~GB joint-PMF array)
Produces, into this folder: inference_data_<lik>.nc, prior_predictive.png, diagnose_<lik>.txt,
rootogram_<lik>.png, draw_calibration.png, identifiability.png, results_summary.json.

Usage:  python run_sfmmo.py            # Poisson + NegBin, default 2-cycle window
"""
import json
import warnings
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import pymc as pm
import arviz as az
from scipy.stats import poisson, nbinom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = "03_SFMMO/20_review/analysis"
CSV = "03_SFMMO/10_data/data_byPlayer__SFM_II__TM__WM.csv"
SEED = sum(map(ord, "sfmmo-review-devA"))
rng = np.random.default_rng(SEED)

# Training window for diagnostics. After "drop first season per league" this keeps the
# most recent cycle(s). Model properties checked here (convergence, draw calibration,
# attack/defense coupling, Poisson-vs-NegBin) are window-invariant.
TRAIN_SEASONS = ["WMQ2014", "WM2014", "WMQ2018", "WM2018", "WMQ2022", "WM2022"]

FACTORS_CS = [  # devVersion 'A' (full)
    "points_diff", "goalsscored_cum_team", "goalsscored_cum_opp",
    "goalsconceded_cum_team", "goalsconceded_cum_opp",
    "teamMOM__S", "oppMOM__S", "teamMOM__M", "oppMOM__M", "teamMOM__L", "oppMOM__L",
    "teamMOM__S_L", "oppMOM__S_L",
    "FD_teamMOM__S", "FD_oppMOM__S", "FD_teamMOM__M", "FD_oppMOM__M",
    "FD_teamMOM__L", "FD_oppMOM__L",
    "elo_team", "elo_opp", "elo_team_opp",
]
OTHER_FACTORS = ["home_pitch"]
FACTORS = OTHER_FACTORS + FACTORS_CS
FACTORS_G = [f for f in FACTORS if f != "home_pitch"]
IDVAR = ["id_match", "name_team", "name_opp", "name_league", "id_league", "season", "gameday", "kick_off"]
YVAR = "match_outcome"


def update_elo(r_home, r_away, result, K=20, home_adv=50):
    exp_home = 1 / (1 + 10 ** ((r_away - r_home - home_adv) / 400))
    s_home = {2: 1.0, 1: 0.5, 0: 0.0}[result]
    return (r_home + K * (s_home - exp_home),
            r_away + K * ((1 - s_home) - (1 - exp_home)))


def prepare_features():
    """Faithful transcription of Grand-Loop feature engineering (full data, pre-split)."""
    data_raw = pd.read_csv(CSV)
    data_raw["kick_off"] = pd.to_datetime(data_raw["kick_off"])
    data_raw = data_raw.sort_values(["name_player", "season", "kick_off"])
    keep = ["points_team", "points_opp", "goalsscored_inGame_team", "goalsscored_inGame_opp",
            "goalsscored_cum_team", "goalsscored_cum_opp", "goalsconceded_cum_team", "goalsconceded_cum_opp",
            "home_pitch", "goalsscored_rank_team", "goalsconceded_rank_opp",
            "id_match", "name_team", "name_opp", "name_league", "id_league", "season", "gameday", "kick_off",
            "goalsscored_rank_opp", "goalsconceded_rank_team", "goalsscored_diff",
            "goal_balance_team", "goal_balance_opp", "goal_balance_diff", "points_diff",
            "tm_marketvalue_team_squad", "tm_marketvalue_opp_squad"]
    cd = (data_raw.drop_duplicates(subset=["id_match", "home_pitch"])[keep]
          .sort_values(["name_league", "kick_off"]).reset_index(drop=True))
    cd = cd.loc[cd["id_match"].duplicated(keep=False), :]          # both home & away observed
    cd["match_outcome"] = cd["goalsscored_inGame_team"].copy()
    cd["gameday_orig"] = cd["gameday"].copy()
    cd["gameday"] = [int(float(i.split("_")[1][2:])) for i in cd["id_match"].values]

    # --- ELO (per league/season, sequential) — values identical to upstream, write vectorized
    cd["match_outcome__home"] = 1
    cd.loc[cd["goalsscored_inGame_team"] > cd["goalsscored_inGame_opp"], "match_outcome__home"] = 2
    cd.loc[cd["goalsscored_inGame_team"] < cd["goalsscored_inGame_opp"], "match_outcome__home"] = 0
    elo_team = {}
    elo_opp = {}
    for ll in cd["name_league"].unique():
        ELO = {t: 1500 for t in cd.loc[cd["name_league"] == ll, "name_team"].unique()}
        seasons = cd.loc[cd["name_league"] == ll, "season"].unique().tolist()
        for si, ss in enumerate(seasons):
            sl = cd[(cd["name_league"] == ll) & (cd["season"] == ss)].sort_values("kick_off")
            if si > 0:
                prev = cd.loc[(cd["name_league"] == ll) & (cd["season"] == seasons[si - 1]), "name_team"].unique().tolist()
                for t in ELO:
                    ELO[t] = ELO[t] * 0.75 + 1500 * 0.25 if t in prev else 1300
            for gg in range(sl.shape[0]):
                h, a = sl["name_team"].iloc[gg], sl["name_opp"].iloc[gg]
                gi = sl.index[gg]
                elo_team[gi], elo_opp[gi] = ELO[h], ELO[a]
                ELO[h], ELO[a] = update_elo(ELO[h], ELO[a], sl["match_outcome__home"].iloc[gg])
    cd["elo_team"] = pd.Series(elo_team)
    cd["elo_opp"] = pd.Series(elo_opp)
    cd["elo_diff"] = cd["elo_team"] - cd["elo_opp"]
    cd["elo_team_opp"] = cd["elo_team"] * cd["elo_opp"]
    cd["goal_appeal"] = cd["goalsconceded_rank_opp"] - cd["goalsscored_rank_team"]

    # --- Team momentum (EWMA of FD points, halflives 1/4/8) + FD-of-momentum + interactions
    cd["home_team"] = cd["name_team"]
    cd = cd.set_index(["kick_off", "season", "id_match", "home_team"])
    mom_cols = ["teamMOM__S", "teamMOM__M", "teamMOM__L", "oppMOM__S", "oppMOM__M", "oppMOM__L", "team__FD_points",
                "FD_teamMOM__S", "FD_teamMOM__M", "FD_teamMOM__L", "FD_oppMOM__S", "FD_oppMOM__M", "FD_oppMOM__L"]
    cd[mom_cols] = np.nan
    for tt in cd["name_team"].unique():
        td = cd.loc[(cd["name_team"] == tt) | (cd["name_opp"] == tt), :].copy()
        tp = pd.DataFrame(np.nan, index=td.index, columns=["home", "points"])
        tp["points"] = np.where(td["name_team"] == tt, td["points_team"], td["points_opp"])
        tp["home"] = np.where(td["name_team"] == tt, 1, 0)
        tp["FD_points"] = tp.groupby("season")["points"].diff()
        for hl, suf in [(1, "S"), (4, "M"), (8, "L")]:
            tp[f"MOM__{suf}"] = tp.groupby(level="season")["FD_points"].ewm(halflife=hl).mean().droplevel(level=0)
            tp[f"FD_MOM__{suf}"] = tp.groupby(level="season")[f"MOM__{suf}"].diff()
        ih = cd[cd["name_team"] == tt].index.intersection(tp.index)
        cd.loc[ih, ["teamMOM__S", "teamMOM__M", "teamMOM__L", "FD_teamMOM__S", "FD_teamMOM__M", "FD_teamMOM__L", "team__FD_points"]] = \
            tp.loc[tp["home"] == 1, ["MOM__S", "MOM__M", "MOM__L", "FD_MOM__S", "FD_MOM__M", "FD_MOM__L", "FD_points"]].values
        io = cd[cd["name_opp"] == tt].index.intersection(tp.index)
        cd.loc[io, ["oppMOM__S", "oppMOM__M", "oppMOM__L", "FD_oppMOM__S", "FD_oppMOM__M", "FD_oppMOM__L"]] = \
            tp.loc[tp["home"] == 0, ["MOM__S", "MOM__M", "MOM__L", "FD_MOM__S", "FD_MOM__M", "FD_MOM__L"]].values
    cd["teamMOM__S_L"] = cd["teamMOM__S"] * cd["teamMOM__L"]
    cd["oppMOM__S_L"] = cd["oppMOM__S"] * cd["oppMOM__L"]
    cd = cd.reset_index().drop("home_team", axis=1)

    # --- Market values
    cd["tm_marketvalue_ratio"] = cd["tm_marketvalue_team_squad"] / cd["tm_marketvalue_opp_squad"]
    for c in ["tm_marketvalue_team_squad", "tm_marketvalue_opp_squad"]:
        cd[c] = np.where(cd[c] <= 0, 0, np.log(cd[c]))

    # --- Winsorize goals at 7 (NOTE: upstream clips for fitting only)
    cd["match_outcome__orig"] = cd["match_outcome"].copy()
    cd["match_outcome"] = np.where(cd["match_outcome"] > 7, 7, cd["match_outcome"])
    return cd[IDVAR + [YVAR] + FACTORS].dropna().reset_index(drop=True)


def make_train(cd, train_seasons):
    df = cd.loc[cd["season"].isin(train_seasons), :].dropna()
    df = df.loc[df.groupby("name_league")["season"].transform("min") != df["season"], :]  # drop first season/league
    df = df.sort_values(["name_league", "kick_off"]).reset_index(drop=True)
    # cross-sectional standardization by gameday (do__scaleCS=True path)
    scaled = (df.groupby("gameday")[FACTORS_CS]
              .apply(lambda x: (x - x.mean()) / x.std())
              .reset_index().set_index("level_1").drop("gameday", axis=1))
    df[FACTORS_CS] = scaled
    return df.dropna().reset_index(drop=True)


def build_model(df, likelihood):
    teams = sorted(set(df["name_team"]).union(df["name_opp"]))
    t2i = {t: i for i, t in enumerate(teams)}
    leagues = sorted(set(df["name_league"]))
    l2i = {l: i for i, l in enumerate(leagues)}
    coords = {"factor_g": FACTORS_G, "obs_id": df.index, "teams": teams, "leagues": leagues}
    with pm.Model(coords=coords) as m:
        X_gf = pm.Data("X_gf", df[FACTORS_G].to_numpy(), dims=("obs_id", "factor_g"))
        X_home = pm.Data("X_home", df["home_pitch"].to_numpy().astype(int), dims="obs_id")
        Y = pm.Data("Y", df[YVAR].to_numpy(), dims="obs_id")
        idx_team = pm.Data("idx_team", df["name_team"].map(t2i).to_numpy(), dims="obs_id")
        idx_opp = pm.Data("idx_opp", df["name_opp"].map(t2i).to_numpy(), dims="obs_id")
        idx_league = pm.Data("idx_league", df["name_league"].map(l2i).to_numpy(), dims="obs_id")

        sigma_alpha = pm.Gamma("sigma_alpha", alpha=2, beta=4)
        alpha = pm.Deterministic("alpha", pm.ZeroSumNormal("alpha_raw", sigma=1, dims="teams") * sigma_alpha, dims="teams")
        sigma_delta = pm.Gamma("sigma_delta", alpha=2, beta=4)
        delta = pm.Deterministic("delta", pm.Normal("delta_raw", 0, 1, dims="teams") * sigma_delta, dims="teams")
        mu_gamma = pm.Normal("mu_gamma", mu=0.30, sigma=0.20)
        sigma_gamma = pm.Gamma("sigma_gamma", alpha=2, beta=20)
        beta_home = pm.Deterministic("beta_home", mu_gamma + pm.Normal("gamma_raw", 0, 1, dims="teams") * sigma_gamma, dims="teams")
        kappa = pm.ZeroSumNormal("kappa", sigma=0.3, dims="leagues")
        beta = pm.Normal("beta", mu=0, sigma=0.3, dims="factor_g")
        eta = pm.Deterministic("eta", alpha[idx_team] - delta[idx_opp] + kappa[idx_league]
                               + X_home * beta_home[idx_team] + pt.dot(X_gf, beta.T), dims="obs_id")
        if likelihood == "nbinom":
            scale = pm.Gamma("scale", alpha=20, beta=0.1)
            pm.NegativeBinomial("match_outcome", mu=pm.math.exp(eta), alpha=scale, observed=Y, dims="obs_id")
        else:
            pm.Poisson("match_outcome", mu=pm.math.exp(eta), observed=Y, dims="obs_id")
    return m


def draw_calibration(idata, df, k_max=8, n_sub=400):
    """Model-implied P(draw)/P(home)/P(away) per match via thinned posterior (no GB array)."""
    eta = idata.posterior["eta"].stack(samples=("chain", "draw"))
    S = eta.sizes["samples"]
    sub = rng.choice(S, size=min(n_sub, S), replace=False)
    lam = np.exp(eta.isel(samples=sub).values)              # (n_obs, n_sub)
    pos = {oid: i for i, oid in enumerate(df.index)}
    rows = []
    for mid, g in df.groupby("id_match"):
        h = g[g["home_pitch"] == 1]
        a = g[g["home_pitch"] == 0]
        if len(h) != 1 or len(a) != 1:
            continue
        lh, la = lam[pos[h.index[0]]], lam[pos[a.index[0]]]            # (n_sub,)
        ks = np.arange(k_max + 1)
        ph = poisson.pmf(ks[:, None], lh[None, :])                    # (K, n_sub)
        pa = poisson.pmf(ks[:, None], la[None, :])
        # Vectorized over samples via cumulative goal PMFs (identical to outer-product, no GB array):
        ph_cum, pa_cum = np.cumsum(ph, 0), np.cumsum(pa, 0)
        ph_tot, pa_tot = ph.sum(0), pa.sum(0)
        p_draw = (ph * pa).sum(0)                                     # (n_sub,)
        p_home = (pa * (ph_tot[None, :] - ph_cum)).sum(0)            # home goals > away goals
        oh, oa = int(h["match_outcome"].iloc[0]), int(a["match_outcome"].iloc[0])
        obs = 1 if oh == oa else (2 if oh > oa else 0)                # 1=draw,2=home,0=away
        rows.append({"id_match": mid, "league": g["name_league"].iloc[0],
                     "p_draw": p_draw.mean(), "p_home": p_home.mean(),
                     "obs": obs, "is_draw": int(obs == 1)})
    return pd.DataFrame(rows)


def main():
    print(f"[seed={SEED}] preparing features ...")
    cd = prepare_features()
    df = make_train(cd, TRAIN_SEASONS)
    print(f"train obs={len(df)}  matches={df['id_match'].nunique()}  teams={len(set(df['name_team'])|set(df['name_opp']))}  leagues={df['name_league'].nunique()}")
    print(f"observed goal mean={df[YVAR].mean():.3f}  var={df[YVAR].var():.3f}  (var/mean={df[YVAR].var()/df[YVAR].mean():.2f})")
    summary = {"seed": SEED, "train_seasons": TRAIN_SEASONS, "n_obs": int(len(df)),
               "n_matches": int(df["id_match"].nunique()),
               "goal_mean": float(df[YVAR].mean()), "goal_var": float(df[YVAR].var())}

    idatas = {}
    for lik in ["poisson", "nbinom"]:
        print(f"\n===== {lik} =====")
        m = build_model(df, lik)
        if lik == "poisson":  # prior predictive once
            with m:
                pp = pm.sample_prior_predictive(samples=500, random_seed=rng)
            gp = pp.prior_predictive["match_outcome"].values.ravel()
            summary["prior_pred"] = {"mean": float(np.mean(gp)), "p95": float(np.quantile(gp, .95)),
                                     "p_gt10": float(np.mean(gp > 10)), "max": float(np.max(gp))}
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(np.clip(gp, 0, 20), bins=np.arange(0, 21), density=True, alpha=.8)
            ax.set(title=f"Prior predictive goals (mean={np.mean(gp):.2f}, P(>10)={np.mean(gp>10):.3f})",
                   xlabel="goals", ylabel="density")
            fig.tight_layout(); fig.savefig(f"{HERE}/prior_predictive.png", dpi=110); plt.close(fig)

        with m:
            idata = pm.sample(nuts_sampler="nutpie", draws=1000, tune=1000, target_accept=0.9,
                              random_seed=rng, progressbar=False)
            pm.compute_log_likelihood(idata, model=m)
        idata.to_netcdf(f"{HERE}/inference_data_{lik}.nc")    # SAVE before post-processing
        idatas[lik] = idata

        diag = az.summary(idata, var_names=["sigma_alpha", "sigma_delta", "mu_gamma", "sigma_gamma", "beta"])
        rhat_max = float(diag["r_hat"].max()); ess_min = float(diag["ess_bulk"].min())
        ndiv = int(idata.sample_stats["diverging"].sum())
        with open(f"{HERE}/diagnose_{lik}.txt", "w") as f:
            f.write(diag.to_string())
        print(f"  rhat_max={rhat_max:.3f}  ess_bulk_min={ess_min:.0f}  divergences={ndiv}")
        summary[lik] = {"rhat_max": rhat_max, "ess_bulk_min": ess_min, "divergences": ndiv}

        try:
            cal = draw_calibration(idata, df)
            obs_draw = float(cal["is_draw"].mean()); pred_draw = float(cal["p_draw"].mean())
            summary[lik]["draw_obs"] = obs_draw
            summary[lik]["draw_pred"] = pred_draw
            cal.to_csv(f"{HERE}/draw_calibration_{lik}.csv", index=False)
            print(f"  P(draw): observed={obs_draw:.3f}  model={pred_draw:.3f}  (gap={obs_draw-pred_draw:+.3f})")
        except Exception as e:
            print(f"  draw_calibration failed: {e}")

    # --- Draw-calibration plot (both likelihoods)
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels, obs, pred = [], [], []
        for lik in ["poisson", "nbinom"]:
            if "draw_obs" in summary.get(lik, {}):
                labels.append(lik); obs.append(summary[lik]["draw_obs"]); pred.append(summary[lik]["draw_pred"])
        x = np.arange(len(labels))
        ax.bar(x - .2, obs, .4, label="observed draw rate")
        ax.bar(x + .2, pred, .4, label="model-predicted P(draw)")
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("draw probability")
        ax.set_title("Draw calibration: observed vs model"); ax.legend()
        fig.tight_layout(); fig.savefig(f"{HERE}/draw_calibration.png", dpi=110); plt.close(fig)
    except Exception as e:
        print(f"draw plot failed: {e}")

    # --- Identifiability illustration (attack/defense coupling + baseline absorption)
    try:
        ip = idatas["poisson"].posterior
        a_mean = ip["alpha"].mean(("chain", "draw")).values
        d_mean = ip["delta"].mean(("chain", "draw")).values
        mean_alpha = ip["alpha"].mean("teams").values.ravel()   # ~0 by ZeroSum
        mean_delta = ip["delta"].mean("teams").values.ravel()   # absorbs baseline if !=0
        corr = float(np.corrcoef(a_mean, d_mean)[0, 1])
        summary["identifiability"] = {"corr_alpha_delta": corr,
                                      "mean_delta_post": float(mean_delta.mean()),
                                      "mean_alpha_post": float(mean_alpha.mean())}
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(a_mean, d_mean, s=12, alpha=.6)
        axes[0].set(xlabel="posterior mean alpha (attack)", ylabel="posterior mean delta (defense)",
                    title=f"per-team attack vs defense (corr={corr:+.2f})")
        axes[1].hist(mean_alpha, bins=40, alpha=.7, label=f"mean(alpha)  ~{mean_alpha.mean():+.2f}")
        axes[1].hist(mean_delta, bins=40, alpha=.7, label=f"mean(delta)  ~{mean_delta.mean():+.2f}")
        axes[1].axvline(0, color="k", lw=.8)
        axes[1].set(title="ZeroSum alpha (~0) vs free delta (absorbs baseline)", xlabel="team-averaged effect")
        axes[1].legend()
        fig.tight_layout(); fig.savefig(f"{HERE}/identifiability.png", dpi=110); plt.close(fig)
        print(f"  attack/defense corr={corr:+.2f}  mean(delta)={mean_delta.mean():+.2f}  mean(alpha)={mean_alpha.mean():+.2f}")
    except Exception as e:
        print(f"identifiability plot failed: {e}")

    # --- LOO model comparison Poisson vs NegBin
    try:
        cmp = az.compare({k: v for k, v in idatas.items()}, ic="loo")
        cmp.to_csv(f"{HERE}/loo_compare.csv")
        summary["loo_compare"] = {"best": str(cmp.index[0]),
                                  "elpd_diff": float(cmp["elpd_diff"].iloc[1]),
                                  "dse": float(cmp["dse"].iloc[1])}
        print("\nLOO comparison:\n", cmp[["rank", "elpd_loo", "elpd_diff", "dse", "weight"]].to_string())
    except Exception as e:
        print(f"LOO compare failed: {e}")

    with open(f"{HERE}/results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved results_summary.json")


if __name__ == "__main__":
    main()
