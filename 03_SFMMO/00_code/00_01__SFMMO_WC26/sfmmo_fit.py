#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
#  SFMMO Dev World-Cup Edition -- FIT SCRIPT (DevK, expanding-window)
# =============================================================================
#  Faithful port of Max's Colab notebook
#      03_SFMMO/00_code/SFMMOwm__dev_EW.ipynb
#  (cell-by-cell source: .claude/jobs/cc7b39b8/tmp/dev_src.py).
#
#  His modeling logic is preserved VERBATIM. The ONLY deliberate changes are the
#  patches flagged with `# IMPROVEMENT:` and the Colab->repo path fixes flagged
#  with `# PATH:` / `# RUN-BLOCKER:`. Everything else is copied as-is, including
#  feature engineering, the per-league Elo (`update_elo`/`compute_league_elo`),
#  momentum features, cross-sectional standardization, the hardcoded
#  confederation Elo multipliers (0.8/0.8/0.75/0.6), the expanding-window split,
#  the 25th-percentile cold-start anchor for unseen teams, the PyMC structure,
#  the goal-PMF -> W/D/L collapse, and the saved pickle format.
#
#  Output pickle structure (consumed by sfmmo_eval.py), UNCHANGED:
#      {'factors': [...],
#       'dict_preds': {fold: {'Yhat':   DataFrame[['0','1','2']],
#                             'Y__SFM': DataFrame[['id_match','match_outcome']]}}}
#
#  Run (full):
#      uv run --directory /Users/alex_andorra/tptm_alex/portfolio/football-modeling/03_SFMMO \
#          --group model python 00_code/sfmmo_fit.py
#  Run (smoke, fast check -- last fold only, draws=200/tune=200/chains=2):
#      SFMMO_SMOKE=1 uv run --directory .../03_SFMMO --group model python 00_code/sfmmo_fit.py
# =============================================================================


# ===== cell 1 =====
# IMPROVEMENT (patch 1, STRIP COLAB): removed `from google.colab import drive`
# and `drive.mount('/content/drive')` -- not applicable outside Colab.

# ===== cell 2 =====
# IMPROVEMENT (patch 1, STRIP COLAB): removed the `!pip install ...` lines
# (pymc, numpyro, pymc-bart, plotly, graphviz, arviz, jax[cuda]). Dependencies
# are provided by the `model` uv group (pyproject.toml).

# ===== cell 3 =====

import os  # PATH: needed for repo-relative path resolution
import json  # PATH: needed for the per-fold diagnostics dump


# --- Usual Libraries:
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- PyMC & Affiliates
# import arviz as az
import pymc as pm
import pytensor.tensor as pt
import arviz as az

# RUN-BLOCKER: the following cell-3 imports are absent from the `model` uv group
# and are NOT used anywhere on the DevK fit path (BART is the dead devVersion=='Z'
# branch; plotly/graphviz/seaborn/arviz_plots/arviz_stats are only used by the
# notebook's evaluation/plotting cells, which are a separate stage handled by
# sfmmo_eval.py). Importing them unguarded crashes the script before it can run.
# They are guarded so faithful references still resolve if the package is present
# and otherwise fail loudly only if actually used. No modeling behavior changes.
try:
    import arviz_plots as azp
except Exception:
    azp = None
try:
    import arviz_stats
except Exception:
    arviz_stats = None

# --- BART
try:
    import pymc_bart as pmb
except Exception:
    pmb = None


# --- Stats
from scipy.stats import nbinom, poisson

try:
    from sklearn.preprocessing import StandardScaler
except Exception:
    StandardScaler = None

# --- Interactive Plots:
try:
    import plotly.graph_objects as go
    import plotly
except Exception:
    go = None
    plotly = None

# --- Plotting:
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:
    sns = None

seed = sum(map(ord, "sfm"))
rng = np.random.default_rng(seed)

pm.__version__

# ===== cell 4 =====
np.__version__


# =============================================================================
# PATH: repo-relative path resolution (replaces the Colab `directory = './'` and
#       the non-existent `.../102_Development/data_byPlayer__SFM_II__TM__WM.csv`).
# 03_SFMMO is one level up from 00_code; the model-data CSV lives under 10_data/.
# =============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))  # .../03_SFMMO/00_code/00_01__SFMMO_WC26
_SFMMO_ROOT = os.path.dirname(os.path.dirname(_HERE))  # .../03_SFMMO (two levels up since the WC26 subfolder move)

# PATH: the notebook reads `{directory}/10_data/102_Development/data_byPlayer...csv`,
#       but the CSV actually lives at 10_data/data_byPlayer__SFM_II__TM__WM.csv.
DATA_CSV = os.path.join(_SFMMO_ROOT, "10_data", "data_byPlayer__SFM_II__TM__WM.csv")

# PATH: output dir for predictions + diagnostics (created below).
OUT_DIR = os.path.join(_SFMMO_ROOT, "10_data", "102_Development")
os.makedirs(OUT_DIR, exist_ok=True)  # PATH: ensure the output dir exists

assert os.path.exists(DATA_CSV), f"[ERROR]: model-data CSV not found at {DATA_CSV}"


# =============================================================================
# SMOKE mode: fast run-check. SFMMO_SMOKE=1 -> draws=200, tune=200, chains=2,
# and only the LAST fold (train through WM2022 -> validate WMQ2026). Default
# (no env var) = full: all folds, full draws.
# =============================================================================
SMOKE = os.environ.get("SFMMO_SMOKE") == "1"
# Robustness variant: SFMMO_STRICT_HOLDOUT=1 freezes the per-league Elo on the held-out
# season (no Elo update on held-out outcomes) -> a strict pre-tournament forecast. The default
# (sequential / "vs-the-closing-line") lets prior held-out matches update Elo, matching the
# information set the bookmaker closing odds are formed on. (Cumulative-goals features come
# pre-computed from the data CSV and are unchanged by this switch.)
STRICT_HOLDOUT = os.environ.get("SFMMO_STRICT_HOLDOUT") == "1"
_VARIANT = "_strict" if STRICT_HOLDOUT else ""


# ===== cell 5 =====
# -------------------------------------- USER INTERACTION -------------------------------------- #

# --- Set the directory to the datafile ('data_byPlayer__SFM_II__TM__WM.csv'):
# PATH: `directory` is no longer used for I/O (replaced by DATA_CSV / OUT_DIR above);
#       kept here only so the verbatim notebook code below has the symbol in scope.
directory = "./"

# --- Which 'seasons' are in the Training-Set, which are in the Validation-Set?
# --- --- Each Training-/Val-Set combination needs to be a separate list
dict_EW = {
    "trainT": [
        ["WMQ2010", "WM2010", "WMQ2014", "WM2014"],
        ["WMQ2010", "WM2010", "WMQ2014", "WM2014", "WMQ2018"],
        ["WMQ2010", "WM2010", "WMQ2014", "WM2014", "WMQ2018", "WM2018"],
        ["WMQ2010", "WM2010", "WMQ2014", "WM2014", "WMQ2018", "WM2018", "WMQ2022"],
        ["WMQ2010", "WM2010", "WMQ2014", "WM2014", "WMQ2018", "WM2018", "WMQ2022", "WM2022"],
    ],
    "valT": [["WMQ2018"], ["WM2018"], ["WMQ2022"], ["WM2022"], ["WMQ2026"]],
}

# --- --- Only for the 'production version'
# dict_EW = {'trainT':[['WMQ2010','WM2010','WMQ2014','WM2014','WMQ2018','WM2018','WMQ2022','WM2022','WMQ2026']],'valT':[['']]}


# --- Cross-Sectional Standardization, or whole-sample Standardization?
do__scaleCS = True

# --- Which Dev-Version?
devVersion = "K"

# -------------------------------------- USER INTERACTION -------------------------------------- #

# IMPROVEMENT (SMOKE mode): restrict to the LAST fold only (train through WM2022
# -> validate WMQ2026) for a fast run-check. Full pipeline is unchanged by default.
if SMOKE:
    dict_EW = {"trainT": [dict_EW["trainT"][-1]], "valT": [dict_EW["valT"][-1]]}
    print(
        "\n[SMOKE] last fold only:",
        dict_EW["valT"][0][0],
        "(train through",
        dict_EW["trainT"][0][-1] + ")\n",
    )


# ===== cell 7 =====
def build_confederation_map(data):
    """team -> confederation, inferred from the qualifier league(s) the team appears in."""
    conf_leagues = [
        l for l in data["name_league"].unique() if str(l).startswith("wm-qualifikation")
    ]
    q = data[data["name_league"].isin(conf_leagues)]
    counts = {}
    for col in ["name_team", "name_opp"]:
        for team, league in zip(q[col], q["name_league"]):
            counts.setdefault(team, {}).setdefault(league, 0)
            counts[team][league] += 1
    return {t: max(cs, key=cs.get) for t, cs in counts.items()}  # most-frequent confederation


# ===== cell 8 =====
def compute_league_elo(
    complete_data, K=20, home_adv=50, regress_to=1500, regress_w=0.0, freeze_seasons=None
):

    # --- Run over Leagues
    N_leagues = complete_data["name_league"].unique().tolist()

    for ll in N_leagues:
        # print(f'\nELO-Ratings for league: {ll}')

        # --- Store the ELO ratings:
        ELO_rating = {
            key: 1500
            for key in complete_data.loc[complete_data["name_league"] == ll, "name_team"].unique()
        }

        # --- Available Seasons:
        N_seasons = (
            complete_data.loc[complete_data["name_league"] == ll, "season"].unique().tolist()
        )

        # --- Run across the season:
        for ss in N_seasons:
            # --- Extract the season:
            ll_ss_data = (
                complete_data[
                    (complete_data["name_league"] == ll) & (complete_data["season"] == ss)
                ]
                .copy()
                .sort_values("kick_off")
            )

            # --- At the beginning of the season, adjust the rating:
            ss_idx = N_seasons.index(ss)
            if ss_idx > 0:
                # --- Teams in previous season:
                ss_t1__teams = (
                    complete_data.loc[
                        (complete_data["name_league"] == ll)
                        & (complete_data["season"] == N_seasons[ss_idx - 1]),
                        "name_team",
                    ]
                    .unique()
                    .tolist()
                )

                for team in ELO_rating.keys():
                    if team in ss_t1__teams:
                        ELO_rating[team] = (
                            ELO_rating[team] * (1 - regress_w) + regress_to * regress_w
                        )
                    else:
                        ELO_rating[team] = 1300

            # --- Run across the season:
            for gg in range(ll_ss_data.shape[0]):
                # --- Home Team:
                gg_home = ll_ss_data["name_team"].iloc[gg]

                # --- Away Team:
                gg_away = ll_ss_data["name_opp"].iloc[gg]

                # --- Get Current ELO Ratings:
                gg_home__elo = ELO_rating[gg_home]
                gg_away__elo = ELO_rating[gg_away]

                # --- Insert Current ELO Ratings:
                gg_idx = ll_ss_data.index[gg]
                complete_data.loc[gg_idx, "elo_team"] = gg_home__elo
                complete_data.loc[gg_idx, "elo_opp"] = gg_away__elo

                # --- Update ELO Ratings (robustness): skip the update on frozen/held-out seasons so a
                #     held-out match's result cannot leak into a later held-out match's pre-game Elo.
                if freeze_seasons is None or ss not in freeze_seasons:
                    ELO_rating[gg_home], ELO_rating[gg_away] = update_elo(
                        gg_home__elo,
                        gg_away__elo,
                        ll_ss_data["match_outcome__home"].iloc[gg],
                        K=K,
                        home_adv=home_adv,
                    )

    return complete_data


# ===== cell 9 =====
def compute_global_elo(complete_data, K=20, home_adv=50, regress_w=0.0, regress_to=1500.0):
    """
    Global cross-competition ELO. Processes each MATCH once, in true chronological order.
    Updates ratings only on PLAYED matches (OOS rows with NaN goals are read, not updated —
    so all WC group games use the same pre-tournament rating; no fake-draw contamination).
    Writes elo_team / elo_opp (pre-game) onto BOTH perspective rows of complete_data.
    regress_w > 0 applies a mild regression toward `regress_to` at each new calendar year.
    """
    teams = set(complete_data["name_team"]).union(complete_data["name_opp"])
    elo = {t: 1500.0 for t in teams}

    # one row per match (home perspective), global chronological order
    m = complete_data[complete_data["home_pitch"] == 1].sort_values("kick_off")

    pre = {}  # id_match -> (home_elo, away_elo) entering the match
    prev_year = None
    for _, row in m.iterrows():
        h, a = row["name_team"], row["name_opp"]

        yr = row["kick_off"].year
        if regress_w > 0 and prev_year is not None and yr != prev_year:
            for t in elo:
                elo[t] = (1 - regress_w) * elo[t] + regress_w * regress_to
        prev_year = yr

        eh, ea = elo[h], elo[a]
        pre[row["id_match"]] = (eh, ea)

        # update only on played matches (result known)
        if pd.notna(row["goalsscored_inGame_team"]) and pd.notna(row["goalsscored_inGame_opp"]):
            elo[h], elo[a] = update_elo(
                eh, ea, int(row["match_outcome__home"]), K=K, home_adv=home_adv
            )

    # write pre-game ratings back onto both perspective rows
    et, eo = [], []
    for _, row in complete_data.iterrows():
        eh, ea = pre[row["id_match"]]
        if row["home_pitch"] == 1:
            et.append(eh)
            eo.append(ea)  # home perspective
        else:
            et.append(ea)
            eo.append(eh)  # away perspective
    complete_data["elo_team"] = et
    complete_data["elo_opp"] = eo
    return complete_data


# ===== cell 10 =====
def _cs_scale(x):
    sd = x.std()
    sd = sd.where(sd > 0, 1.0)
    return (x - x.mean()) / sd


# ===== cell 11 =====
def compute_log_likelihood(predictions, actuals):
    """
    predictions: array of shape (n_samples, n_categories)
                 e.g., [[0.4, 0.35, 0.2, 0.05], ...] for one match
    actuals: array of actual outcomes [0, 1, 2, 3, ...]
    """
    log_lik = 0
    for i, actual in enumerate(actuals):
        # Get probability assigned to actual outcome
        prob_actual = predictions[i][actual]
        log_lik += np.log(prob_actual + 1e-10)  # Add small epsilon to avoid log(0)

    return log_lik


def log_loss_categorical(probs, actuals, eps=1e-10):
    return -np.mean(np.log(probs[np.arange(len(actuals)), actuals] + eps))


def multi_class_brier_score(predictions, actuals, n_classes=4):
    """
    predictions: (n_samples, n_classes) probability matrix
    actuals: (n_samples,) actual outcomes
    """
    # Convert actuals to one-hot encoding
    one_hot = np.zeros((len(actuals), n_classes))
    one_hot[np.arange(len(actuals)), actuals] = 1

    # Compute Brier score
    brier = np.mean(np.sum((predictions - one_hot) ** 2, axis=1))
    return brier


def ranked_probability_score(predictions, actuals, n_classes=4):
    """
    Ranked Probability Score for ordinal outcomes
    """
    rps = 0
    for i, actual in enumerate(actuals):
        # Cumulative probabilities
        pred_cumsum = np.cumsum(predictions[i])

        # Actual cumulative (one-hot converted to cumulative)
        actual_cumsum = np.zeros(n_classes)
        actual_cumsum[actual:] = 1

        # RPS for this prediction
        rps += np.sum((pred_cumsum - actual_cumsum) ** 2)

    return rps / len(actuals)


def expected_calibration_error(predictions, actuals, n_bins=10):
    """
    ECE for probabilistic predictions
    """
    # Get predicted probabilities for actual class
    pred_probs = predictions[np.arange(len(actuals)), actuals]

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)

    ece = 0
    for i in range(n_bins):
        # Find predictions in this bin
        in_bin = (pred_probs >= bin_edges[i]) & (pred_probs < bin_edges[i + 1])

        if np.sum(in_bin) > 0:
            # Average predicted probability in bin
            avg_pred = np.mean(pred_probs[in_bin])

            # Actual accuracy in bin (for max predicted class)
            max_pred = np.argmax(predictions[in_bin], axis=1)
            avg_actual = np.mean(max_pred == actuals[in_bin])

            # Weighted by bin size
            ece += np.abs(avg_pred - avg_actual) * np.sum(in_bin)

    return ece / len(actuals)


def ordinal_accuracy(predictions, actuals):
    """
    Percentage of times the highest probability category was correct
    """
    pred_classes = np.argmax(predictions, axis=1)
    return np.mean(pred_classes == actuals)


def ordinal_mae(predictions, actuals):
    """
    MAE treating categories as ordinal
    """
    pred_classes = np.argmax(predictions, axis=1)
    return np.mean(np.abs(pred_classes - actuals))


def update_elo(r_home, r_away, result, K=20, home_adv=50):
    exp_home = 1 / (1 + 10 ** ((r_away - r_home - home_adv) / 400))
    if result == 2:  # home win
        s_home = 1.0
    elif result == 1:  # draw
        s_home = 0.5
    elif result == 0:  # away win
        s_home = 0.0
    else:
        raise ValueError("Invalid result value. Must be 0, 1, or 2.")

    r_home_new = r_home + K * (s_home - exp_home)
    r_away_new = r_away + K * ((1 - s_home) - (1 - exp_home))
    return r_home_new, r_away_new


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


# ===== cell 12 =====
def rootogram(obs_counts, pp_counts, max_val=8, title=""):
    """
    obs_counts : 1D int array of observed goal counts        (n_fixtures,)
    pp_counts  : 2D int array of posterior predictive counts (n_fixtures, n_samples)
    """
    obs_counts = np.asarray(obs_counts, dtype=int)
    pp_counts = np.asarray(pp_counts, dtype=int)

    xs = np.arange(max_val + 1)
    obs_freq = np.bincount(obs_counts, minlength=max_val + 1)[: max_val + 1]

    n_samp = min(pp_counts.shape[1], 200)
    exp_freq = np.array(
        [np.bincount(pp_counts[:, s], minlength=max_val + 1)[: max_val + 1] for s in range(n_samp)]
    )  # shape (n_samp, max_val+1)

    exp_mean = exp_freq.mean(axis=0)
    exp_low = np.quantile(exp_freq, 0.03, axis=0)
    exp_up = np.quantile(exp_freq, 0.97, axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(xs, np.sqrt(obs_freq), alpha=0.5, label="observed")
    ax.plot(xs, np.sqrt(exp_mean), "ro-", label="expected (PP mean)")
    ax.fill_between(xs, np.sqrt(exp_low), np.sqrt(exp_up), color="red", alpha=0.2, label="94% HDI")
    ax.set_xlabel("goals")
    ax.set_ylabel("√frequency")
    ax.set_title(title)
    ax.legend()
    plt.show()

    print(f"\nRMSE: {np.round(np.sqrt(np.mean((obs_freq - exp_mean))), 3)}")
    print(f"Diff: {np.round((obs_freq - exp_mean) / obs_freq, 2)}\n")


# ===== cell 13 =====
# IMPROVEMENT (patch 4): k_max default 5 -> 15 (removes the probability-mass
# truncation on extreme-lambda fixtures).
def get__perMatch_jointPMF(eta, idx_home, idx_away, scale=None, k_max=15):
    """
    For every row in the training set, return the posterior-predictive PMF
    P(Y = k) for k = 0..k_max with credible intervals.

    Returns a long-format DataFrame: (obs_id, k, mean, lo, hi).
    """

    # --- 1. Stack chain × draw → samples
    lam_samples = np.exp(eta.stack(samples=("chain", "draw")).values)  # (n_obs, n_samples)

    # --- Extract Home-Away Match-Ups
    lam_h = lam_samples[idx_home, :]  # (n_fixtures, n_samples)
    lam_a = lam_samples[idx_away, :]

    # --- Probability of Number of Goals to Evaluate
    N_goals = np.arange(k_max + 1)

    # --- 2. PMFs per sample
    if scale is not None:
        scale_samples = scale.stack(samples=("chain", "draw")).values  # (n_samples,)
        # nbinom.pmf with broadcasting: (k_max+1, n_fixtures, n_samples)
        pmf_h = nbinom.pmf(
            N_goals[:, None, None],
            n=scale_samples[None, None, :],
            p=scale_samples[None, None, :] / (scale_samples[None, None, :] + lam_h[None, :, :]),
        )  # (k_max+1, n_fixtures, n_samples)
        pmf_a = nbinom.pmf(
            N_goals[:, None, None],
            n=scale_samples[None, None, :],
            p=scale_samples[None, None, :] / (scale_samples[None, None, :] + lam_a[None, :, :]),
        )
    else:
        # poisson.pmf with broadcasting: (k_max+1, n_fixtures, n_samples)
        pmf_h = poisson.pmf(N_goals[:, None, None], mu=lam_h[None, :, :])
        pmf_a = poisson.pmf(N_goals[:, None, None], mu=lam_a[None, :, :])

    # --- 3. Outer product per fixture per sample → joint
    # einsum: 'h f s, a f s -> f s h a'
    joint = np.einsum("hfs,afs->fsha", pmf_h, pmf_a)

    return joint


def get__home_WDL(matchup_jointPMF, cred_region=0.9):
    """
    matchup_jointPMF: (n_samples, K, K) joint PMF for one fixture
    """
    # Per-sample W/D/L
    diag_idx = np.arange(matchup_jointPMF.shape[-1])

    p_draw = matchup_jointPMF[:, diag_idx, diag_idx].sum(axis=-1)  # (n_samples,)
    p_loss = np.triu(matchup_jointPMF, k=1).sum(axis=(-2, -1))  # away > home (home loss)
    p_win = np.tril(matchup_jointPMF, k=-1).sum(axis=(-2, -1))  # home > away

    alpha_lo = (1 - cred_region) / 2
    alpha_hi = 1 - alpha_lo

    df = pd.DataFrame(index=["W", "D", "L"], columns=["low", "mid", "up"], dtype=float)
    for label, arr in [("W", p_win), ("D", p_draw), ("L", p_loss)]:
        df.loc[label, "mid"] = arr.mean()
        df.loc[label, "low"] = np.quantile(arr, alpha_lo)
        df.loc[label, "up"] = np.quantile(arr, alpha_hi)

    return df


# =============================================================================
# 1   The Grand Loop  (cell 16)
# =============================================================================
def main():

    # ======================================== The Grand Loop ======================================== #

    # for devVersion in ['A','B','C','D','E']:

    dict_preds = {key[0]: {} for key in dict_EW["valT"]}
    dict_fitEval = {key[0]: {} for key in dict_EW["valT"]}

    # IMPROVEMENT (patch 2): collect per-fold R-hat / divergence diagnostics to dump next to the pickle.
    diagnostics = {}

    for t in range(len(dict_EW["trainT"])):
        # --- Extract Training-End and Validation Season:
        train_end = dict_EW["trainT"][t]
        if type(dict_EW["valT"][t]) == list:
            val_seasons = dict_EW["valT"][t]
        elif type(dict_EW["valT"][t]) == str:
            val_seasons = [dict_EW["valT"][t]]

        # --------------------------------- 00. Data Preparation --------------------------------- #
        # PATH: read from the real CSV location under 10_data/ (the notebook's
        #       .../102_Development/data_byPlayer...csv does not exist).
        data_raw = pd.read_csv(DATA_CSV)

        # --- Kick 'wm-qualifikation-ozeanien': too few seasons in the training-set
        # data_raw = data_raw[~(data_raw['name_league'] == 'wm-qualifikation-ozeanien')]

        # --- Pre-Processing, Part I:
        data_raw["kick_off"] = pd.to_datetime(data_raw["kick_off"])
        data_raw = data_raw.sort_values(["name_player", "season", "kick_off"])

        # --- Confederations:
        # confed_map   = build_confederation_map(data_raw)
        confed_map = {
            key: []
            for key in [
                f"wm-qualifikation-{c}"
                for c in [
                    "asien",
                    "europa",
                    "afrika",
                    "nordamerika-mittelamerika",
                    "suedamerika",
                    "ozeanien",
                ]
            ]
        }
        for c in confed_map.keys():
            confed_map[c] = (
                data_raw.loc[data_raw["name_league"] == c, "name_team"].unique().tolist()
            )

        # --- Columns to Keep:
        keepCols = [
            "points_team",
            "points_opp",
            "goalsscored_inGame_team",
            "goalsscored_inGame_opp",
            "goalsscored_cum_team",
            "goalsscored_cum_opp",
            "goalsconceded_cum_team",
            "goalsconceded_cum_opp",
            "home_pitch",
            "goalsscored_rank_team",
            "goalsconceded_rank_opp",
            "id_match",
            "name_team",
            "name_opp",
            "name_league",
            "id_league",
            "season",
            "gameday",
            "kick_off",
            "goalsscored_rank_opp",
            "goalsconceded_rank_team",
            "goalsscored_diff",
            "goal_balance_team",
            "goal_balance_opp",
            "goal_balance_diff",
            "points_diff",
            "tm_marketvalue_team_squad",
            "tm_marketvalue_opp_squad",
        ]

        # --- Extract unique Matches:
        complete_data = (
            data_raw.drop_duplicates(subset=["id_match", "home_pitch"])[keepCols]
            .copy()
            .sort_values(["name_league", "kick_off"])
            .reset_index(drop=True)
        )

        # --- Keep only Matches for which we observe both Home & Away Team:
        complete_data = complete_data.loc[complete_data["id_match"].duplicated(keep=False), :]

        # Note:       We need two observations per match. This is preserved here.
        #             The home team is identified by the 'home_pitch' indicator.
        # Important:  Each observation is to be seen from the perspective of the TEAM --- but that does not say if TEAM plays at home!
        #             Hence, we can model the target easily, by taking the TEAM variable as the anchor!

        # --- Target: Number of Goals
        complete_data["match_outcome"] = complete_data["goalsscored_inGame_team"].copy()

        # -------------------------------------------- Feature Engineering -- Part I -------------------------------------------- #

        # --- Some Game-Day-Adjustments are necessary -- especially for cross-sectional scaling if a Gameday is '14.1', give it some grace and set it to '14'.
        complete_data["gameday_orig"] = complete_data["gameday"].copy()
        complete_data["gameday"] = [
            int(float(i.split("_")[1][2:])) for i in complete_data["id_match"].values
        ]

        # -------------------------------------------- ELO Ratings -- Global -------------------------------------------- #
        complete_data[["elo_team", "elo_opp"]] = np.nan

        # --- For Compatibility:
        complete_data["match_outcome__home"] = 1
        complete_data.loc[
            complete_data["goalsscored_inGame_team"] > complete_data["goalsscored_inGame_opp"],
            "match_outcome__home",
        ] = 2
        complete_data.loc[
            complete_data["goalsscored_inGame_team"] < complete_data["goalsscored_inGame_opp"],
            "match_outcome__home",
        ] = 0

        # --- Instantiate 'elo':
        complete_data["elo_team"] = np.nan
        complete_data["elo_opp"] = np.nan

        # --- Compute 'elo':
        # complete_data = compute_global_elo(complete_data, regress_w=0.25)
        complete_data = compute_league_elo(
            complete_data,
            K=20,
            home_adv=50,
            regress_to=1500,
            regress_w=0.25,
            freeze_seasons=(val_seasons if STRICT_HOLDOUT else None),
        )

        # --- 'elo' weighting:
        if 1 == 1:
            complete_data["elo_team"] = np.where(
                complete_data["name_team"].isin(confed_map["wm-qualifikation-asien"]),
                complete_data["elo_team"] * 0.8,
                complete_data["elo_team"],
            )
            complete_data["elo_opp"] = np.where(
                complete_data["name_opp"].isin(confed_map["wm-qualifikation-asien"]),
                complete_data["elo_opp"] * 0.8,
                complete_data["elo_opp"],
            )

            complete_data["elo_team"] = np.where(
                complete_data["name_team"].isin(confed_map["wm-qualifikation-afrika"]),
                complete_data["elo_team"] * 0.8,
                complete_data["elo_team"],
            )
            complete_data["elo_opp"] = np.where(
                complete_data["name_opp"].isin(confed_map["wm-qualifikation-afrika"]),
                complete_data["elo_opp"] * 0.8,
                complete_data["elo_opp"],
            )

            complete_data["elo_team"] = np.where(
                complete_data["name_team"].isin(
                    confed_map["wm-qualifikation-nordamerika-mittelamerika"]
                ),
                complete_data["elo_team"] * 0.75,
                complete_data["elo_team"],
            )
            complete_data["elo_opp"] = np.where(
                complete_data["name_opp"].isin(
                    confed_map["wm-qualifikation-nordamerika-mittelamerika"]
                ),
                complete_data["elo_opp"] * 0.75,
                complete_data["elo_opp"],
            )

            complete_data["elo_team"] = np.where(
                complete_data["name_team"].isin(confed_map["wm-qualifikation-ozeanien"]),
                complete_data["elo_team"] * 0.6,
                complete_data["elo_team"],
            )
            complete_data["elo_opp"] = np.where(
                complete_data["name_opp"].isin(confed_map["wm-qualifikation-ozeanien"]),
                complete_data["elo_opp"] * 0.6,
                complete_data["elo_opp"],
            )

        complete_data["elo_diff"] = complete_data["elo_team"] - complete_data["elo_opp"]
        complete_data["elo_team_opp"] = complete_data["elo_team"] * complete_data["elo_opp"]

        # ======================================== Feature Engineering --- Part II ======================================== #

        # --- Goal Appeal (of the match):
        complete_data["goal_appeal"] = (
            complete_data["goalsconceded_rank_opp"] - complete_data["goalsscored_rank_team"]
        )

        # ---------------------- Team Momentum - exponentially-weighted MA of previous goals/outcomes ---------------------- #

        # --- For Compatibility: ---> BUT BE CAREFUL TO DROP IT AS IT IS JUST A HELPER!
        complete_data["home_team"] = complete_data["name_team"].copy()

        complete_data = complete_data.set_index(["kick_off", "season", "id_match", "home_team"])
        complete_data[
            ["teamMOM__S", "teamMOM__M", "teamMOM__L", "oppMOM__S", "oppMOM__M", "oppMOM__L"]
        ] = np.nan
        complete_data[
            [
                "FD_teamMOM__S",
                "FD_teamMOM__M",
                "FD_teamMOM__L",
                "FD_oppMOM__S",
                "FD_oppMOM__M",
                "FD_oppMOM__L",
            ]
        ] = np.nan

        for tt in complete_data["name_team"].unique():
            # --- Extract the data for team 'tt'
            # tt_data = complete_data.loc[(complete_data['name_team'] == tt) | (complete_data['name_opp'] == tt),:].copy()
            tt_data = complete_data.loc[complete_data["name_team"] == tt, :].copy()

            # --- Calculate the Cumulative Points:
            tt_points = pd.DataFrame(index=tt_data.index)
            tt_points["points"] = tt_data["points_team"]

            # --- FD of points
            tt_points["FD_points"] = tt_points.groupby(level="season")["points"].diff()
            # --- Calculate MOMENTUM: EWMA of points in previous games (halflife == one appearance)
            tt_points["MOM__S"] = (
                tt_points.groupby(level="season")["FD_points"]
                .ewm(halflife=1)
                .mean()
                .droplevel(level=0)
            )
            tt_points["MOM__M"] = (
                tt_points.groupby(level="season")["FD_points"]
                .ewm(halflife=4)
                .mean()
                .droplevel(level=0)
            )
            tt_points["MOM__L"] = (
                tt_points.groupby(level="season")["FD_points"]
                .ewm(halflife=8)
                .mean()
                .droplevel(level=0)
            )

            # --- Calculate Delta MOMENTUM: first-difference of MOMENTUM
            tt_points["FD_MOM__S"] = tt_points.groupby(level="season")["MOM__S"].diff()
            tt_points["FD_MOM__M"] = tt_points.groupby(level="season")["MOM__M"].diff()
            tt_points["FD_MOM__L"] = tt_points.groupby(level="season")["MOM__L"].diff()

            # --- Team & Opponent Columns:
            team_cols = [
                "teamMOM__S",
                "teamMOM__M",
                "teamMOM__L",
                "FD_teamMOM__S",
                "FD_teamMOM__M",
                "FD_teamMOM__L",
            ]
            opp_cols = [
                "oppMOM__S",
                "oppMOM__M",
                "oppMOM__L",
                "FD_oppMOM__S",
                "FD_oppMOM__M",
                "FD_oppMOM__L",
            ]

            # --- Source Columns:
            src_cols = ["MOM__S", "MOM__M", "MOM__L", "FD_MOM__S", "FD_MOM__M", "FD_MOM__L"]

            # --- Team:
            complete_data.loc[tt_data.index, team_cols] = tt_points[src_cols].values

            # --- Opponent:
            mom_by_match = tt_points[src_cols].copy()
            mom_by_match.index = tt_points.index.get_level_values("id_match")
            # --- --- Get the rows:
            opp_rows = complete_data.index[complete_data["name_opp"] == tt]
            # --- --- Assign the Opponent's data:
            opp_ids = opp_rows.get_level_values("id_match")
            complete_data.loc[opp_rows, opp_cols] = mom_by_match.loc[opp_ids].values

        # --- Some Two-Way Interactions:
        complete_data["teamMOM__S_L"] = complete_data["teamMOM__S"] * complete_data["teamMOM__L"]
        complete_data["oppMOM__S_L"] = complete_data["oppMOM__S"] * complete_data["oppMOM__L"]

        # --- Momentum Difference:
        complete_data["MOM__S__diff"] = complete_data["teamMOM__S"] * complete_data["oppMOM__S"]

        # --- Reset index & IMPORTANT: DROP 'home_team'
        complete_data = complete_data.reset_index().drop("home_team", axis=1)

        # ======================================== Feature Engineering -- Part III ======================================== #

        # -------------------------------------------- Transfermarket Market-Values -------------------------------------------- #

        # --- TM Ratio:
        complete_data["tm_marketvalue_ratio"] = (
            complete_data["tm_marketvalue_team_squad"] / complete_data["tm_marketvalue_opp_squad"]
        )

        # --- Log-Transform the Raw Values
        complete_data["tm_marketvalue_team_squad"] = np.where(
            complete_data["tm_marketvalue_team_squad"] <= 0,
            0,
            np.log(complete_data["tm_marketvalue_team_squad"]),
        )
        complete_data["tm_marketvalue_opp_squad"] = np.where(
            complete_data["tm_marketvalue_opp_squad"] <= 0,
            0,
            np.log(complete_data["tm_marketvalue_opp_squad"]),
        )

        # ======================================== Define the Factors ======================================== #

        if devVersion in ["A"]:
            # --- Numerical Factors:
            factors_CS = [
                "points_diff",
                "goalsscored_cum_team",
                "goalsscored_cum_opp",
                "goalsconceded_cum_team",
                "goalsconceded_cum_opp",
                "teamMOM__S",
                "oppMOM__S",
                #'MOM__S__diff',
                #'teamMOM__M','oppMOM__M',
                #'teamMOM__L','oppMOM__L',
                #'teamMOM__S_L', 'oppMOM__S_L',
                #'FD_teamMOM__S','FD_oppMOM__S',
                #'FD_teamMOM__M','FD_oppMOM__M',
                #'FD_teamMOM__L','FD_oppMOM__L',
                "elo_team",
                "elo_opp",
                "elo_team_opp",
                #'elo_diff'
            ]

        elif devVersion in ["B"]:
            # --- Numerical Factors:
            factors_CS = [
                "points_diff",
                "goalsscored_cum_team",
                "goalsscored_cum_opp",
                "goalsconceded_cum_team",
                "goalsconceded_cum_opp",
                #'teamMOM__S','oppMOM__S',
                "MOM__S__diff",
                #'teamMOM__M','oppMOM__M',
                #'teamMOM__L','oppMOM__L',
                #'teamMOM__S_L', 'oppMOM__S_L',
                #'FD_teamMOM__S','FD_oppMOM__S',
                #'FD_teamMOM__M','FD_oppMOM__M',
                #'FD_teamMOM__L','FD_oppMOM__L',
                "elo_team",
                "elo_opp",
                "elo_team_opp",
                #'elo_diff'
            ]

        elif devVersion in ["C"]:
            # --- Numerical Factors:
            factors_CS = [
                "points_diff",
                "goalsscored_cum_team",
                "goalsscored_cum_opp",
                "goalsconceded_cum_team",
                "goalsconceded_cum_opp",
                "teamMOM__S",
                "oppMOM__S",
                #'MOM__S__diff',
                #'teamMOM__M','oppMOM__M',
                #'teamMOM__L','oppMOM__L',
                #'teamMOM__S_L', 'oppMOM__S_L',
                #'FD_teamMOM__S','FD_oppMOM__S',
                #'FD_teamMOM__M','FD_oppMOM__M',
                #'FD_teamMOM__L','FD_oppMOM__L',
                "elo_team",
                "elo_opp",
                #'elo_team_opp'
                "elo_diff",
            ]

        elif devVersion in ["D"]:
            # --- Numerical Factors:
            factors_CS = [
                "points_diff",
                "goalsscored_cum_team",
                "goalsscored_cum_opp",
                "goalsconceded_cum_team",
                "goalsconceded_cum_opp",
                #'teamMOM__S','oppMOM__S',
                "MOM__S__diff",
                #'teamMOM__M','oppMOM__M',
                #'teamMOM__L','oppMOM__L',
                #'teamMOM__S_L', 'oppMOM__S_L',
                #'FD_teamMOM__S','FD_oppMOM__S',
                #'FD_teamMOM__M','FD_oppMOM__M',
                #'FD_teamMOM__L','FD_oppMOM__L',
                "elo_team",
                "elo_opp",
                #'elo_team_opp'
                "elo_diff",
            ]

        elif devVersion in ["E"]:
            # --- Numerical Factors:
            factors_CS = [
                "points_diff",
                "goalsscored_cum_team",
                "goalsscored_cum_opp",
                "goalsconceded_cum_team",
                "goalsconceded_cum_opp",
                "teamMOM__S",
                "oppMOM__S",
                #'MOM__S__diff',
                #'teamMOM__M','oppMOM__M',
                #'teamMOM__L','oppMOM__L',
                #'teamMOM__S_L', 'oppMOM__S_L',
                "FD_teamMOM__S",
                "FD_oppMOM__S",
                #'FD_teamMOM__M','FD_oppMOM__M',
                #'FD_teamMOM__L','FD_oppMOM__L',
                "elo_team",
                "elo_opp",
                "elo_team_opp",
                #'elo_diff'
            ]

        elif devVersion in ["F"]:
            # --- Numerical Factors:
            factors_CS = [
                "points_diff",
                "goalsscored_cum_team",
                "goalsscored_cum_opp",
                "goalsconceded_cum_team",
                "goalsconceded_cum_opp",
                #'teamMOM__S','oppMOM__S',
                "MOM__S__diff",
                #'teamMOM__M','oppMOM__M',
                #'teamMOM__L','oppMOM__L',
                #'teamMOM__S_L', 'oppMOM__S_L',
                "FD_teamMOM__S",
                "FD_oppMOM__S",
                #'FD_teamMOM__M','FD_oppMOM__M',
                #'FD_teamMOM__L','FD_oppMOM__L',
                "elo_team",
                "elo_opp",
                "elo_team_opp",
                #'elo_diff'
            ]

        elif devVersion in ["G"]:
            # --- Numerical Factors:
            factors_CS = [
                "points_diff",
                "goalsscored_cum_team",
                "goalsscored_cum_opp",
                "goalsconceded_cum_team",
                "goalsconceded_cum_opp",
                #'teamMOM__S','oppMOM__S',
                "MOM__S__diff",
                #'teamMOM__M','oppMOM__M',
                #'teamMOM__L','oppMOM__L',
                #'teamMOM__S_L', 'oppMOM__S_L',
                "FD_teamMOM__S",
                "FD_oppMOM__S",
                #'FD_teamMOM__M','FD_oppMOM__M',
                #'FD_teamMOM__L','FD_oppMOM__L',
                "elo_team",
                "elo_opp",
                #'elo_team_opp'
                "elo_diff",
            ]

        elif devVersion in ["H"]:
            # --- Numerical Factors:
            factors_CS = [
                "points_diff",
                "goalsscored_cum_team",
                "goalsscored_cum_opp",
                "goalsconceded_cum_team",
                "goalsconceded_cum_opp",
                #'teamMOM__S','oppMOM__S',
                #'MOM__S__diff',
                #'teamMOM__M','oppMOM__M',
                #'teamMOM__L','oppMOM__L',
                #'teamMOM__S_L', 'oppMOM__S_L',
                #'FD_teamMOM__S','FD_oppMOM__S',
                #'FD_teamMOM__M','FD_oppMOM__M',
                #'FD_teamMOM__L','FD_oppMOM__L',
                "elo_team",
                "elo_opp",
                #'elo_team_opp'
                "elo_diff",
            ]

        elif devVersion in ["I"]:
            # --- Numerical Factors:
            factors_CS = ["points_diff", "elo_team", "elo_opp", "elo_team_opp"]

        elif devVersion in ["J"]:
            # --- Numerical Factors:
            factors_CS = [
                "points_diff",
                "goalsscored_cum_team",
                "goalsscored_cum_opp",
                "goalsconceded_cum_team",
                "goalsconceded_cum_opp",
                #'teamMOM__S','oppMOM__S',
                #'MOM__S__diff',
                #'teamMOM__M','oppMOM__M',
                #'teamMOM__L','oppMOM__L',
                #'teamMOM__S_L', 'oppMOM__S_L',
                #'FD_teamMOM__S','FD_oppMOM__S',
                #'FD_teamMOM__M','FD_oppMOM__M',
                #'FD_teamMOM__L','FD_oppMOM__L',
                "elo_team",
                "elo_opp",
                "elo_team_opp",
                #'elo_diff'
            ]

        elif devVersion in ["K"]:
            # --- Numerical Factors:
            factors_CS = [
                "points_diff",
                "goalsscored_cum_team",
                "goalsscored_cum_opp",
                "goalsconceded_cum_team",
                "goalsconceded_cum_opp",
                #'teamMOM__S','oppMOM__S',
                #'MOM__S__diff',
                #'teamMOM__M','oppMOM__M',
                #'teamMOM__L','oppMOM__L',
                #'teamMOM__S_L', 'oppMOM__S_L',
                #'FD_teamMOM__S','FD_oppMOM__S',
                #'FD_teamMOM__M','FD_oppMOM__M',
                #'FD_teamMOM__L','FD_oppMOM__L',
                "elo_team",
                "elo_opp",
                #'elo_team_opp'
                #'elo_diff'
            ]

        # --- Other Factors:
        other_factors = ["home_pitch"]

        # --- Concatenate:
        factors = other_factors + factors_CS

        # ======================================== Target & ID vars ======================================== #

        IDvar = [
            "id_match",
            "name_team",
            "name_opp",
            "name_league",
            "id_league",
            "season",
            "gameday",
            "kick_off",
        ]
        Yvar = "match_outcome"

        # ======================================== Some Data Preprocessing ======================================== #

        # --- 0.01 Winsorize the Number of Goals in a Game:
        complete_data["match_outcome__orig"] = complete_data["match_outcome"].copy()
        complete_data["match_outcome"] = np.where(
            complete_data["match_outcome"] > 7, 7, complete_data["match_outcome"]
        )

        # --- 0.02 Special Treatment:
        complete_data[factors] = complete_data[factors].fillna(0)

        # --- 0.03 Special Treatment:
        complete_data = complete_data[IDvar + [Yvar] + factors].dropna().reset_index(drop=True)

        # --- 0.1 Validation Set:
        data_oos = complete_data.loc[complete_data["season"].isin(val_seasons), :].dropna()
        data_oos = data_oos.loc[
            data_oos["name_league"].isin(
                complete_data.loc[complete_data["season"].isin(train_end), "name_league"]
            ),
            :,
        ]

        # --- 0.2 Get the Training-Data only:
        complete_data = complete_data.loc[complete_data["season"].isin(train_end), :].dropna()

        # --- 0.3 Kick the first Season by League:
        complete_data = complete_data.loc[
            complete_data.groupby("name_league")["season"].transform("min")
            != complete_data["season"],
            :,
        ]

        # --- 0.4 Some Type-Setting:
        complete_data["kick_off"] = pd.to_datetime(
            complete_data["kick_off"], yearfirst=True
        ).dt.normalize()

        # --- 0.5 Final Sorting for Convenience:
        complete_data = complete_data.sort_values(["name_league", "kick_off"]).reset_index(
            drop=True
        )

        # ======================================== Cross-Sectional Standardization (by Gameday) ======================================== #

        if do__scaleCS:
            # --- For future use:
            # train_means = complete_data.groupby('gameday')[factors_CS].mean()
            # train_stds  = complete_data.groupby('gameday')[factors_CS].std()
            train_means = complete_data.groupby(["season", "id_league", "gameday"])[
                factors_CS
            ].mean()
            train_stds = complete_data.groupby(["season", "id_league", "gameday"])[factors_CS].std()

            # --- Conduct actual scaling:
            # data__scaleCS = complete_data.groupby('gameday')[factors_CS].apply(lambda x: (x - x.mean()) / x.std()).reset_index().set_index('level_1').drop('gameday',axis=1)
            data__scaleCS = (
                complete_data.groupby(["season", "id_league", "gameday"])[factors_CS]
                .apply(_cs_scale)
                .reset_index()
                .set_index("level_3")
                .drop(["season", "id_league", "gameday"], axis=1)
            )

            print("\nScaling data cross-sectionally!\n")

            # --- Merge:
            complete_data[factors_CS] = data__scaleCS

        # --- Drop NA:
        complete_data = complete_data.dropna()

        # ======================================== 1.1 Factor Standardization ======================================== #

        if not do__scaleCS:
            factors_CS_train = complete_data[factors_CS].copy()

            # --- Do the Standardization
            scaler = StandardScaler()
            factors_CS_sdz = pd.DataFrame(
                scaler.fit_transform(factors_CS_train), columns=factors_CS
            )

            # --- Add the non-numeric factor to the standardized DataFrame
            factors_sdz = factors_CS_sdz.copy()
            factors_sdz[other_factors] = complete_data[other_factors].copy()

            # --- Ensure that the order is the same as the PyMC coords later on
            factors_sdz = factors_sdz[factors]

            print("\nScaling data across the Whole Sample!\n")

        else:
            # --- Standardization already done! Ensure that the order is the same as the PyMC coords later on:
            factors_sdz = complete_data[factors].copy()

            # print('\nData already scaled cross-sectionally!\n')

        # --- No Home-Pitch Effect at World Cups:
        factors_sdz.loc[complete_data["id_league"] == "WM", "home_pitch"] = 0

        # =========================== Changing Variables for Train- & Test-Set =========================== #

        # --- Teams:
        names_teams = sorted(set(complete_data["name_team"]).union(complete_data["name_opp"]))
        team_to_idx = {t: i for i, t in enumerate(names_teams)}

        # --- Home & Away Team Indices:
        idx_home = complete_data[complete_data["home_pitch"] == 1].index
        idx_away = complete_data[complete_data["home_pitch"] == 0].index

        # --- Leagues:
        names_leagues = sorted(set(complete_data["name_league"]))
        league_to_idx = {l: i for i, l in enumerate(names_leagues)}

        # --- Global Factors
        factors_g = [f for f in factors if f != "home_pitch"]

        # --- Set the Coords:
        COORDS = {
            "factor_g": factors_g,
            "obs_id": complete_data.index,
            "outcome_categories": [len(np.unique(complete_data[Yvar])) - 1],
            "teams": names_teams,
            "leagues": names_leagues,
        }

        # ========================================= The Team-Factors ========================================= #

        with pm.Model(coords=COORDS) as SFMMO__dev:
            # --- Set the Data:
            X_gf = pm.Data(
                "X_gf", factors_sdz[factors_g].copy().to_numpy(), dims=("obs_id", "factor_g")
            )
            X_home = pm.Data(
                "X_home", factors_sdz["home_pitch"].copy().to_numpy().astype(int), dims="obs_id"
            )
            Y = pm.Data("Y", complete_data[Yvar].copy().to_numpy(), dims="obs_id")
            idx_team = pm.Data(
                "idx_team", complete_data["name_team"].map(team_to_idx).to_numpy(), dims="obs_id"
            )
            idx_opp = pm.Data(
                "idx_opp", complete_data["name_opp"].map(team_to_idx).to_numpy(), dims="obs_id"
            )
            idx_home = pm.Data(
                "idx_home", (complete_data["home_pitch"] == 1).to_numpy(), dims="obs_id"
            )
            idx_league = pm.Data(
                "idx_league",
                complete_data["name_league"].map(league_to_idx).to_numpy(),
                dims="obs_id",
            )

            # --- Set the Model Priors:

            if devVersion in ["Z"]:
                # --- BART:
                gX = pmb.BART("gX", X, Y, m=50, shape=(2, "obs_id"))

            else:
                # --- Global intercept on the goals scale:  exp(mu) ≈ baseline goals per team
                mu = pm.Normal("mu", mu=np.log(1.4), sigma=0.3)

                # IMPROVEMENT (patch 3): tightened team-effect prior (prior-predictive check showed the
                # Gamma(2,4) x ZSN(1) scale implied absurd goals: q99.9~113, max~9000).
                # Replaces the Gamma-hyperprior x raw construction for both attack (alpha) and
                # defence (delta) with a fixed-scale ZeroSumNormal(sigma=0.30).
                # --- Team-Level Home-Fixed-Effect:
                alpha = pm.ZeroSumNormal("alpha", sigma=0.30, dims="teams")

                # --- Team-Level Away-Fixed-Effect:
                delta = pm.ZeroSumNormal("delta", sigma=0.30, dims="teams")

                # --- Team-Level Home Advantage:
                mu_gamma = pm.Normal("mu_gamma", mu=0.30, sigma=0.20)
                sigma_gamma = pm.Gamma(
                    "sigma_gamma", alpha=2, beta=20
                )  # pm.HalfNormal("sigma_gamma", sigma=0.10)
                # --- --- Non-centered for sampling stability:
                gamma_raw = pm.Normal("gamma_raw", 0, 1, dims="teams")
                beta_home = pm.Deterministic(
                    "beta_home",
                    mu_gamma + gamma_raw * sigma_gamma,
                    dims="teams",
                )

                # --- League Fixed Effect:
                kappa = pm.ZeroSumNormal("kappa", sigma=0.3, dims="leagues")
                # sigma_alpha_league = pm.Gamma("sigma_alpha_league", alpha=2, beta=4, dims="leagues")

                # --- Factors are pooled across teams
                beta = pm.Normal("beta", mu=0, sigma=0.1, dims="factor_g")

        # ==================== Set the Model: Bayesian Time-Series Regression (Classificatuion) ==================== #

        with SFMMO__dev:
            # --- Conditional Mean:
            # eta = pm.Deterministic('eta',alpha[idx_team] - delta[idx_opp] + kappa[idx_league] + X_home * beta_home[idx_team] + pt.dot(X_gf, beta.T))
            eta = pm.Deterministic(
                "eta",
                mu
                + alpha[idx_team]
                - delta[idx_opp]
                + kappa[idx_league]
                + X_home * beta_home[idx_team]
                + pt.dot(X_gf, beta.T),
            )

            if 1 == 2:
                # --- Variance:
                scale = pm.Gamma("scale", alpha=20, beta=0.1)

                # --- Likelihood:
                pm.NegativeBinomial(
                    "match_outcome", mu=pm.math.exp(eta), alpha=scale, observed=Y, dims="obs_id"
                )

            else:
                # --- Likelihood:
                pm.Poisson("match_outcome", mu=pm.math.exp(eta), observed=Y, dims="obs_id")

        # ============================================= Inference! ============================================= #

        # ============================== Inference (nutpie) ==============================
        # IMPROVEMENT (Bayesian workflow): nutpie sampler. Do NOT hardcode the chain count for
        # the real run -- let nutpie pick the platform's multi-chain default (smoke keeps
        # chains=2 only as a fast dev check). Descriptive, fixed seed for reproducibility.
        # The tightened fixed-scale team priors removed the hierarchical funnel, so target_accept
        # comes down from 0.99 to 0.9 (Alex's June review saw 0 divergences at 0.9).
        N_draws = 200 if SMOKE else 4000
        N_tune = 200 if SMOKE else 1000
        RANDOM_SEED = sum(map(ord, "sfmmo"))

        with SFMMO__dev:
            if devVersion in ["Z"]:
                idata = pm.sample(draws=N_draws, tune=N_tune, cores=4, random_seed=RANDOM_SEED)
            else:
                sample_kwargs = dict(
                    nuts_sampler="nutpie",
                    target_accept=0.9,
                    draws=N_draws,
                    tune=N_tune,
                    random_seed=RANDOM_SEED,
                )
                if SMOKE:
                    sample_kwargs["chains"] = 2  # dev-only fast check (still multi-chain)
                idata = pm.sample(**sample_kwargs)

            # nutpie silently ignores idata_kwargs for log_likelihood / log_prior, so compute
            # them explicitly: log_likelihood -> LOO, log_prior -> prior-sensitivity (psense).
            try:
                pm.compute_log_likelihood(idata, model=SFMMO__dev)
                pm.compute_log_prior(idata, model=SFMMO__dev)
            except Exception as _e:
                print(f"[WARN] compute_log_likelihood/log_prior failed: {_e}")

        # IMPROVEMENT (Bayesian workflow): arviz_stats.diagnose() is the first diagnostic --
        # R-hat, ESS, divergences, tree-depth and E-BFMI in a single call.
        if arviz_stats is not None:
            try:
                print(f"\n[diagnose fold={val_seasons[0]}]")
                arviz_stats.diagnose(idata)
            except Exception as _e:
                print(f"[WARN] arviz_stats.diagnose failed: {_e}")
        try:
            rhat_ds = az.rhat(idata)
            max_rhat = float(max(float(rhat_ds[v].max()) for v in rhat_ds.data_vars))
        except Exception as _e:
            max_rhat = float("nan")
            print(f"[WARN] could not compute R-hat: {_e}")
        try:
            n_divergences = int(idata.sample_stats.diverging.sum())
        except Exception as _e:
            n_divergences = -1
            print(f"[WARN] could not read divergences: {_e}")
        try:
            n_chains = int(idata.posterior.sizes["chain"])
        except Exception:
            n_chains = -1
        print(
            f"\n[DIAGNOSTICS fold={val_seasons[0]}] max R-hat = {max_rhat:.4f} | "
            f"divergences = {n_divergences} | chains = {n_chains}\n"
        )
        diagnostics[val_seasons[0]] = {
            "max_rhat": max_rhat,
            "n_divergences": n_divergences,
            "n_chains": n_chains,
            "n_draws": int(N_draws),
            "n_tune": int(N_tune),
        }

        # ============================= Some Preparation for Post-Estimation Processing ============================= #

        # NOTE (version fix): the notebook added an in-sample posterior-predictive via
        # `idata.extend(pm.sample_posterior_predictive(idata))`. That arviz-0.x API is gone
        # in this stack (InferenceData is a DataTree, no `.extend`), and the in-sample PPC is
        # NOT used for the OOS predictions below — those come from
        # sample_posterior_predictive(predictions=True) on `eta_oos`. It only fed the
        # notebook's unported in-sample rootogram diagnostics, so it is skipped.

        if data_oos.shape[0] == 0:
            print("\n[NOTE]: No OOS data available ... could be intentional though !")
            break

        # ===================================== OOS-Data: Preparation ===================================== #

        # --- 0.1 Some Type-Setting:
        data_oos["kick_off"] = pd.to_datetime(data_oos["kick_off"], yearfirst=True).dt.normalize()

        # --- 0.2 Final Sorting for Convenience:
        data_oos = data_oos.sort_values(["name_league", "kick_off"]).reset_index(drop=True)

        if do__scaleCS:
            data__scaleCS = (
                data_oos.groupby(["season", "id_league", "gameday"])[factors_CS]
                .apply(_cs_scale)
                .reset_index()
                .set_index("level_3")
                .drop(["season", "id_league", "gameday"], axis=1)
            )

            # --- Merge:
            data_oos[factors_CS] = data__scaleCS

            # data_oos[factors_CS] = data_oos.apply(lambda row: (row[factors_CS] - train_means.loc[row['gameday']]) / train_stds.loc[row['gameday']], axis=1)

            print("\nScaling Data Cross-Sectionally!\n")

        else:
            # --- Standardization already done! Ensure that the order is the same as the PyMC coords later on:
            print("\nNo Sclaing Applied!\n")

        factors_sdz__oos = data_oos[factors].copy()

        # --- No Home-Pitch Effect at World Cups:
        factors_sdz__oos.loc[data_oos["id_league"] == "WM", "home_pitch"] = 0

        # ================================== Out-of-Sample Prediction ================================== #

        # --- Identify the Teams:
        teams_in_oos = set(data_oos["name_team"]).union(data_oos["name_opp"])
        truly_new_teams = sorted(t for t in teams_in_oos if t not in names_teams)

        # --- Combined coord: training teams first, then new teams
        names_teams_all = list(names_teams) + truly_new_teams
        team_to_idx_all = {t: i for i, t in enumerate(names_teams_all)}

        n_train = len(names_teams)
        n_new = len(truly_new_teams)

        # --- Identify the Leagues (sanity check: any leagues unseen in training?)
        leagues_in_oos = sorted(set(data_oos["name_league"]))
        unknown_leagues = [l for l in leagues_in_oos if l not in league_to_idx]
        if unknown_leagues:
            raise ValueError(
                f"OOS data contains leagues not seen in training: {unknown_leagues}. "
                f"You'll need to handle these explicitly (e.g., kappa__new with a prior)."
            )

        # ---------------------------- Update the Model ---------------------------- #
        with SFMMO__dev:
            # New coords
            if n_new > 0:
                SFMMO__dev.add_coord("team__new", truly_new_teams)
            SFMMO__dev.add_coord("obs_id__oos", data_oos.index)

            # OOS data containers
            X_gf_oos = pm.Data(
                "X_gf_oos",
                factors_sdz__oos.loc[data_oos.index, factors_g].to_numpy(),
                dims=("obs_id__oos", "factor_g"),
            )
            X_home_oos = pm.Data(
                "X_home_oos",
                factors_sdz__oos.loc[data_oos.index, "home_pitch"].to_numpy().astype(int),
                dims="obs_id__oos",
            )
            idx_team_oos = pm.Data(
                "idx_team_oos",
                data_oos["name_team"].map(team_to_idx_all).to_numpy(),
                dims="obs_id__oos",
            )
            idx_opp_oos = pm.Data(
                "idx_opp_oos",
                data_oos["name_opp"].map(team_to_idx_all).to_numpy(),
                dims="obs_id__oos",
            )
            idx_home_oos = pm.Data(
                "idx_home_oos",
                (data_oos["home_pitch"] == 1).to_numpy().astype(int),
                dims="obs_id__oos",
            )
            idx_league_oos = pm.Data(
                "idx_league_oos",  # ← NEW
                data_oos["name_league"].map(league_to_idx).to_numpy(),
                dims="obs_id__oos",
            )
            Y_oos = pm.Data("Y_oos", data_oos[Yvar].to_numpy().astype(int), dims="obs_id__oos")

            # ============ New-team parameters (only if any) ============
            if n_new > 0:
                # Reference shared hyperparameters and globals from training
                # RUN-BLOCKER (consequence of patch 3): the original referenced
                # SFMMO__dev.sigma_alpha / .sigma_delta as the new-team prior sigma.
                # Patch 3 removed those Gamma hyperpriors, so we substitute the fixed
                # 0.30 scale now used for alpha/delta (same value the trained team
                # effects carry). The 25th-pct cold-start anchor below is UNCHANGED.
                sigma_alpha_hp = 0.30
                sigma_delta_hp = 0.30
                mu_gamma_hp = SFMMO__dev.mu_gamma
                sigma_gamma_hp = SFMMO__dev.sigma_gamma

                # Empirical anchors for new teams (25th-pct heuristic; consider promoted-team mean instead)
                alpha_anchor = (
                    idata["posterior"]["alpha"]
                    .stack(samples=("chain", "draw"))
                    .median("samples")
                    .quantile(0.25)
                    .to_numpy()
                )
                delta_anchor = (
                    idata["posterior"]["delta"]
                    .stack(samples=("chain", "draw"))
                    .median("samples")
                    .quantile(0.25)
                    .to_numpy()
                )

                alpha__new = pm.Normal(
                    "alpha__new", mu=alpha_anchor, sigma=sigma_alpha_hp, dims="team__new"
                )
                delta__new = pm.Normal(
                    "delta__new", mu=delta_anchor, sigma=sigma_delta_hp, dims="team__new"
                )
                gamma_raw__new = pm.Normal("gamma_raw__new", 0, 1, dims="team__new")
                beta_home__new = pm.Deterministic(
                    "beta_home__new",
                    mu_gamma_hp + gamma_raw__new * sigma_gamma_hp,
                    dims="team__new",
                )

                # Combined parameter vectors via concatenation
                alpha_all = pt.concatenate([alpha, alpha__new])
                delta_all = pt.concatenate([delta, delta__new])
                beta_home_all = pt.concatenate([beta_home, beta_home__new])
            else:
                # No new teams this OOS window — combined is just the trained
                alpha_all = alpha
                delta_all = delta
                beta_home_all = beta_home

            # ============ Single OOS linear predictor and likelihood ============
            eta_oos = pm.Deterministic(
                "eta_oos",
                mu
                + alpha_all[idx_team_oos]
                - delta_all[idx_opp_oos]
                + kappa[idx_league_oos]
                + X_home_oos * beta_home_all[idx_team_oos]
                + pt.dot(X_gf_oos, beta),
                dims="obs_id__oos",
            )

            if 1 == 2:
                pm.NegativeBinomial(
                    "match_outcome_oos",
                    mu=pm.math.exp(eta_oos),
                    alpha=scale,
                    observed=Y_oos,
                    dims="obs_id__oos",
                )
            else:
                pm.Poisson(
                    "match_outcome_oos",
                    mu=pm.math.exp(eta_oos),
                    observed=Y_oos,
                    dims="obs_id__oos",
                )

        # ============ One prediction call for everything ============
        with SFMMO__dev:
            oos_preds = pm.sample_posterior_predictive(
                idata,
                predictions=True,
                var_names=["eta_oos", "match_outcome_oos"]
                + (["alpha__new", "delta__new", "beta_home__new"] if n_new > 0 else []),
                compile_kwargs={"mode": "NUMBA"},
            )

        # ===================================== Collect Results: Match-Up Goals Matrix ===================================== #

        # --- Assemble the Match-IDs and corresponding Match-Ups:
        Y__eval = (
            data_oos[["id_match", "name_team", "name_opp", "home_pitch", "kick_off"] + [Yvar]]
            .copy()
            .reset_index(names="index__data_oos")
        )

        # --- Get Home & Away Indices within the Samples:
        idxSamples_home = (
            Y__eval.loc[Y__eval["home_pitch"] == 1, "id_match"].reset_index().set_index("id_match")
        )
        idxSamples_away = (
            Y__eval.loc[Y__eval["home_pitch"] == 0, "id_match"].reset_index().set_index("id_match")
        )

        idxSamples = pd.merge(
            idxSamples_home,
            idxSamples_away,
            left_index=True,
            right_index=True,
            suffixes=("__home", "__away"),
        )
        idxSamples_home = idxSamples["index__home"].values
        idxSamples_away = idxSamples["index__away"].values

        # --- Get the Joint Posterior of the Goal-Matrix
        # IMPROVEMENT (patch 4): k_max 5 -> 15 (removes the probability-mass truncation on extreme-lambda fixtures).
        jointPMF = get__perMatch_jointPMF(
            eta=oos_preds["predictions"]["eta_oos"],
            # scale=idata['posterior']['scale'],
            idx_home=idxSamples_home,
            idx_away=idxSamples_away,
            k_max=15,
        )

        # ===================================== Evaluate Predictions ===================================== #

        Y__SFMMO = pd.DataFrame(columns=["id_match", "match_outcome"])
        Yhat = pd.DataFrame(columns=["0", "1", "2"])

        for m in tqdm(range(len(idxSamples_home))):
            # --- Observed Outcome:
            m_Y = (
                Y__eval.loc[(idxSamples_home[m], idxSamples_away[m]), :]
                .copy()
                .sort_values("home_pitch", ascending=False)
            )
            m_Y__outcome = (
                2
                if m_Y["match_outcome"].diff().values[1] < 0
                else 0
                if m_Y["match_outcome"].diff().values[1] > 0
                else 1
            )
            Y__SFMMO = pd.concat(
                [
                    Y__SFMMO,
                    pd.DataFrame(
                        {"id_match": [m_Y["id_match"].iloc[0]], "match_outcome": [m_Y__outcome]}
                    ),
                ],
                axis=0,
            ).reset_index(drop=True)

            # --- Predicted Probabilities
            m_Yhat = get__home_WDL(jointPMF[m, :, :, :])["mid"].iloc[::-1].values
            Yhat = pd.concat(
                [Yhat, pd.DataFrame(m_Yhat, index=["0", "1", "2"]).T], axis=0
            ).reset_index(drop=True)

        dict_preds[val_seasons[0]] = {"Yhat": Yhat, "Y__SFM": Y__SFMMO}

        # ====================================== For In-Sample Evaluation / Sampling Evaluation ====================================== #
        idx_home = (
            complete_data.loc[complete_data["home_pitch"] == 1, "id_match"]
            .reset_index()
            .set_index("id_match")
        )
        idx_away = (
            complete_data.loc[complete_data["home_pitch"] == 0, "id_match"]
            .reset_index()
            .set_index("id_match")
        )

        idx_lookup = pd.merge(
            idx_home, idx_away, left_index=True, right_index=True, suffixes=("_home", "_away")
        )

        dict_fitEval[val_seasons[0]] = {"idata": idata, "idx_lookup": idx_lookup}

        # IMPROVEMENT (patch 2): persist the per-fold InferenceData next to the pickle
        # (e.g. ..._idata_WMQ2026.nc) for auditable diagnostics.
        try:
            idata_path = os.path.join(
                OUT_DIR,
                f"Evaluation__SFMMOwm_Dev{devVersion}__scaleCS__EW__improved{_VARIANT}_idata_{val_seasons[0]}.nc",
            )
            idata.to_netcdf(idata_path)
            print(f"[SAVE] idata -> {idata_path}")
        except Exception as _e:
            print(f"[WARN] could not save idata for fold {val_seasons[0]}: {_e}")

    # ================================== Export ================================== #
    # IMPROVEMENT (patch 6, RUN-BLOCKER): the notebook gated this save behind
    # `if 1==2:` (cells 16 & 17, disabled) so nothing was ever written. Re-enabled
    # as an unconditional save of the IMPROVED DevK pickle, under a NEW filename so
    # it does NOT overwrite the committed ...DevK__scaleCS__EW.pkl.
    import cloudpickle

    if do__scaleCS:
        scale__type = "_scaleCS"  # --- cross-sectional
    else:
        scale__type = "_scaleWS"  # --- whole-sample

    # PATH + IMPROVEMENT (patch 2): new "__improved" filename in 10_data/102_Development/.
    # The verbatim string `Evaluation__SFMMOwm_Dev{devVersion}_{scale__type}__EW`
    # with scale__type='_scaleCS' reproduces the eval's `...DevK__scaleCS__EW` stem;
    # we append `__improved` to avoid clobbering the committed file.
    pickle_filepath = os.path.join(
        OUT_DIR, f"Evaluation__SFMMOwm_Dev{devVersion}_{scale__type}__EW__improved{_VARIANT}.pkl"
    )
    dict_to_save = {"factors": factors, "dict_preds": dict_preds}

    with open(pickle_filepath, "wb") as f:
        cloudpickle.dump(dict_to_save, f)

    print(f"\n[SAVE] predictions pickle -> {pickle_filepath}")

    # IMPROVEMENT (patch 2): also dump a small diagnostics json (max R-hat + divergence count per fold).
    diag_path = os.path.join(
        OUT_DIR,
        f"Evaluation__SFMMOwm_Dev{devVersion}_{scale__type}__EW__improved{_VARIANT}_diagnostics.json",
    )
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"[SAVE] diagnostics json -> {diag_path}")

    print(f"\nFine. Version: {devVersion}\n\n")

    return dict_preds, dict_fitEval, diagnostics


if __name__ == "__main__":
    main()
