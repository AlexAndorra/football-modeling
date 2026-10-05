#!/usr/bin/env python
# coding: utf-8
"""
==========================================================================================
 006_060 — Weekly Match-Outcome Forecasts for the LEAGUE SFMMO
==========================================================================================

League sibling of `006_050__Predictions_MatchOutcome__SFMMOwm.py` (the World-Cup script).
Same architecture, proven over the WC campaign: **fit once, predict many.**

    SEASON BUNDLE (fitted once, on GPU, by SFMMO__dev_EW.ipynb with FIT_PRODUCTION=True)
        -> 10_data/01_Models/SFMMO_DevK__scaleCS__train<YYYYYY>__PROD.pkl
    THIS SCRIPT (weekly, on CPU, seconds)
        -> re-runs feature engineering on the UPDATED data (results roll in -> ELO moves),
           pushes the upcoming fixtures through the model's OOS graph, applies the
           Dixon-Coles correction, and writes the website/app feed.

The posterior never moves during the season. That is not a shortcut: it is exactly the
protocol the expanding-window validation measured (train through season t, predict season
t+1 with features rolling and parameters frozen), so the published error bars mean what
they say.

WHAT IT DOES
------------
1.  Loads the season bundle (posterior draws, model graph, team index, CS-scaling moments,
    Dixon-Coles rho).
2.  Loads the byPlayer data (history + the upcoming season's fixtures), reduces to one row
    per team-match, re-derives gameday and ELO over the FULL history so upcoming fixtures
    carry current ratings.
3.  Slices the unplayed fixtures, cross-sectionally standardises them with the BUNDLE's
    training moments (never with OOS moments -- that would leak), applies the M1
    missing-factor policy (undefined form -> league average in standardized space).
4.  Extends the model graph with the OOS containers (new/promoted teams get the anchored
    priors) and samples the posterior predictive.
5.  **ETA PARITY GATE** -- reconstructs eta in NumPy exactly as a downstream consumer would
    and asserts it equals the graph's eta_oos. This is the check that would have caught the
    WC `mu` bug on day one; nothing is exported if it fails.
6.  Builds the joint scoreline PMF (k_max=15), applies Dixon-Coles tau with the bundle's
    rho, and derives W/D/L with credible bands, expected goals and the most likely score.
7.  Archives the previous board into `_vintages/` BEFORE overwriting, then writes the feed.

USAGE
-----
    python 006_060__Predictions_MatchOutcome__SFMMO.py

Set TARGET_SEASON to the season being forecast. Everything else is read from the bundle.
==========================================================================================
"""

import os
import re
import shutil
import pickle
import cloudpickle
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import poisson

import pymc as pm
import pytensor.tensor as pt


# ============================================================================= #
#                              USER INTERACTION                                  #
# ============================================================================= #

directory = '/Users/maximilian/Dropbox/Max/51_SoccerAnalytics'

BUNDLE_PATH   = f'{directory}/10_data/01_Models/SFMMO_DevK__scaleCS__train202526__PROD.pkl'
HIST_PATH     = f'{directory}/10_data/106_Website/data_byPlayer.csv'       # played history
# NOTE: the weekly pipeline refreshes the NON-TM file; the __TM variant (market values) is
# only rebuilt for model fitting. Model K uses no Transfermarkt columns, and the K features
# are identical across the two files (verified: max|diff| = 0 on 79,845 shared rows).
OOS_PATH      = f'{directory}/10_data/106_Website/data_byPlayer__OOS.csv'          # upcoming fixtures
TARGET_SEASON = '2026/27'          # the season whose unplayed fixtures we forecast
HORIZON_DAYS  = 14                 # forecast fixtures kicking off within this many days.
#   The OOS file carries the WHOLE season (~1,670 fixtures). Forecasting all of them is both
#   wasteful and wrong-headed: a May fixture predicted in August carries no form information,
#   and it would freeze that near-useless number into the ledger. A rolling horizon means each
#   fixture is frozen shortly before kickoff, with maximal information.
#   Chosen DATE-based, not round-based (the La Liga lesson: rounds interleave when matches are
#   postponed). 14 days covers the next matchday plus midweek games, with slack if a weekly run
#   is skipped -- widen it if runs are ever missed, since a fixture that kicks off without ever
#   entering the horizon has no frozen forecast (the run warns loudly if that happens).
#   The window also has a LOWER bound: a fixture that has already kicked off and still has no
#   result is never forecast again (its frozen row stands) -- see select_forecast_window.

K_MAX         = 15                 # scoreline grid (M2: 5 truncated ~11% of joint mass)
CRED_REGION   = 0.90               # credible band for the W/D/L probabilities
USE_DIXON_COLES = True             # apply tau with the bundle's fitted rho

ARCHIVE_VINTAGES = True
OUT_DIR     = f'{directory}/10_data/106_Website'
VINTAGE_DIR = f'{OUT_DIR}/_vintages'

OUT_MATCH_CSV = f'{OUT_DIR}/SFMMO_predictions__matches.csv'
OUT_GRID_CSV  = f'{OUT_DIR}/SFMMO_predictions__scorelines.csv'
OUT_TEAM_CSV  = f'{OUT_DIR}/SFMMO_predictions__team_goals.csv'
OUT_PKL       = f'{OUT_DIR}/SFMMO_predictions__prod.pkl'
FROZEN_LEDGER = f'{OUT_DIR}/SFMMO_predictions__frozen.csv'   # last PRE-MATCH forecast per fixture

# ============================================================================= #


def archive_existing_outputs(paths, vintage_dir=VINTAGE_DIR):
    """Snapshot the previous board before it is overwritten (WC lesson L6: a forecast that
    isn't archived before the next refresh never existed). Tagged with the file's own mtime;
    never clobbers an existing archive."""
    archived = []
    os.makedirs(vintage_dir, exist_ok=True)
    for p in paths:
        if not os.path.exists(p):
            continue
        stem, ext = os.path.splitext(os.path.basename(p))
        mtime = datetime.fromtimestamp(os.path.getmtime(p))
        dest = os.path.join(vintage_dir, f'{stem}__{mtime:%Y-%m-%d}{ext}')
        if os.path.exists(dest):
            dest = os.path.join(vintage_dir, f'{stem}__{mtime:%Y-%m-%d_%H%M%S}{ext}')
        shutil.copy2(p, dest)
        archived.append(os.path.basename(dest))
    return archived


def update_elo(r_home, r_away, result, K=20, home_adv=50):
    """ELO update, home side FIRST (it receives the +50). result: 2 home win, 1 draw, 0 away."""
    exp_home = 1 / (1 + 10 ** ((r_away - r_home - home_adv) / 400))
    s_home = 1.0 if result == 2 else 0.5 if result == 1 else 0.0
    return r_home + K * (s_home - exp_home), r_away + K * ((1 - s_home) - (1 - exp_home))


def mirror_missing_sides(cd):
    """Some upcoming fixtures arrive with only ONE perspective row: promoted teams have no
    players in the universe yet, so their side is absent and the both-sides filter would drop
    the whole fixture (7 of 48 on 2026/27 matchday 1). Each row already carries BOTH sides'
    features (`*_team` and `*_opp`), so the complement is reconstructed by swapping -- EXACT,
    not approximate. Upstream fix: add promoted squads to the player universe."""
    per = cd.groupby('id_match')['home_pitch'].nunique()
    single = per[per == 1].index
    if not len(single):
        return cd
    src = cd[cd['id_match'].isin(single)].copy()
    pairs = [(c, c.replace('_team', '_opp')) for c in cd.columns
             if c.endswith('_team') and c.replace('_team', '_opp') in cd.columns]
    mir = src.copy()
    for a, b in pairs:
        mir[a], mir[b] = src[b].values, src[a].values
    mir['name_team'], mir['name_opp'] = src['name_opp'].values, src['name_team'].values
    mir['home_pitch'] = 1 - src['home_pitch']
    for c in cd.columns:
        if c.endswith('_diff'):
            mir[c] = -src[c].values
    print(f"  [mirror] {len(single)} fixture(s) had a single perspective row (promoted teams "
          f"absent from the player universe) -> complement reconstructed by swapping: "
          f"{sorted(single)[:4]}{' ...' if len(single) > 4 else ''}")
    return pd.concat([cd, mir], ignore_index=True)


def build_match_level(data_raw):
    """byPlayer -> one row per (match, side), the notebook's `complete_data`."""
    keep = ['points_team', 'points_opp', 'goalsscored_inGame_team', 'goalsscored_inGame_opp',
            'goalsscored_cum_team', 'goalsscored_cum_opp', 'goalsconceded_cum_team',
            'goalsconceded_cum_opp', 'home_pitch', 'id_match', 'name_team', 'name_opp',
            'name_league', 'id_league', 'season', 'gameday', 'kick_off', 'points_diff']
    data_raw = data_raw.copy()
    data_raw['kick_off'] = pd.to_datetime(data_raw['kick_off'])
    data_raw = data_raw.sort_values(['name_player', 'season', 'kick_off'])
    # played rows win the dedup, so a stale OOS copy can never revert a finished match
    cd = (data_raw.sort_values('goalsscored_inGame_team', na_position='last', kind='stable')
          .drop_duplicates(subset=['id_match', 'home_pitch'])[keep]
          .copy().sort_values(['name_league', 'kick_off']).reset_index(drop=True))
    cd = mirror_missing_sides(cd)
    cd = cd.loc[cd['id_match'].duplicated(keep=False), :]
    cd['match_outcome'] = cd['goalsscored_inGame_team'].copy()
    cd['gameday'] = [int(float(i.split('_')[1][2:])) for i in cd['id_match'].values]
    return cd


def compute_elo(cd):
    """Per-league ELO over the FULL history. ONE update per match, home side attributed
    (the 2026-08 fix: iterating perspective rows double-updated AND leaked the match's own
    result into the second row's features). Unplayed fixtures take a fake-draw update, so
    upcoming gamedays carry sensible current ratings -- documented WC behaviour."""
    cd[['elo_team', 'elo_opp']] = np.nan
    cd['match_outcome__home'] = 1
    cd.loc[cd['goalsscored_inGame_team'] > cd['goalsscored_inGame_opp'], 'match_outcome__home'] = 2
    cd.loc[cd['goalsscored_inGame_team'] < cd['goalsscored_inGame_opp'], 'match_outcome__home'] = 0

    for ll in cd['name_league'].unique():
        R = {t: 1500.0 for t in cd.loc[cd['name_league'] == ll, 'name_team'].unique()}
        seasons = cd.loc[cd['name_league'] == ll, 'season'].unique().tolist()
        for si, ss in enumerate(seasons):
            sl = cd[(cd['name_league'] == ll) & (cd['season'] == ss)].sort_values('kick_off', kind='stable')
            if si > 0:
                prev = set(cd.loc[(cd['name_league'] == ll) & (cd['season'] == seasons[si - 1]), 'name_team'])
                for t in R:
                    R[t] = R[t] * 0.75 + 1500 * 0.25 if t in prev else 1300.0
            home_rows = sl[sl['home_pitch'] == 1]
            away_idx = sl.loc[sl['home_pitch'] == 0].reset_index().set_index('id_match')['index']
            for r in home_rows.itertuples():
                h, a = r.name_team, r.name_opp
                rh, ra = R[h], R[a]
                cd.loc[r.Index, 'elo_team'] = rh
                cd.loc[r.Index, 'elo_opp'] = ra
                ai = away_idx.get(r.id_match)
                if ai is not None:
                    cd.loc[ai, 'elo_team'] = ra
                    cd.loc[ai, 'elo_opp'] = rh
                R[h], R[a] = update_elo(rh, ra, r.match_outcome__home)
    return cd


def kicked_off_mask(ko, now):
    """True where a fixture has already kicked off: on an earlier day, or today with a confirmed
    time that has passed. An unconfirmed slot is stored as 00:00 local (not a midnight kickoff),
    so a same-day 00:00 stamp cannot be judged and counts as not yet kicked off.
    006_040 carries a copy of this rule; a parity test keeps the two identical."""
    now = pd.Timestamp(now)
    today = now.normalize()
    ko = pd.to_datetime(ko)
    day = ko.dt.normalize()
    timed = ko != day                                   # a real kickoff time, not a 00:00 placeholder
    return (day < today) | ((day == today) & timed & (ko < now))


def season_complete(cd, season):
    """True when every league of `season` has played its full double round-robin: n*(n-1)
    DISTINCT (home, away) fixtures between its n own teams. A play-off guest from the division
    below (two legs, far fewer matches than anyone else) is not one of the n, and a fixture
    listed twice counts once. Once no unplayed row is left, this tells "the season is over"
    (ship the final results) apart from "the fixture feed was not updated" (stop and fix it)."""
    home = cd[(cd['season'] == season) & (cd['home_pitch'] == 1)]
    if not len(home):
        return False
    for _, g in home.groupby('name_league'):
        apps = pd.concat([g['name_team'], g['name_opp']]).value_counts()
        core = apps[apps >= apps.max() / 2].index
        played = g[g['match_outcome'].notna() & g['name_team'].isin(core) & g['name_opp'].isin(core)]
        if len(played[['name_team', 'name_opp']].drop_duplicates()) < len(core) * (len(core) - 1):
            return False
    return True


def select_forecast_window(oos, now, horizon_days):
    """Split the UNPLAYED target-season rows into (window, deferred, stale) and return the horizon.

    stale    : already kicked off but still carries no result (results feed lagging, postponement
               not yet re-dated, ...). NEVER re-forecast: a forecast made after kick-off is not a
               receipt, and by then ELO carries results the pre-match forecast could not see
               (matches that kicked off before or alongside it). Re-forecasting would overwrite
               the genuine pre-match row in the ledger (WC lesson L5). A postponed fixture is
               forecast again as soon as the feed carries its new kick-off.
    window   : anchor .. anchor + horizon_days (day-inclusive) -> forecast in this run
    deferred : beyond the horizon -> forecast by a later run, closer to kickoff; also any row with
               no kick_off yet (it waits for a date; main() warns)

    Anchor = max(today, earliest unplayed) -- Max's rule (64f86df), so an international break
    forecasts the next round instead of exiting empty -- taken over fixtures that have NOT
    kicked off: a stale fixture can neither be re-forecast nor drag the window backwards.
    006_040 uses the same rule, so the two boards cover the same fixtures."""
    now = pd.Timestamp(now)
    today = now.normalize()
    ko = pd.to_datetime(oos['kick_off'])
    day = ko.dt.normalize()
    undated = ko.isna()
    kicked_off = kicked_off_mask(ko, now) & ~undated
    live = ~kicked_off & ~undated
    stale = oos[kicked_off].copy()
    anchor = max(today, day[live].min()) if live.any() else today
    horizon = anchor + pd.Timedelta(days=horizon_days)
    deferred = oos[(live & (day > horizon)) | undated].copy()
    window = oos[live & (day >= anchor) & (day <= horizon)].copy()
    return window, deferred, stale, horizon


def merge_frozen_ledger(led, fresh, key):
    """Ledger update rule: a fixture forecast in THIS run (present in `fresh`) takes the new
    row -- it is still unplayed, so this is new information, still pre-match. Every other ledger
    row is kept verbatim. Refuses duplicate keys on either side: a duplicate would make the
    feed's `.loc[key]` lookup return a Series and ship garbage probabilities without an error."""
    for name, df in (('ledger', led), ('fresh', fresh)):
        dup = df.duplicated(subset=key, keep=False)
        if dup.any():
            raise ValueError(f"duplicate {key} rows in the {name}: "
                             f"{df.loc[dup, key].drop_duplicates().values.tolist()[:3]}")
    kept = led.merge(fresh[key].assign(_refresh=1), on=key, how='left')
    kept = kept[kept['_refresh'].isna()].drop(columns='_refresh')     # NOT refreshed -> verbatim
    return pd.concat([kept, fresh], ignore_index=True)


def stale_feed_rows(stale_home, ledger, key, pcols):
    """Feed rows for fixtures that kicked off without a recorded result (one row per fixture,
    home perspective): shipped as 'upcoming' with the probabilities FROZEN before kickoff, never
    a re-forecast. Returns (rows, missing) -- `missing` names fixtures with no frozen row at all
    (they never entered a run while unplayed; that receipt cannot be reconstructed honestly)."""
    lref = ledger.set_index(key)
    rows, missing = [], []
    for r in stale_home.itertuples():
        k = (r.season, r.name_team, r.name_opp)
        if k not in lref.index:
            missing.append(f"{r.name_team} v {r.name_opp}")
            continue
        rec = dict(id_match=r.id_match, name_league=r.name_league, season=r.season,
                   gameday=r.gameday, kick_off=r.kick_off,
                   home_team=r.name_team, away_team=r.name_opp,
                   home_goals=np.nan, away_goals=np.nan, status='upcoming',
                   elo_home=round(float(getattr(r, 'elo_team', np.nan)), 1),
                   elo_away=round(float(getattr(r, 'elo_opp', np.nan)), 1))
        for c in pcols:
            rec[c] = lref.loc[k, c]
        rec['forecast_frozen_at'] = lref.loc[k, 'forecast_frozen_at']
        rows.append(rec)
    return rows, missing


def carry_forward_board_rows(prev, stale_home):
    """Scoreline grids and team-goal rows for stale fixtures, taken from the PREVIOUS board
    (`prev` = the last run's exported dict) so they do not vanish from those two feeds while the
    result is pending. Matched on the team pair, never on id_match. Returns (grid, team, missing)."""
    ids = {(h, a): i for h, a, i in zip(stale_home['name_team'], stale_home['name_opp'],
                                         stale_home['id_match'])}
    g, t = prev['scorelines'], prev['team_goals']
    grid = g[[(h, a) in ids for h, a in zip(g['home_team'], g['away_team'])]].copy()
    grid['id_match'] = [ids[(h, a)] for h, a in zip(grid['home_team'], grid['away_team'])]
    # the home side's row is (team, opponent, is_home=1); the away side's is the mirror. The
    # return leg (opponent at home) is a DIFFERENT fixture and must not be picked up.
    home_key = [(tm, op) if h == 1 else (op, tm)
                for tm, op, h in zip(t['team'], t['opponent'], t['is_home'])]
    team = t[[k in ids for k in home_key]].copy()
    team['id_match'] = [ids[k] for k in home_key if k in ids]
    found = set(zip(grid['home_team'], grid['away_team']))
    missing = [f"{h} v {a}" for (h, a) in ids if (h, a) not in found]
    return grid, team, sorted(missing)


def fixture_lambdas(eta_da, idx_home, idx_away):
    """Per-fixture home/away scoring rates per posterior draw. (n_fixtures, n_samples)."""
    lam = np.exp(eta_da.stack(samples=('chain', 'draw')).values)     # (n_obs, n_samples)
    return lam[idx_home, :], lam[idx_away, :]


def joint_grid_one(lam_h_s, lam_a_s, k_max=K_MAX):
    """Joint scoreline PMF for ONE fixture: (n_samples, k_max+1, k_max+1).

    Built lazily per fixture on purpose. Materialising all fixtures at once is fine for a
    single matchday (~48) but not for a full season: 1,672 fixtures x 8,000 draws x 16x16
    would be ~27 GB. Per fixture it is ~16 MB."""
    ks = np.arange(k_max + 1)
    pmf_h = poisson.pmf(ks[:, None], mu=lam_h_s[None, :])            # (k+1, n_samples)
    pmf_a = poisson.pmf(ks[:, None], mu=lam_a_s[None, :])
    return np.einsum('hs,as->sha', pmf_h, pmf_a)


def apply_dc_tau(joint_s, lam_h_s, lam_a_s, rho):
    """Dixon-Coles low-score correction for one fixture. Mass-preserving by construction."""
    g = joint_s.copy()
    g[:, 0, 0] *= np.clip(1.0 - lam_h_s * lam_a_s * rho, 1e-12, None)
    g[:, 0, 1] *= np.clip(1.0 + lam_h_s * rho, 1e-12, None)
    g[:, 1, 0] *= np.clip(1.0 + lam_a_s * rho, 1e-12, None)
    g[:, 1, 1] *= (1.0 - rho)
    return g


def wdl_from_grid(g, cred=CRED_REGION):
    """(n_samples, K, K) -> per-outcome mid/low/up. axis -2 = home goals, -1 = away."""
    d = np.arange(g.shape[-1])
    p_draw = g[:, d, d].sum(axis=-1)
    p_away = np.triu(g, k=1).sum(axis=(-2, -1))
    p_home = np.tril(g, k=-1).sum(axis=(-2, -1))
    lo, hi = (1 - cred) / 2, 1 - (1 - cred) / 2
    out = {}
    for lbl, arr in [('home_win', p_home), ('draw', p_draw), ('away_win', p_away)]:
        out[lbl] = (arr.mean(), np.quantile(arr, lo), np.quantile(arr, hi))
    return out


def forecast_fixtures(oos, B, rho, factors_CS, factors_g):
    """Model forecasts for the fixtures in the window -> (df_matches, grid_rows, team_rows,
    eta-parity deviation). Moved verbatim out of main() so that a run with nothing to forecast
    can still rebuild the feed without touching the model."""
    meta, model, idata = B['meta'], B['model'], B['idata']
    team_to_idx, names_teams = B['team_to_idx'], B['names_teams']
    train_means, train_stds = B['train_means'], B['train_stds']

    # keep the RAW ELO ratings for display before standardization overwrites the columns
    oos['elo_home_raw'] = oos['elo_team'].to_numpy()
    oos['elo_away_raw'] = oos['elo_opp'].to_numpy()

    # standardize with the BUNDLE's training moments (never with OOS moments)
    stds_safe = train_stds.replace(0.0, np.nan)
    gd_avail = train_means.index
    oos['_gd_use'] = oos['gameday'].clip(upper=int(gd_avail.max()))
    oos[factors_CS] = oos.apply(
        lambda r: (r[factors_CS] - train_means.loc[r['_gd_use']]) / stds_safe.loc[r['_gd_use']], axis=1)
    oos[factors_CS] = oos[factors_CS].replace([np.inf, -np.inf], np.nan)
    _na = oos[factors_CS].isna()
    if _na.any().any():
        print(f"  [M1] {int(_na.sum().sum())} undefined factor-cells across {int(_na.any(axis=1).sum())} "
              f"rows -> league average (0 in standardized space)")
    oos[factors_CS] = oos[factors_CS].fillna(0.0)
    Xg = oos[factors_g].to_numpy(dtype=float)
    Xh = oos['home_pitch'].to_numpy(dtype=float)

    # ----------------------- 4. OOS graph ----------------------- #
    new_teams = sorted(set(oos['name_team']) | set(oos['name_opp']) - set(names_teams))
    new_teams = [t for t in new_teams if t not in team_to_idx]
    all_teams = list(names_teams) + new_teams
    t2i_all = {t: i for i, t in enumerate(all_teams)}
    n_new = len(new_teams)
    print(f"  promoted/unseen teams: {n_new}{' ' + str(new_teams) if n_new else ''}")

    post = idata.posterior
    with model:
        if n_new:
            model.add_coord("team__new", new_teams)
        model.add_coord("obs_id__oos", oos.index)
        X_gf_oos = pm.Data("X_gf_oos", Xg, dims=("obs_id__oos", "factor_g"))
        X_home_oos = pm.Data("X_home_oos", Xh, dims="obs_id__oos")
        idx_team_oos = pm.Data("idx_team_oos", oos['name_team'].map(t2i_all).to_numpy(), dims="obs_id__oos")
        idx_opp_oos = pm.Data("idx_opp_oos", oos['name_opp'].map(t2i_all).to_numpy(), dims="obs_id__oos")

        if n_new:
            a_anchor = float(post['alpha'].stack(s=('chain', 'draw')).median('s').quantile(0.25))
            d_anchor = float(post['delta'].stack(s=('chain', 'draw')).median('s').quantile(0.25))
            alpha__new = pm.Normal("alpha__new", mu=a_anchor, sigma=0.30, dims="team__new")
            delta__new = pm.Normal("delta__new", mu=d_anchor, sigma=0.30, dims="team__new")
            gr__new = pm.Normal("gamma_raw__new", 0, 1, dims="team__new")
            bh__new = pm.Deterministic("beta_home__new",
                                       model['mu_gamma'] + gr__new * model['sigma_gamma'],
                                       dims="team__new")
            alpha_all = pt.concatenate([model['alpha'], alpha__new])
            delta_all = pt.concatenate([model['delta'], delta__new])
            bhome_all = pt.concatenate([model['beta_home'], bh__new])
        else:
            alpha_all, delta_all, bhome_all = model['alpha'], model['delta'], model['beta_home']

        eta_oos = pm.Deterministic(
            "eta_oos",
            model['mu']
            + alpha_all[idx_team_oos]
            - delta_all[idx_opp_oos]
            + X_home_oos * bhome_all[idx_team_oos]
            + pt.dot(X_gf_oos, model['beta']),
            dims="obs_id__oos")

    with model:
        preds = pm.sample_posterior_predictive(
            idata, predictions=True, random_seed=meta.get('seed', 326),
            var_names=["eta_oos"] + (["alpha__new", "delta__new", "beta_home__new"] if n_new else []))

    # ----------------------- 5. ETA PARITY GATE ----------------------- #
    S = lambda v: post[v].stack(s=('chain', 'draw')).values
    P = preds['predictions']
    SP = lambda v: P[v].stack(s=('chain', 'draw')).values
    mu_s, beta_s = S('mu'), S('beta')
    alpha_s, delta_s, bhome_s = S('alpha'), S('delta'), S('beta_home')
    if n_new:
        alpha_s = np.concatenate([alpha_s, SP('alpha__new')], axis=0)
        delta_s = np.concatenate([delta_s, SP('delta__new')], axis=0)
        bhome_s = np.concatenate([bhome_s, SP('beta_home__new')], axis=0)
    ti = oos['name_team'].map(t2i_all).to_numpy()
    oi = oos['name_opp'].map(t2i_all).to_numpy()
    eta_hat = (mu_s[None, :] + alpha_s[ti] - delta_s[oi]
               + Xh[:, None] * bhome_s[ti] + Xg @ beta_s)
    eta_graph = SP('eta_oos')
    dev = float(np.abs(eta_hat - eta_graph).max())
    print(f"\n[eta parity] max |reconstruction - graph| = {dev:.3e}")
    assert dev < 1e-8, (f"ETA PARITY FAILED ({dev:.3e}): the NumPy reconstruction disagrees with "
                        f"the model graph — a term is missing from one of them (the mu-bug class). "
                        f"NOTHING EXPORTED.")
    print("[eta parity] PASS — safe to export.")

    # ----------------------- 6. scorelines + W/D/L ----------------------- #
    pair = (oos.reset_index().pivot_table(index='id_match', columns='home_pitch',
                                          values='index', aggfunc='first')
            .rename(columns={1: 'pos_home', 0: 'pos_away'}).dropna().astype(int))
    idx_home = pair['pos_home'].to_numpy()
    idx_away = pair['pos_away'].to_numpy()
    lam_h, lam_a = fixture_lambdas(preds['predictions']['eta_oos'], idx_home, idx_away)
    print(f"  building {len(pair)} joint grids lazily ({lam_h.shape[1]:,} draws, "
          f"{K_MAX+1}x{K_MAX+1} each) ...")

    rows, grid_rows, team_rows = [], [], []
    q_lo, q_hi = (1 - CRED_REGION) / 2, 1 - (1 - CRED_REGION) / 2
    meta_h = oos.loc[idx_home].reset_index(drop=True)
    meta_a = oos.loc[idx_away].reset_index(drop=True)
    for f in range(len(pair)):
        g = joint_grid_one(lam_h[f], lam_a[f])
        if rho is not None:
            g = apply_dc_tau(g, lam_h[f], lam_a[f], rho)
        w = wdl_from_grid(g)
        gm = g.mean(axis=0)
        gq_lo = np.quantile(g, q_lo, axis=0)      # per-cell credible band (the WC feed had these)
        gq_up = np.quantile(g, q_hi, axis=0)
        ml = np.unravel_index(np.argmax(gm), gm.shape)
        rows.append(dict(
            id_match=meta_h.loc[f, 'id_match'], name_league=meta_h.loc[f, 'name_league'],
            season=meta_h.loc[f, 'season'], gameday=meta_h.loc[f, 'gameday'],
            kick_off=meta_h.loc[f, 'kick_off'],
            home_team=meta_h.loc[f, 'name_team'], away_team=meta_h.loc[f, 'name_opp'],
            p_home_win=w['home_win'][0], p_home_win_lo=w['home_win'][1], p_home_win_up=w['home_win'][2],
            p_draw=w['draw'][0], p_draw_lo=w['draw'][1], p_draw_up=w['draw'][2],
            p_away_win=w['away_win'][0], p_away_win_lo=w['away_win'][1], p_away_win_up=w['away_win'][2],
            exp_goals_home=float(lam_h[f].mean()), exp_goals_away=float(lam_a[f].mean()),
            ml_score_home=int(ml[0]), ml_score_away=int(ml[1]),
            elo_home=round(float(meta_h.loc[f, 'elo_home_raw']), 1),
            elo_away=round(float(meta_h.loc[f, 'elo_away_raw']), 1)))
        for hh in range(K_MAX + 1):
            for aa in range(K_MAX + 1):
                if gm[hh, aa] > 1e-4:
                    grid_rows.append(dict(id_match=meta_h.loc[f, 'id_match'],
                                          home_team=meta_h.loc[f, 'name_team'],
                                          away_team=meta_h.loc[f, 'name_opp'],
                                          home_goals=hh, away_goals=aa,
                                          p_mid=float(gm[hh, aa]),
                                          p_lo=float(gq_lo[hh, aa]), p_up=float(gq_up[hh, aa])))

        # --- per-team goal distribution P(0/1/2/3+) with bands (the match-detail section)
        for side, lam_s, tm, opp in [('home', lam_h[f], meta_h.loc[f, 'name_team'], meta_h.loc[f, 'name_opp']),
                                     ('away', lam_a[f], meta_h.loc[f, 'name_opp'], meta_h.loc[f, 'name_team'])]:
            pk = np.stack([poisson.pmf(k, lam_s) for k in (0, 1, 2)]
                          + [1.0 - poisson.cdf(2, lam_s)])          # (4, n_samples)
            rec = dict(id_match=meta_h.loc[f, 'id_match'], name_league=meta_h.loc[f, 'name_league'],
                       gameday=meta_h.loc[f, 'gameday'], kick_off=meta_h.loc[f, 'kick_off'],
                       team=tm, opponent=opp, is_home=int(side == 'home'),
                       exp_goals=float(lam_s.mean()))
            for i, lbl in enumerate(['0', '1', '2', '3plus']):
                rec[f'p_goals_{lbl}'] = float(pk[i].mean())
                rec[f'p_goals_{lbl}_lo'] = float(np.quantile(pk[i], q_lo))
                rec[f'p_goals_{lbl}_up'] = float(np.quantile(pk[i], q_hi))
            team_rows.append(rec)
    df_matches = pd.DataFrame(rows).sort_values(['name_league', 'kick_off']).reset_index(drop=True)
    return df_matches, grid_rows, team_rows, dev


def main():
    # ----------------------- 1. bundle ----------------------- #
    print(f"Loading season bundle:\n  {BUNDLE_PATH}")
    with open(BUNDLE_PATH, 'rb') as f:
        B = cloudpickle.load(f)
    meta = B['meta']      # model, idata, team maps and training moments: see forecast_fixtures()
    rho = B['rho'] if USE_DIXON_COLES else None
    factors_CS = meta['factors_CS']
    factors = meta['factors']
    factors_g = [f for f in factors if f != 'home_pitch']
    print(f"  devVersion {meta['devVersion']} | trained through {meta['train_end']} "
          f"({meta['n_train_rows']:,} rows) | rho {B['rho']:+.4f} | created {meta['created']}")

    # ----------------------- 2. data + features ----------------------- #
    print(f"\nLoading data:\n  HIST {HIST_PATH}\n  OOS  {OOS_PATH}")
    hist_raw = pd.read_csv(HIST_PATH, low_memory=False)
    oos_raw = pd.read_csv(OOS_PATH, low_memory=False)
    print(f"  history rows {len(hist_raw):,} | upcoming rows {len(oos_raw):,} "
          f"({oos_raw['id_match'].nunique()} fixtures)")
    raw = pd.concat([hist_raw, oos_raw], axis=0, ignore_index=True)
    cd = build_match_level(raw)
    print(f"  {len(cd):,} team-match rows | seasons {cd['season'].min()} .. {cd['season'].max()}")
    cd = compute_elo(cd)

    # ----------------------- 3. OOS slice + scaling ----------------------- #
    oos = cd[(cd['season'] == TARGET_SEASON) & (cd['match_outcome'].isna())].copy()
    if not len(oos) and season_complete(cd, TARGET_SEASON):
        _next = cd[(cd['season'] > TARGET_SEASON) & cd['match_outcome'].isna()]
        if len(_next):      # the new season's fixtures are in: shipping the old one would skip them
            raise SystemExit(f"{TARGET_SEASON} is complete and {_next['id_match'].nunique()} unplayed "
                             f"fixture(s) of {sorted(_next['season'].unique())} are waiting -- bump "
                             f"TARGET_SEASON (and the bundle) before running.")
        print(f"  {TARGET_SEASON} is complete -- nothing left to forecast; shipping the final results")
    elif not len(oos):
        raise SystemExit(f"No unplayed {TARGET_SEASON} fixtures found and the season is not complete — "
                         f"nothing to forecast. "
                         f"(Has the fixture data been rolled into {os.path.basename(OOS_PATH)}?)")
    n_all = oos['id_match'].nunique()
    oos, deferred, stale, horizon = select_forecast_window(oos, pd.Timestamp.now(), HORIZON_DAYS)
    if len(stale):
        _sh = stale[stale['home_pitch'] == 1]
        print(f"  ⚠️  {_sh['id_match'].nunique()} fixture(s) already kicked off but carry NO "
              f"result yet (results feed lagging? postponement not yet re-dated?) -- NOT re-forecast, "
              f"their frozen forecast stands: "
              f"{[f'{h} v {a}' for h, a in zip(_sh['name_team'], _sh['name_opp'])][:4]}")
    _ud = deferred[deferred['kick_off'].isna() & (deferred['home_pitch'] == 1)]
    if len(_ud):
        print(f"  ⚠️  {_ud['id_match'].nunique()} fixture(s) have NO kick_off in the feed -- not "
              f"forecast until it dates them: "
              f"{[f'{h} v {a}' for h, a in zip(_ud['name_team'], _ud['name_opp'])][:4]}")
    if not len(oos):
        # With the anchor at the next fixture not yet kicked off, an empty window means nothing
        # dated is still to come: only kicked-off fixtures awaiting results (and undated ones).
        # Nothing to forecast, but results that landed since the last run still belong in the
        # feed, and the stale fixtures keep their frozen rows -- so the feed is rebuilt without
        # the model, and the ledger untouched.
        if not os.path.exists(OUT_PKL):
            raise SystemExit(f"Nothing to forecast ({n_all} unplayed fixture(s), none still to come with "
                             f"a date) and no previous board at {OUT_PKL} to carry scorelines from.")
        print(f"  nothing to forecast: {stale['id_match'].nunique()} fixture(s) kicked off and await "
              f"results, {_ud['id_match'].nunique()} undated -- rebuilding the feed from results + "
              f"frozen forecasts only")
        df_matches, grid_rows, team_rows, dev = pd.DataFrame(), [], [], None
    else:
        oos = oos.sort_values(['name_league', 'kick_off']).reset_index(drop=True)
        n_fix = oos['id_match'].nunique()
        print(f"  horizon: {HORIZON_DAYS} days (to {horizon:%Y-%m-%d}) -> forecasting {n_fix} of "
              f"{n_all} unplayed fixtures; {deferred['id_match'].nunique()} deferred to later runs")
        print(f"\nUpcoming fixtures in {TARGET_SEASON}: {n_fix} "
              f"({', '.join(f'{k} {v}' for k, v in oos[oos.home_pitch == 1].groupby('name_league').size().items())})")
        df_matches, grid_rows, team_rows, dev = forecast_fixtures(oos, B, rho, factors_CS, factors_g)


    # ------------------------------------------------------------------ #
    #  FROZEN FORECAST LEDGER  +  full-season feed (played + upcoming)
    #
    #  Two things the downstream stack (validation / receipts / pick'em) needs and that a
    #  naive "forecast the unplayed" feed cannot give:
    #    (a) results must APPEAR in the feed once a match is played -- otherwise played
    #        fixtures silently leave the file;
    #    (b) the probabilities shown for a played match must be the ones FROZEN BEFORE it
    #        kicked off, never a re-forecast. By the next run the ELO has absorbed the
    #        result, so a re-forecast is hindsight (WC lesson L5: it flattered accuracy by
    #        ~4pp). The ledger below stores each fixture's LAST pre-match forecast; once the
    #        match is played that row is never overwritten again.
    # ------------------------------------------------------------------ #
    PCOLS = ['p_home_win', 'p_home_win_lo', 'p_home_win_up', 'p_draw', 'p_draw_lo', 'p_draw_up',
             'p_away_win', 'p_away_win_lo', 'p_away_win_up', 'exp_goals_home', 'exp_goals_away',
             'ml_score_home', 'ml_score_away']
    # KEY ON THE TEAMS, NOT id_match. When a fixture is postponed out of a round, the source
    # RENUMBERS the survivors (La Liga 2026-08: Celta-Osasuna left GD1, so G6->G5, G7->G6).
    # An id-keyed ledger then hands each match its NEIGHBOUR's frozen forecast -- silent, and
    # fatal to the receipts. (season, home_team, away_team) is unique in a double round-robin
    # and immune to both renumbering and date changes.
    KEY = ['season', 'home_team', 'away_team']
    # Full timestamp, not a date. A date-only stamp cannot be resolved against a same-day
    # kick-off: 26 ledger rows were frozen ON their fixture's kick-off date, and the claim that
    # they were frozen before it rested on knowing that runs happen in the morning, not on
    # anything in the file. With a time, "frozen before kick-off" is auditable from the ledger
    # alone. ISO format keeps the lexical sort order of the older date-only stamps.
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    led = None
    if os.path.exists(FROZEN_LEDGER):
        led = pd.read_csv(FROZEN_LEDGER)
        if not all(k in led.columns for k in KEY):
            raise SystemExit(f"{FROZEN_LEDGER} predates the team-keyed schema. Rebuild it from "
                             f"the archived vintages before running (see rebuild_frozen_ledger.py).")
    if len(df_matches):
        fresh = df_matches[KEY + ['id_match'] + PCOLS].copy()
        fresh['forecast_frozen_at'] = stamp
        # first run: same guard (a duplicate key in the board must never reach the ledger)
        ledger = merge_frozen_ledger(led if led is not None else fresh.iloc[0:0], fresh, KEY)
        ledger.to_csv(FROZEN_LEDGER, index=False)
    elif led is not None:   # nothing forecast this run: the ledger is read (same guard), never rewritten
        ledger = merge_frozen_ledger(led, led.iloc[0:0], KEY)
    else:
        ledger = pd.DataFrame(columns=KEY + ['id_match'] + PCOLS + ['forecast_frozen_at'])

    # played fixtures of the target season: result + the forecast frozen before kickoff
    played = cd[(cd['season'] == TARGET_SEASON) & (cd['home_pitch'] == 1)
                & (cd['match_outcome'].notna())].copy()
    feed_rows = []
    if len(played):
        lref = ledger.set_index(KEY)
        pk = list(zip(played['season'], played['name_team'], played['name_opp']))
        missing = [f"{h} v {a}" for (_, h, a) in pk if (_, h, a) not in lref.index]
        if missing:
            print(f"  ⚠️  {len(missing)} played fixture(s) have NO frozen forecast (they were never "
                  f"in a run while unplayed) — results shipped without probabilities: {missing[:3]}")
        for r in played.itertuples():
            rec = dict(id_match=r.id_match, name_league=r.name_league, season=r.season,
                       gameday=r.gameday, kick_off=r.kick_off,
                       home_team=r.name_team, away_team=r.name_opp,
                       home_goals=float(r.match_outcome), away_goals=float(r.goalsscored_inGame_opp),
                       status='finished',
                       elo_home=round(float(r.elo_team), 1), elo_away=round(float(r.elo_opp), 1))
            k = (r.season, r.name_team, r.name_opp)
            if k in lref.index:
                for c in PCOLS:
                    rec[c] = lref.loc[k, c]
                rec['forecast_frozen_at'] = lref.loc[k, 'forecast_frozen_at']
            feed_rows.append(rec)
    if len(stale):
        _rows, _miss = stale_feed_rows(stale[stale['home_pitch'] == 1], ledger, KEY, PCOLS)
        feed_rows.extend(_rows)
        if _miss:
            print(f"  ⚠️  {len(_miss)} kicked-off fixture(s) have NO frozen forecast and no result "
                  f"-- absent from the feed until the result lands: {_miss[:3]}")
    df_upcoming = df_matches.copy()
    df_upcoming['home_goals'] = np.nan
    df_upcoming['away_goals'] = np.nan
    df_upcoming['status'] = 'upcoming'
    df_upcoming['forecast_frozen_at'] = stamp
    _parts = [p for p in (pd.DataFrame(feed_rows), df_upcoming) if len(p)]
    if not _parts:
        raise SystemExit("Nothing to ship: no result, no frozen forecast and nothing forecast this run.")
    df_matches = (pd.concat(_parts, ignore_index=True)
                  .sort_values(['name_league', 'kick_off', 'id_match']).reset_index(drop=True))
    for c in PCOLS + ['forecast_frozen_at']:      # a feed of results without any frozen forecast
        if c not in df_matches:
            df_matches[c] = np.nan
    for c in ['ml_score_home', 'ml_score_away', 'home_goals', 'away_goals']:
        df_matches[c] = df_matches[c].astype('Int64')     # nullable int: render 2, not 2.0
    print(f"  feed: {int((df_matches['status'] == 'finished').sum())} finished (results + frozen "
          f"forecast) + {int((df_matches['status'] == 'upcoming').sum())} upcoming")
    df_grid = pd.DataFrame(grid_rows)
    df_team = pd.DataFrame(team_rows)
    if len(stale) and not os.path.exists(OUT_PKL):
        print(f"  [stale] no previous board on disk -- kicked-off fixtures awaiting results have no "
              f"scoreline/team-goal rows this run")
    if len(stale) and os.path.exists(OUT_PKL):      # previous board still on disk (archived below)
        with open(OUT_PKL, 'rb') as f:
            _prev = pickle.load(f)
        _g, _t, _miss = carry_forward_board_rows(_prev, stale[stale['home_pitch'] == 1])
        # (a run with nothing forecast has empty, column-less frames here: _g / _t alone)
        df_grid = pd.concat([d for d in (df_grid, _g) if len(d.columns)], ignore_index=True)
        df_team = pd.concat([d for d in (df_team, _t) if len(d.columns)], ignore_index=True)
        print(f"  [stale] scorelines/team-goals carried forward from the previous board for "
              f"{len(set(zip(_g['home_team'], _g['away_team'])))} fixture(s)"
              + (f"; not on the previous board: {_miss[:3]}" if _miss else ""))
    if not len(df_grid.columns) or not len(df_team.columns):
        # nothing forecast and nothing carried forward: keep the previous board's columns, so the
        # site reads an empty table rather than a header-less file
        with open(OUT_PKL, 'rb') as f:
            _prev = pickle.load(f)
        df_grid = df_grid if len(df_grid.columns) else _prev['scorelines'].iloc[0:0]
        df_team = df_team if len(df_team.columns) else _prev['team_goals'].iloc[0:0]
    df_team = df_team.sort_values(['name_league', 'kick_off', 'id_match', 'is_home'],
                                  ascending=[True, True, True, False]).reset_index(drop=True)

    # ------------------- 6b. NORMALISATION GATE (pre-export) ------------------- #
    # W/D/L must sum to 1 on EVERY row, not on average: a mean over ~200 rows hides a single
    # broken fixture, and this check used to run after to_csv -- i.e. it could only ever report
    # a bad file, never stop one. It now gates export, like the eta parity check.
    # Tolerance: the joint grid is truncated at k_max, so a little mass is legitimately lost.
    # It is largest on fixtures involving PROMOTED teams -- their new-team priors are wide, so
    # some posterior draws put lambda high enough for the k_max tail to bite (2026/27 worst
    # case: Man City v Coventry, 3.6e-5). SUM_TOL sits well above that floor and far below any
    # real normalisation bug, which would be orders of magnitude larger.
    SUM_TOL = 1e-3
    _has = df_matches[['p_home_win', 'p_draw', 'p_away_win']].notna().all(axis=1)
    _s = df_matches.loc[_has, ['p_home_win', 'p_draw', 'p_away_win']].sum(axis=1)
    if _has.any():
        _worst = df_matches.loc[_s.sub(1).abs().idxmax()]
        print(f"\n[row sums] {int(_has.sum())} forecast rows | min {_s.min():.9f} max {_s.max():.9f} | "
              f"max deficit {(1 - _s.min()):.2e} (k_max={K_MAX} truncation) | "
              f"worst: {_worst['home_team']} v {_worst['away_team']}")
    if not np.isclose(_s, 1.0, atol=SUM_TOL, rtol=0).all():
        bad = df_matches.loc[_has][~np.isclose(_s, 1.0, atol=SUM_TOL, rtol=0)]
        raise AssertionError(
            f"[row sums] FAIL — {len(bad)} row(s) off 1.0 by more than {SUM_TOL:.0e}; "
            f"nothing exported. Worst: {bad.iloc[0]['home_team']} v {bad.iloc[0]['away_team']}")
    print("[row sums] PASS — safe to export.")

    # ----------------------- 7. archive + export ----------------------- #
    os.makedirs(OUT_DIR, exist_ok=True)
    if ARCHIVE_VINTAGES:
        arch = archive_existing_outputs([OUT_MATCH_CSV, OUT_GRID_CSV, OUT_TEAM_CSV, OUT_PKL])
        print(f"\nArchived {len(arch)} previous output(s) -> {VINTAGE_DIR}/" if arch
              else "\n[vintage] no previous board to archive (first run)")

    out = dict(meta=dict(bundle=os.path.basename(BUNDLE_PATH), devVersion=meta['devVersion'],
                         train_end=meta['train_end'], target_season=TARGET_SEASON,
                         rho=rho, k_max=K_MAX, cred_region=CRED_REGION,
                         run=datetime.now().strftime('%Y-%m-%d %H:%M'), eta_parity=dev),
               matches=df_matches, scorelines=df_grid, team_goals=df_team)
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(out, f)
    df_matches.to_csv(OUT_MATCH_CSV, index=False)
    df_grid.to_csv(OUT_GRID_CSV, index=False)
    df_team.to_csv(OUT_TEAM_CSV, index=False)

    print(f"\n================== DONE ==================")
    print(f"Fixtures forecast : {len(df_matches)}")
    print(f"Saved matches csv : {OUT_MATCH_CSV}")
    print(f"Saved grid csv    : {OUT_GRID_CSV}  ({len(df_grid):,} cells, with credible bands)")
    print(f"Saved team csv    : {OUT_TEAM_CSV}  ({len(df_team)} team-matches)")
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print("\n" + df_matches[['name_league', 'home_team', 'away_team', 'p_home_win', 'p_draw',
                                 'p_away_win', 'exp_goals_home', 'exp_goals_away',
                                 'ml_score_home', 'ml_score_away']].head(12).round(3).to_string(index=False))
    _p = df_matches.loc[_has, ['p_home_win', 'p_draw', 'p_away_win']]
    print(f"\nsanity ({int(_has.sum())} rows with a forecast): "
          f"home-win share {_p['p_home_win'].mean():.3f} | draw share {_p['p_draw'].mean():.3f}")
    return out


if __name__ == '__main__':
    main()
