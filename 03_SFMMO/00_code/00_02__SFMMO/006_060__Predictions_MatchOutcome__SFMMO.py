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
HIST_PATH     = f'{directory}/10_data/106_Website/data_byPlayer__SFM_II__TM.csv'   # played history
OOS_PATH      = f'{directory}/10_data/106_Website/data_byPlayer__OOS.csv'          # upcoming fixtures
TARGET_SEASON = '2026/27'          # the season whose unplayed fixtures we forecast

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


def joint_pmf(eta_da, idx_home, idx_away, k_max=K_MAX):
    """Per-fixture joint scoreline PMF. Returns (joint[f, s, h, a], lam_h, lam_a)."""
    lam = np.exp(eta_da.stack(samples=('chain', 'draw')).values)     # (n_obs, n_samples)
    lam_h = lam[idx_home, :]
    lam_a = lam[idx_away, :]
    ks = np.arange(k_max + 1)
    pmf_h = poisson.pmf(ks[:, None, None], mu=lam_h[None, :, :])
    pmf_a = poisson.pmf(ks[:, None, None], mu=lam_a[None, :, :])
    return np.einsum('hfs,afs->fsha', pmf_h, pmf_a), lam_h, lam_a


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


def main():
    # ----------------------- 1. bundle ----------------------- #
    print(f"Loading season bundle:\n  {BUNDLE_PATH}")
    with open(BUNDLE_PATH, 'rb') as f:
        B = cloudpickle.load(f)
    meta = B['meta']
    model = B['model']
    idata = B['idata']
    team_to_idx = B['team_to_idx']
    names_teams = B['names_teams']
    train_means, train_stds = B['train_means'], B['train_stds']
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
    if not len(oos):
        raise SystemExit(f"No unplayed {TARGET_SEASON} fixtures found — nothing to forecast. "
                         f"(Has the fixture data been rolled into {os.path.basename(OOS_PATH)}?)")
    oos = oos.sort_values(['name_league', 'kick_off']).reset_index(drop=True)
    n_fix = oos['id_match'].nunique()
    print(f"\nUpcoming fixtures in {TARGET_SEASON}: {n_fix} "
          f"({', '.join(f'{k} {v}' for k, v in oos[oos.home_pitch == 1].groupby('name_league').size().items())})")

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
    joint, lam_h, lam_a = joint_pmf(preds['predictions']['eta_oos'], idx_home, idx_away)
    print(f"  joint grids: {joint.shape[0]} fixtures x {joint.shape[1]} draws, "
          f"{K_MAX+1}x{K_MAX+1} (mass {joint.sum(axis=(2,3)).mean():.6f})")

    rows, grid_rows, team_rows = [], [], []
    q_lo, q_hi = (1 - CRED_REGION) / 2, 1 - (1 - CRED_REGION) / 2
    meta_h = oos.loc[idx_home].reset_index(drop=True)
    meta_a = oos.loc[idx_away].reset_index(drop=True)
    for f in range(len(pair)):
        g = joint[f]
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
    df_grid = pd.DataFrame(grid_rows)
    df_team = pd.DataFrame(team_rows).sort_values(['name_league', 'kick_off', 'id_match', 'is_home'],
                                                  ascending=[True, True, True, False]).reset_index(drop=True)

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
    print(f"\nsanity: mean row sum {(df_matches[['p_home_win','p_draw','p_away_win']].sum(axis=1)).mean():.6f} | "
          f"home-win share {df_matches['p_home_win'].mean():.3f} | draw share {df_matches['p_draw'].mean():.3f}")
    return out


if __name__ == '__main__':
    main()
