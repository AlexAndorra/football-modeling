#!/usr/bin/env python
# coding: utf-8
"""
==========================================================================================
 006_061 — Season Outcome Odds (title / top-4 / relegation) for the LEAGUE SFMMO
==========================================================================================

Weekly companion to `006_060`. Where 006_060 forecasts individual fixtures, this simulates
the REST OF THE SEASON and returns each team's probability of the title, a top-4 finish and
relegation, plus expected points.

RESULTS-AWARE (the WC group-sim pattern): every match already played is PINNED to its real
result; only the remaining fixtures are simulated. So the board conditions on the season so
far automatically and tightens week by week.

Run it on the same cadence as 006_060. Each run:
  * writes the current board (`SFMMO_season_odds.csv`),
  * archives a DATED snapshot into `_vintages/`,
  * appends to a running tracker (`SFMMO_season_odds__tracker.csv`) so the website's
    title-odds chart has a time series to draw.

METHOD AND ITS LIMITS (state these on the site, as with the WC board)
--------------------------------------------------------------------
*  Team strength per posterior draw: lam(i->j) = exp(mu + ATT_i + DEF_j [+ beta_home_i]),
   with ATT_i = alpha_i + b_eloT*z(elo_i) and DEF_j = -delta_j + b_eloO*z(elo_j).
*  Fixtures are taken as a full double round-robin (every pair, home and away) rather than
   the published calendar: for a full league season these coincide, and it keeps the script
   independent of fixture-list availability. Matches already played are matched by team pair
   and pinned.
*  Form features (points-diff, cumulative goals) are held at league average, exactly as in
   the WC knockout reconstruction. Folding actual form in was tested there and REJECTED
   (fragile under multicollinearity); ELO carries recency robustly instead.
*  ELO is current (updated through the last played match), so strength reflects the season
   so far; it does not update further inside a simulated season.
==========================================================================================
"""

import os
import shutil
import importlib.util
from datetime import datetime

import numpy as np
import pandas as pd
import cloudpickle


directory = '/Users/maximilian/Dropbox/Max/51_SoccerAnalytics'

BUNDLE_PATH   = f'{directory}/10_data/01_Models/SFMMO_DevK__scaleCS__train202526__PROD.pkl'
TARGET_SEASON = '2026/27'
N_SIM         = None          # None = all posterior draws (posterior-consistent)
TOP_N         = 4             # "top-4" definition
RELEGATED     = {'bundesliga': 3, 'la-liga': 3, 'ligue-1': 3, 'premier-league': 3, 'serie-a': 3}

OUT_DIR      = f'{directory}/10_data/106_Website'
VINTAGE_DIR  = f'{OUT_DIR}/_vintages'
OUT_CSV      = f'{OUT_DIR}/SFMMO_season_odds.csv'
TRACKER_CSV  = f'{OUT_DIR}/SFMMO_season_odds__tracker.csv'

SEED = 326

# --- reuse 006_060's data/feature helpers rather than duplicating them (M6: one source) ---
_spec = importlib.util.spec_from_file_location(
    'sfmmo_predict', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '006_060__Predictions_MatchOutcome__SFMMO.py'))
_p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p)


def simulate_league(teams, played, sp, rng, n_sim, n_releg):
    """Monte-Carlo the season for one league. `played` = {(home, away): (hg, ag)} already
    played; every other ordered pair is simulated. One simulated season per posterior draw."""
    n = len(teams)
    tix = {t: i for i, t in enumerate(teams)}
    a, d, bh = sp['alpha'], sp['delta'], sp['beta_home']
    mu, mug, beta = sp['mu'], sp['mu_gamma'], sp['beta']
    t2i, fg = sp['team_to_idx'], sp['factors_g']
    iET, iEO = fg.index('elo_team'), fg.index('elo_opp')
    ndraw = mu.shape[0]
    n_sim = ndraw if n_sim is None else min(n_sim, ndraw)

    # unseen (promoted) teams -> the notebook's anchored new-team priors
    aA = np.quantile(np.median(a, axis=1), 0.25)
    dA = np.quantile(np.median(d, axis=1), 0.25)
    A  = np.stack([a[t2i[t]] if t in t2i else np.full(ndraw, aA) for t in teams])[:, :n_sim]
    D  = np.stack([d[t2i[t]] if t in t2i else np.full(ndraw, dA) for t in teams])[:, :n_sim]
    BH = np.stack([bh[t2i[t]] if t in t2i else mug for t in teams])[:, :n_sim]
    mu_s = mu[:n_sim]
    z  = np.array([(sp['elo_now'][t] - sp['elo_mu']) / sp['elo_sd'] for t in teams])
    zo = np.array([(sp['elo_now'][t] - sp['eloopp_mu']) / sp['eloopp_sd'] for t in teams])
    ATT = A + z[:, None] * beta[iET][:n_sim][None, :]
    DEF = -D + zo[:, None] * beta[iEO][:n_sim][None, :]

    hh, aa = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    m = hh != aa
    H, Aw = hh[m], aa[m]
    lam_h = np.exp(mu_s[None, :] + ATT[H] + DEF[Aw] + BH[H])
    lam_a = np.exp(mu_s[None, :] + ATT[Aw] + DEF[H])
    hg = rng.poisson(lam_h)
    ag = rng.poisson(lam_a)

    # PIN played fixtures to their real scores (identical across every simulated season)
    n_pinned = 0
    for f, (i, j) in enumerate(zip(H, Aw)):
        res = played.get((teams[i], teams[j]))
        if res is not None:
            hg[f, :], ag[f, :] = int(res[0]), int(res[1])
            n_pinned += 1

    hp = np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
    ap = np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
    PTS = np.zeros((n, n_sim)); GD = np.zeros((n, n_sim))
    np.add.at(PTS, H, hp);       np.add.at(PTS, Aw, ap)
    np.add.at(GD, H, hg - ag);   np.add.at(GD, Aw, ag - hg)
    key = PTS * 1e6 + GD * 1e2 + rng.random((n, n_sim))     # pts -> GD -> random tiebreak
    rank = (-key).argsort(axis=0).argsort(axis=0)           # 0 = champion

    return pd.DataFrame({
        'team': teams,
        'p_title':  (rank == 0).mean(axis=1),
        'p_top4':   (rank < TOP_N).mean(axis=1),
        'p_releg':  (rank >= n - n_releg).mean(axis=1),
        'exp_pts':  PTS.mean(axis=1).round(1),
        'pts_lo':   np.quantile(PTS, 0.05, axis=1).round(1),
        'pts_up':   np.quantile(PTS, 0.95, axis=1).round(1),
        'elo_now':  [round(sp['elo_now'][t], 1) for t in teams],
        'new_team': [t not in t2i for t in teams],
    }), n_pinned, n_sim


def main():
    print(f"Loading bundle: {os.path.basename(BUNDLE_PATH)}")
    with open(BUNDLE_PATH, 'rb') as f:
        B = cloudpickle.load(f)
    meta, post = B['meta'], B['idata'].posterior
    S = lambda v: post[v].stack(s=('chain', 'draw')).values
    sp = dict(mu=S('mu'), alpha=S('alpha'), delta=S('delta'), beta_home=S('beta_home'),
              mu_gamma=S('mu_gamma'), beta=S('beta'), team_to_idx=B['team_to_idx'],
              factors_g=[f for f in meta['factors'] if f != 'home_pitch'],
              elo_mu=float(B['train_means']['elo_team'].mean()),
              elo_sd=float(B['train_stds']['elo_team'].mean()),
              eloopp_mu=float(B['train_means']['elo_opp'].mean()),
              eloopp_sd=float(B['train_stds']['elo_opp'].mean()))
    print(f"  devVersion {meta['devVersion']} | trained through {meta['train_end']} | "
          f"{sp['mu'].shape[0]:,} posterior draws")

    print("\nLoading data + rebuilding ELO through the last played match ...")
    raw = pd.concat([pd.read_csv(_p.HIST_PATH, low_memory=False),
                     pd.read_csv(_p.OOS_PATH, low_memory=False)], ignore_index=True)
    cd = _p.compute_elo(_p.build_match_level(raw))

    season = cd[cd['season'] == TARGET_SEASON]
    if not len(season):
        raise SystemExit(f"No {TARGET_SEASON} rows found.")
    home = season[season['home_pitch'] == 1]
    played_rows = home[home['match_outcome'].notna()]
    print(f"  {TARGET_SEASON}: {home['id_match'].nunique()} fixtures known, "
          f"{played_rows['id_match'].nunique()} already played")

    rng = np.random.default_rng(SEED)
    stamp = datetime.now().strftime('%Y-%m-%d')
    boards = []
    for lg, g in home.groupby('name_league'):
        teams = sorted(set(g['name_team']) | set(g['name_opp']))
        played = {(r.name_team, r.name_opp): (r.match_outcome, r.goalsscored_inGame_opp)
                  for r in g[g['match_outcome'].notna()].itertuples()}
        # current ELO per team: the latest rating seen in this season's rows
        elo_now = {}
        for r in g.itertuples():
            elo_now[r.name_team] = r.elo_team
            elo_now[r.name_opp] = r.elo_opp
        sp_lg = dict(sp, elo_now=elo_now)
        df, n_pin, n_sim = simulate_league(teams, played, sp_lg, rng, N_SIM, RELEGATED.get(lg, 3))
        df.insert(0, 'league', lg)
        df.insert(0, 'as_of', stamp)
        boards.append(df.sort_values('p_title', ascending=False).reset_index(drop=True))
        print(f"  {lg:16s} {len(teams)} teams | {n_pin} of {len(teams)*(len(teams)-1)} fixtures pinned "
              f"| {n_sim:,} simulated seasons | sum p_title {df['p_title'].sum():.3f}")

    board = pd.concat(boards, ignore_index=True)

    os.makedirs(VINTAGE_DIR, exist_ok=True)
    if os.path.exists(OUT_CSV):     # archive the outgoing board before overwriting
        mt = datetime.fromtimestamp(os.path.getmtime(OUT_CSV)).strftime('%Y-%m-%d')
        dest = os.path.join(VINTAGE_DIR, f'SFMMO_season_odds__{mt}.csv')
        if os.path.exists(dest):
            dest = dest.replace('.csv', datetime.fromtimestamp(os.path.getmtime(OUT_CSV)).strftime('_%H%M%S.csv'))
        shutil.copy2(OUT_CSV, dest)
        print(f"\nArchived previous board -> {os.path.basename(dest)}")
    board.to_csv(OUT_CSV, index=False)

    # running tracker: one row per (as_of, league, team) -> the site's title-odds line chart
    cols = ['as_of', 'league', 'team', 'p_title', 'p_top4', 'p_releg', 'exp_pts']
    trk = board[cols]
    if os.path.exists(TRACKER_CSV):
        old = pd.read_csv(TRACKER_CSV)
        trk = pd.concat([old[old['as_of'] != stamp], trk], ignore_index=True)   # idempotent per day
    trk.to_csv(TRACKER_CSV, index=False)

    print(f"\n================== DONE ==================")
    print(f"Saved board   : {OUT_CSV}  ({len(board)} teams)")
    print(f"Saved tracker : {TRACKER_CSV}  ({len(trk)} rows, {trk['as_of'].nunique()} snapshot date(s))")
    with pd.option_context('display.width', 200):
        for lg, g in board.groupby('league'):
            top = g.nlargest(3, 'p_title')
            print(f"  {lg:16s} " + " | ".join(f"{r.team} {r.p_title:.1%}" for r in top.itertuples()))
    return board


if __name__ == '__main__':
    main()
