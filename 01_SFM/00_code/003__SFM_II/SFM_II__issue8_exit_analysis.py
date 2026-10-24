"""Issue #8: does the career GP (f_long) help or hurt in a player's exit season?

Pre-registered comparison (SFM_II__dev_EW.ipynb, variant dict): C_ELO vs C_ELO_noLong, both
re-run on current data with RUN_TAG='__issue8', exit-season log score vs the rest, SEs clustered
by player.

Per row d = loss(challenger) - loss(full), joined on (val_season, id_match, name_player), so
d > 0 means the term the challenger drops HELPS. Exit season = the player's last season in
data_byPlayer (the data runs to the current season, so every scored season 2021/22-2023/24 has
a known future). Tenure = career season index (1-based count of distinct seasons up to that one).

usage: python SFM_II__issue8_exit_analysis.py <full.pkl> <challenger.pkl> <data_byPlayer.csv>
"""
import sys

import numpy as np
import pandas as pd


def cluster_ols(y, X, groups):
    """OLS coefficients with player-clustered (CR1) standard errors."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    codes, uniq = pd.factorize(groups)
    scores = np.zeros((len(uniq), X.shape[1]))
    np.add.at(scores, codes, X * u[:, None])
    G, n, k = len(uniq), len(y), X.shape[1]
    c = G / (G - 1) * (n - 1) / (n - k)
    V = c * XtX_inv @ (scores.T @ scores) @ XtX_inv
    return beta, np.sqrt(np.diag(V)), G


def paired_t(d):
    # --- the notebook's selection statistic: row-level t of mean(d) vs 0
    d = np.asarray(d, dtype=float)
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def main(full_pkl, chal_pkl, data_csv):
    full = pd.read_pickle(full_pkl)
    chal = pd.read_pickle(chal_pkl)
    keys = ['val_season', 'id_match', 'name_player']
    a = full['perrow'][keys + ['season', 'is_new', 'logloss', 'rps']]
    b = chal['perrow'][keys + ['logloss', 'rps']]
    m = a.merge(b, on=keys, suffixes=('_full', '_chal'), validate='one_to_one')
    assert len(m) == len(a) == len(b), (len(m), len(a), len(b))
    m['d'] = m['logloss_chal'] - m['logloss_full']
    m['d_rps'] = m['rps_chal'] - m['rps_full']

    seasons = pd.read_csv(data_csv, usecols=['name_player', 'season']).drop_duplicates()
    last = seasons.groupby('name_player')['season'].max().rename('last_season')
    seasons = seasons.sort_values(['name_player', 'season'])
    seasons['tenure'] = seasons.groupby('name_player').cumcount() + 1
    m = m.merge(last, on='name_player', how='left').merge(seasons, on=['name_player', 'season'], how='left')
    assert m['last_season'].notna().all() and m['tenure'].notna().all()
    m['exit'] = (m['season'] == m['last_season']).astype(float)

    print(f"full = {full['devVersion']}   challenger = {chal['devVersion']}   rows = {len(m):,}")
    for lbl, d in [('full', full), ('challenger', chal)]:
        print(f"\n  diagnostics, {lbl}:\n{d['diagnostics'][['divergences', 'max_rhat', 'min_ess_bulk']].to_string()}")

    print(f"\n--- selection rule (row-level paired t, challenger - full; < 0 favours the challenger)")
    for col in ['d', 'd_rps']:
        print(f"  {'logloss' if col == 'd' else 'rps':<8} mean {m[col].mean()*1e3:+.3f} x1e-3   t = {paired_t(m[col]):+.2f}")

    print(f"\n--- d = logloss(challenger) - logloss(full), x1e-3; d > 0 => the dropped term helps; "
          f"SEs clustered by player")

    def row(lbl, sub):
        beta, se, G = cluster_ols(sub['d'].to_numpy(), np.ones((len(sub), 1)), sub['name_player'].to_numpy())
        print(f"  {lbl:<28} n={len(sub):>6,}  players={G:>5,}  mean d = {beta[0]*1e3:+7.3f}  "
              f"(se {se[0]*1e3:.3f}, z={beta[0]/se[0]:+.2f})")

    row('all', m)
    for s, g in m.groupby('val_season'):
        row(f'fold {s}', g)
    row('exit season', m[m['exit'] == 1])
    row('other seasons', m[m['exit'] == 0])
    fitted = m[~m['is_new']]
    row('exit season, fitted only', fitted[fitted['exit'] == 1])
    row('other seasons, fitted only', fitted[fitted['exit'] == 0])
    for lo, hi in [(1, 4), (5, 9), (10, 14), (15, 99)]:
        g = m[(m['tenure'] >= lo) & (m['tenure'] <= hi)]
        for e, lbl in [(1, 'exit'), (0, 'other')]:
            gg = g[g['exit'] == e]
            if len(gg) > 50:
                row(f'tenure {lo}-{hi if hi < 99 else "+"} {lbl}', gg)

    # --- exit minus other, players clustered (a player can contribute to both groups)
    X = np.column_stack([np.ones(len(m)), m['exit'].to_numpy()])
    beta, se, G = cluster_ols(m['d'].to_numpy(), X, m['name_player'].to_numpy())
    print(f"\n  exit - other: {beta[1]*1e3:+.3f} x1e-3 (se {se[1]*1e3:.3f}, z={beta[1]/se[1]:+.2f}, players={G:,})")


if __name__ == '__main__':
    main(*sys.argv[1:4])
