# ======================================================================================== #
#                        SAR / PAR  --  SFM II · C_ELO  (light path)                        #
# ======================================================================================== #
#
# Sibling of 006_041__SAR_PAR__SFM_OG.py, on the light engine: reads the LIGHT bundle
# (posterior draws + graph-exported f_within/f_long grids) and reconstructs everything in
# NumPy.  No pymc, no pytensor, no model rehydration.
#
# TWO SUBSTANTIVE CHANGES vs the OG script -- both deliberate, both from the 2026 revision:
#
# 1. THE ESTIMAND IS FIXED (referee report M5).  The OG script computes its counterfactual as
#    `np.zeros_like(factor_data)` -- which zeroes EVERY factor, including home_pitch and the
#    position dummies.  That is not "all teams equal"; it is "every match away, every player
#    recoded to the reference position".  Here (decision of 2026-08-04):
#       equalized  : standardized CONTEXT factors -> 0 (= the cross-sectional average)
#                    home_pitch                   -> its empirical mean (average schedule)
#       held       : player ATTRIBUTES at observed values (position_FOR, cumulative goals,
#                    goal share)
#    elo_diff is CONTEXT (it is the team-vs-opponent strength gap) and is equalized.
#    Replacement level = the AVERAGE PLAYER -- a documented deviation from the paper.
#    NOTE: C_ELO carries no momentum factor, so the historical-average / form-neutral SAR
#    pair of the dev notebook collapses to a single SAR here.  Nothing is lost.
#
# 2. THE ARTIFACT IS ~5 ORDERS OF MAGNITUDE SMALLER.  The OG script pickles the full
#    obs-level posterior predictive: the live 041_SARPAR__prod.pkl is **10.5 GB**.  The
#    website only ever needs SAR/PAR *per player per draw* -- which is exactly what
#    006_042__SAR_PAR_funcCalc computes from it -- so this script does that aggregation
#    here, in chunks, and ships ~50 MB.  A drop-in `get__SAR_PAR__SFM_II()` is provided at
#    the bottom for 006_042 to call.
#
# Runs locally in the sfmII venv in a couple of minutes.
# ======================================================================================== #

import os
import pickle

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse

seed = sum(map(ord, "sfm"))
rng = np.random.default_rng(seed)


# ----------------------------- USER INTERACTION ----------------------------- #

# --- Set the directory to the data folder:
directory = '/Users/maximilian/Dropbox/Max/51_SoccerAnalytics'

# --- Fitted bundle stem (the LIGHT companion is read: {stem}__LIGHT.pkl):
SFM_model__NAME = 'SFM_II_FinalC_ELO_scaleCS__2526'

# --- Which factors are CONTEXT (equalized in the SAR counterfactual)?  Everything else in
# --- factors_team is a player ATTRIBUTE and is held at its observed value.
CONTEXT_FACTORS = ['points_diff', 'goalsconceded_rank_opp', 'goal_appeal', 'elo_diff']

# --- Per-row goal outcome used for the boards:
# ---   'expected' -> E[goals | draw] = sum_k k * P(k)   (analytic, no simulation noise)
# ---   'sampled'  -> a categorical draw per row          (reproduces the OG script's object)
# --- 'expected' is the better estimator of the underlying quantity: it removes
# --- observation-level sampling noise and keeps posterior uncertainty, so the published
# --- credible bands get NARROWER than the OG ones.  That is a visible change on the site --
# --- switch to 'sampled' if you want the old spread.
GOALS_MODE = 'expected'

# --- Row chunk for the accumulation (memory ~ S x chunk x 4 floats):
CHUNK = 20_000

# ----------------------------- USER INTERACTION ----------------------------- #


# ======================================== Light Engine ======================================== #
# (identical algebra to 006_040__Predictions_ScoringProb__SFM_II -- validated against the
#  PyMC graph on 2026-08-14 to 2.2e-16 on quantiles)

def softplus(x):
    return np.logaddexp(0.0, x)


def ordered_probs(eta_s, cut_s):
    """eta_s: (S, n); cut_s: (S, n, K-1) -> (S, n, K). Mirrors pm.OrderedLogistic.compute_p."""
    cdf = 1.0 / (1.0 + np.exp(-(cut_s - eta_s[..., None])))
    return np.concatenate([cdf[..., :1], np.diff(cdf, axis=-1), 1.0 - cdf[..., -1:]], axis=-1)


def player_effect_and_cutpoints(L):
    D = L['draws']
    bsig = np.sqrt(L['intercept_sigma']**2
                   + D['player_effect_diversity']**2 / L['n_players_train'])
    pe = bsig[:, None] * D['baseline'][:, None] + D['player_effect_raw']
    delta = D['delta_mean'][:, None, :] + D['delta_sigma'][:, None, :] * D['delta_player']
    cut = np.concatenate([np.full((pe.shape[0], pe.shape[1], 1), L['cutpoint_offset']),
                          L['cutpoint_offset'] + np.cumsum(softplus(delta), axis=-1)], axis=-1)
    return pe, cut


def update_elo(r_home, r_away, result, K=20, home_adv=50):
    exp_home = 1 / (1 + 10 ** ((r_away - r_home - home_adv) / 400))
    s_home = {2: 1.0, 1: 0.5, 0: 0.0}[result]
    return r_home + K * (s_home - exp_home), r_away + K * ((1 - s_home) - (1 - exp_home))


def _cs_scale_cols(x):
    return x.apply(lambda col: (col - col.mean()) / col.std()
                   if (len(x) > 1 and col.std() > 0) else col * 0.0)


# ======================================== Load & Prove ======================================== #

with open(f'{directory}/10_data/01_Models/{SFM_model__NAME}__LIGHT.pkl', 'rb') as f:
    L = pickle.load(f)

factors_CS   = L['factor_standardize']
factors_team = L['factors_team']
players_train = pd.Index(L['players_ordered'])
elo_cfg = L['elo']
_dc = L['data_contract']
print(f'bundle       : {SFM_model__NAME}__LIGHT.pkl')
print(f'trained thru : {_dc.get("train_end")} | {_dc.get("n_rows"):,} rows | {len(players_train):,} players')
assert 'elo_diff' in factors_CS, 'not a C_ELO bundle'
assert all(f in factors_team for f in CONTEXT_FACTORS), 'a CONTEXT_FACTOR is not in the design matrix'

# --- golden self-test: the engine must reproduce the export bit-for-bit
_g = L['golden']
_pe, _cut = player_effect_and_cutpoints(L)
_eta = (_pe[:, _g['player_codes']] + L['f_within'][:, _g['gd_idx']] + L['f_long'][:, _g['season_idx']]
        + np.einsum('nf,sf->sn', _g['X'], L['draws']['beta__team']))
_dev = float(np.abs(ordered_probs(_eta, _cut[:, _g['player_codes']]) - _g['probs']).max())
assert _dev < 1e-10, f'GOLDEN ROWS FAILED ({_dev:.3e}) -- engine drift. DO NOT PUBLISH.'
print(f'golden rows  : PASS ({_dev:.1e})')
del _eta


# ======================================== Data & Features ======================================== #
# (mirrors training exactly; the boards are computed over the TRAINING window)

d = pd.read_csv(f'{directory}/10_data/106_Website/data_byPlayer__SFM_II.csv')
d['kick_off'] = pd.to_datetime(d['kick_off'])
d = d.sort_values(['name_player', 'season', 'kick_off']).reset_index(drop=True)
d['gameday'] = d['gameday'].astype(int)
d = d[d.season <= _dc['train_end']].copy()

d['goals_in_match'] = d['goals_in_match'].astype(int)
d['goals_cats'] = np.where(d['goals_in_match'] >= 3, 3, d['goals_in_match'])
d['goal_appeal'] = d['goalsconceded_rank_opp'] - d['goalsscored_rank_team']

d = d.sort_values(['name_player', 'kick_off'])
d['position_player'] = (d.groupby(['season', 'name_player'])['position_player']
                        .transform(lambda s: s.bfill().ffill()))
d['position_player'] = d.groupby('name_player')['position_player'].ffill()
d = d.sort_values(['name_player', 'season', 'kick_off']).reset_index(drop=True)
d['position_FOR'] = np.where(d['position_player'] == 'Sturm', 1, 0)

d = d.loc[d['goalsscored_share_player_team'] <= 1, :].reset_index(drop=True)
d['season_nbr'] = d.groupby(['name_player'])['season'].transform(lambda x: x.factorize(sort=True)[0])

# --- ELO over the training window (same recursion as training) ---
_m = (d[d.home_pitch == 1].drop_duplicates('id_match')
      [['id_match', 'name_league', 'season', 'kick_off', 'name_team', 'name_opp',
        'goalsscored_inGame_team', 'goalsscored_inGame_opp']]
      .rename(columns={'name_team': 'home', 'name_opp': 'away',
                       'goalsscored_inGame_team': 'g_home', 'goalsscored_inGame_opp': 'g_away'}))
_missing = set(d.id_match.unique()) - set(_m.id_match)
if _missing:
    _aw = (d[d.id_match.isin(_missing) & (d.home_pitch == 0)].drop_duplicates('id_match')
           [['id_match', 'name_league', 'season', 'kick_off', 'name_opp', 'name_team',
             'goalsscored_inGame_opp', 'goalsscored_inGame_team']])
    _aw.columns = _m.columns
    _m = pd.concat([_m, _aw], ignore_index=True)
_m['res'] = np.select([_m.g_home > _m.g_away, _m.g_home < _m.g_away], [2, 0], default=1)
_rows = []
for _ll, _g_ in _m.groupby('name_league', sort=False):
    _R, _seen = {}, set()
    for _ss, _sg in _g_.sort_values(['kick_off', 'id_match']).groupby('season', sort=False):
        if _seen:
            _ret = set(_R)
            for _t in set(_sg.home) | set(_sg.away):
                _R[_t] = _R[_t] * 0.75 + 1500 * 0.25 if _t in _ret else 1300.0
        else:
            for _t in set(_sg.home) | set(_sg.away):
                _R.setdefault(_t, 1500.0)
        _seen.add(_ss)
        for _i in _sg.index:
            _h, _a = _m.at[_i, 'home'], _m.at[_i, 'away']
            _R.setdefault(_h, 1500.0); _R.setdefault(_a, 1500.0)
            _rows.append((_m.at[_i, 'id_match'], _R[_h], _R[_a]))
            _R[_h], _R[_a] = update_elo(_R[_h], _R[_a], _m.at[_i, 'res'],
                                        K=elo_cfg['K'], home_adv=elo_cfg['home_adv'])
d = d.merge(pd.DataFrame(_rows, columns=['id_match', 'elo_home', 'elo_away']), on='id_match', how='left')
d['elo_team'] = np.where(d.home_pitch == 1, d.elo_home, d.elo_away)
d['elo_opp'] = np.where(d.home_pitch == 1, d.elo_away, d.elo_home)
d['elo_diff'] = d['elo_team'] - d['elo_opp']
assert d['elo_diff'].notna().all()

# --- cross-sectional standardization (season x gameday, factor-wise) ---
d[factors_CS] = d.groupby(['season', 'gameday'], group_keys=False)[factors_CS].apply(_cs_scale_cols)
assert d[factors_CS].notna().all().all()
print(f'rows         : {len(d):,}')


# ======================================== Design matrices ======================================== #

X_obs = d[factors_team].astype(np.float64).to_numpy()          # --- PAR: observed context
X_cf = X_obs.copy()                                            # --- SAR: equalized context
for _j, _f in enumerate(factors_team):
    if _f in CONTEXT_FACTORS:
        X_cf[:, _j] = 0.0                                      # --- standardized -> CS average
    elif _f == 'home_pitch':
        X_cf[:, _j] = X_obs[:, _j].mean()                      # --- average schedule
    # else: player attribute -> held at observed
print('SAR counterfactual:')
for _j, _f in enumerate(factors_team):
    _what = ('EQUALIZED -> 0' if _f in CONTEXT_FACTORS else
             f'-> mean {X_cf[0, _j]:.3f}' if _f == 'home_pitch' else 'held at observed')
    print(f'   {_f:32s} {_what}')

pl_codes = pd.Categorical(d['name_player'], categories=players_train).codes
assert (pl_codes >= 0).all(), 'a board row references a player absent from the bundle'
gd_codes = pd.Categorical(d['gameday'], categories=L['unique_gamedays']).codes
ss_codes = d['season_nbr'].to_numpy()
assert (gd_codes >= 0).all() and ss_codes.max() < len(L['unique_seasons'])


# ======================================== Chunked accumulation ======================================== #

pe, cut = player_effect_and_cutpoints(L)
S, P = pe.shape[0], len(players_train)
beta = L['draws']['beta__team']
goal_vals = np.arange(cut.shape[-1] + 1, dtype=np.float64)     # [0, 1, 2, 3]

# --- one-hot row->player map, for an exact per-player sum via one sparse matmul per chunk
onehot = sparse.csr_matrix((np.ones(len(d)), (np.arange(len(d)), pl_codes)), shape=(len(d), P))

acc = {'PAR': np.zeros((S, P)), 'SAR': np.zeros((S, P))}
tot = {'PAR': np.zeros(S), 'SAR': np.zeros(S)}
n_rows_player = np.asarray(onehot.sum(axis=0)).ravel()

for a in range(0, len(d), CHUNK):
    b = slice(a, min(a + CHUNK, len(d)))
    base = pe[:, pl_codes[b]] + L['f_within'][:, gd_codes[b]] + L['f_long'][:, ss_codes[b]]
    cut_b = cut[:, pl_codes[b]]
    for key, Xm in (('PAR', X_obs), ('SAR', X_cf)):
        eta = base + np.einsum('nf,sf->sn', Xm[b], beta)
        pr = ordered_probs(eta, cut_b)                          # (S, chunk, K)
        if GOALS_MODE == 'expected':
            g = pr @ goal_vals                                  # (S, chunk)
        else:
            cdf = np.cumsum(pr, axis=-1)
            u = rng.random(size=pr.shape[:2])[..., None]
            g = (u > cdf).sum(axis=-1).astype(np.float64)
        acc[key] += (onehot[b].T @ g.T).T                       # (S, P) exact per-player sums
        tot[key] += g.sum(axis=1)
    if (a // CHUNK) % 5 == 0:
        print(f'   ... {min(a + CHUNK, len(d)):>7,} / {len(d):,} rows')

# --- SAR / PAR = per-player mean  -  overall mean, per draw

boards = {}
for key in ('PAR', 'SAR'):
    per_player = acc[key] / n_rows_player[None, :]
    overall = (tot[key] / len(d))[:, None]
    boards[key] = (per_player - overall).astype(np.float32)     # (S, P)


# ======================================== Export ======================================== #
# Aggregated per player per draw -- what the website actually plots. Reshaped to
# (chain, draw, name_player) so the xarray idiom downstream is unchanged.

_n_chains = int(L['provenance'].get('n_chains', 4))
_n_draws = S // _n_chains
assert _n_chains * _n_draws == S, 'draw count is not divisible by the chain count'

def _as_da(arr, name):
    return xr.DataArray(arr.reshape(_n_chains, _n_draws, P),
                        dims=('chain', 'draw', 'name_player'),
                        coords={'chain': np.arange(_n_chains), 'draw': np.arange(_n_draws),
                                'name_player': np.asarray(players_train)},
                        name=name)

dict_SARPAR = {
    'SAR': _as_da(boards['SAR'], 'SAR'),
    'PAR': _as_da(boards['PAR'], 'PAR'),
    'n_rows_player': pd.Series(n_rows_player, index=np.asarray(players_train), name='n_rows'),
    'meta': {'model': SFM_model__NAME, 'goals_mode': GOALS_MODE,
             'context_factors': CONTEXT_FACTORS, 'factors_team': factors_team,
             'replacement': 'average player (documented deviation from the paper)',
             'estimand': 'SAR: context equalized, player attributes held. PAR: observed context.',
             'train_end': _dc.get('train_end'), 'n_rows': int(len(d)),
             'note': 'AGGREGATED per player per draw -- the OG artifact stored the full '
                     'obs-level predictive (10.5 GB). Use get__SAR_PAR__SFM_II() below.'},
}

_out = f'{directory}/00_code/006_Website/01__SFMcom/SFMwebsite__v2/static/data/041_SARPAR__prod__SFM_II.pkl'
with open(_out, 'wb') as f:
    pickle.dump(dict_SARPAR, f, protocol=4)

print(f'\nWritten: {_out}  ({os.path.getsize(_out)/1e6:.1f} MB)')
_med = boards['SAR'].mean(axis=0)
_top = np.argsort(-_med)[:10]
print('\ntop-10 by mean SAR (capped goals/appearance above the average player):')
for _i in _top:
    print(f'   {players_train[_i]:32s} {_med[_i]:+.4f}')
print('\n[SUCCESS]: SAR/PAR exported.')


# ======================================================================================== #
# Drop-in for 006_042__SAR_PAR_funcCalc.get__SAR_PAR -- the aggregation now happens upstream,
# so this only slices and summarizes.
# ======================================================================================== #
def get__SAR_PAR__SFM_II(cred_region=0.9, directory=directory):
    with open(f'{directory}/00_code/006_Website/01__SFMcom/SFMwebsite__v2/'
              f'static/data/041_SARPAR__prod__SFM_II.pkl', 'rb') as fh:
        dd = pickle.load(fh)
    lo, up = (1 - cred_region) / 2, 1 - (1 - cred_region) / 2
    out = {}
    for key in ('SAR', 'PAR'):
        da = dd[key]
        out[key] = pd.DataFrame({
            'low': da.quantile(lo, dim=('chain', 'draw')).to_numpy(),
            'mid': da.quantile(0.5, dim=('chain', 'draw')).to_numpy(),
            'up': da.quantile(up, dim=('chain', 'draw')).to_numpy(),
        }, index=da['name_player'].to_numpy()).sort_values('mid', ascending=False)
    out['draws'] = {k: dd[k] for k in ('SAR', 'PAR')}      # --- for the violin plots
    out['meta'] = dd['meta']
    return out
