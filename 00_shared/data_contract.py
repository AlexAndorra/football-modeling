"""The training-data fingerprint: proof that the rows a bundle was fitted on are still the rows it
is served on.

The bundle carries no copy of its data. Until 2026-10 the only contract was the training row count
and the goal-category mix, which a repair that changes VALUES while keeping the counts passes
silently (the La Liga 2025/26 brace repair moves the mix by about 1e-4, at the threshold). The data
behind this model has been repaired many times, so the serving scripts now also compare a
fingerprint of the training rows.

What is hashed: the 19 columns of data_byPlayer that the model's features are built from, for
seasons <= train_end, one hash per season, row order and number formatting normalised ('3' and
'3.0' hash alike), other columns and current-season rows ignored. A fingerprint mismatch names the
seasons that changed.

Where the expected fingerprint comes from:
  - the bundle's data_contract['fingerprint'], written by the export in SFM_II__dev_EW.ipynb, which
    runs in Colab and so carries an INLINE copy of training_fingerprint / _canon (a test keeps the
    two copies identical);
  - for a bundle exported before fingerprints existed, a baseline file next to it
    (<bundle>__fingerprint.json) recorded with record_baseline() from the serving CSV while the
    count/mix contract passed.

A mismatch stops the run, like the count/mix contract, unless ALLOW_CONTRACT_MISMATCH overrides it
for one run. If a historical repair is deliberately accepted WITHOUT a refit, record a new
baseline with a reason: record_baseline(csv, train_end, bundle_path, reason='...').
"""

import datetime as dt
import hashlib
import json
import os

import pandas as pd

FINGERPRINT_COLS = [
    'goal', 'goals_in_match', 'goalsscored_inGame_team', 'goalsscored_inGame_opp', 'home_pitch',
    'goalsscored_rank_team', 'goalsconceded_rank_opp', 'goalsscored_cum_player', 'id_match',
    'name_team', 'name_opp', 'name_league', 'season', 'gameday', 'kick_off', 'points_diff',
    'goalsscored_share_player_team', 'name_player', 'position_player']


def _canon(col):
    """A column as canonical text: numbers as %.12g (so '3', '3.0' and 3.0000000000001 agree), text
    as is, missing as ''."""
    num = pd.to_numeric(col, errors='coerce')
    if num.notna().sum() == (col != '').sum():
        return num.map(lambda v: '' if pd.isna(v) else format(v, '.12g'))
    return col


def training_fingerprint(csv_path, train_end):
    """{'version', 'columns', 'train_end', 'n_rows', 'sha256', 'by_season'} for the rows of
    `csv_path` with season <= train_end."""
    t = pd.read_csv(csv_path, dtype=str, keep_default_na=False, usecols=FINGERPRINT_COLS)
    t = t[t['season'] <= train_end][FINGERPRINT_COLS].apply(_canon)
    t = t.sort_values(['id_match', 'name_player'], kind='mergesort')
    by_season = {s: hashlib.sha256(g.to_csv(index=False, header=False).encode()).hexdigest()[:16]
                 for s, g in t.groupby('season', sort=True)}
    overall = hashlib.sha256(json.dumps(by_season, sort_keys=True).encode()).hexdigest()[:16]
    return {'version': 1, 'columns': FINGERPRINT_COLS, 'train_end': train_end,
            'n_rows': int(len(t)), 'sha256': overall, 'by_season': by_season}


def _baseline_path(bundle_path):
    return os.path.splitext(bundle_path)[0] + '__fingerprint.json'


def expected_fingerprint(data_contract, bundle_path):
    """(fingerprint, source) -- from the bundle itself, else from its baseline file, else (None, None)."""
    if data_contract.get('fingerprint'):
        return data_contract['fingerprint'], 'bundle'
    p = _baseline_path(bundle_path)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)['fingerprint'], os.path.basename(p)
    return None, None


def changed_seasons(expected, actual):
    seasons = sorted(set(expected['by_season']) | set(actual['by_season']))
    return [s for s in seasons if expected['by_season'].get(s) != actual['by_season'].get(s)]


def check(csv_path, data_contract, bundle_path, allow=False):
    """Compare the serving CSV with the bundle's expected fingerprint. Raises on a mismatch unless
    `allow` (the run's ALLOW_CONTRACT_MISMATCH). Returns a one-line status."""
    expected, source = expected_fingerprint(data_contract, bundle_path)
    if expected is None:
        msg = ('training fingerprint: NONE to check against (bundle predates fingerprints and has no '
               'baseline file) -- count/mix contract only')
        print(f'   !! {msg}')
        return msg
    actual = training_fingerprint(csv_path, expected['train_end'])
    if actual['sha256'] == expected['sha256']:
        msg = f'training fingerprint: OK ({actual["sha256"]}, {actual["n_rows"]:,} rows, vs {source})'
        print(msg)
        return msg
    bad = changed_seasons(expected, actual)
    why = (f'training fingerprint MISMATCH vs {source}: {len(bad)} season(s) changed since the '
           f'bundle was fitted / baselined: {bad}. The training data is not the data this model was '
           f'fitted on. REFIT, or -- if the change is a reviewed repair accepted without a refit -- '
           f'record a new baseline with data_contract.record_baseline(..., reason=...).')
    if not allow:
        raise RuntimeError(why + ' Override for one run with ALLOW_CONTRACT_MISMATCH=True.')
    print(f'   !! {why}\n   !! ALLOW_CONTRACT_MISMATCH=True -- served on changed data, deliberately.')
    return why


def record_baseline(csv_path, train_end, bundle_path, reason):
    """Write <bundle>__fingerprint.json from the current CSV. Only for a bundle without a fingerprint
    of its own, or to accept a reviewed historical repair without a refit; `reason` is required."""
    if not reason:
        raise ValueError('record_baseline needs a reason')
    out = {'recorded': dt.datetime.now().astimezone().isoformat(timespec='seconds'),
           'reason': reason, 'csv': str(csv_path), 'bundle': os.path.basename(bundle_path),
           'fingerprint': training_fingerprint(csv_path, train_end)}
    with open(_baseline_path(bundle_path), 'w') as f:
        json.dump(out, f, indent=1)
    return out
