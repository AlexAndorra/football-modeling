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
baseline with a reason AND the repaired data's counts, because a repair that changes the target
moves the goal-category mix too:
    record_baseline(csv, train_end, bundle_path, reason='...', counts=training_counts(csv, train_end))
check_counts() then compares against the baseline's accepted counts instead of the bundle's. For
a bundle with a fingerprint of its own (fitted on the repaired data) none of this applies.
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


def fingerprint_frame(df, train_end, columns=FINGERPRINT_COLS, keys=('id_match', 'name_player')):
    """{'version', 'columns', 'train_end', 'n_rows', 'sha256', 'by_season'} for the rows of `df` with
    season <= train_end, over `columns`, sorted by `keys`. For a writer whose model inputs are not
    data_byPlayer's player rows (e.g. a match-level frame), pass its own columns and keys."""
    t = df.loc[df['season'] <= train_end, list(columns)]
    t = t.astype(object).where(t.notna(), '').astype(str).apply(_canon)
    t = t.sort_values(list(keys), kind='mergesort')
    by_season = {s: hashlib.sha256(g.to_csv(index=False, header=False).encode()).hexdigest()[:16]
                 for s, g in t.groupby('season', sort=True)}
    overall = hashlib.sha256(json.dumps(by_season, sort_keys=True).encode()).hexdigest()[:16]
    return {'version': 1, 'columns': list(columns), 'train_end': train_end,
            'n_rows': int(len(t)), 'sha256': overall, 'by_season': by_season}


def training_fingerprint(csv_path, train_end):
    """The SFM fingerprint: data_byPlayer's model-input columns, read as text from `csv_path`."""
    t = pd.read_csv(csv_path, dtype=str, keep_default_na=False, usecols=FINGERPRINT_COLS)
    return fingerprint_frame(t, train_end)


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


def training_counts(csv_path, train_end):
    """The count/mix contract's numbers, from the CSV as 006_040 and 006_041 serve it: training
    seasons, rows with goalsscored_share_player_team <= 1, goals_in_match capped at 3."""
    t = pd.read_csv(csv_path, usecols=['season', 'goals_in_match', 'goalsscored_share_player_team'])
    t = t[(t['season'] <= train_end) & (t['goalsscored_share_player_team'] <= 1)]
    mix = t['goals_in_match'].astype(int).clip(upper=3).value_counts(normalize=True).sort_index()
    return {'n_rows': int(len(t)), 'goal_cats_share': {int(k): float(round(v, 6)) for k, v in mix.items()}}


def expected_counts(data_contract, bundle_path):
    """(n_rows, goal_cats_share, source) -- a reviewed baseline's accepted counts if its file carries
    them, else the bundle's own."""
    p = _baseline_path(bundle_path)
    if os.path.exists(p):
        with open(p) as f:
            counts = json.load(f).get('counts')
        if counts:
            return (counts['n_rows'], {int(k): v for k, v in counts['goal_cats_share'].items()},
                    os.path.basename(p))
    return (data_contract.get('n_rows'),
            {int(k): v for k, v in data_contract.get('goal_cats_share', {}).items()}, 'bundle')


def check_counts(n_rows, goal_cats_share, data_contract, bundle_path, allow=False):
    """The count/mix contract for the training rows a script is about to serve: its own row count and
    goal-category mix against the expected ones (tolerance 1e-4 per category). Raises on a mismatch
    unless `allow`. Returns a one-line status."""
    exp_n, exp_mix, source = expected_counts(data_contract, bundle_path)
    n_ok = n_rows == exp_n
    mix_ok = all(abs(goal_cats_share.get(k, 0) - v) < 1e-4 for k, v in exp_mix.items())
    msg = (f'bundle/CSV contract: rows {exp_n:,} vs {n_rows:,} {"OK" if n_ok else "MISMATCH"} | '
           f'category mix {"OK" if mix_ok else "MISMATCH"}' + ('' if source == 'bundle' else f' (vs {source})'))
    print(msg)
    if n_ok and mix_ok:
        return msg
    why = (f'bundle/CSV contract FAILED vs {source}: training rows {exp_n:,} vs {n_rows:,}, category '
           f'mix {"OK" if mix_ok else "MISMATCH"}. The dataset was rebuilt since this bundle was '
           f'fitted / baselined, so serving it would publish numbers from a model fitted on different '
           f'data. REFIT before serving, or accept a reviewed repair with record_baseline(..., counts=...).')
    if not allow:
        raise RuntimeError(why + ' Override for one run with ALLOW_CONTRACT_MISMATCH=True.')
    print(f'   !! {why}\n   !! ALLOW_CONTRACT_MISMATCH=True -- serving a STALE MODEL deliberately. Not a pass.')
    return why


def record_baseline(csv_path, train_end, bundle_path, reason, counts=None):
    """Write <bundle>__fingerprint.json from the current CSV. Only for a bundle without a fingerprint
    of its own, or to accept a reviewed historical repair without a refit; `reason` is required.
    `counts` (training_counts of the same CSV) accepts the repaired row count and goal-category mix
    as well; leave it out only when the repair cannot move them."""
    if not reason:
        raise ValueError('record_baseline needs a reason')
    out = {'recorded': dt.datetime.now().astimezone().isoformat(timespec='seconds'),
           'reason': reason, 'csv': str(csv_path), 'bundle': os.path.basename(bundle_path),
           'fingerprint': training_fingerprint(csv_path, train_end)}
    if counts is not None:
        out['counts'] = counts
    with open(_baseline_path(bundle_path), 'w') as f:
        json.dump(out, f, indent=1)
    return out
