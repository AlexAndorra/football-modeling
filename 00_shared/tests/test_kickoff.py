"""The shared kick-off module against the shared kick-off vectors, through the real path.

Each vector row is staged the way production meets it: a feed FILE on disk (absent, header-only,
or one fixture), read by `read_kickoff_feed`, joined by `attach_kickoff`, decided by
`may_have_started`. Only `expect_hold` is checked; `expect_prune_drop` belongs to 006_021.

The vectors file is a frozen snapshot; quote its row count and hash with any result
(`test_vectors_identity` prints both).
"""

import hashlib
import pathlib
import sys

import pandas as pd
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import kickoff  # noqa: E402

VECTORS = HERE / 'kickoff_vectors.csv'
V = pd.read_csv(VECTORS, dtype=str, keep_default_na=False)
KEY = '2026/27|x|A|B'


def _true(s):
    return str(s).strip().lower() == 'true'


def _stage_feed(r, tmp_path):
    path = tmp_path / 'fixtures_2026-27__kickoff.csv'
    if _true(r['feed_present']):
        rows = ([{'season': '2026-27', 'league': 'x', 'team_home': 'A', 'team_away': 'B',
                  'kick_off_utc': r['feed_kick_off_utc'],
                  'time_confirmed': r['feed_time_confirmed']}]
                if r['feed_kick_off_utc'] else [])
        pd.DataFrame(rows, columns=kickoff._FEED_COLS).to_csv(path, index=False)
    return path


def test_vectors_identity():
    sha = hashlib.sha256(VECTORS.read_bytes()).hexdigest()[:12]
    print(f'{len(V)} rows | sha256 {sha}')
    assert len(V) and V['case_id'].is_unique


@pytest.mark.parametrize('r', [r for _, r in V.iterrows()], ids=list(V['case_id']))
def test_hold_rule(r, tmp_path):
    feed = kickoff.read_kickoff_feed(_stage_feed(r, tmp_path))
    assert feed.attrs['missing'] is (not _true(r['feed_present']))
    fx = kickoff.attach_kickoff(
        pd.DataFrame({'fixture_key': [KEY], 'kick_off': [r['oos_date'] or None]}), feed)
    held = kickoff.may_have_started(fx['kick_off'], fx['kick_off_utc'], fx['time_confirmed'],
                                    pd.Timestamp(r['now_utc']))
    assert bool(held.iloc[0]) is _true(r['expect_hold']), r['description']


def test_naive_clock_is_refused():
    one = pd.Series([pd.NaT])
    with pytest.raises(ValueError, match='tz-aware'):
        kickoff.may_have_started(one, one, pd.Series([pd.NA]), pd.Timestamp('2026-09-18 10:00'))


def test_feed_listing_a_fixture_twice_is_refused(tmp_path):
    p = tmp_path / 'f.csv'
    row = {'season': '2026-27', 'league': 'x', 'team_home': 'A', 'team_away': 'B',
           'kick_off_utc': '2026-09-18 18:45:00+00:00', 'time_confirmed': True}
    pd.DataFrame([row, row]).to_csv(p, index=False)
    with pytest.raises(ValueError, match='listed twice'):
        kickoff.read_kickoff_feed(p)


def test_feed_with_wrong_shape_fails_loudly(tmp_path):
    p = tmp_path / 'f.csv'
    pd.DataFrame([{'season': '2026-27', 'league': 'x'}]).to_csv(p, index=False)
    with pytest.raises(ValueError, match='lacks'):
        kickoff.read_kickoff_feed(p)


def test_window_split():
    now = pd.Timestamp('2026-09-18 10:00', tz=kickoff.BERLIN)
    oos = pd.DataFrame({'kick_off': ['2026-09-17', '2026-09-20', '2026-10-30', None]},
                       index=['stale', 'window', 'deferred', 'undated'])
    started = pd.Series([True, False, False, True], index=oos.index)
    window, deferred, stale, horizon = kickoff.select_forecast_window(oos, started, now, 14)
    assert list(window.index) == ['window']
    assert sorted(deferred.index) == ['deferred', 'undated']
    assert list(stale.index) == ['stale']
    assert horizon == pd.Timestamp('2026-10-04')    # anchored on the earliest LIVE day, 20 Sep


def test_window_anchor_skips_an_international_break():
    now = pd.Timestamp('2026-10-05 10:00', tz=kickoff.BERLIN)
    oos = pd.DataFrame({'kick_off': ['2026-10-17', '2026-10-25']})
    window, deferred, _, horizon = kickoff.select_forecast_window(oos, [False, False], now, 3)
    assert horizon == pd.Timestamp('2026-10-20')
    assert len(window) == 1 and len(deferred) == 1
