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
                  'time_confirmed': r['feed_time_confirmed'],
                  'first_kick_off_utc': r.get('feed_first_kick_off_utc', '')}]
                if r['feed_kick_off_utc'] else [])
        pd.DataFrame(rows, columns=kickoff._FEED_COLS + ['first_kick_off_utc']).to_csv(path, index=False)
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
                                    pd.Timestamp(r['now_utc']),
                                    first_kick_off_utc=fx['first_kick_off_utc'])
    assert bool(held.iloc[0]) is _true(r['expect_hold']), r['description']


# The board's narrow question mirrors the prune's "certainties only" rule. The prune also reads the
# OOS row's own TIME, which the date-only OOS file given to the writers does not carry, so the rows
# whose prune answer depends on that time are out of scope here.
NEEDS_OOS_TIME = {'stale_feed_fresh_oos__layers_differ',
                  'stale_feed_confirmed_oos_placeholder__err_to_omission',
                  # BY DESIGN (SFM_data, 2026-10-13): a re-dated replay is upcoming, so the prune
                  # keeps it; certainly_started says "kicked off" from first_kick_off_utc, and the
                  # guards HOLD its forecast. Different layers, different jobs.
                  'abandoned_then_redated'}


@pytest.mark.parametrize('r', [r for _, r in V.iterrows()], ids=list(V['case_id']))
def test_certainly_started_never_exceeds_the_hold(r, tmp_path):
    """Certain implies may-have-started, for every vector: the board can never drop a fixture the
    ledger would still refresh."""
    feed = kickoff.read_kickoff_feed(_stage_feed(r, tmp_path))
    fx = kickoff.attach_kickoff(
        pd.DataFrame({'fixture_key': [KEY], 'kick_off': [r['oos_date'] or None]}), feed)
    args = (fx['kick_off'], fx['kick_off_utc'], fx['time_confirmed'], pd.Timestamp(r['now_utc']))
    kw = {'first_kick_off_utc': fx['first_kick_off_utc']}
    certain = kickoff.certainly_started(*args, **kw).iloc[0]
    maybe = kickoff.may_have_started(*args, **kw).iloc[0]
    assert not (certain and not maybe)
    if r['case_id'] not in NEEDS_OOS_TIME:
        assert bool(certain) is _true(r['expect_prune_drop']), r['description']


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



# --- a fixture that kicked off is final, even when re-dated ------------------------------------

def _feed_file(tmp_path, rows, with_first=True):
    """rows: (home, away, kick_off_utc, confirmed, first_kick_off_utc)"""
    cols = kickoff._FEED_COLS + (['first_kick_off_utc'] if with_first else [])
    path = tmp_path / 'feed.csv'
    pd.DataFrame([{'season': '2026-27', 'league': 'x', 'team_home': h, 'team_away': a,
                   'kick_off_utc': t, 'time_confirmed': c, 'first_kick_off_utc': f}
                  for h, a, t, c, f in rows]
                 )[cols].to_csv(path, index=False)
    return kickoff.read_kickoff_feed(path)


NOW = pd.Timestamp('2026-10-20 10:00', tz='Europe/Berlin')


def test_an_older_feed_without_the_column_means_nothing_is_known_to_have_kicked_off(tmp_path):
    feed = _feed_file(tmp_path, [('A', 'B', '2026-10-28T18:00:00Z', True, '')], with_first=False)
    assert feed['first_kick_off_utc'].isna().all()


def test_a_recorded_first_kick_off_overrides_a_future_date(tmp_path):
    # abandoned 10 Oct, re-dated to 28 Oct: only first_kick_off_utc still tells the truth
    feed = _feed_file(tmp_path, [('A', 'B', '2026-10-28T18:00:00Z', True, '2026-10-10T12:00:00Z'),
                                 ('C', 'D', '2026-10-28T18:00:00Z', True, '')])
    args = (pd.Series([pd.NaT, pd.NaT]), feed['kick_off_utc'], feed['time_confirmed'])
    for rule in (kickoff.may_have_started, kickoff.certainly_started):
        assert list(rule(*args, NOW, first_kick_off_utc=feed['first_kick_off_utc'])) == [True, False]
        assert list(rule(*args, NOW)) == [False, False]          # without it: the trap


def test_a_board_written_before_the_first_kick_off_is_still_pre_kick_off(tmp_path):
    # rebuild_frozen_ledger asks with a board's write time as `now`
    feed = _feed_file(tmp_path, [('A', 'B', '2026-10-28T18:00:00Z', True, '2026-10-10T12:00:00Z')])
    before = pd.Timestamp('2026-09-22 07:44', tz='Europe/Berlin')
    held = kickoff.may_have_started(pd.Series([pd.NaT]), feed['kick_off_utc'], feed['time_confirmed'],
                                    before, first_kick_off_utc=feed['first_kick_off_utc'])
    assert not held.iloc[0]


def test_record_started_keeps_what_kicked_off_and_skips_the_merely_uncertain(tmp_path):
    feed = _feed_file(tmp_path, [('A', 'B', '2026-10-19T13:00:00Z', True, ''),    # confirmed, passed
                                 ('C', 'D', '2026-10-18T13:00:00Z', False, ''),   # placeholder, day over
                                 ('E', 'F', '2026-10-21T13:00:00Z', False, ''),   # placeholder, tomorrow
                                 ('G', 'H', '2026-10-20T18:45:00Z', True, ''),    # tonight
                                 ('I', 'J', '2026-10-28T18:00:00Z', True, '2026-10-10T12:00:00Z')])
    reg, new = kickoff.record_started(kickoff.load_started(tmp_path / 'none.csv'), feed, NOW)
    assert sorted(k.split('|')[2] for k in new) == ['A', 'C', 'I'] and list(reg.columns) == kickoff.STARTED_COLS
    assert reg.set_index('fixture_key').loc['2026/27|x|I|J', 'kick_off_utc'].startswith('2026-10-10 12:00')


def test_a_recorded_fixture_stays_recorded_when_the_feed_re_dates_it(tmp_path):
    first = _feed_file(tmp_path, [('A', 'B', '2026-10-10T12:00:00Z', True, '')])
    reg, _ = kickoff.record_started(kickoff.load_started(tmp_path / 'none.csv'), first, NOW)
    redated = _feed_file(tmp_path, [('A', 'B', '2026-11-04T19:00:00Z', True, '')])   # marker lost
    reg2, new = kickoff.record_started(reg, redated, NOW + pd.Timedelta(days=20))
    assert new == [] and len(reg2) == 1 and reg2.loc[0, 'kick_off_utc'].startswith('2026-10-10 12:00')
