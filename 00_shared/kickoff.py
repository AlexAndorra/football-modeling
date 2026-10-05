"""The receipts rule's kick-off check, shared by every forecast writer (006_040, 006_060).

A forecast made after kick-off is never a receipt. Each writer therefore asks one question of every
unplayed fixture before it refreshes the fixture's ledger row: MAY IT HAVE STARTED? If yes, the row
is held at its last pre-kick-off forecast (or, if it never had one, gets none). Holding only costs
freshness; refreshing too late costs a receipt, so every uncertain case answers yes.

One reader of the facts (`read_kickoff_feed`) and one rule (`may_have_started`). Until 2026-10 each
writer carried its own copy of the rule, and the shared kick-off vectors found a latent defect in
every copy. Other layers (006_021's prune) may apply a DIFFERENT rule on purpose, so that they fail
in different ways, but they should read the facts through `read_kickoff_feed`.

Facts, as the data actually carries them:
  - the OOS file's `kick_off` is a DATE (Berlin calendar), never a time;
  - the kick-off feed (fixtures_<season>__kickoff.csv) carries the time, as UTC, plus a
    `time_confirmed` flag. An unconfirmed time is a placeholder: it can sit on a LATER day than
    the real slot (Monaco v Lens: stored Sat 15:00, played Fri 20:45);
  - the feed file can be missing, and a fixture can be missing from it.
"""

import os

import pandas as pd

BERLIN = 'Europe/Berlin'

# MEASURED, NOT GUESSED -- do NOT cut to 2. Across 57 weekly 2025/26 OOS snapshots (n=555 fixtures
# inside the 14-day window) a real kick-off landed at most 2 days EARLIER than its placeholder date,
# never 3 (drift -2:27, -1:97, 0:415, +1:13); 2026/27 agrees. 3 = worst observed case + 1 spare day.
UNCONFIRMED_HOLD_DAYS = 3

_FEED_COLS = ['season', 'league', 'team_home', 'team_away', 'kick_off_utc', 'time_confirmed']


def fixture_key(season, league, home, away):
    """The stable fixture identity, ordered by venue: 'season|league|home|away' with season as
    '2026/27'. id_match is NOT stable (a postponement renumbers every later fixture)."""
    return season.astype(str).str.replace('-', '/') + '|' + league + '|' + home + '|' + away


def read_kickoff_feed(path):
    """The kick-off feed as one row per fixture: fixture_key, kick_off_utc (tz-aware UTC),
    time_confirmed (nullable boolean).

    A MISSING FILE IS NOT AN ERROR: it returns an empty feed, so every fixture takes the unconfirmed
    rule from its OOS date (agreed 2026-10-05; the OOS date equals the feed's date for every fixture,
    so losing the feed loses only the time and the confirmation). `attrs['missing']` says which case
    it was, for the caller's log. A file that EXISTS but has the wrong shape fails loudly."""
    if not os.path.exists(path):
        feed = pd.DataFrame({'fixture_key': pd.Series(dtype=object),
                             'kick_off_utc': pd.Series(dtype='datetime64[ns, UTC]'),
                             'time_confirmed': pd.Series(dtype='boolean')})
        feed.attrs['missing'] = True
        return feed
    k = pd.read_csv(path)
    absent = [c for c in _FEED_COLS if c not in k.columns]
    if absent:
        raise ValueError(f'{path}: kick-off feed lacks {absent}')
    ko = pd.to_datetime(k['kick_off_utc'], utc=True)
    conf = k['time_confirmed'].map(
        lambda v: pd.NA if pd.isna(v) else str(v).strip().lower() in ('true', '1')).astype('boolean')
    feed = pd.DataFrame({'fixture_key': fixture_key(k['season'], k['league'], k['team_home'],
                                                    k['team_away']),
                         'kick_off_utc': ko, 'time_confirmed': conf})
    dup = feed['fixture_key'].duplicated(keep=False)
    if dup.any():
        raise ValueError(f'{path}: {feed.loc[dup, "fixture_key"].nunique()} fixture(s) listed twice, '
                         f'e.g. {feed.loc[dup, "fixture_key"].iloc[0]}')
    feed.attrs['missing'] = False
    return feed


def attach_kickoff(rows, feed, key='fixture_key'):
    """Left-join the feed onto `rows` by fixture key. A fixture with no feed row gets NaT / <NA>,
    which `may_have_started` treats exactly like a missing feed file."""
    out = rows.merge(feed.rename(columns={'fixture_key': key}), on=key, how='left')
    out.index = rows.index
    out['time_confirmed'] = out['time_confirmed'].astype('boolean')
    return out


def may_have_started(oos_date, kick_off_utc, time_confirmed, now):
    """True where a fixture MAY have kicked off, so its ledger row must not be refreshed.

        confirmed feed time          -> kick_off_utc <= now (inclusive: at the instant, held)
        anything else                -> from (placeholder day - UNCONFIRMED_HOLD_DAYS) onwards,
          (unconfirmed time, no feed     where the placeholder day is the feed time's Berlin date,
           row, no feed file)            or the OOS date when there is no feed time
        no date anywhere             -> always (nothing can prove it has not started)

    `now` must be tz-aware; "today" is the Berlin calendar day, so neither the machine's timezone
    nor a fixed UTC offset (wrong after 25 Oct) can shift it. Arguments are aligned Series."""
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError('may_have_started needs a tz-aware `now`; a naive clock is how the '
                         'ledger stamps ended up 2 hours late')
    today = now.tz_convert(BERLIN).normalize().tz_localize(None)
    ko = pd.to_datetime(kick_off_utc, utc=True)
    conf = pd.Series(time_confirmed, index=ko.index).astype('boolean').fillna(False).astype(bool)
    oos_day = pd.to_datetime(pd.Series(oos_date, index=ko.index)).dt.normalize()
    exact = conf & ko.notna()
    feed_day = ko.dt.tz_convert(BERLIN).dt.normalize().dt.tz_localize(None)
    placeholder_day = feed_day.where(ko.notna(), oos_day)
    from_day = placeholder_day - pd.Timedelta(days=UNCONFIRMED_HOLD_DAYS)
    uncertain = placeholder_day.isna() | (from_day <= today)
    return (exact & (ko <= now)) | (~exact & uncertain)


def select_forecast_window(oos, started, now, horizon_days):
    """Split UNPLAYED target-season rows into (window, deferred, stale) and return the horizon.

    stale    : may have started (`started`, from may_have_started) and is dated -- never re-forecast;
               its ledger row is held at the last pre-kick-off forecast.
    window   : anchor .. anchor + horizon_days (day-inclusive) -> forecast in this run.
    deferred : beyond the horizon, or undated (it waits for a date; nothing is written for it).

    Anchor = max(today, earliest live fixture day), so an international break forecasts the next
    round instead of exiting empty (Max's rule, 64f86df). Both writers use this, so the two boards
    cover the same fixtures. Days are Berlin calendar days, like the OOS dates."""
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError('select_forecast_window needs a tz-aware `now`')
    today = now.tz_convert(BERLIN).normalize().tz_localize(None)
    day = pd.to_datetime(oos['kick_off']).dt.normalize()
    undated = day.isna()
    started = pd.Series(started, index=oos.index).astype(bool)
    stale_m = started & ~undated
    live = ~started & ~undated
    anchor = max(today, day[live].min()) if live.any() else today
    horizon = anchor + pd.Timedelta(days=horizon_days)
    window = oos[live & (day >= anchor) & (day <= horizon)].copy()
    deferred = oos[(live & (day > horizon)) | undated].copy()
    return window, deferred, oos[stale_m].copy(), horizon
