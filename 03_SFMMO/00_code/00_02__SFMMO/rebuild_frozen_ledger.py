#!/usr/bin/env python
"""Rebuild SFMMO_predictions__frozen.csv from the archived boards -- a RECOVERY tool.

Every weekly board (the feed 006_060 writes; the next run archives it into _vintages/) carries the
fixtures it forecast. Replaying the boards oldest to newest and keeping each fixture's LAST
pre-kick-off row reconstructs the ledger: the same rule 006_060 applies live, where an unplayed
fixture's row is refreshed run by run until it may have started. First used to move the ledger off
id_match keys: a postponement renumbers a round, and id-keyed rows were attached to the wrong
matches (La Liga 2026-08-22: Deportivo-Elche was handed Celta-Osasuna's probability).

SECOND WRITER. This is the only file other than 006_060 that produces the ledger, so it asks the
same kick-off question through the same module: a board row is taken only if its fixture could not
have started when that board was written. "When" is the board file's modification time -- the
archive is a copy2, which keeps it -- read as an instant, so the machine's timezone cannot shift
it. (An earlier version compared a date-only kick-off with that full time, so every fixture of the
board's own day counted as started: a 08:40 freeze of a 20:45 kick-off was dropped, and five
fixtures lost their receipt entirely.)

It NEVER replaces the live ledger. It writes SFMMO_predictions__frozen__rebuilt.csv beside it and
prints how the two differ; swapping them is a decision for a person who has read that diff.

    python rebuild_frozen_ledger.py        (reads the data root, writes the state root)
"""

import glob
import os
import pathlib
import re
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / '00_shared'))
import kickoff
import runroots

TARGET_SEASON = '2026/27'
PCOLS = ['p_home_win', 'p_home_win_lo', 'p_home_win_up', 'p_draw', 'p_draw_lo', 'p_draw_up',
         'p_away_win', 'p_away_win_lo', 'p_away_win_up', 'exp_goals_home', 'exp_goals_away',
         'ml_score_home', 'ml_score_away']
KEY = ['season', 'home_team', 'away_team']


def board_time(path, dated=True):
    """When a board was written: (stamp, now). `now` is the file's mtime as a tz-aware instant;
    `stamp` is that instant in Berlin local time, the format 006_060 writes. An ARCHIVED board's
    name carries its Berlin date; if that disagrees with the mtime, the copy did not keep its mtime
    and the board cannot be dated to the minute -> None (its rows are not used)."""
    now = pd.Timestamp(os.path.getmtime(path), unit='s', tz='UTC')
    local = now.tz_convert(kickoff.BERLIN)
    if dated:
        named = re.search(r'__(\d{4}-\d{2}-\d{2})', os.path.basename(path)).group(1)
        if local.strftime('%Y-%m-%d') != named:
            return None
    return local.strftime('%Y-%m-%d %H:%M:%S'), now


def rebuild(boards, feed):
    """boards: [(path, stamp, now)]. Returns (ledger, dropped) -- one row per fixture, the latest
    board row written before the fixture may have started; `dropped` counts, per board, the rows
    that were still 'upcoming' (no result yet) but may already have started."""
    rows, dropped = [], {}
    for path, stamp, now in boards:
        d = pd.read_csv(path, float_precision='round_trip')   # exact: these become receipts
        if 'status' in d.columns:                      # newer boards carry played rows too
            d = d[d['status'] == 'upcoming']
        d = d[d['p_home_win'].notna()]
        if not len(d):
            continue
        d = d.assign(fixture_key=kickoff.fixture_key(d['season'], d['name_league'],
                                                     d['home_team'], d['away_team']))
        r = kickoff.attach_kickoff(d, feed)
        late = kickoff.may_have_started(r['kick_off'], r['kick_off_utc'], r['time_confirmed'], now)
        if late.any():
            dropped[os.path.basename(path)] = int(late.sum())
        rows.append(d[~late][KEY + ['id_match'] + PCOLS].assign(forecast_frozen_at=stamp))
    led = pd.concat(rows, ignore_index=True)
    # STABLE sort: boards are concatenated oldest first, so on a tied stamp keep='last' is the
    # later board, not whichever quicksort happened to put last.
    led = led.sort_values('forecast_frozen_at', kind='stable').drop_duplicates(KEY, keep='last')
    for c in ('ml_score_home', 'ml_score_away'):   # a board with played rows reads them as float
        led[c] = led[c].astype('Int64')              # the ledger writes 2, not 2.0
    return led.reset_index(drop=True), dropped


def compare(rebuilt, live):
    """How the rebuilt ledger differs from the live one: fixtures in only one of them, the largest
    probability difference, and fixtures whose freeze DAY differs (live stamps written before
    2026-09-15 are date-only, so the time cannot be compared)."""
    m = live.merge(rebuilt, on=KEY, how='outer', suffixes=('_live', '_rebuilt'), indicator=True)
    both = m[m['_merge'] == 'both']
    day = both['forecast_frozen_at_live'].str[:10] != both['forecast_frozen_at_rebuilt'].str[:10]
    return dict(
        only_live=m.loc[m['_merge'] == 'left_only', KEY].values.tolist(),
        only_rebuilt=m.loc[m['_merge'] == 'right_only', KEY].values.tolist(),
        max_prob_diff=float(max((both[f'{c}_live'] - both[f'{c}_rebuilt']).abs().max()
                                for c in PCOLS)) if len(both) else 0.0,
        freeze_day_differs=both.loc[day, KEY].values.tolist(),
    )


def main():
    roots = runroots.load_roots()
    print(roots.describe())
    src = roots.data / '10_data' / '106_Website'
    out_dir = roots.state / '10_data' / '106_Website'
    feed = kickoff.read_kickoff_feed(src / f"fixtures_{TARGET_SEASON.replace('/', '-')}__kickoff.csv")
    if feed.attrs['missing']:
        print(f"  ⚠️  no kick-off feed -- every fixture is judged from {kickoff.UNCONFIRMED_HOLD_DAYS} "
              f"days before its board date, so the rebuild will drop rows the live ledger kept")

    boards = []
    for v in sorted(glob.glob(str(src / '_vintages' / 'SFMMO_predictions__matches__*.csv'))):
        t = board_time(v)
        if t is None:
            print(f"  ⚠️  {os.path.basename(v)}: mtime does not match the filename date -- cannot be "
                  f"dated to the minute, not used")
            continue
        boards.append((v, *t))
    current = src / 'SFMMO_predictions__matches.csv'     # newest board: archived only by the next run
    if current.exists():
        boards.append((str(current), *board_time(current, dated=False)))
    boards.sort(key=lambda b: b[1])
    print(f"rebuilding from {len(boards)} board(s), {boards[0][1]} .. {boards[-1][1]}")

    led, dropped = rebuild(boards, feed)
    for name, n in dropped.items():
        print(f"  {name}: {n} row(s) still 'upcoming' but possibly started when written -- not used")
    out = out_dir / 'SFMMO_predictions__frozen__rebuilt.csv'
    led.to_csv(out, index=False)
    print(f"\nwrote {len(led)} fixtures -> {out}")

    live_path = out_dir / 'SFMMO_predictions__frozen.csv'
    if live_path.exists():
        diff = compare(led, pd.read_csv(live_path, float_precision='round_trip'))
        print(f"vs the live ledger: {len(diff['only_live'])} fixture(s) only live, "
              f"{len(diff['only_rebuilt'])} only rebuilt, max probability diff "
              f"{diff['max_prob_diff']:.2e}, {len(diff['freeze_day_differs'])} freeze-day difference(s)")
        for k in ('only_live', 'only_rebuilt', 'freeze_day_differs'):
            if diff[k]:
                print(f"  {k}: {diff[k][:5]}")
    print("The live ledger was NOT changed. Replace it by hand only after reading the diff above.")


if __name__ == '__main__':
    main()
