"""The ledger rebuild (rebuild_frozen_ledger.py) asks the shared kick-off rule of every board row.

Pinned on the two incidents the rule exists for, replayed from archived boards: a morning freeze of
an evening kick-off is a receipt (Monaco v Lens, 18 Sep), a board written after kick-off is not
(Hull v Man Utd, 22 Aug).
"""

import importlib.util
import os
import pathlib

import pandas as pd

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "00_02__SFMMO" / "rebuild_frozen_ledger.py"
spec = importlib.util.spec_from_file_location("rebuild_frozen_ledger", SCRIPT)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
KEY = R.KEY


def _board(dir_, written_utc, rows, name=None):
    """An archived board written at `written_utc`: rows of (home, away, kick-off date, p_home)."""
    when = pd.Timestamp(written_utc)
    local = when.tz_convert("Europe/Berlin")
    path = dir_ / (name or f"SFMMO_predictions__matches__{local:%Y-%m-%d_%H%M%S}.csv")
    pd.DataFrame(
        [{**dict.fromkeys(R.PCOLS, 0.0),
          **dict(id_match=f"{h}{a}", name_league="premier-league", season="2026/27", gameday=1,
                 kick_off=ko, home_team=h, away_team=a, status="upcoming", p_home_win=p)}
         for h, a, ko, p in rows]
    ).to_csv(path, index=False)  # fmt: skip
    os.utime(path, (when.timestamp(), when.timestamp()))
    return path


def _feed(dir_, rows):
    path = dir_ / "kickoff.csv"
    pd.DataFrame(
        [dict(season="2026-27", league="premier-league", team_home=h, team_away=a,
              kick_off_utc=t, time_confirmed=True) for h, a, t in rows]
    ).to_csv(path, index=False)  # fmt: skip
    return R.kickoff.read_kickoff_feed(path)


def _boards(*paths):
    return [(str(p), *R.board_time(p)) for p in paths]


def test_a_morning_freeze_of_an_evening_kick_off_is_kept(tmp_path):
    # Monaco v Lens: frozen 18 Sep 08:40 Berlin, kicked off 20:45 the same day
    feed = _feed(tmp_path, [("Monaco", "Lens", "2026-09-18T18:45:00Z")])
    b1 = _board(tmp_path, "2026-09-15T06:57:28Z", [("Monaco", "Lens", "2026-09-18", 0.50)])
    b2 = _board(tmp_path, "2026-09-18T06:40:39Z", [("Monaco", "Lens", "2026-09-18", 0.55)])
    led, dropped = R.rebuild(_boards(b1, b2), feed)
    row = led.set_index(KEY).loc[("2026/27", "Monaco", "Lens")]
    assert row["p_home_win"] == 0.55 and row["forecast_frozen_at"] == "2026-09-18 08:40:39"
    assert dropped == {}


def test_a_board_written_after_kick_off_is_not_a_receipt(tmp_path):
    # Hull v Man Utd: kicked off 22 Aug 13:30 Berlin; the 16:48 board still listed it as upcoming
    feed = _feed(tmp_path, [("Hull", "Man Utd", "2026-08-22T11:30:00Z")])
    b1 = _board(tmp_path, "2026-08-15T14:41:51Z", [("Hull", "Man Utd", "2026-08-22", 0.40)])
    b2 = _board(tmp_path, "2026-08-22T14:48:30Z", [("Hull", "Man Utd", "2026-08-22", 0.40)])
    led, dropped = R.rebuild(_boards(b1, b2), feed)
    row = led.set_index(KEY).loc[("2026/27", "Hull", "Man Utd")]
    assert row["forecast_frozen_at"] == "2026-08-15 16:41:51"  # the last board BEFORE kick-off
    assert dropped == {b2.name: 1}


def test_a_fixture_seen_only_after_kick_off_gets_no_row(tmp_path):
    feed = _feed(tmp_path, [("Rayo", "Alaves", "2026-08-22T17:00:00Z")])
    b = _board(tmp_path, "2026-08-22T18:00:00Z",
               [("Rayo", "Alaves", "2026-08-22", 0.4), ("A", "B", "2026-09-30", 0.6)])  # fmt: skip
    led, _ = R.rebuild(_boards(b), feed)
    assert list(led["home_team"]) == ["A"]  # results-only for Rayo v Alaves, never a late receipt


def test_rebuilt_rows_carry_the_board_text_exactly(tmp_path):
    # read with round_trip: the default parser would turn this board value into ...0102
    feed = _feed(tmp_path, [("A", "B", "2026-09-30T13:00:00Z")])
    b = _board(tmp_path, "2026-09-18T06:40:39Z", [("A", "B", "2026-09-30", 0.21200813546201028)])
    led, _ = R.rebuild(_boards(b), feed)
    out = tmp_path / "rebuilt.csv"
    led.to_csv(out, index=False)
    row = out.read_text().splitlines()[1]
    assert ",0.21200813546201028," in row
    assert row.endswith(",0,0,2026-09-18 08:40:39")  # most-likely score as 0, not 0.0


def test_board_time_is_an_instant_and_its_stamp_is_berlin_local(tmp_path):
    b = _board(tmp_path, "2026-10-25T22:30:00Z", [("A", "B", "2026-10-29", 0.5)])  # after DST ends
    stamp, now = R.board_time(b)
    assert stamp == "2026-10-25 23:30:00" and now == pd.Timestamp("2026-10-25T22:30:00Z")


def test_a_board_whose_mtime_was_not_kept_is_not_dated(tmp_path):
    b = _board(tmp_path, "2026-09-18T06:40:39Z", [("A", "B", "2026-09-30", 0.5)],
               name="SFMMO_predictions__matches__2026-09-15.csv")  # fmt: skip
    assert R.board_time(b) is None


def test_compare_reports_values_days_and_missing_fixtures():
    def led(rows):
        return pd.DataFrame(
            [{**dict.fromkeys(R.PCOLS, 0.0),
              **dict(season="2026/27", home_team=h, away_team=a, id_match="x", p_home_win=p,
                     forecast_frozen_at=s)}
             for h, a, p, s in rows]
        )  # fmt: skip

    live = led(
        [("A", "B", 0.5, "2026-09-15"), ("C", "D", 0.4, "2026-09-18"), ("E", "F", 0.3, "2026-09-18")]
    )
    rebuilt = led([("A", "B", 0.5, "2026-09-15 08:57:28"), ("C", "D", 0.45, "2026-09-15 08:57:28")])
    d = R.compare(rebuilt, live)
    assert d["only_live"] == [["2026/27", "E", "F"]] and d["only_rebuilt"] == []
    assert abs(d["max_prob_diff"] - 0.05) < 1e-12
    assert d["freeze_day_differs"] == [["2026/27", "C", "D"]]
