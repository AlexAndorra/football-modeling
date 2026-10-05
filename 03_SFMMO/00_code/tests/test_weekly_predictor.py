"""Receipts-integrity guards of the weekly league predictor (006_060).

The frozen ledger is only honest if a fixture that has ALREADY kicked off is never re-forecast:
a forecast made after kick-off is not a receipt, and it can see results the pre-match one could
not (WC lesson L5). These tests pin the window selection, the ledger merge, and the hand-over of
stale fixtures to the feed.
"""

import importlib.util
import io
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "00_02__SFMMO"
    / "006_060__Predictions_MatchOutcome__SFMMO.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("sfmmo_weekly", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():  # pymc/pytensor import chatter is not under test
        warnings.simplefilter("ignore")
        spec.loader.exec_module(mod)
    return mod


W = _load()
KEY = ["season", "home_team", "away_team"]
PCOLS = ["p_home_win", "p_draw", "p_away_win"]
NOW = pd.Timestamp("2026-09-07 10:00", tz="Europe/Berlin")  # a Monday-morning run


def _oos(rows):
    """Team-perspective OOS rows in the REAL shape: kick_off is a DATE. The time lives in the
    kick-off feed (_feed), as it does in production."""
    return pd.DataFrame(
        rows, columns=["id_match", "name_team", "name_opp", "kick_off", "match_outcome"]
    ).assign(
        kick_off=lambda d: pd.to_datetime(d["kick_off"]).dt.normalize(),
        season="2026/27", name_league="premier-league", home_pitch=1,
    )  # fmt: skip


def _feed(tmp_path, rows, league="premier-league"):
    """A kick-off feed file on disk, read back through the shared reader: (home, away, kick-off
    UTC, confirmed) per fixture."""
    path = tmp_path / "fixtures_2026-27__kickoff.csv"
    pd.DataFrame(
        [dict(season="2026-27", league=league, team_home=h, team_away=a, kick_off_utc=t,
              time_confirmed=c) for h, a, t, c in rows]
    ).to_csv(path, index=False)  # fmt: skip
    return W.kickoff.read_kickoff_feed(path)


# --- forecast window ------------------------------------------------------------------
# The kick-off RULE is the shared module's and is tested there against the kick-off vectors.
# These tests pin how 006_060 asks it: real data shapes, the hold rule, both perspectives.


def test_window_excludes_fixtures_that_may_have_kicked_off(tmp_path):
    oos = _oos(
        [
            ("g1", "A", "B", "2026-09-06", np.nan),  # played yesterday, result not in yet
            ("g2", "C", "D", "2026-09-07", np.nan),  # tonight, 20:45
            ("g3", "E", "F", "2026-09-12", np.nan),  # inside the horizon
            ("g4", "G", "H", "2026-09-30", np.nan),  # beyond the horizon
        ]
    )
    feed = _feed(tmp_path, [("A", "B", "2026-09-06T13:00:00Z", True),
                            ("C", "D", "2026-09-07T18:45:00Z", True),
                            ("E", "F", "2026-09-12T13:30:00Z", True),
                            ("G", "H", "2026-09-30T13:30:00Z", True)])  # fmt: skip
    window, deferred, stale, _ = W.select_forecast_window(oos, feed, NOW, horizon_days=14)
    assert list(window["id_match"]) == ["g2", "g3"]
    assert list(deferred["id_match"]) == ["g4"]
    assert list(stale["id_match"]) == ["g1"]


def test_window_holds_an_unconfirmed_fixture_from_three_days_before_its_placeholder(tmp_path):
    # Monaco v Lens: stored unconfirmed on the Saturday, played on the Friday night
    oos = _oos([("g1", "A", "B", "2026-09-10", np.nan), ("g2", "C", "D", "2026-09-11", np.nan)])
    feed = _feed(tmp_path, [("A", "B", "2026-09-10T13:00:00Z", False),
                            ("C", "D", "2026-09-11T13:00:00Z", False)])  # fmt: skip
    window, _, stale, _ = W.select_forecast_window(oos, feed, NOW, horizon_days=14)
    assert list(stale["id_match"]) == ["g1"] and list(window["id_match"]) == ["g2"]


def test_window_asks_the_hold_rule_not_the_boards_narrow_one(tmp_path):
    # Not CERTAINLY started (a board would still show it), but this feed is a receipts channel:
    # a fixture that may have started never ships a fresh number.
    oos = _oos([("g1", "A", "B", "2026-09-09", np.nan)])
    feed = _feed(tmp_path, [("A", "B", "2026-09-09T13:00:00Z", False)])
    rows = W.kickoff.attach_kickoff(oos.assign(fixture_key=W.fixture_keys(oos)), feed)
    assert not W.kickoff.certainly_started(
        rows["kick_off"], rows["kick_off_utc"], rows["time_confirmed"], NOW
    ).any()
    window, _, stale, _ = W.select_forecast_window(oos, feed, NOW, horizon_days=14)
    assert list(stale["id_match"]) == ["g1"] and window.empty


def test_window_without_a_kickoff_feed_holds_from_three_days_before_the_oos_date(tmp_path):
    feed = W.kickoff.read_kickoff_feed(tmp_path / "no_such_feed.csv")
    oos = _oos([("g1", "A", "B", "2026-09-08", np.nan), ("g2", "C", "D", "2026-09-12", np.nan)])
    window, _, stale, _ = W.select_forecast_window(oos, feed, NOW, horizon_days=14)
    assert feed.attrs["missing"]
    assert list(stale["id_match"]) == ["g1"] and list(window["id_match"]) == ["g2"]


def test_both_perspectives_of_a_fixture_take_the_home_ordered_key(tmp_path):
    # Keyed naively, the away row (B, A) would find no feed row, fall back to its OOS date --
    # today, inside the hold -- and be held while its home row is forecast.
    oos = pd.concat(
        [_oos([("g1", "A", "B", "2026-09-07", np.nan)]),
         _oos([("g1", "B", "A", "2026-09-07", np.nan)]).assign(home_pitch=0)],
        ignore_index=True,
    )  # fmt: skip
    feed = _feed(tmp_path, [("A", "B", "2026-09-07T18:45:00Z", True)])  # tonight
    window, _, stale, _ = W.select_forecast_window(oos, feed, NOW, horizon_days=14)
    assert sorted(window["home_pitch"]) == [0, 1] and stale.empty


def test_window_horizon_is_inclusive_of_the_last_day(tmp_path):
    oos = _oos([("g0", "C", "D", "2026-09-07", np.nan), ("g1", "A", "B", "2026-09-21", np.nan)])
    feed = _feed(tmp_path, [("C", "D", "2026-09-07T18:00:00Z", True),   # tonight: anchor = today
                            ("A", "B", "2026-09-21T18:00:00Z", True)])  # NOW + 14 days, evening
    window, _, _, _ = W.select_forecast_window(oos, feed, NOW, horizon_days=14)
    assert list(window["id_match"]) == ["g0", "g1"]


# --- ledger merge ---------------------------------------------------------------------


def _ledger(rows):
    return pd.DataFrame(rows, columns=KEY + ["id_match"] + PCOLS + ["forecast_frozen_at"])


def test_merge_keeps_ledger_rows_absent_from_fresh_verbatim():
    led = _ledger([("2026/27", "A", "B", "g1", 0.5, 0.3, 0.2, "2026-09-01")])
    fresh = _ledger([("2026/27", "C", "D", "g2", 0.4, 0.3, 0.3, "2026-09-07")])
    out = W.merge_frozen_ledger(led, fresh, KEY)
    ab = out.set_index(KEY).loc[("2026/27", "A", "B")]
    assert ab["p_home_win"] == 0.5 and ab["forecast_frozen_at"] == "2026-09-01"
    assert len(out) == 2


def test_merge_refreshes_rows_present_in_fresh():
    led = _ledger([("2026/27", "A", "B", "g1", 0.5, 0.3, 0.2, "2026-09-01")])
    fresh = _ledger([("2026/27", "A", "B", "g1", 0.6, 0.25, 0.15, "2026-09-07")])
    out = W.merge_frozen_ledger(led, fresh, KEY)
    assert len(out) == 1 and out.loc[0, "p_home_win"] == 0.6


@pytest.mark.parametrize("side", ["ledger", "fresh"])
def test_merge_refuses_duplicate_keys(side):
    dup = _ledger(
        [
            ("2026/27", "A", "B", "g1", 0.5, 0.3, 0.2, "2026-09-01"),
            ("2026/27", "A", "B", "g9", 0.4, 0.3, 0.3, "2026-09-02"),
        ]
    )
    clean = _ledger([("2026/27", "C", "D", "g2", 0.4, 0.3, 0.3, "2026-09-07")])
    led, fresh = (dup, clean) if side == "ledger" else (clean, dup)
    with pytest.raises(ValueError, match="duplicate"):
        W.merge_frozen_ledger(led, fresh, KEY)


def test_stale_fixture_frozen_row_survives_a_run_end_to_end(tmp_path):
    """The scenario the guard exists for: result feed lags, the run happens anyway."""
    led = _ledger([("2026/27", "A", "B", "g1", 0.5, 0.3, 0.2, "2026-09-01")])
    oos = _oos([("g1", "A", "B", "2026-09-06", np.nan), ("g2", "C", "D", "2026-09-12", np.nan)])
    feed = _feed(tmp_path, [("A", "B", "2026-09-06T13:00:00Z", True),   # kicked off, no result
                            ("C", "D", "2026-09-12T13:00:00Z", True)])  # fmt: skip
    window, _, stale, _ = W.select_forecast_window(oos, feed, NOW, horizon_days=14)
    fresh = _ledger([("2026/27", "C", "D", "g2", 0.4, 0.3, 0.3, "2026-09-07")])
    assert set(fresh["id_match"]) == set(window["id_match"])  # only the window is forecast
    out = W.merge_frozen_ledger(led, fresh, KEY)
    ab = out.set_index(KEY).loc[("2026/27", "A", "B")]
    assert ab["p_home_win"] == 0.5 and ab["forecast_frozen_at"] == "2026-09-01"
    assert list(stale["id_match"]) == ["g1"]


# --- stale fixtures in the feed -------------------------------------------------------


def test_stale_rows_enter_the_feed_with_their_frozen_numbers():
    led = _ledger([("2026/27", "A", "B", "g1", 0.5, 0.3, 0.2, "2026-09-01")])
    stale = pd.DataFrame(
        {
            "id_match": ["g1", "g7"],
            "name_league": ["l", "l"],
            "season": ["2026/27", "2026/27"],
            "gameday": [3, 3],
            "kick_off": pd.to_datetime(["2026-09-06 15:00", "2026-09-06 17:00"]),
            "name_team": ["A", "X"],
            "name_opp": ["B", "Y"],
        }
    )
    rows, missing = W.stale_feed_rows(stale, led, KEY, PCOLS)
    assert [r["id_match"] for r in rows] == ["g1"]
    assert rows[0]["status"] == "upcoming"
    assert rows[0]["p_home_win"] == 0.5 and rows[0]["forecast_frozen_at"] == "2026-09-01"
    assert rows[0]["home_team"] == "A" and rows[0]["away_team"] == "B"
    assert missing == ["X v Y"]


# --- red-team round 1 additions -------------------------------------------------------


def test_window_late_on_match_day_holds_the_started_and_the_unconfirmed(tmp_path):
    late = pd.Timestamp("2026-09-07 22:00", tz="Europe/Berlin")
    oos = _oos(
        [
            ("g1", "A", "B", "2026-09-07", np.nan),  # kicked off 7 h ago, no result
            ("g2", "C", "D", "2026-09-07", np.nan),  # unconfirmed slot today -> cannot be judged
            ("g3", "E", "F", "2026-09-07", np.nan),  # later tonight, confirmed -> forecast
        ]
    )
    feed = _feed(tmp_path, [("A", "B", "2026-09-07T13:00:00Z", True),
                            ("C", "D", "2026-09-07T13:00:00Z", False),
                            ("E", "F", "2026-09-07T20:30:00Z", True)])  # fmt: skip
    window, _, stale, _ = W.select_forecast_window(oos, feed, late, horizon_days=14)
    assert list(stale["id_match"]) == ["g1", "g2"]
    assert list(window["id_match"]) == ["g3"]


def test_stale_rows_carry_elo_when_available():
    led = _ledger([("2026/27", "A", "B", "g1", 0.5, 0.3, 0.2, "2026-09-01")])
    stale = pd.DataFrame(
        {
            "id_match": ["g1"],
            "name_league": ["l"],
            "season": ["2026/27"],
            "gameday": [3],
            "kick_off": pd.to_datetime(["2026-09-06 15:00"]),
            "name_team": ["A"],
            "name_opp": ["B"],
            "elo_team": [1600.04],
            "elo_opp": [1450.06],
        }
    )
    rows, _ = W.stale_feed_rows(stale, led, KEY, PCOLS)
    assert rows[0]["elo_home"] == 1600.0 and rows[0]["elo_away"] == 1450.1


def test_previous_board_rows_are_carried_forward_for_stale_fixtures():
    prev = {
        "scorelines": pd.DataFrame(
            {
                "id_match": ["g1", "g1", "g5"],
                "home_team": ["A", "A", "X"],
                "away_team": ["B", "B", "Y"],
                "home_goals": [0, 1, 0],
                "away_goals": [0, 0, 0],
                "p_mid": [0.1, 0.2, 0.3],
            }
        ),
        "team_goals": pd.DataFrame(
            {
                "id_match": ["g1", "g1", "g5", "g5"],
                "team": ["A", "B", "X", "Y"],
                "opponent": ["B", "A", "Y", "X"],
                "is_home": [1, 0, 1, 0],
                "p_goals_0": [0.2, 0.3, 0.4, 0.5],
            }
        ),
    }
    stale = pd.DataFrame(
        {"id_match": ["g1", "g8"], "name_team": ["A", "Q"], "name_opp": ["B", "R"]}
    )
    grid, team, missing = W.carry_forward_board_rows(prev, stale)
    assert len(grid) == 2 and set(grid["home_team"]) == {"A"}
    assert len(team) == 2 and set(team["team"]) == {"A", "B"}
    assert missing == ["Q v R"]


# --- red-team round 2 additions -------------------------------------------------------


def test_window_defers_rows_without_kick_off(tmp_path):
    # an undated fixture is neither forecast nor treated as kicked off: it waits for a date
    # (main() warns), rather than crashing the weekly run or vanishing silently
    oos = _oos([("g1", "A", "B", "2026-09-12", np.nan), ("g2", "C", "D", "2026-09-13", np.nan)])
    oos.loc[1, "kick_off"] = pd.NaT
    feed = _feed(tmp_path, [("A", "B", "2026-09-12T13:00:00Z", True)])
    window, deferred, stale, horizon = W.select_forecast_window(oos, feed, NOW, horizon_days=14)
    assert (
        list(window["id_match"]) == ["g1"] and list(deferred["id_match"]) == ["g2"] and stale.empty
    )
    assert horizon == pd.Timestamp("2026-09-26")


def test_carry_forward_ignores_the_return_leg_and_remaps_renumbered_ids():
    prev = {
        "scorelines": pd.DataFrame(
            {
                "id_match": ["g6", "g9"],
                "home_team": ["A", "B"],
                "away_team": ["B", "A"],
                "home_goals": [0, 0],
                "away_goals": [0, 0],
                "p_mid": [0.1, 0.2],
            }
        ),
        "team_goals": pd.DataFrame(
            {
                "id_match": ["g6", "g6", "g9", "g9"],
                "team": ["A", "B", "B", "A"],
                "opponent": ["B", "A", "A", "B"],
                "is_home": [1, 0, 1, 0],
                "p_goals_0": [0.2, 0.3, 0.4, 0.5],
            }
        ),
    }
    stale = pd.DataFrame({"id_match": ["g5"], "name_team": ["A"], "name_opp": ["B"]})  # renumbered
    grid, team, missing = W.carry_forward_board_rows(prev, stale)
    assert list(grid["id_match"]) == ["g5"] and list(grid["p_mid"]) == [0.1]
    assert sorted(team["p_goals_0"]) == [0.2, 0.3] and set(team["id_match"]) == {"g5"}
    assert missing == []


def test_window_anchors_at_the_earliest_forecastable_fixture_during_a_break(tmp_path):
    # International break: nothing for 18 days. A today-anchored window would come back empty
    # and the run would exit instead of forecasting the next round (Max, 64f86df).
    oos = _oos(
        [
            ("g1", "A", "B", "2026-09-25", np.nan),
            ("g2", "C", "D", "2026-09-26", np.nan),
            ("g3", "E", "F", "2026-10-12", np.nan),  # 14 days past the 09-25 anchor + 3
        ]
    )
    feed = _feed(tmp_path, [("A", "B", "2026-09-25T13:00:00Z", True),
                            ("C", "D", "2026-09-26T13:00:00Z", True),
                            ("E", "F", "2026-10-12T13:00:00Z", True)])  # fmt: skip
    window, deferred, stale, horizon = W.select_forecast_window(oos, feed, NOW, horizon_days=14)
    assert list(window["id_match"]) == ["g1", "g2"]
    assert list(deferred["id_match"]) == ["g3"]
    assert horizon == pd.Timestamp("2026-10-09")
    assert len(stale) == 0


def test_window_anchor_ignores_kicked_off_fixtures(tmp_path):
    # A postponed / unrecorded fixture carrying a past kick-off must neither be re-forecast nor
    # pull the anchor backwards: the anchor is the earliest fixture that is still forecastable.
    oos = _oos(
        [
            ("g0", "X", "Y", "2026-08-30", np.nan),  # kicked off, no result
            ("g1", "A", "B", "2026-09-25", np.nan),
            ("g2", "E", "F", "2026-10-12", np.nan),
        ]
    )
    feed = _feed(tmp_path, [("X", "Y", "2026-08-30T13:00:00Z", True),
                            ("A", "B", "2026-09-25T13:00:00Z", True),
                            ("E", "F", "2026-10-12T13:00:00Z", True)])  # fmt: skip
    window, deferred, stale, horizon = W.select_forecast_window(oos, feed, NOW, horizon_days=14)
    assert list(stale["id_match"]) == ["g0"]
    assert list(window["id_match"]) == ["g1"]
    assert list(deferred["id_match"]) == ["g2"]
    assert horizon == pd.Timestamp("2026-10-09")


def test_window_starts_at_the_next_forecastable_fixture(tmp_path):
    # Max's rule (64f86df): anchor = max(today, earliest unplayed), here over fixtures that have
    # not kicked off. Same rule as 006_040, so the two boards cover the same fixtures.
    oos = _oos([("g1", "A", "B", "2026-09-08", np.nan), ("g2", "C", "D", "2026-09-22", np.nan)])
    feed = _feed(tmp_path, [("A", "B", "2026-09-08T13:00:00Z", True),
                            ("C", "D", "2026-09-22T13:00:00Z", True)])  # fmt: skip
    window, deferred, stale, horizon = W.select_forecast_window(oos, feed, NOW, horizon_days=14)
    assert horizon == pd.Timestamp("2026-09-22")
    assert list(window["id_match"]) == ["g1", "g2"]
    assert deferred.empty


def test_window_anchor_is_today_when_a_fixture_is_still_to_come_today(tmp_path):
    oos = _oos([("g1", "A", "B", "2026-09-07", np.nan), ("g2", "C", "D", "2026-09-22", np.nan)])
    feed = _feed(tmp_path, [("A", "B", "2026-09-07T18:00:00Z", True),
                            ("C", "D", "2026-09-22T13:00:00Z", True)])  # fmt: skip
    window, deferred, stale, horizon = W.select_forecast_window(oos, feed, NOW, horizon_days=14)
    assert horizon == pd.Timestamp("2026-09-21")
    assert list(window["id_match"]) == ["g1"]
    assert list(deferred["id_match"]) == ["g2"]


def test_merge_with_an_empty_ledger_is_the_first_run_path():
    # First run (no ledger file yet) must go through the same guard: a duplicate key in the
    # fresh board would otherwise be written and make every later `.loc[key]` return a Series.
    empty = _ledger([]).iloc[0:0]
    clean = _ledger([("2026/27", "C", "D", "g2", 0.4, 0.3, 0.3, "2026-09-07")])
    out = W.merge_frozen_ledger(empty, clean, KEY)
    assert len(out) == 1 and list(out["home_team"]) == ["C"]
    dup = _ledger(
        [
            ("2026/27", "A", "B", "g1", 0.5, 0.3, 0.2, "2026-09-07"),
            ("2026/27", "A", "B", "g9", 0.4, 0.3, 0.3, "2026-09-07"),
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        W.merge_frozen_ledger(empty, dup, KEY)


# --- main(), end to end on toy files ----------------------------------------------------
# The bundle and the season CSVs live outside the repo, so main() is driven here with toy
# files and a stand-in for the model step: everything AROUND the model (window, ledger,
# feed, export) runs for real.

FULL_PCOLS = [
    "p_home_win", "p_home_win_lo", "p_home_win_up", "p_draw", "p_draw_lo", "p_draw_up",
    "p_away_win", "p_away_win_lo", "p_away_win_up", "exp_goals_home", "exp_goals_away",
    "ml_score_home", "ml_score_away",
]  # fmt: skip


def _raw(id_match, home, away, kick_off, gh, ga):
    base = dict(
        id_match=id_match, name_league="PL", id_league=1, season="2026/27", gameday=0,
        kick_off=kick_off, points_team=0, points_opp=0, points_diff=0,
        goalsscored_cum_team=0, goalsscored_cum_opp=0,
        goalsconceded_cum_team=0, goalsconceded_cum_opp=0,
    )  # fmt: skip
    return [
        dict(base, name_player=f"{home}1", name_team=home, name_opp=away, home_pitch=1,
             goalsscored_inGame_team=gh, goalsscored_inGame_opp=ga),
        dict(base, name_player=f"{away}1", name_team=away, name_opp=home, home_pitch=0,
             goalsscored_inGame_team=ga, goalsscored_inGame_opp=gh),
    ]  # fmt: skip


def _frozen(home, away, id_match, p_home, stamp):
    row = dict(season="2026/27", home_team=home, away_team=away, id_match=id_match)
    row.update({c: 0.0 for c in FULL_PCOLS})
    row.update(
        p_home_win=p_home, p_draw=0.3, p_away_win=0.7 - p_home, ml_score_home=1, ml_score_away=1
    )
    row["forecast_frozen_at"] = stamp
    return row


@pytest.fixture
def season_files(tmp_path, monkeypatch):
    import cloudpickle

    # real shapes: the OOS file carries the DATE, the kick-off feed the confirmed UTC time
    future = (pd.Timestamp.now(tz="Europe/Berlin") + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    hist = pd.DataFrame(_raw("PL_GD01_AB", "A", "B", "2026-09-01", 2, 1))
    oos = pd.DataFrame(
        _raw("PL_GD02_CD", "C", "D", "2026-09-20", np.nan, np.nan)  # kicked off, no result
        + _raw("PL_GD03_EF", "E", "F", future, np.nan, np.nan)
    )
    kickoff_feed = pd.DataFrame(
        [dict(season="2026-27", league="PL", team_home=h, team_away=a, kick_off_utc=t,
              time_confirmed=True)
         for h, a, t in [("C", "D", "2026-09-20T13:00:00Z"), ("E", "F", f"{future}T18:00:00Z")]]
    )  # fmt: skip
    ledger = pd.DataFrame(
        [
            _frozen("A", "B", "PL_GD01_AB", 0.5, "2026-08-30"),
            _frozen("C", "D", "PL_GD02_CD", 0.4, "2026-09-18"),
        ]
    )
    prev_board = dict(
        scorelines=pd.DataFrame(
            [dict(id_match="PL_GD02_CD", home_team="C", away_team="D", home_goals=1, away_goals=1,
                  p_mid=0.12, p_lo=0.1, p_up=0.14)]
        ),
        team_goals=pd.DataFrame(
            [dict(id_match="PL_GD02_CD", name_league="PL", gameday=2, kick_off="2026-09-20 15:00",
                  team=t, opponent=o, is_home=h, exp_goals=1.2)
             for t, o, h in [("C", "D", 1), ("D", "C", 0)]]
        ),
    )  # fmt: skip
    bundle = dict(
        meta=dict(devVersion="K", train_end="2025/26", n_train_rows=1, created="toy",
                  factors_CS=[], factors=["home_pitch"], seed=1),
        model=None, idata=None, team_to_idx={}, names_teams=[],
        train_means=pd.DataFrame(), train_stds=pd.DataFrame(), rho=0.0,
    )  # fmt: skip
    p = {k: tmp_path / v for k, v in dict(
        BUNDLE_PATH="bundle.pkl", HIST_PATH="hist.csv", OOS_PATH="oos.csv",
        FROZEN_LEDGER="frozen.csv", OUT_MATCH_CSV="matches.csv", OUT_GRID_CSV="grid.csv",
        OUT_TEAM_CSV="team.csv", OUT_PKL="board.pkl", KICKOFF_PATH="kickoff.csv").items()}  # fmt: skip
    with open(p["BUNDLE_PATH"], "wb") as f:
        cloudpickle.dump(bundle, f)
    with open(p["OUT_PKL"], "wb") as f:
        cloudpickle.dump(prev_board, f)
    hist.to_csv(p["HIST_PATH"], index=False)
    oos.to_csv(p["OOS_PATH"], index=False)
    ledger.to_csv(p["FROZEN_LEDGER"], index=False)
    kickoff_feed.to_csv(p["KICKOFF_PATH"], index=False)
    for k, v in p.items():
        monkeypatch.setattr(W, k, str(v))
    monkeypatch.setattr(W, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(W, "ARCHIVE_VINTAGES", False)
    return p


def test_main_rebuilds_the_feed_when_only_kicked_off_fixtures_remain(season_files, monkeypatch):
    # Sourcery on PR #3: with nothing left to forecast, main() used to exit before the feed,
    # so results that landed since the last run never reached the site.
    oos = pd.read_csv(season_files["OOS_PATH"])
    oos[oos["id_match"] == "PL_GD02_CD"].to_csv(season_files["OOS_PATH"], index=False)

    def no_model(*a, **k):
        raise AssertionError("nothing is in the window: the model must not run")

    monkeypatch.setattr(W, "forecast_fixtures", no_model)
    ledger_before = season_files["FROZEN_LEDGER"].read_bytes()
    W.main()
    assert season_files["FROZEN_LEDGER"].read_bytes() == ledger_before  # nothing forecast
    feed = pd.read_csv(season_files["OUT_MATCH_CSV"]).set_index("id_match")
    assert (
        feed.loc["PL_GD01_AB", "status"] == "finished" and feed.loc["PL_GD01_AB", "home_goals"] == 2
    )
    assert feed.loc["PL_GD01_AB", "p_home_win"] == 0.5  # the forecast frozen before kick-off
    assert feed.loc["PL_GD02_CD", "status"] == "upcoming"
    assert feed.loc["PL_GD02_CD", "p_home_win"] == 0.4
    assert feed.loc["PL_GD02_CD", "forecast_frozen_at"] == "2026-09-18"
    assert list(pd.read_csv(season_files["OUT_GRID_CSV"])["id_match"]) == ["PL_GD02_CD"]
    assert set(pd.read_csv(season_files["OUT_TEAM_CSV"])["team"]) == {"C", "D"}


def _fake_model(seen):
    """A stand-in for forecast_fixtures: one fixture at p_home_win 0.55, plus its board rows."""

    def fake_model(oos, *a, **k):
        seen["ids"] = sorted(oos["id_match"].unique())
        h = oos[oos["home_pitch"] == 1].iloc[0]
        row = _frozen(h["name_team"], h["name_opp"], h["id_match"], 0.55, None)
        row.pop("forecast_frozen_at")
        row.update(
            name_league="PL", gameday=3, kick_off=h["kick_off"], elo_home=1500.0, elo_away=1500.0
        )
        grid = [dict(id_match=h["id_match"], home_team="E", away_team="F", home_goals=1, away_goals=0,
                     p_mid=0.1, p_lo=0.08, p_up=0.12)]  # fmt: skip
        team = [dict(id_match=h["id_match"], name_league="PL", gameday=3, kick_off=h["kick_off"],
                     team=t, opponent=o, is_home=i, exp_goals=1.0)
                for t, o, i in [("E", "F", 1), ("F", "E", 0)]]  # fmt: skip
        return pd.DataFrame([row]), grid, team, 0.0

    return fake_model


def test_main_forecasts_only_fixtures_that_have_not_kicked_off(season_files, monkeypatch):
    seen = {}
    monkeypatch.setattr(W, "forecast_fixtures", _fake_model(seen))
    W.main()
    assert seen["ids"] == ["PL_GD03_EF"]  # the kicked-off C v D is never sent to the model
    led = pd.read_csv(season_files["FROZEN_LEDGER"]).set_index("id_match")
    assert led.loc["PL_GD02_CD", "forecast_frozen_at"] == "2026-09-18"  # untouched
    assert led.loc["PL_GD03_EF", "p_home_win"] == 0.55
    feed = pd.read_csv(season_files["OUT_MATCH_CSV"]).set_index("id_match")
    assert feed.loc[["PL_GD01_AB", "PL_GD02_CD", "PL_GD03_EF"], "status"].tolist() == [
        "finished", "upcoming", "upcoming",
    ]  # fmt: skip
    assert feed.loc["PL_GD02_CD", "p_home_win"] == 0.4


def test_main_discards_a_forecast_whose_fixture_reached_its_hold_point_during_the_run(
    season_files, monkeypatch
):
    # The model samples for minutes: a run that starts at 20:40 for a 20:45 kick-off must not
    # write the 20:46 forecast. The rule is asked again after sampling; here the clock "moves"
    # between the two asks and E v F (no frozen row) gets no receipt and no feed row.
    seen, asks = {}, []
    rule = W.may_have_started

    def clock_moves(oos, feed, now):
        asks.append(now)
        started = rule(oos, feed, now)
        return started if len(asks) == 1 else started | True

    monkeypatch.setattr(W, "may_have_started", clock_moves)
    monkeypatch.setattr(W, "forecast_fixtures", _fake_model(seen))
    ledger_before = season_files["FROZEN_LEDGER"].read_bytes()
    W.main()
    assert len(asks) == 2 and seen["ids"] == ["PL_GD03_EF"]  # it WAS forecast...
    assert season_files["FROZEN_LEDGER"].read_bytes() == ledger_before  # ...and never frozen
    feed = pd.read_csv(season_files["OUT_MATCH_CSV"])
    assert "PL_GD03_EF" not in set(feed["id_match"])  # ...nor shipped
    assert "PL_GD03_EF" not in set(pd.read_csv(season_files["OUT_GRID_CSV"])["id_match"])
    assert feed.set_index("id_match").loc["PL_GD02_CD", "p_home_win"] == 0.4


def test_main_writes_finished_and_held_ledger_rows_back_byte_for_byte(season_files, monkeypatch):
    # The default float parser is off by up to 1 ULP: this value, from the live ledger, reads
    # back as 0.1657176973373485. A receipt must not change by a digit, so the ledger is read
    # exactly and a row the run does not refresh is written back as it was read.
    drifts = "0.16571769733734856"
    assert repr(pd.read_csv(io.StringIO(f"x\n{drifts}"))["x"][0]) != drifts  # the trap is real
    led = pd.read_csv(season_files["FROZEN_LEDGER"])
    led["p_draw"] = float(drifts)
    led["p_away_win"] = 1 - led["p_home_win"] - led["p_draw"]  # rows still sum to 1 (export gate)
    led.to_csv(season_files["FROZEN_LEDGER"], index=False)
    before = season_files["FROZEN_LEDGER"].read_text().splitlines()[1:]
    assert len(before) == 2 and all(drifts in line for line in before)
    monkeypatch.setattr(W, "forecast_fixtures", _fake_model({}))
    W.main()
    after = set(season_files["FROZEN_LEDGER"].read_text().splitlines())
    assert all(line in after for line in before)  # A v B finished, C v D held: untouched


def test_main_with_nothing_to_forecast_still_refuses_a_duplicated_ledger(season_files, monkeypatch):
    oos = pd.read_csv(season_files["OOS_PATH"])
    oos[oos["id_match"] == "PL_GD02_CD"].to_csv(season_files["OOS_PATH"], index=False)
    led = pd.read_csv(season_files["FROZEN_LEDGER"])
    pd.concat([led, led.iloc[[1]]]).to_csv(season_files["FROZEN_LEDGER"], index=False)
    monkeypatch.setattr(
        W, "forecast_fixtures", lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    )
    with pytest.raises(ValueError, match="duplicate"):
        W.main()


def test_main_with_only_undated_fixtures_left_ships_readable_tables(season_files, monkeypatch):
    oos = pd.read_csv(season_files["OOS_PATH"])
    oos = oos[oos["id_match"] == "PL_GD03_EF"].assign(kick_off=np.nan)
    oos.to_csv(season_files["OOS_PATH"], index=False)
    monkeypatch.setattr(
        W, "forecast_fixtures", lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    )
    W.main()
    assert list(pd.read_csv(season_files["OUT_MATCH_CSV"])["id_match"]) == ["PL_GD01_AB"]
    grid = pd.read_csv(season_files["OUT_GRID_CSV"])  # header-only, not an unreadable empty file
    assert grid.empty and "p_mid" in grid.columns
    assert pd.read_csv(season_files["OUT_TEAM_CSV"]).empty


# --- season end -------------------------------------------------------------------------------


def _season(season_files, played):
    hist = pd.DataFrame([r for args in played for r in _raw(*args)])
    hist.to_csv(season_files["HIST_PATH"], index=False)
    pd.read_csv(season_files["OOS_PATH"]).iloc[0:0].to_csv(season_files["OOS_PATH"], index=False)


def test_main_ships_the_final_results_once_the_season_is_complete(season_files, monkeypatch):
    # two-team league: A v B and B v A is the whole double round-robin
    _season(
        season_files,
        [
            ("PL_GD01_AB", "A", "B", "2026-09-01 15:00", 2, 1),
            ("PL_GD02_BA", "B", "A", "2027-05-20 15:00", 0, 0),
        ],
    )
    monkeypatch.setattr(
        W, "forecast_fixtures", lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    )
    ledger_before = season_files["FROZEN_LEDGER"].read_bytes()
    W.main()
    assert season_files["FROZEN_LEDGER"].read_bytes() == ledger_before
    feed = pd.read_csv(season_files["OUT_MATCH_CSV"]).set_index("id_match")
    assert feed["status"].tolist() == ["finished", "finished"]
    assert feed.loc["PL_GD01_AB", "p_home_win"] == 0.5 and feed.loc["PL_GD02_BA", "home_goals"] == 0
    grid = pd.read_csv(season_files["OUT_GRID_CSV"])
    assert grid.empty and "p_mid" in grid.columns


def test_main_still_stops_when_the_fixture_feed_is_empty_mid_season(season_files, monkeypatch):
    # B v A is still to play but missing from the OOS file: a pipeline problem, not a season end
    _season(season_files, [("PL_GD01_AB", "A", "B", "2026-09-01 15:00", 2, 1)])
    monkeypatch.setattr(
        W, "forecast_fixtures", lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    )
    with pytest.raises(SystemExit, match="No unplayed"):
        W.main()


def test_main_stops_when_the_next_season_is_waiting_and_target_season_was_not_bumped(
    season_files, monkeypatch
):
    _season(
        season_files,
        [
            ("PL_GD01_AB", "A", "B", "2026-09-01 15:00", 2, 1),
            ("PL_GD02_BA", "B", "A", "2027-05-20 15:00", 0, 0),
        ],
    )
    nxt = pd.DataFrame(_raw("PL28_GD01_AB", "A", "B", "2027-08-20 20:00", np.nan, np.nan)).assign(
        season="2027/28"
    )
    nxt.to_csv(season_files["OOS_PATH"], index=False)
    with pytest.raises(SystemExit, match="TARGET_SEASON"):
        W.main()


def _cd(rows):
    """home-perspective match rows: (league, home, away, outcome)"""
    return pd.DataFrame(
        [dict(season="2026/27", home_pitch=1, name_league=lg, name_team=h, name_opp=a, match_outcome=o)
         for lg, h, a, o in rows]
    )  # fmt: skip


def _round_robin(teams, league="L"):
    return [(league, h, a, 1.0) for h in teams for a in teams if h != a]


def test_season_complete_ignores_relegation_play_off_guests():
    rr = _round_robin(["A", "B", "C", "D"])
    playoff = [("L", "D", "Z2", 1.0), ("L", "Z2", "D", 0.0)]  # second-division side, two legs
    assert W.season_complete(_cd(rr + playoff), "2026/27")


def test_season_complete_counts_fixtures_not_rows():
    rr = _round_robin(["A", "B", "C"])
    missing = [r for r in rr if (r[1], r[2]) != ("C", "B")]
    twice = missing + [missing[0]]  # one played fixture listed twice must not stand in for C v B
    assert not W.season_complete(_cd(twice), "2026/27")
    assert W.season_complete(_cd(rr), "2026/27")


# --- gameday_label ----------------------------------------------------------------------


def test_gameday_label_keeps_the_half_the_integer_gameday_floors_away():
    ids = ["PL1-S2627_GD2_G5", "PL1-S2627_GD2.5_G1", "BL1-S2627_GD10_G3"]
    label, n_fallback = W.gameday_label(ids, [2, 2, 10])
    assert list(label) == ["2", "2.5", "10"] and n_fallback == 0


def test_gameday_label_falls_back_to_the_integer_gameday_without_a_token():
    label, n_fallback = W.gameday_label(["PL1-S2627_GD3_G1", "no-token"], [3, 7])
    assert list(label) == ["3", "7"] and n_fallback == 1


def test_gameday_label_does_not_match_a_longer_round_number():
    # the token is delimited on both sides: GD1 must not be read out of GD10 or GD1x
    label, _ = W.gameday_label(["X_GD10_G1"], [10])
    assert list(label) == ["10"]
