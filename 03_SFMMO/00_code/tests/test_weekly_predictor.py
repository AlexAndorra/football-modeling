"""Receipts-integrity guards of the weekly league predictor (006_060).

The frozen ledger is only honest if a fixture that has ALREADY kicked off is never re-forecast:
a forecast made after kick-off is not a receipt, and it can see results the pre-match one could
not (WC lesson L5). These tests pin the window selection, the ledger merge, and the hand-over of
stale fixtures to the feed.
"""

import importlib.util
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
NOW = pd.Timestamp("2026-09-07 10:00")


def _oos(rows):
    return pd.DataFrame(
        rows, columns=["id_match", "name_team", "name_opp", "kick_off", "match_outcome"]
    ).assign(kick_off=lambda d: pd.to_datetime(d["kick_off"]), season="2026/27")


# --- forecast window ------------------------------------------------------------------


def test_window_excludes_fixtures_that_already_kicked_off():
    oos = _oos(
        [
            ("g1", "A", "B", "2026-09-06 15:00", np.nan),  # played yesterday, result not in yet
            ("g2", "C", "D", "2026-09-07 20:00", np.nan),  # tonight
            ("g3", "E", "F", "2026-09-12 15:00", np.nan),  # inside the horizon
            ("g4", "G", "H", "2026-09-30 15:00", np.nan),  # beyond the horizon
        ]
    )
    window, deferred, stale, _ = W.select_forecast_window(oos, NOW, horizon_days=14)
    assert list(window["id_match"]) == ["g2", "g3"]
    assert list(deferred["id_match"]) == ["g4"]
    assert list(stale["id_match"]) == ["g1"]


def test_window_keeps_same_day_fixture_with_date_only_stamp():
    # the fixture feed stores an unconfirmed slot as 00:00 local -- NOT a midnight kick-off
    oos = _oos([("g1", "A", "B", "2026-09-07 00:00", np.nan)])
    window, deferred, stale, _ = W.select_forecast_window(oos, NOW, horizon_days=14)
    assert list(window["id_match"]) == ["g1"] and stale.empty


def test_window_horizon_is_inclusive_of_the_last_day():
    oos = _oos(
        [
            ("g0", "C", "D", "2026-09-07 20:00", np.nan),  # tonight -> anchor is today
            ("g1", "A", "B", "2026-09-21 20:00", np.nan),  # NOW + 14 days, evening
        ]
    )
    window, deferred, stale, _ = W.select_forecast_window(oos, NOW, horizon_days=14)
    assert list(window["id_match"]) == ["g0", "g1"]


@pytest.mark.parametrize(
    "kick_off, expected",
    [
        ("2026-09-06 15:00", True),  # yesterday
        ("2026-09-06 00:00", True),  # yesterday, unconfirmed slot: the day itself has passed
        ("2026-09-07 09:00", True),  # today, confirmed time already passed
        ("2026-09-07 10:00", False),  # today, kicking off right now: not yet
        ("2026-09-07 20:00", False),  # tonight
        ("2026-09-07 00:00", False),  # today, unconfirmed slot: cannot be judged
        ("2026-09-08 15:00", False),  # tomorrow
    ],
)
def test_kicked_off_mask(kick_off, expected):
    ko = pd.Series(pd.to_datetime([kick_off]))
    assert bool(W.kicked_off_mask(ko, NOW).iloc[0]) is expected


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


def test_stale_fixture_frozen_row_survives_a_run_end_to_end():
    """The scenario the guard exists for: result feed lags, the run happens anyway."""
    led = _ledger([("2026/27", "A", "B", "g1", 0.5, 0.3, 0.2, "2026-09-01")])
    oos = _oos(
        [
            ("g1", "A", "B", "2026-09-06 15:00", np.nan),  # kicked off, no result yet
            ("g2", "C", "D", "2026-09-12 15:00", np.nan),
        ]
    )
    window, _, stale, _ = W.select_forecast_window(oos, NOW, horizon_days=14)
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


def test_window_marks_same_day_fixture_with_a_real_time_in_the_past_as_stale():
    late = pd.Timestamp("2026-09-07 22:00")
    oos = _oos(
        [
            ("g1", "A", "B", "2026-09-07 15:00", np.nan),  # kicked off 7 h ago, no result
            ("g2", "C", "D", "2026-09-07 00:00", np.nan),  # unconfirmed slot -> keep
            ("g3", "E", "F", "2026-09-07 22:30", np.nan),  # later tonight -> keep
        ]
    )
    window, _, stale, _ = W.select_forecast_window(oos, late, horizon_days=14)
    assert list(stale["id_match"]) == ["g1"]
    assert list(window["id_match"]) == ["g2", "g3"]


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
    stale = pd.DataFrame({"id_match": ["g1", "g8"], "name_team": ["A", "Q"], "name_opp": ["B", "R"]})
    grid, team, missing = W.carry_forward_board_rows(prev, stale)
    assert len(grid) == 2 and set(grid["home_team"]) == {"A"}
    assert len(team) == 2 and set(team["team"]) == {"A", "B"}
    assert missing == ["Q v R"]


# --- red-team round 2 additions -------------------------------------------------------


def test_window_refuses_rows_without_kick_off():
    oos = _oos([("g1", "A", "B", "2026-09-12 15:00", np.nan)])
    oos.loc[0, "kick_off"] = pd.NaT
    with pytest.raises(ValueError, match="no kick_off"):
        W.select_forecast_window(oos, NOW, horizon_days=14)


def test_carry_forward_ignores_the_return_leg_and_remaps_renumbered_ids():
    prev = {
        "scorelines": pd.DataFrame(
            {"id_match": ["g6", "g9"], "home_team": ["A", "B"], "away_team": ["B", "A"],
             "home_goals": [0, 0], "away_goals": [0, 0], "p_mid": [0.1, 0.2]}
        ),
        "team_goals": pd.DataFrame(
            {"id_match": ["g6", "g6", "g9", "g9"], "team": ["A", "B", "B", "A"],
             "opponent": ["B", "A", "A", "B"], "is_home": [1, 0, 1, 0], "p_goals_0": [0.2, 0.3, 0.4, 0.5]}
        ),
    }
    stale = pd.DataFrame({"id_match": ["g5"], "name_team": ["A"], "name_opp": ["B"]})  # renumbered
    grid, team, missing = W.carry_forward_board_rows(prev, stale)
    assert list(grid["id_match"]) == ["g5"] and list(grid["p_mid"]) == [0.1]
    assert sorted(team["p_goals_0"]) == [0.2, 0.3] and set(team["id_match"]) == {"g5"}
    assert missing == []


def test_window_anchors_at_the_earliest_forecastable_fixture_during_a_break():
    # International break: nothing for 18 days. A today-anchored window would come back empty
    # and the run would exit instead of forecasting the next round (Max, 64f86df).
    oos = _oos(
        [
            ("g1", "A", "B", "2026-09-25 15:00", np.nan),
            ("g2", "C", "D", "2026-09-26 15:00", np.nan),
            ("g3", "E", "F", "2026-10-12 15:00", np.nan),  # 14 days past the 09-25 anchor + 3
        ]
    )
    window, deferred, stale, horizon = W.select_forecast_window(oos, NOW, horizon_days=14)
    assert list(window["id_match"]) == ["g1", "g2"]
    assert list(deferred["id_match"]) == ["g3"]
    assert horizon == pd.Timestamp("2026-10-09")
    assert len(stale) == 0


def test_window_anchor_ignores_kicked_off_fixtures():
    # A postponed / unrecorded fixture carrying a past kick-off must neither be re-forecast nor
    # pull the anchor backwards: the anchor is the earliest fixture that is still forecastable.
    oos = _oos(
        [
            ("g0", "X", "Y", "2026-08-30 15:00", np.nan),  # kicked off, no result
            ("g1", "A", "B", "2026-09-25 15:00", np.nan),
            ("g2", "E", "F", "2026-10-12 15:00", np.nan),
        ]
    )
    window, deferred, stale, horizon = W.select_forecast_window(oos, NOW, horizon_days=14)
    assert list(stale["id_match"]) == ["g0"]
    assert list(window["id_match"]) == ["g1"]
    assert list(deferred["id_match"]) == ["g2"]
    assert horizon == pd.Timestamp("2026-10-09")


def test_window_starts_at_the_next_forecastable_fixture():
    # Max's rule (64f86df): anchor = max(today, earliest unplayed), here over fixtures that have
    # not kicked off. Same rule as 006_040, so the two boards cover the same fixtures.
    oos = _oos([("g1", "A", "B", "2026-09-08 15:00", np.nan), ("g2", "C", "D", "2026-09-22 15:00", np.nan)])
    window, deferred, stale, horizon = W.select_forecast_window(oos, NOW, horizon_days=14)
    assert horizon == pd.Timestamp("2026-09-22")
    assert list(window["id_match"]) == ["g1", "g2"]
    assert deferred.empty


def test_window_anchor_is_today_when_a_fixture_is_still_to_come_today():
    oos = _oos([("g1", "A", "B", "2026-09-07 20:00", np.nan), ("g2", "C", "D", "2026-09-22 15:00", np.nan)])
    window, deferred, stale, horizon = W.select_forecast_window(oos, NOW, horizon_days=14)
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
