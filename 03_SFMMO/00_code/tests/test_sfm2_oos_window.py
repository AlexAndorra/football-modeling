"""Receipts guards of the SFM II weekly scoring notebook (006_040).

A fixture that has kicked off without a recorded result must never be forecast again: a
forecast made after kick-off is not a receipt, and the ledger would replace the frozen
pre-match row with it. The notebook cannot import 006_060, so it carries copies of the
kicked-off rule and the horizon rule; the parity tests keep the two products on the same
fixtures. The ledger and run-block tests exec the notebook's own code straight from the .ipynb.
"""

import ast
import importlib.util
import json
import os
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
NB = ROOT / "01_SFM" / "00_code" / "003__SFM_II" / "006_040__Predictions_ScoringProb__SFM_II.ipynb"
SCRIPT = (
    ROOT / "03_SFMMO" / "00_code" / "00_02__SFMMO" / "006_060__Predictions_MatchOutcome__SFMMO.py"
)


def _cells():
    with NB.open() as f:
        return ["".join(c["source"]) for c in json.load(f)["cells"] if c["cell_type"] == "code"]


def _engine():
    src = next(s for s in _cells() if "def update_elo" in s)
    ns = {"np": np, "pd": pd, "cred_region": 0.9}
    exec(compile(src, str(NB), "exec"), ns)
    return ns


def _ledger_cell():
    return next(s for s in _cells() if "def update_frozen_ledger" in s)


def _ledger_functions():
    tree = ast.parse(_ledger_cell())
    keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
    ns = {"np": np, "pd": pd, "os": os, "LEDGER_PATH": "unused.csv", "players_train": []}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(NB), "exec"), ns)
    return ns


def _sfmmo():
    spec = importlib.util.spec_from_file_location("sfmmo_weekly_parity", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(mod)
    return mod


E, W = _engine(), _sfmmo()
NOW = pd.Timestamp("2026-09-07 10:00")


# --- parity with 006_060 -------------------------------------------------------------------

STAMPS = [
    "2026-09-06 15:00", "2026-09-06 00:00", "2026-09-07 09:00", "2026-09-07 10:00",
    "2026-09-07 20:00", "2026-09-07 00:00", "2026-09-08 15:00", "2026-09-21 20:00",
    "2026-09-22 00:00", "2026-09-22 15:00", "2026-10-12 15:00",
]  # fmt: skip


def test_kicked_off_rule_is_identical_to_006_060():
    ko = pd.Series(pd.to_datetime(STAMPS))
    assert E["kicked_off_mask"](ko, NOW).tolist() == W.kicked_off_mask(ko, NOW).tolist()


@pytest.mark.parametrize(
    "stamps",
    [
        STAMPS,  # a stale fixture, today, the last day of the window, beyond it
        ["2026-09-08 15:00", "2026-09-22 15:00", "2026-09-23 15:00"],  # next fixture tomorrow
        ["2026-09-25 15:00", "2026-09-26 15:00", "2026-10-12 15:00"],  # international break
        ["2026-08-30 15:00", "2026-09-25 15:00", "2026-10-12 15:00"],  # break + stale fixture
        ["2026-09-06 15:00", "2026-09-05 15:00"],  # nothing live left
    ],
)
def test_horizon_covers_the_same_fixtures_as_006_060(stamps):
    oos = pd.DataFrame(
        {"id_match": [f"g{i}" for i in range(len(stamps))], "kick_off": pd.to_datetime(stamps)}
    )
    window, deferred, stale, horizon = W.select_forecast_window(oos, NOW, horizon_days=14)
    keep, kicked, t0 = E["oos_horizon"](oos["kick_off"], NOW, 14)
    assert sorted(oos.loc[keep & ~kicked, "id_match"]) == sorted(window["id_match"])
    assert sorted(oos.loc[kicked, "id_match"]) == sorted(stale["id_match"])
    assert t0 + pd.Timedelta(days=14) == horizon


def test_kicked_off_rows_stay_in_the_data():
    # they leave the board, not the season x gameday cross-section of the rest of their round
    ko = pd.Series(pd.to_datetime(["2026-09-06 15:00", "2026-09-07 20:00"]))
    keep, kicked, _ = E["oos_horizon"](ko, NOW, 14)
    assert keep.tolist() == [True, True] and kicked.tolist() == [True, False]


# --- ledger: a kicked-off fixture keeps its frozen row ---------------------------------------


def _board(rows):
    out = {}
    for player, gameday, team, opp, p0 in rows:
        mid = pd.DataFrame(
            [[p0, (1 - p0) * 0.7, (1 - p0) * 0.2, (1 - p0) * 0.1]], index=[f"2026/27__{gameday}"]
        )
        stats = pd.DataFrame({"name_team": [team], "name_opp": [opp], "name_league": ["L"]})
        out[player] = dict(mid=mid, low=mid * 0.9, up=mid * 1.1, match_stats=stats, model_seen=True)
    return out


def _scored(rows):
    cols = ["id_match", "name_player", "gameday", "name_team", "name_opp", "home_pitch", "kick_off"]
    return pd.DataFrame(rows, columns=cols).assign(season="2026/27", name_league="L")


G1 = ("g1", "p1", 5, "A", "B", 1, pd.Timestamp("2026-09-06 15:00"))
G2 = ("g2", "p2", 5, "C", "D", 1, pd.Timestamp("2026-09-07 20:00"))
NO_PLAYED = pd.DataFrame(
    columns=["id_match", "name_player", "season", "gameday", "name_league", "name_team", "name_opp",
             "goals_in_match", "home_pitch"]
)  # fmt: skip


def test_kicked_off_fixture_keeps_its_frozen_row_until_the_result_lands(tmp_path):
    L = _ledger_functions()
    path = str(tmp_path / "frozen.csv")
    run = L["update_frozen_ledger"]
    # run 1, before the weekend: both fixtures forecast
    run(_board([("p1", 5, "A", "B", 0.70), ("p2", 5, "C", "D", 0.60)]), _scored([G1, G2]), NO_PLAYED,
        "m", ledger_path=path, now="run1")  # fmt: skip
    # run 2: A v B kicked off, no result yet -> off the board, its row must stand
    out = run(_board([("p2", 5, "C", "D", 0.50)]), _scored([G1, G2]), NO_PLAYED, "m",
              ledger_path=path, now="run2", stale_ids={"g1"}).set_index("id_match")  # fmt: skip
    assert out.loc["g1", "p0_mid"] == 0.70 and out.loc["g1", "forecast_frozen_at"] == "run1"
    assert out.loc["g1", "status"] == "upcoming"
    assert out.loc["g2", "p0_mid"] == 0.50 and out.loc["g2", "forecast_frozen_at"] == "run2"
    # run 3: the result lands -> the row standing at kick-off is frozen with the outcome
    played = _scored([G1]).assign(goals_in_match=1)[NO_PLAYED.columns]
    out = run(_board([("p2", 5, "C", "D", 0.45)]), _scored([G2]), played, "m",
              ledger_path=path, now="run3").set_index("id_match")  # fmt: skip
    assert out.loc["g1", "status"] == "finished" and out.loc["g1", "actual_goals"] == 1
    assert out.loc["g1", "p0_mid"] == 0.70 and out.loc["g1", "forecast_frozen_at"] == "run1"


def test_a_board_row_for_a_kicked_off_fixture_never_replaces_the_frozen_one(tmp_path):
    # belt and braces: even if a kicked-off fixture reached the board, the ledger keeps run 1
    L = _ledger_functions()
    path = str(tmp_path / "frozen.csv")
    run = L["update_frozen_ledger"]
    run(
        _board([("p1", 5, "A", "B", 0.70)]),
        _scored([G1]),
        NO_PLAYED,
        "m",
        ledger_path=path,
        now="run1",
    )
    out = run(_board([("p1", 5, "A", "B", 0.20)]), _scored([G1]), NO_PLAYED, "m",
              ledger_path=path, now="run2", stale_ids={"g1"}).set_index("id_match")  # fmt: skip
    assert out.loc["g1", "p0_mid"] == 0.70 and out.loc["g1", "forecast_frozen_at"] == "run1"
