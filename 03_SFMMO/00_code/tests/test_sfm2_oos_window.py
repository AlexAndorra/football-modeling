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
        ["2026-09-08 15:00", None, "2026-09-06 15:00"],  # an undated fixture: never forecast
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


# --- the run block: SMOKE never writes the ledger -------------------------------------------


def _run_block():
    tree = ast.parse(_ledger_cell())
    block = [n for n in tree.body if isinstance(n, ast.If)][-1]
    return compile(ast.Module(body=[block], type_ignores=[]), str(NB), "exec")


@pytest.mark.parametrize("smoke", [True, False])
def test_smoke_run_never_writes_the_ledger(smoke):
    calls = []
    data = pd.DataFrame(
        {"id_match": ["g1"], "name_player": ["p1"], "season": ["2026/27"], "gameday": [5],
         "name_league": ["L"], "name_team": ["A"], "name_opp": ["B"], "goals_in_match": [0],
         "home_pitch": [1], "is_oos": [True]}
    )  # fmt: skip
    ns = dict(
        SMOKE=smoke, dict_PLAYERS={"oos": {"p1": {}}}, data__all=data, SFM_model__NAME="m",
        STALE_MATCH_IDS={"g0"}, update_frozen_ledger=lambda *a, **k: calls.append(k),
    )  # fmt: skip
    exec(_run_block(), ns)
    assert len(calls) == (0 if smoke else 1)
    if not smoke:
        assert calls[0]["stale_ids"] == {"g0"}


# --- ledger identity: each leg keeps its own receipt ------------------------------------------

DUMMY = (
    "gX",
    "z_player",
    4,
    "X",
    "Y",
    1,
    pd.Timestamp("2026-09-27 15:00"),
)  # keeps the board non-empty


def _played(rows, goals):
    return _scored(rows).assign(goals_in_match=goals)[NO_PLAYED.columns]


def test_the_return_leg_gets_its_own_receipt(tmp_path):
    # A pair meets twice. The first leg's finished row must never be relabelled as the return leg
    # (which would throw away the return leg's forecast and never record its result).
    L = _ledger_functions()
    path = str(tmp_path / "frozen.csv")
    run = L["update_frozen_ledger"]
    first = ("gF", "a_player", 3, "A", "B", 1, pd.Timestamp("2026-09-20 15:00"))  # A at home
    ret = ("gR", "a_player", 22, "A", "B", 0, pd.Timestamp("2027-02-20 15:00"))  # B at home
    dummy = _board([("z_player", 4, "X", "Y", 0.9)])
    run(
        _board([("a_player", 3, "A", "B", 0.70)]),
        _scored([first]),
        NO_PLAYED,
        "m",
        ledger_path=path,
        now="run1",
    )
    run(dummy, _scored([DUMMY]), _played([first], 1), "m", ledger_path=path, now="run2")
    out = run(_board([("a_player", 22, "A", "B", 0.40)]), _scored([ret]), _played([first, DUMMY], [1, 0]), "m",
              ledger_path=path, now="run3").set_index("id_match")  # fmt: skip
    assert out.loc["gF", "status"] == "finished" and out.loc["gF", "p0_mid"] == 0.70
    assert out.loc["gF", "actual_goals"] == 1 and out.loc["gF", "gameday"] == 3
    assert out.loc["gR", "status"] == "upcoming" and out.loc["gR", "p0_mid"] == 0.40
    out = run(dummy, _scored([DUMMY]), _played([first, DUMMY, ret], [1, 0, 3]), "m",
              ledger_path=path, now="run4").set_index("id_match")  # fmt: skip
    assert out.loc["gR", "status"] == "finished" and out.loc["gR", "actual_goals"] == 3
    assert out.loc["gR", "p0_mid"] == 0.40 and out.loc["gF", "p0_mid"] == 0.70


def test_a_renumbered_fixture_is_still_frozen_when_played(tmp_path):
    # Max's self-heal (La Liga 2026-08: a postponement renumbered G6 -> G5) must keep working.
    L = _ledger_functions()
    path = str(tmp_path / "frozen.csv")
    run = L["update_frozen_ledger"]
    g6 = ("g6", "p1", 6, "A", "B", 1, pd.Timestamp("2026-09-27 15:00"))
    g5 = ("g5", "p1", 5, "A", "B", 1, pd.Timestamp("2026-09-27 15:00"))
    run(
        _board([("p1", 6, "A", "B", 0.70)]),
        _scored([g6]),
        NO_PLAYED,
        "m",
        ledger_path=path,
        now="run1",
    )
    out = run(_board([("z_player", 4, "X", "Y", 0.9)]), _scored([DUMMY]), _played([g5], 2), "m",
              ledger_path=path, now="run2").set_index("id_match")  # fmt: skip
    assert "g6" not in out.index
    assert out.loc["g5", "status"] == "finished" and out.loc["g5", "p0_mid"] == 0.70
    assert out.loc["g5", "actual_goals"] == 2


def test_a_held_over_tie_in_the_same_gameday_keeps_its_own_id(tmp_path):
    # La Liga labels a held-over round-1 tie '2.5' -> floored to gameday 2, the same bucket as the
    # team's round-2 match. The board row for the live tie must not be pinned to the other id.
    L = _ledger_functions()
    path = str(tmp_path / "frozen.csv")
    run = L["update_frozen_ledger"]
    r2 = ("g_r2", "p1", 2, "A", "B", 1, pd.Timestamp("2026-08-22 15:00"))  # kicked off, no result
    held = ("g_held", "p1", 2, "A", "C", 1, pd.Timestamp("2026-08-26 20:00"))
    run(
        _board([("p1", 2, "A", "B", 0.70)]),
        _scored([r2, held]),
        NO_PLAYED,
        "m",
        ledger_path=path,
        now="run1",
    )
    out = run(_board([("p1", 2, "A", "C", 0.55)]), _scored([r2, held]), NO_PLAYED, "m",
              ledger_path=path, now="run2", stale_ids={"g_r2"}).set_index("id_match")  # fmt: skip
    assert out.loc["g_r2", "p0_mid"] == 0.70 and out.loc["g_r2", "forecast_frozen_at"] == "run1"
    assert out.loc["g_held", "p0_mid"] == 0.55 and out.loc["g_held", "forecast_frozen_at"] == "run2"


# --- wiring: kicked-off fixtures leave the board ----------------------------------------------


def test_load_cell_flags_kicked_off_fixtures_and_keeps_them_in_the_data(tmp_path):
    src = next(s for s in _cells() if s.startswith("# -------------------------- Load & Combine"))
    web = tmp_path / "10_data" / "106_Website"
    web.mkdir(parents=True)
    now = pd.Timestamp.now().floor("min")
    cols = ["id_match", "name_player", "season", "gameday", "kick_off", "goals_in_match"]
    pd.DataFrame([("gP", "p1", "2026/27", 1, now - pd.Timedelta(days=9), 1)], columns=cols).to_csv(
        web / "data_byPlayer.csv", index=False
    )
    oos = [
        (
            "gS",
            "p1",
            "2026/27",
            2,
            (now - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=15),
            np.nan,
        ),
        (
            "gL",
            "p1",
            "2026/27",
            3,
            (now + pd.Timedelta(days=2)).normalize() + pd.Timedelta(hours=15),
            np.nan,
        ),
        ("gU", "p1", "2026/27", 4, pd.NaT, np.nan),  # no kick-off yet: not forecast
    ]
    pd.DataFrame(oos, columns=cols).to_csv(web / "data_byPlayer__OOS.csv", index=False)
    ns = dict(
        pd=pd,
        np=np,
        os=os,
        directory=str(tmp_path),
        FORECAST_HORIZON_DAYS=14,
        oos_horizon=E["oos_horizon"],
    )
    exec(compile(src, str(NB), "exec"), ns)
    assert ns["STALE_MATCH_IDS"] == {"gS"}
    assert set(ns["data__all"].loc[ns["data__all"].is_oos, "id_match"]) == {"gS", "gL"}


def test_the_oos_board_leaves_kicked_off_fixtures_out():
    tree = ast.parse(
        next(s for s in _cells() if "dict_PLAYERS = {d: {} for d in datasets_to_process}" in s)
    )
    oos_sel = [
        ast.unparse(n)
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and "data__new" in ast.unparse(n.targets[0])
    ]
    assert any(
        "is_oos" in s and "~data__all.id_match.isin(STALE_MATCH_IDS)" in s for s in oos_sel
    ), oos_sel


def test_a_row_of_unknown_orientation_is_not_repointed():
    # an old ledger row without home_pitch must not be guessed onto the other leg
    L = _ledger_functions()
    first = _scored([("gF", "a_player", 3, "A", "B", 1, pd.Timestamp("2026-09-20 15:00"))])
    ret = _scored([("gR", "a_player", 22, "A", "B", 0, pd.Timestamp("2027-02-20 15:00"))])
    fn = next(
        n
        for n in ast.walk(ast.parse(_ledger_cell()))
        if isinstance(n, ast.FunctionDef) and n.name == "_repoint"
    )
    ns = {"np": np, "pd": pd}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(NB), "exec"), ns)
    old = first.assign(home_pitch=np.nan)
    out = ns["_repoint"](old, pd.concat([first, ret]))
    assert out["id_match"].tolist() == ["gF"]
    out = ns["_repoint"](first.assign(id_match="g_old"), pd.concat([first, ret]))
    assert out["id_match"].tolist() == ["gF"]  # known orientation -> re-pointed to its own leg
    assert L  # the cell's functions still load together


def test_a_universe_row_of_unknown_orientation_is_ignored():
    fn = next(
        n
        for n in ast.walk(ast.parse(_ledger_cell()))
        if isinstance(n, ast.FunctionDef) and n.name == "_repoint"
    )
    ns = {"np": np, "pd": pd}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(NB), "exec"), ns)
    ret = _scored([("gR", "a_player", 22, "A", "B", 0, pd.Timestamp("2027-02-20 15:00"))])
    first_unknown = _scored(
        [("gX", "a_player", 3, "A", "B", np.nan, pd.Timestamp("2026-09-20 15:00"))]
    )
    # gX has no orientation: read as an away row it would key as B v A, the return leg
    out = ns["_repoint"](ret, pd.concat([first_unknown, ret]))
    assert out["id_match"].tolist() == ["gR"]


# --- season end: an empty board still grades what was played -----------------------------------


def test_with_nothing_on_the_board_played_rows_are_still_frozen(tmp_path):
    L = _ledger_functions()
    path = str(tmp_path / "frozen.csv")
    run = L["update_frozen_ledger"]
    last = ("gL", "p1", 38, "A", "B", 1, pd.Timestamp("2027-05-23 15:00"))
    later = (
        "gP",
        "p2",
        30,
        "C",
        "D",
        1,
        pd.Timestamp("2027-05-26 20:00"),
    )  # postponed, still to play
    run(_board([("p1", 38, "A", "B", 0.70), ("p2", 30, "C", "D", 0.60)]), _scored([last, later]), NO_PLAYED, "m",
        ledger_path=path, now="run1")  # fmt: skip
    out = run({}, _scored([])[:0], _played([last], 2), "m", ledger_path=path, now="run2",
              hold_pending=True).set_index("id_match")  # fmt: skip
    assert out.loc["gL", "status"] == "finished" and out.loc["gL", "actual_goals"] == 2
    assert out.loc["gL", "p0_mid"] == 0.70 and out.loc["gL", "forecast_frozen_at"] == "run1"
    assert out.loc["gP", "status"] == "upcoming" and out.loc["gP", "forecast_frozen_at"] == "run1"


@pytest.mark.parametrize(
    "board, datasets, ledger_exists, expected",
    [
        ({"p1": {}}, ["oos"], True, "board"),  # a normal week
        ({}, ["oos"], True, "hold"),  # nothing on the board: grade, hold the rest
        ({}, ["oos"], False, None),  # no ledger yet: nothing to grade
        ({}, ["train"], True, None),  # not the weekly job: never touch the ledger
    ],
)
def test_run_block_routes_an_empty_board(board, datasets, ledger_exists, expected, tmp_path):
    calls = []
    ledger = tmp_path / "frozen.csv"
    if ledger_exists:
        ledger.write_text("id_match\n")
    data = pd.DataFrame(
        {"id_match": ["g1"], "name_player": ["p1"], "season": ["2026/27"], "gameday": [5],
         "name_league": ["L"], "name_team": ["A"], "name_opp": ["B"], "goals_in_match": [0],
         "home_pitch": [1], "is_oos": [False]}
    )  # fmt: skip
    ns = dict(
        SMOKE=False, dict_PLAYERS={"oos": board}, data__all=data, SFM_model__NAME="m", STALE_MATCH_IDS=set(),
        datasets_to_process=datasets, os=os, LEDGER_PATH=str(ledger),
        update_frozen_ledger=lambda b, *a, **k: calls.append((b, k)),
    )  # fmt: skip
    exec(_run_block(), ns)
    if expected is None:
        assert calls == []
    elif expected == "board":
        assert calls[0][0] == board and calls[0][1]["stale_ids"] == set()
    else:
        assert calls[0][0] == {} and calls[0][1]["hold_pending"] is True


def test_a_board_that_resolves_to_no_rows_never_rewrites_the_ledger(tmp_path):
    L = _ledger_functions()
    path = tmp_path / "frozen.csv"
    run = L["update_frozen_ledger"]
    run(
        _board([("p1", 5, "A", "B", 0.70)]),
        _scored([G1]),
        NO_PLAYED,
        "m",
        ledger_path=str(path),
        now="run1",
    )
    before = path.read_bytes()
    with pytest.raises(ValueError, match="no ledger rows"):
        run(
            _board([("p1", 9, "A", "B", 0.70)]),
            _scored([G1]),
            NO_PLAYED,
            "m",
            ledger_path=str(path),
            now="run2",
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize("hold", [True, False])
def test_a_reused_id_never_freezes_the_wrong_fixture(tmp_path, hold):
    # C v D (g5) is postponed and leaves this run's data; upstream renumbers A v B g6 -> g5 and
    # it is played. C v D's pending row must not be frozen with A v B's result.
    L = _ledger_functions()
    path = str(tmp_path / "frozen.csv")
    run = L["update_frozen_ledger"]
    cd = ("g5", "p2", 5, "C", "D", 1, pd.Timestamp("2026-09-27 15:00"))
    ab_old = ("g6", "p1", 6, "A", "B", 1, pd.Timestamp("2026-10-04 15:00"))
    ab_new = ("g5", "p1", 5, "A", "B", 1, pd.Timestamp("2026-10-04 15:00"))
    run(_board([("p2", 5, "C", "D", 0.60), ("p1", 6, "A", "B", 0.70)]), _scored([cd, ab_old]), NO_PLAYED, "m",
        ledger_path=path, now="run1")  # fmt: skip
    board, scored = (
        ({}, _scored([])[:0])
        if hold
        else (_board([("z_player", 4, "X", "Y", 0.9)]), _scored([DUMMY]))
    )
    out = run(
        board, scored, _played([ab_new], 2), "m", ledger_path=path, now="run2", hold_pending=hold
    )
    ab = out[out.name_player == "p1"].iloc[0]
    assert ab["id_match"] == "g5" and ab["status"] == "finished" and ab["actual_goals"] == 2
    cd_rows = out[out.name_player == "p2"]
    assert not (cd_rows["status"] == "finished").any()  # never graded with someone else's result
    if hold:
        assert cd_rows.iloc[0]["p0_mid"] == 0.60 and cd_rows.iloc[0]["status"] == "upcoming"
