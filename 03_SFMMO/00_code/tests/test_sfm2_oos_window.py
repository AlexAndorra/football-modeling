"""Receipts guards of the SFM II weekly scoring notebook (006_040).

A fixture that may have kicked off must never be forecast again: a forecast made after kick-off is
not a receipt, and the ledger would replace the frozen pre-match row with it. The rule itself lives
in 00_shared/kickoff.py (one rule for both weekly writers, tested against the shared kick-off
vectors in 00_shared/tests). These tests exec the notebook's own code straight from the .ipynb and
check that it is WIRED to that rule, with inputs shaped like the real ones: a date-only OOS file and
a separate kick-off feed carrying UTC times and a confirmed flag.
"""

import ast
import json
import os
import pathlib
import pickle
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
NB = ROOT / "01_SFM" / "00_code" / "003__SFM_II" / "006_040__Predictions_ScoringProb__SFM_II.ipynb"
sys.path.insert(0, str(ROOT / "00_shared"))
import kickoff  # noqa: E402


def _cells():
    with NB.open() as f:
        return ["".join(c["source"]) for c in json.load(f)["cells"] if c["cell_type"] == "code"]


def _ledger_cell():
    return next(s for s in _cells() if "def update_frozen_ledger" in s)


def _ledger_functions(directory):
    tree = ast.parse(_ledger_cell())
    keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
    ns = {"np": np, "pd": pd, "os": os, "LEDGER_PATH": "unused.csv", "players_train": [],
          "kickoff": kickoff, "directory": str(directory), "KICKOFF_FEED": None}  # fmt: skip
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(NB), "exec"), ns)
    return ns


def _berlin(t):
    return pd.Timestamp(t, tz=kickoff.BERLIN)


# --- the notebook uses the shared rule and carries no copy of its own ------------------------


def test_the_notebook_uses_the_shared_kick_off_rule():
    src = "\n".join(_cells())
    assert "kickoff.certainly_started(" in src  # the board: certain kick-offs leave it
    assert "kickoff.may_have_started(" in src  # the ledger: anything uncertain is held
    for copy in ("def kicked_off_mask", "def oos_horizon", "UNCONFIRMED_MARGIN_DAYS"):
        assert copy not in src, f"006_040 carries its own copy again: {copy}"


# --- test data, in the real shapes -----------------------------------------------------------


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


def _key(df):
    h = df["home_pitch"] == 1
    return kickoff.fixture_key(df["season"], df["name_league"],
                               np.where(h, df["name_team"], df["name_opp"]),
                               np.where(h, df["name_opp"], df["name_team"]))  # fmt: skip


def _feed(scored, confirmed=True):
    """The kick-off feed as read_kickoff_feed returns it: one row per fixture, UTC, confirmed."""
    fx = scored.drop_duplicates("id_match").reset_index(drop=True)
    ko = pd.to_datetime(fx["kick_off"]).dt.tz_localize(kickoff.BERLIN).dt.tz_convert("UTC")
    feed = pd.DataFrame({"fixture_key": _key(fx), "kick_off_utc": ko,
                         "time_confirmed": pd.array([confirmed] * len(fx), dtype="boolean")})  # fmt: skip
    feed = feed.drop_duplicates("fixture_key")
    feed.attrs["missing"] = False
    return feed


NO_FEED = _feed(_scored([]))
NO_FEED.attrs["missing"] = True

G1 = ("g1", "p1", 5, "A", "B", 1, pd.Timestamp("2026-09-06 15:00"))
G2 = ("g2", "p2", 5, "C", "D", 1, pd.Timestamp("2026-09-07 20:00"))
DUMMY = ("gX", "z_player", 4, "X", "Y", 1, pd.Timestamp("2026-09-27 15:00"))  # keeps a board non-empty
NO_PLAYED = pd.DataFrame(
    columns=["id_match", "name_player", "season", "gameday", "name_league", "name_team", "name_opp",
             "goals_in_match", "home_pitch"]
)  # fmt: skip
EARLY = "2026-08-01 10:00"  # before every test fixture: nothing has kicked off


def _played(rows, goals):
    return _scored(rows).assign(goals_in_match=goals)[NO_PLAYED.columns]


class Ledger:
    """update_frozen_ledger as the notebook runs it: the FULL OOS file staged on disk (date-only
    kick_off, read from the data root), the kick-off feed, and a tz-aware run clock."""

    def __init__(self, tmp_path):
        self.web = tmp_path / "10_data" / "106_Website"
        self.web.mkdir(parents=True)
        self.path = str(tmp_path / "frozen.csv")
        self.fn = _ledger_functions(tmp_path)["update_frozen_ledger"]

    def run(self, board, scored, played, now, at=EARLY, feed=None, oos=None, **kw):
        o = (scored if oos is None else oos).copy()
        o["kick_off"] = pd.to_datetime(o["kick_off"]).dt.strftime("%Y-%m-%d")  # date-only
        o.to_csv(self.web / "data_byPlayer__OOS.csv", index=False)
        return self.fn(board, scored, played, "m", ledger_path=self.path, now=now,
                       feed=_feed(scored) if feed is None else feed, run_time=_berlin(at), **kw)  # fmt: skip


# --- ledger: a fixture that may have kicked off keeps its row ----------------------------------


def test_kicked_off_fixture_keeps_its_frozen_row_until_the_result_lands(tmp_path):
    lg = Ledger(tmp_path)
    # run 1, before the weekend: both fixtures forecast
    lg.run(_board([("p1", 5, "A", "B", 0.70), ("p2", 5, "C", "D", 0.60)]), _scored([G1, G2]), NO_PLAYED,
           "run1", at="2026-09-05 10:00")  # fmt: skip
    # run 2: A v B kicked off (confirmed 15:00 yesterday), no result yet -> its row must stand
    out = lg.run(_board([("p2", 5, "C", "D", 0.50)]), _scored([G1, G2]), NO_PLAYED, "run2",
                 at="2026-09-07 10:00").set_index("id_match")  # fmt: skip
    assert out.loc["g1", "p0_mid"] == 0.70 and out.loc["g1", "forecast_frozen_at"] == "run1"
    assert out.loc["g1", "status"] == "upcoming"
    assert out.loc["g2", "p0_mid"] == 0.50 and out.loc["g2", "forecast_frozen_at"] == "run2"
    # run 3: the result lands -> the row standing at kick-off is frozen with the outcome
    out = lg.run(_board([("p2", 5, "C", "D", 0.45)]), _scored([G2]), _played([G1], 1), "run3",
                 at="2026-09-08 10:00", feed=_feed(_scored([G1, G2]))).set_index("id_match")  # fmt: skip
    assert out.loc["g1", "status"] == "finished" and out.loc["g1", "actual_goals"] == 1
    assert out.loc["g1", "p0_mid"] == 0.70 and out.loc["g1", "forecast_frozen_at"] == "run1"


def test_a_board_row_for_a_kicked_off_fixture_never_replaces_the_frozen_one(tmp_path):
    # belt and braces: even if a kicked-off fixture reached the board, the ledger keeps run 1
    lg = Ledger(tmp_path)
    lg.run(_board([("p1", 5, "A", "B", 0.70)]), _scored([G1]), NO_PLAYED, "run1", at="2026-09-05 10:00")
    out = lg.run(_board([("p1", 5, "A", "B", 0.20)]), _scored([G1]), NO_PLAYED, "run2",
                 at="2026-09-07 10:00").set_index("id_match")  # fmt: skip
    assert out.loc["g1", "p0_mid"] == 0.70 and out.loc["g1", "forecast_frozen_at"] == "run1"


# The three incidents, at the ledger: a same-day confirmed kick-off (Hull v Man Utd, 22 Aug: run at
# 16:48 over a 13:30 kick-off), Monaco v Lens kicked off that evening, and Monaco v Lens stored as
# an UNCONFIRMED Saturday placeholder while really played on the Friday. Plus the control.
INCIDENTS = [
    ("hull_v_manutd_same_day", "2026-08-22 13:30", True, "2026-08-22 16:48", True),
    ("monaco_v_lens_same_evening", "2026-09-18 20:45", True, "2026-09-18 21:00", True),
    ("monaco_v_lens_placeholder", "2026-09-19 15:00", False, "2026-09-18 21:00", True),
    ("control_confirmed_tonight", "2026-09-18 20:45", True, "2026-09-18 10:40", False),
]


@pytest.mark.parametrize("name, feed_ko, confirmed, at, held", INCIDENTS, ids=[i[0] for i in INCIDENTS])
def test_incidents_at_the_ledger(tmp_path, name, feed_ko, confirmed, at, held):
    fx = ("g1", "p1", 5, "A", "B", 1, pd.Timestamp(feed_ko))
    lg = Ledger(tmp_path)
    lg.run(_board([("p1", 5, "A", "B", 0.70)]), _scored([fx]), NO_PLAYED, "run1", at=EARLY)
    out = lg.run(_board([("p1", 5, "A", "B", 0.20)]), _scored([fx]), NO_PLAYED, "run2", at=at,
                 feed=_feed(_scored([fx]), confirmed=confirmed)).set_index("id_match")  # fmt: skip
    assert out.loc["g1", "p0_mid"] == (0.70 if held else 0.20), name


@pytest.mark.parametrize("days_out, held", [(1, True), (5, False)])
def test_without_a_feed_every_fixture_is_unconfirmed(tmp_path, days_out, held):
    at = pd.Timestamp("2026-09-18 08:40")
    fx = ("g1", "p1", 5, "A", "B", 1, at.normalize() + pd.Timedelta(days=days_out, hours=15))
    lg = Ledger(tmp_path)
    lg.run(_board([("p1", 5, "A", "B", 0.70)]), _scored([fx]), NO_PLAYED, "run1", at=EARLY)
    out = lg.run(_board([("p1", 5, "A", "B", 0.20)]), _scored([fx]), NO_PLAYED, "run2", at=at,
                 feed=NO_FEED).set_index("id_match")  # fmt: skip
    assert out.loc["g1", "p0_mid"] == (0.70 if held else 0.20)


def test_a_pending_fixture_absent_from_oos_and_feed_is_held_not_dropped(tmp_path):
    # nothing can prove it has not started, so its standing forecast stays
    lg = Ledger(tmp_path)
    lg.run(_board([("p1", 5, "A", "B", 0.70)]), _scored([G1]), NO_PLAYED, "run1", at=EARLY)
    out = lg.run(_board([("z_player", 4, "X", "Y", 0.9)]), _scored([DUMMY]), NO_PLAYED, "run2",
                 at=EARLY).set_index("id_match")  # fmt: skip
    assert out.loc["g1", "p0_mid"] == 0.70 and out.loc["g1", "status"] == "upcoming"


# --- the run block: SMOKE never writes the ledger -------------------------------------------


def _run_block():
    tree = ast.parse(_ledger_cell())
    block = [n for n in tree.body if isinstance(n, ast.If)][-1]
    return compile(ast.Module(body=[block], type_ignores=[]), str(NB), "exec")


FEED, RT = _feed(_scored([G1])), _berlin("2026-09-07 10:00")


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
        KICKOFF_FEED=FEED, RUN_TIME=RT, update_frozen_ledger=lambda *a, **k: calls.append(k),
    )  # fmt: skip
    exec(_run_block(), ns)
    assert len(calls) == (0 if smoke else 1)
    if not smoke:
        assert calls[0]["feed"] is FEED and calls[0]["run_time"] == RT


# --- ledger identity: each leg keeps its own receipt ------------------------------------------


def test_the_return_leg_gets_its_own_receipt(tmp_path):
    # A pair meets twice. The first leg's finished row must never be relabelled as the return leg
    # (which would throw away the return leg's forecast and never record its result).
    lg = Ledger(tmp_path)
    first = ("gF", "a_player", 3, "A", "B", 1, pd.Timestamp("2026-09-20 15:00"))  # A at home
    ret = ("gR", "a_player", 22, "A", "B", 0, pd.Timestamp("2027-02-20 15:00"))  # B at home
    dummy = _board([("z_player", 4, "X", "Y", 0.9)])
    lg.run(_board([("a_player", 3, "A", "B", 0.70)]), _scored([first]), NO_PLAYED, "run1")
    lg.run(dummy, _scored([DUMMY]), _played([first], 1), "run2")
    out = lg.run(_board([("a_player", 22, "A", "B", 0.40)]), _scored([ret]), _played([first, DUMMY], [1, 0]),
                 "run3").set_index("id_match")  # fmt: skip
    assert out.loc["gF", "status"] == "finished" and out.loc["gF", "p0_mid"] == 0.70
    assert out.loc["gF", "actual_goals"] == 1 and out.loc["gF", "gameday"] == 3
    assert out.loc["gR", "status"] == "upcoming" and out.loc["gR", "p0_mid"] == 0.40
    out = lg.run(dummy, _scored([DUMMY]), _played([first, DUMMY, ret], [1, 0, 3]),
                 "run4").set_index("id_match")  # fmt: skip
    assert out.loc["gR", "status"] == "finished" and out.loc["gR", "actual_goals"] == 3
    assert out.loc["gR", "p0_mid"] == 0.40 and out.loc["gF", "p0_mid"] == 0.70


def test_a_renumbered_fixture_is_still_frozen_when_played(tmp_path):
    # Max's self-heal (La Liga 2026-08: a postponement renumbered G6 -> G5) must keep working.
    lg = Ledger(tmp_path)
    g6 = ("g6", "p1", 6, "A", "B", 1, pd.Timestamp("2026-09-27 15:00"))
    g5 = ("g5", "p1", 5, "A", "B", 1, pd.Timestamp("2026-09-27 15:00"))
    lg.run(_board([("p1", 6, "A", "B", 0.70)]), _scored([g6]), NO_PLAYED, "run1")
    out = lg.run(_board([("z_player", 4, "X", "Y", 0.9)]), _scored([DUMMY]), _played([g5], 2),
                 "run2").set_index("id_match")  # fmt: skip
    assert "g6" not in out.index
    assert out.loc["g5", "status"] == "finished" and out.loc["g5", "p0_mid"] == 0.70
    assert out.loc["g5", "actual_goals"] == 2


def test_a_held_over_tie_in_the_same_gameday_keeps_its_own_id(tmp_path):
    # La Liga labels a held-over round-1 tie '2.5' -> floored to gameday 2, the same bucket as the
    # team's round-2 match. The board row for the live tie must not be pinned to the other id.
    lg = Ledger(tmp_path)
    r2 = ("g_r2", "p1", 2, "A", "B", 1, pd.Timestamp("2026-08-22 15:00"))  # kicked off, no result
    held = ("g_held", "p1", 2, "A", "C", 1, pd.Timestamp("2026-08-26 20:00"))
    lg.run(_board([("p1", 2, "A", "B", 0.70)]), _scored([r2, held]), NO_PLAYED, "run1", at="2026-08-20 10:00")
    out = lg.run(_board([("p1", 2, "A", "C", 0.55)]), _scored([r2, held]), NO_PLAYED, "run2",
                 at="2026-08-24 10:00").set_index("id_match")  # fmt: skip
    assert out.loc["g_r2", "p0_mid"] == 0.70 and out.loc["g_r2", "forecast_frozen_at"] == "run1"
    assert out.loc["g_held", "p0_mid"] == 0.55 and out.loc["g_held", "forecast_frozen_at"] == "run2"


# --- wiring: certainly kicked-off fixtures leave the board --------------------------------------


def test_load_cell_flags_kicked_off_fixtures_and_keeps_them_in_the_data(tmp_path):
    src = next(s for s in _cells() if s.startswith("# -------------------------- Load & Combine"))
    web = tmp_path / "10_data" / "106_Website"
    web.mkdir(parents=True)
    now = pd.Timestamp.now(tz=kickoff.BERLIN).tz_localize(None).floor("min")
    cols = ["id_match", "name_player", "season", "name_league", "gameday", "name_team", "name_opp",
            "home_pitch", "kick_off", "goals_in_match"]  # fmt: skip
    played_day = (now - pd.Timedelta(days=9)).strftime("%Y-%m-%d")  # both real files: date-only
    pd.DataFrame([("gP", "p1", "2026/27", "L", 1, "A", "Z", 1, played_day, 1)],
                 columns=cols).to_csv(web / "data_byPlayer.csv", index=False)  # fmt: skip
    yday = (now - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=15)
    soon = (now + pd.Timedelta(days=2)).normalize() + pd.Timedelta(hours=15)
    oos = pd.DataFrame([("gS", "p1", "2026/27", "L", 2, "A", "B", 1, yday, np.nan),
                        ("gL", "p1", "2026/27", "L", 3, "A", "C", 0, soon, np.nan),
                        ("gU", "p1", "2026/27", "L", 4, "A", "D", 1, pd.NaT, np.nan)],  # undated
                       columns=cols)  # fmt: skip
    feed = oos.dropna(subset=["kick_off"])
    pd.DataFrame({"season": "2026-27", "league": "L",
                  "team_home": np.where(feed.home_pitch == 1, feed.name_team, feed.name_opp),
                  "team_away": np.where(feed.home_pitch == 1, feed.name_opp, feed.name_team),
                  "kick_off_utc": pd.to_datetime(feed.kick_off).dt.tz_localize(kickoff.BERLIN)
                                    .dt.tz_convert("UTC").astype(str),
                  "time_confirmed": True}).to_csv(web / "fixtures_2026-27__kickoff.csv", index=False)  # fmt: skip
    oos.assign(kick_off=pd.to_datetime(oos.kick_off).dt.strftime("%Y-%m-%d")).to_csv(
        web / "data_byPlayer__OOS.csv", index=False)  # date-only, like the real OOS file
    ns = dict(pd=pd, np=np, os=os, kickoff=kickoff, directory=str(tmp_path), oos_season="2026/27",
              runroots=types.SimpleNamespace(run_clock=lambda r: pd.Timestamp.now(tz=kickoff.BERLIN)), ROOTS=None,
              FORECAST_HORIZON_DAYS=14)  # fmt: skip
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


def _repoint():
    fn = next(n for n in ast.walk(ast.parse(_ledger_cell()))
              if isinstance(n, ast.FunctionDef) and n.name == "_repoint")  # fmt: skip
    ns = {"np": np, "pd": pd}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(NB), "exec"), ns)
    return ns["_repoint"]


def test_a_row_without_a_fixture_key_is_not_repointed():
    # the stored fixture_key is the identity; a row without one keeps its id rather than being
    # guessed onto either leg
    first = _scored([("gF", "a_player", 3, "A", "B", 1, pd.Timestamp("2026-09-20 15:00"))])
    ret = _scored([("gR", "a_player", 22, "A", "B", 0, pd.Timestamp("2027-02-20 15:00"))])
    uni = pd.concat([first, ret])
    out = _repoint()(first.assign(id_match="g_old", fixture_key=np.nan), uni)
    assert out["id_match"].tolist() == ["g_old"]
    out = _repoint()(first.assign(id_match="g_old", fixture_key=_key(first)), uni)
    assert out["id_match"].tolist() == ["gF"]  # known key -> re-pointed to its own leg


def test_a_universe_row_of_unknown_orientation_is_ignored():
    ret = _scored([("gR", "a_player", 22, "A", "B", 0, pd.Timestamp("2027-02-20 15:00"))])
    first_unknown = _scored([("gX", "a_player", 3, "A", "B", np.nan, pd.Timestamp("2026-09-20 15:00"))])
    # gX has no orientation: read as an away row it would key as B v A, the return leg
    out = _repoint()(ret.assign(fixture_key=_key(ret)), pd.concat([first_unknown, ret]))
    assert out["id_match"].tolist() == ["gR"]


# --- season end: an empty board still grades what was played -----------------------------------


def test_with_nothing_on_the_board_played_rows_are_still_frozen(tmp_path):
    lg = Ledger(tmp_path)
    last = ("gL", "p1", 38, "A", "B", 1, pd.Timestamp("2027-05-23 15:00"))
    later = ("gP", "p2", 30, "C", "D", 1, pd.Timestamp("2027-05-26 20:00"))  # postponed, to play
    lg.run(_board([("p1", 38, "A", "B", 0.70), ("p2", 30, "C", "D", 0.60)]), _scored([last, later]), NO_PLAYED,
           "run1", at="2027-05-20 10:00")  # fmt: skip
    out = lg.run({}, _scored([])[:0], _played([last], 2), "run2", at="2027-05-24 10:00",
                 feed=_feed(_scored([last, later])), oos=_scored([later]),
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
        SMOKE=False, dict_PLAYERS={"oos": board}, data__all=data, SFM_model__NAME="m",
        KICKOFF_FEED=FEED, RUN_TIME=RT, datasets_to_process=datasets, os=os, LEDGER_PATH=str(ledger),
        update_frozen_ledger=lambda b, *a, **k: calls.append((b, k)),
    )  # fmt: skip
    exec(_run_block(), ns)
    if expected is None:
        assert calls == []
    elif expected == "board":
        assert calls[0][0] == board and calls[0][1]["feed"] is FEED
    else:
        assert calls[0][0] == {} and calls[0][1]["hold_pending"] is True


def test_a_board_that_resolves_to_no_rows_never_rewrites_the_ledger(tmp_path):
    lg = Ledger(tmp_path)
    lg.run(_board([("p1", 5, "A", "B", 0.70)]), _scored([G1]), NO_PLAYED, "run1")
    before = pathlib.Path(lg.path).read_bytes()
    with pytest.raises(ValueError, match="no ledger rows"):
        lg.run(_board([("p1", 9, "A", "B", 0.70)]), _scored([G1]), NO_PLAYED, "run2")
    assert pathlib.Path(lg.path).read_bytes() == before


@pytest.mark.parametrize("hold", [True, False])
def test_a_reused_id_never_freezes_the_wrong_fixture(tmp_path, hold):
    # C v D (g5) is postponed and leaves this run's data; upstream renumbers A v B g6 -> g5 and
    # it is played. C v D's pending row must not be frozen with A v B's result.
    lg = Ledger(tmp_path)
    cd = ("g5", "p2", 5, "C", "D", 1, pd.Timestamp("2026-09-27 15:00"))
    ab_old = ("g6", "p1", 6, "A", "B", 1, pd.Timestamp("2026-10-04 15:00"))
    ab_new = ("g5", "p1", 5, "A", "B", 1, pd.Timestamp("2026-10-04 15:00"))
    lg.run(_board([("p2", 5, "C", "D", 0.60), ("p1", 6, "A", "B", 0.70)]), _scored([cd, ab_old]), NO_PLAYED,
           "run1")  # fmt: skip
    board, scored = (
        ({}, _scored([])[:0]) if hold else (_board([("z_player", 4, "X", "Y", 0.9)]), _scored([DUMMY]))
    )
    out = lg.run(board, scored, _played([ab_new], 2), "run2", feed=_feed(_scored([DUMMY])),
                 oos=_scored([DUMMY]), hold_pending=hold)  # fmt: skip
    ab = out[out.name_player == "p1"].iloc[0]
    assert ab["id_match"] == "g5" and ab["status"] == "finished" and ab["actual_goals"] == 2
    cd_rows = out[out.name_player == "p2"]
    assert not (cd_rows["status"] == "finished").any()  # never graded with someone else's result
    assert cd_rows.iloc[0]["p0_mid"] == 0.60 and cd_rows.iloc[0]["status"] == "upcoming"


# --- export: the board never ships without its train block ----------------------------------------


def test_export_stops_when_the_train_block_is_neither_computed_nor_cached(tmp_path):
    src = next(s for s in _cells() if "Export the Dictionary" in s)
    ns = dict(SMOKE=False, datasets_to_process=["oos"], SFM_model__NAME="m", os=os, pickle=pickle,
              ROOTS=types.SimpleNamespace(state=tmp_path, publish=tmp_path / "pub"),
              dict_PLAYERS={"oos": {"p": {"model_seen": True}}}, PUBLISH_MODEL_SEEN_ONLY=True)  # fmt: skip
    with pytest.raises(FileNotFoundError, match="neither computed this run nor cached"):
        exec(compile(src, str(NB), "exec"), ns)
    assert not (tmp_path / "pub").exists()


def test_a_held_row_comes_back_byte_for_byte(tmp_path):
    # A receipt is never rewritten -- not even its last float digits (the default CSV parser is
    # off by up to 1 ULP, so a plain read -> write changed every pending line until 2026-10).
    lg = Ledger(tmp_path)
    p0 = 0.13567400305398386  # 17 significant digits: the case the default parser rounds
    lg.run(_board([("p1", 5, "A", "B", p0)]), _scored([G1]), NO_PLAYED, "run1", at="2026-09-05 10:00")
    before = [ln for ln in pathlib.Path(lg.path).read_text().splitlines() if "p1" in ln]
    lg.run(_board([("p1", 5, "A", "B", 0.2)]), _scored([G1]), NO_PLAYED, "run2", at="2026-09-07 10:00")
    after = [ln for ln in pathlib.Path(lg.path).read_text().splitlines() if "p1" in ln]
    assert after == before
