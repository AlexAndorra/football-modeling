"""ELO roll-forward of the SFM II weekly scoring notebook (006_040).

The bundle ships (a) the pre-match ratings of every TRAINED match (`match_table`) and (b) the
per-league state after the last trained match (`ratings_final`). Serving must reuse (a) for
trained matches and advance (b) only over matches played SINCE; re-playing history on top of
(b) hands the model ratings it was never fitted with. These tests exec the notebook's light
engine cell straight from the .ipynb, so they pin the code that actually runs.
"""

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

NB = (
    pathlib.Path(__file__).resolve().parents[3]
    / "01_SFM"
    / "00_code"
    / "003__SFM_II"
    / "006_040__Predictions_ScoringProb__SFM_II.ipynb"
)


def _engine():
    with NB.open() as f:
        cells = [c for c in json.load(f)["cells"] if c["cell_type"] == "code"]
    src = next("".join(c["source"]) for c in cells if "def update_elo" in "".join(c["source"]))
    ns = {"np": np, "pd": pd, "cred_region": 0.9}
    exec(compile(src, str(NB), "exec"), ns)
    return ns


E = _engine()


def _matches(rows):
    cols = [
        "id_match",
        "name_league",
        "season",
        "kick_off",
        "home",
        "away",
        "g_home",
        "g_away",
        "is_oos",
    ]
    m = pd.DataFrame(rows, columns=cols)
    m["kick_off"] = pd.to_datetime(m["kick_off"])
    return m.sort_values(["name_league", "kick_off", "id_match"])


def _cfg(match_table_rows, ratings_final):
    mt = pd.DataFrame(
        match_table_rows,
        columns=["id_match", "name_league", "season", "home", "away", "elo_home", "elo_away"],
    )
    return {
        "K": 20,
        "home_adv": 50,
        "ratings_final": ratings_final,
        "season_start": {"returning": (0.75, 1500), "promoted": 1300},
        "match_table": mt,
    }


@pytest.fixture
def toy():
    # season S1 trained (A, B); season S2 new: A-B played, B-C played (C promoted), C-A unplayed
    m = _matches(
        [
            ("s1g1", "l", "S1", "2025-08-01", "A", "B", 2, 0, False),
            ("s1g2", "l", "S1", "2025-08-08", "B", "A", 1, 1, False),
            ("s2g1", "l", "S2", "2026-08-01", "A", "B", 1, 0, False),
            ("s2g2", "l", "S2", "2026-08-08", "B", "C", 2, 2, False),
            ("s2g3", "l", "S2", "2026-08-15", "C", "A", np.nan, np.nan, True),
        ]
    )
    # deliberately NOT what a recomputation from 1500 would give -> proves the table is read
    cfg = _cfg(
        [
            ("s1g1", "l", "S1", "A", "B", 1510.0, 1490.0),
            ("s1g2", "l", "S1", "B", "A", 1480.0, 1520.0),
        ],
        {"l": {"A": 1520.0, "B": 1480.0}},
    )
    return m, cfg


def test_trained_matches_take_the_bundle_ratings_verbatim(toy):
    m, cfg = toy
    elo, n_fwd, n_frozen = E["roll_elo_forward"](m, cfg)
    e = elo.set_index("id_match")
    assert (e.loc["s1g1", ["elo_home", "elo_away"]].to_numpy() == [1510.0, 1490.0]).all()
    assert (e.loc["s1g2", ["elo_home", "elo_away"]].to_numpy() == [1480.0, 1520.0]).all()


def test_new_season_starts_from_ratings_final_with_the_training_rule(toy):
    m, cfg = toy
    elo, n_fwd, n_frozen = E["roll_elo_forward"](m, cfg)
    e = elo.set_index("id_match")
    a0, b0 = 0.75 * 1520 + 375, 0.75 * 1480 + 375  # returning teams regress to the mean
    assert (e.loc["s2g1", ["elo_home", "elo_away"]].to_numpy() == [a0, b0]).all()
    a1, b1 = E["update_elo"](a0, b0, 2)  # A beat B at home
    assert e.loc["s2g2", "elo_home"] == b1 and e.loc["s2g2", "elo_away"] == 1300.0  # C promoted
    b2, c2 = E["update_elo"](b1, 1300.0, 1)
    assert (e.loc["s2g3", ["elo_home", "elo_away"]].to_numpy() == [c2, a1]).all()  # frozen
    assert (n_fwd, n_frozen) == (2, 1)


def test_history_is_not_replayed_on_top_of_ratings_final(toy):
    """The defect: with 2 trained matches replayed, A/B would enter S2 from a different state."""
    m, cfg = toy
    elo, *_ = E["roll_elo_forward"](m, cfg)
    r_a, r_b = 1520.0, 1480.0
    for res in (2, 1):  # what a replay of s1g1, s1g2 would do to the final state
        r_a, r_b = (
            E["update_elo"](r_a, r_b, res) if res == 2 else E["update_elo"](r_b, r_a, res)[::-1]
        )
    replayed_a0 = 0.75 * r_a + 375
    assert elo.set_index("id_match").loc["s2g1", "elo_home"] != replayed_a0


def test_team_returning_after_a_season_away_is_reset_not_regressed():
    # S1 (A,B) and S2 (A,C) trained; B was away in S2 and comes back in S3 -> promoted rule (1300)
    m = _matches(
        [
            ("s1g1", "l", "S1", "2024-08-01", "A", "B", 1, 0, False),
            ("s2g1", "l", "S2", "2025-08-01", "A", "C", 1, 0, False),
            ("s3g1", "l", "S3", "2026-08-01", "B", "A", np.nan, np.nan, True),
        ]
    )
    cfg = _cfg(
        [
            ("s1g1", "l", "S1", "A", "B", 1500.0, 1500.0),
            ("s2g1", "l", "S2", "A", "C", 1505.0, 1300.0),
        ],
        {"l": {"A": 1520.0, "B": 1490.0, "C": 1285.0}},  # B's stale S1 rating still in the state
    )
    elo, *_ = E["roll_elo_forward"](m, cfg)
    e = elo.set_index("id_match")
    assert e.loc["s3g1", "elo_home"] == 1300.0
    assert e.loc["s3g1", "elo_away"] == 0.75 * 1520 + 375


def test_untrained_match_in_a_trained_season_rolls_forward():
    # a result recorded after the fit, inside the last trained season, updates the state
    m = _matches(
        [
            ("s1g1", "l", "S1", "2025-08-01", "A", "B", 1, 0, False),
            ("s1g9", "l", "S1", "2025-08-20", "B", "A", 0, 1, False),  # not in match_table
            ("s2g1", "l", "S2", "2026-08-01", "A", "B", np.nan, np.nan, True),
        ]
    )
    cfg = _cfg([("s1g1", "l", "S1", "A", "B", 1500.0, 1500.0)], {"l": {"A": 1510.0, "B": 1490.0}})
    elo, n_fwd, n_frozen = E["roll_elo_forward"](m, cfg)
    e = elo.set_index("id_match")
    assert (e.loc["s1g9", ["elo_home", "elo_away"]].to_numpy() == [1490.0, 1510.0]).all()
    b1, a1 = E["update_elo"](1490.0, 1510.0, 0)
    assert (
        e.loc["s2g1", ["elo_home", "elo_away"]].to_numpy() == [0.75 * a1 + 375, 0.75 * b1 + 375]
    ).all()
    assert (n_fwd, n_frozen) == (1, 1)


def test_every_match_gets_a_rating_and_ids_are_unique(toy):
    m, cfg = toy
    elo, *_ = E["roll_elo_forward"](m, cfg)
    assert elo["id_match"].is_unique and set(elo["id_match"]) == set(m["id_match"])
    assert elo[["elo_home", "elo_away"]].notna().all().all()


# --- red-team round 1 additions -------------------------------------------------------


def test_league_absent_from_the_bundle_follows_the_harness_first_season_rule(capsys):
    # harness: every team of a league starts at 1500 and NO start rule applies in its first season
    m = _matches(
        [
            ("n1", "new", "S1", "2026-08-01", "A", "B", 1, 0, False),
            ("n2", "new", "S1", "2026-08-08", "B", "A", np.nan, np.nan, True),
        ]
    )
    cfg = _cfg([("s1g1", "l", "S1", "X", "Y", 1500.0, 1500.0)], {"l": {"X": 1510.0, "Y": 1490.0}})
    elo, n_fwd, n_frozen = E["roll_elo_forward"](m, cfg)
    e = elo.set_index("id_match")
    assert (e.loc["n1", ["elo_home", "elo_away"]].to_numpy() == [1500.0, 1500.0]).all()
    a1, b1 = E["update_elo"](1500.0, 1500.0, 2)
    assert (e.loc["n2", ["elo_home", "elo_away"]].to_numpy() == [b1, a1]).all()
    assert "not in the bundle" in capsys.readouterr().out


def test_untrained_result_inside_a_trained_season_is_reported(capsys):
    m = _matches(
        [
            ("s1g1", "l", "S1", "2025-08-01", "A", "B", 1, 0, False),
            ("s1g9", "l", "S1", "2025-08-20", "B", "A", 0, 1, False),  # not in match_table
        ]
    )
    cfg = _cfg([("s1g1", "l", "S1", "A", "B", 1500.0, 1500.0)], {"l": {"A": 1510.0, "B": 1490.0}})
    E["roll_elo_forward"](m, cfg)
    out = capsys.readouterr().out
    assert "1 played match" in out and "trained season" in out


# --- red-team round 2 additions -------------------------------------------------------


def test_unsorted_input_is_refused(toy):
    m, cfg = toy
    with pytest.raises(AssertionError, match="sorted"):
        E["roll_elo_forward"](m.iloc[::-1], cfg)


def test_returning_team_missing_from_ratings_final_is_an_error(toy):
    m, cfg = toy
    cfg["ratings_final"]["l"].pop("B")  # inconsistent bundle: B played S1 but has no final rating
    with pytest.raises(AssertionError, match="ratings_final"):
        E["roll_elo_forward"](m, cfg)


def test_oos_row_carrying_a_result_is_frozen_and_reported(toy, capsys):
    m, cfg = toy
    m.loc[m["id_match"] == "s2g3", ["g_home", "g_away"]] = [3, 1]  # OOS row with goals
    elo, n_fwd, n_frozen = E["roll_elo_forward"](m, cfg)
    assert (n_fwd, n_frozen) == (2, 1)
    assert "result ignored" in capsys.readouterr().out
