"""The SFMMO's training-data fingerprint (00_shared/data_contract.py at MATCH level).

The bundle carries no copy of its data, so a historical repair that changes a training-season result
would silently serve the model on data it was not fitted on. The fingerprint stops that run.
"""

import importlib.util
import pathlib
import warnings

import pandas as pd
import pytest

SCRIPT = (pathlib.Path(__file__).resolve().parents[1] / "00_02__SFMMO"
          / "006_060__Predictions_MatchOutcome__SFMMO.py")  # fmt: skip


def _load():
    spec = importlib.util.spec_from_file_location("sfmmo_fp", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(mod)
    return mod


W = _load()


def _player_rows(season, id_match, home, away, gh, ga, players=1):
    base = dict(id_match=id_match, name_league="PL", id_league=1, season=season, gameday=1,
                kick_off="2025-08-16", points_team=0, points_opp=0, points_diff=0,
                goalsscored_cum_team=0, goalsscored_cum_opp=0, goalsconceded_cum_team=0,
                goalsconceded_cum_opp=0)  # fmt: skip
    rows = []
    for i in range(players):
        rows.append(dict(base, name_player=f"{home}{i}", name_team=home, name_opp=away, home_pitch=1,
                         goalsscored_inGame_team=gh, goalsscored_inGame_opp=ga))  # fmt: skip
    rows.append(dict(base, name_player=f"{away}0", name_team=away, name_opp=home, home_pitch=0,
                     goalsscored_inGame_team=ga, goalsscored_inGame_opp=gh))  # fmt: skip
    return rows


def _cd(changes=None, players=1):
    rows = (_player_rows("2024/25", "PL-S2425_GD1_G1", "A", "B", 2, 1, players)
            + _player_rows("2025/26", "PL-S2526_GD1_G1", "B", "A", 0, 0)
            + _player_rows("2026/27", "PL-S2627_GD1_G1", "A", "B", 3, 0))
    raw = pd.DataFrame(rows)
    for (season, col), val in (changes or {}).items():
        raw.loc[(raw.season == season) & (raw.home_pitch == 1), col] = val
    return W.build_match_level(raw)


def test_fingerprint_ignores_row_order_and_number_format():
    cd = _cd()
    shuffled = cd.sample(frac=1, random_state=1)
    as_float = cd.astype({"goalsscored_inGame_team": float})
    fp = W.training_fingerprint(cd, "2025/26")
    assert fp == W.training_fingerprint(shuffled, "2025/26") == W.training_fingerprint(as_float, "2025/26")
    assert sorted(fp["by_season"]) == ["2024/25", "2025/26"]  # the current season is not hashed


def test_an_extra_player_row_with_the_same_match_values_changes_nothing():
    # the bundles were fitted on the __TM file and are served from data_byPlayer.csv: they differ
    # in player rows, never in match-level values
    assert W.training_fingerprint(_cd(players=1), "2025/26") == W.training_fingerprint(_cd(players=3), "2025/26")


def test_a_changed_training_result_stops_the_run_and_names_the_season(tmp_path):
    bundle = tmp_path / "SFMMO_bundle.pkl"
    W.record_training_baseline(_cd(), "2025/26", str(bundle), reason="test")
    assert W.check_training_data(None, _cd(), str(bundle)).startswith("training fingerprint: OK")
    repaired = _cd({("2025/26", "goalsscored_inGame_team"): 1})
    with pytest.raises(RuntimeError, match=r"\['2025/26'\]"):
        W.check_training_data(None, repaired, str(bundle))
    assert "MISMATCH" in W.check_training_data(None, repaired, str(bundle), allow=True)


def test_a_current_season_change_is_not_a_training_change(tmp_path):
    bundle = tmp_path / "SFMMO_bundle.pkl"
    W.record_training_baseline(_cd(), "2025/26", str(bundle), reason="test")
    later = _cd({("2026/27", "goalsscored_inGame_team"): 5})
    assert W.check_training_data(None, later, str(bundle)).startswith("training fingerprint: OK")


def test_a_fingerprint_in_the_bundle_wins_over_the_baseline_file(tmp_path):
    bundle = tmp_path / "SFMMO_bundle.pkl"
    W.record_training_baseline(_cd({("2025/26", "goalsscored_inGame_team"): 1}), "2025/26", str(bundle),
                               reason="stale baseline")  # fmt: skip
    contract = {"fingerprint": W.training_fingerprint(_cd(), "2025/26")}
    assert W.check_training_data(contract, _cd(), str(bundle)).startswith("training fingerprint: OK")


def test_a_baseline_needs_a_reason(tmp_path):
    with pytest.raises(ValueError, match="reason"):
        W.record_training_baseline(_cd(), "2025/26", str(tmp_path / "b.pkl"), reason="")
