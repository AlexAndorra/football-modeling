"""The HL=4 shadow arm (PREREG_HL4_prospective_2026-28.md) and the pre-season board rebuild.

A shadow run must never write a live file, and a pre-season board must contain no result of the
season it forecasts.
"""

import importlib.util
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

HERE = pathlib.Path(__file__).resolve().parents[1] / "00_02__SFMMO"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():  # pymc/pytensor import chatter is not under test
        warnings.simplefilter("ignore")
        spec.loader.exec_module(mod)
    return mod


def test_live_run_writes_the_live_folder_with_the_live_bundle(monkeypatch):
    monkeypatch.delenv("SFMMO_SHADOW", raising=False)
    W = _load("w_live", "006_060__Predictions_MatchOutcome__SFMMO.py")
    assert W.SHADOW is None and "_shadow" not in W.OUT_DIR
    assert W.BUNDLE_PATH.endswith(W.BUNDLES[None])


def test_shadow_run_moves_every_output_into_its_own_folder(monkeypatch):
    monkeypatch.setenv("SFMMO_SHADOW", "hl4")
    W = _load("w_hl4", "006_060__Predictions_MatchOutcome__SFMMO.py")
    assert W.BUNDLE_PATH.endswith("SFMMO_DevK_hl4__scaleCS__train202526__PROD__4ch.pkl")
    outputs = [W.OUT_MATCH_CSV, W.OUT_GRID_CSV, W.OUT_TEAM_CSV, W.OUT_PKL, W.FROZEN_LEDGER, W.VINTAGE_DIR]
    assert all("/_shadow/hl4" in o for o in outputs)
    O = _load("o_hl4", "006_061__SeasonOdds__SFMMO.py")  # 006_061 follows 006_060's choice
    assert O.BUNDLE_PATH == O._p.BUNDLE_PATH and "/_shadow/hl4" in O.OUT_CSV and "/_shadow/hl4" in O.TRACKER_CSV


def test_an_unknown_shadow_arm_is_refused(monkeypatch):
    monkeypatch.setenv("SFMMO_SHADOW", "hl3")
    with pytest.raises(SystemExit, match="hl3"):
        _load("w_bad", "006_060__Predictions_MatchOutcome__SFMMO.py")


@pytest.fixture
def PB(monkeypatch):
    monkeypatch.delenv("SFMMO_SHADOW", raising=False)
    return _load("preseason", "preseason_board.py")


def test_preseason_removes_every_result_of_the_season_and_nothing_else(PB):
    cd = pd.DataFrame(
        dict(season=["2025/26", "2025/26", "2026/27", "2026/27"], match_outcome=[2.0, 1.0, 3.0, 0.0],
             goalsscored_inGame_team=[2.0, 1.0, 3.0, 0.0], goalsscored_inGame_opp=[1.0, 2.0, 0.0, 3.0])
    )  # fmt: skip
    out = PB.without_season_results(cd, "2026/27")
    cols = ["match_outcome", "goalsscored_inGame_team", "goalsscored_inGame_opp"]
    assert out.loc[out.season == "2026/27", cols].isna().all().all()
    assert out.loc[out.season == "2025/26", cols].equals(cd.loc[cd.season == "2025/26", cols])
    assert cd[cols].notna().all().all()  # the input is not modified


def test_rank_gate_refuses_a_board_whose_ranks_are_not_a_permutation(PB):
    good = pd.DataFrame(dict(league="serie-a", team=list("ABCDEF"),
                             p_title=[1, 0, 0, 0, 0, 0], p_top4=[1, 1, 1, 1, 0, 0],
                             p_releg=[0, 0, 0, 1, 1, 1]))  # fmt: skip
    PB.O.check_rank_validity(good.astype({c: float for c in ("p_title", "p_top4", "p_releg")}))
    bad = good.assign(p_title=[0.6, 0.3, 0, 0, 0, 0])  # 0.9 champions per season
    with pytest.raises(AssertionError, match="p_title"):
        PB.O.check_rank_validity(bad)


def test_season_board_pins_nothing_once_the_season_is_cleared(PB):
    # two-team league: with the results removed, every fixture is simulated (none pinned)
    teams = ["A", "B"]
    cd = pd.DataFrame(
        dict(season="2026/27", name_league="l", home_pitch=1, name_team=["A", "B"], name_opp=["B", "A"],
             kick_off=pd.to_datetime(["2026-08-20", "2027-05-20"]), match_outcome=[2.0, np.nan],
             goalsscored_inGame_opp=[0.0, np.nan], elo_team=[1500.0, 1500.0], elo_opp=[1500.0, 1500.0],
             id_match=["l_GD1_G1", "l_GD2_G1"])
    )  # fmt: skip
    rng_draws = 50
    sp = dict(mu=np.zeros(rng_draws), alpha=np.zeros((2, rng_draws)), delta=np.zeros((2, rng_draws)),
              beta_home=np.zeros((2, rng_draws)), mu_gamma=np.zeros(rng_draws),
              beta=np.zeros((2, rng_draws)), team_to_idx={t: i for i, t in enumerate(teams)},
              factors_g=["elo_team", "elo_opp"], elo_mu=1500.0, elo_sd=100.0, eloopp_mu=1500.0,
              eloopp_sd=100.0)  # fmt: skip
    pinned_live = PB.O.season_board(cd, sp, stamp="x", target_season="2026/27")
    cleared = PB.without_season_results(cd.assign(goalsscored_inGame_team=cd["match_outcome"]), "2026/27")
    pre = PB.O.season_board(cleared, sp, stamp="x", target_season="2026/27")
    # with A's 2-0 win pinned, A's expected points exceed B's; cleared, they are symmetric
    assert pinned_live.set_index("team").loc["A", "exp_pts"] > pinned_live.set_index("team").loc["B", "exp_pts"]
    assert abs(pre.set_index("team").loc["A", "exp_pts"] - pre.set_index("team").loc["B", "exp_pts"]) < 1.0


@pytest.mark.skipif(__import__("sys").version_info >= (3, 13), reason="pathlib._local exists natively")
def test_a_bundle_pickled_under_python_313_loads(tmp_path, monkeypatch):
    # Colab moved to Python 3.13 (2026-10), whose pickles name PosixPath as pathlib._local.PosixPath
    import pickle
    import sys

    monkeypatch.delitem(sys.modules, "pathlib._local", raising=False)
    py312 = pickle.dumps({"rho": -0.05, "where": pathlib.PosixPath("/content/drive")}, protocol=4)
    py313 = py312.replace(b"\x8c\x07pathlib", b"\x8c\x0epathlib._local")
    assert py313 != py312
    with pytest.raises(ModuleNotFoundError):
        pickle.loads(py313)  # what the production interpreter does without the alias
    f = tmp_path / "bundle.pkl"
    f.write_bytes(py313)
    W = _load("w_313", "006_060__Predictions_MatchOutcome__SFMMO.py")
    assert W.load_bundle_file(f) == {"rho": -0.05, "where": pathlib.PosixPath("/content/drive")}
