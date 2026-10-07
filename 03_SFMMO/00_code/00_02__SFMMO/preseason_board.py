#!/usr/bin/env python
"""The PRE-SEASON title board of a season bundle, rebuilt from today's files.

PREREG_HL4_prospective_2026-28.md scores both arms on their pre-season boards. A bundle contains
no data from the season it forecasts, so its pre-season board can be rebuilt at any time: take
today's inputs, remove every result of the target season, recompute ELO (each team then enters
the simulation at its rating going into its first fixture), and run 006_061's own simulation.
Nothing here re-implements the board: load_bundle / season_board / check_rank_validity are
006_061's functions.

    python preseason_board.py --bundle SFMMO_DevK_hl4__scaleCS__train202526__PROD__4ch.pkl

The bundle is looked up in 10_data/01_Models under the data root. The board is written to the
state root's 10_data/106_Website/_shadow/preseason/ and nowhere else.
"""

import argparse
import importlib.util
import os
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('sfmmo_season_odds', HERE / '006_061__SeasonOdds__SFMMO.py')
O = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(O)
P = O._p     # 006_060: data helpers, paths, roots


def without_season_results(cd, season):
    """Match-level data as it stood before `season`'s first kick-off: every result of that
    season removed. Earlier seasons are untouched."""
    out = cd.copy()
    m = out['season'] == season
    out.loc[m, ['match_outcome', 'goalsscored_inGame_team', 'goalsscored_inGame_opp']] = np.nan
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--bundle', required=True, help='file name in 10_data/01_Models')
    ap.add_argument('--season', default=O.TARGET_SEASON)
    args = ap.parse_args(argv)
    if P.ROOTS is None:
        raise SystemExit('sfm_local.toml missing -- see sfm_local.example.toml')
    print(P.ROOTS.describe())

    bundle = P.ROOTS.data / '10_data' / '01_Models' / args.bundle
    meta, sp = O.load_bundle(bundle)
    if meta['train_end'] >= args.season:
        raise SystemExit(f"{args.bundle} was trained through {meta['train_end']}: it has seen "
                         f"{args.season}, so its board for it is not a pre-season board")

    raw = pd.concat([pd.read_csv(P.HIST_PATH, low_memory=False),
                     pd.read_csv(P.OOS_PATH, low_memory=False)], ignore_index=True)
    cd = P.compute_elo(without_season_results(P.build_match_level(raw), args.season))
    board = O.season_board(cd, sp, stamp='preseason', target_season=args.season)
    O.check_rank_validity(board)

    out_dir = P.ROOTS.state / '10_data' / '106_Website' / '_shadow' / 'preseason'
    os.makedirs(out_dir, exist_ok=True)
    out = out_dir / f"preseason_{args.season.replace('/', '-')}__{bundle.stem}.csv"
    board.to_csv(out, index=False)
    print(f'\nwrote {out}')
    for lg, g in board.groupby('league'):
        top = g.nlargest(3, 'p_title')
        print(f"  {lg:16s} " + ' | '.join(f'{r.team} {r.p_title:.1%}' for r in top.itertuples()))
    return board


if __name__ == '__main__':
    main()
