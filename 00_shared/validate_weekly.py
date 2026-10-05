#!/usr/bin/env python
"""Validate the weekly forecast scripts on the REAL files, without touching production.

Runs 006_040, 006_041, 006_060 and 006_061 as a validation run: inputs are read from the data root
in sfm_local.toml, while every write (ledgers, boards, feeds, pickles) goes to a scratch folder
seeded with copies of production's state first. Then it PROVES production was not touched --
every file in the data folder and the publish folder is hashed before and after -- and compares
each ledger with production's.

    python 00_shared/validate_weekly.py                          # clock = now
    python 00_shared/validate_weekly.py --at "2026-10-10 18:00"  # replay mid-matchday (Berlin)

Run it with the production interpreter (the sfmII venv). Exit code 0 only if every script ran,
production is byte-identical afterwards, no ledger row was dropped and no frozen SFM row changed.
Use --at whenever no fixture lies inside the hold window today: a run that holds nothing proves
nothing about the hold.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import runroots  # noqa: E402

REPO = runroots.REPO
W = '10_data/106_Website'
SFM_NB = REPO / '01_SFM/00_code/003__SFM_II/006_040__Predictions_ScoringProb__SFM_II.ipynb'
SFM_SAR = REPO / '01_SFM/00_code/003__SFM_II/006_041__SAR_PAR__SFM_II.py'
MMO_060 = REPO / '03_SFMMO/00_code/00_02__SFMMO/006_060__Predictions_MatchOutcome__SFMMO.py'
MMO_061 = REPO / '03_SFMMO/00_code/00_02__SFMMO/006_061__SeasonOdds__SFMMO.py'
SFM_STATE = [f'{W}/SFM_predictions__frozen.csv',
             f'{W}/040_ScoringProb__SFM_II_FinalC_ELO_scaleCS__2526__train.pkl']
LEDGERS = {'SFM': (f'{W}/SFM_predictions__frozen.csv', ['fixture_key', 'name_player']),
           'SFMMO': (f'{W}/SFMMO_predictions__frozen.csv', ['season', 'home_team', 'away_team'])}


def _state_files(path):
    """A script's own STATE_FILES list -- single source, read by importing the script."""
    spec = importlib.util.spec_from_file_location(path.stem.replace('-', '_'), path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return list(mod.STATE_FILES)


def _hashes(*folders):
    out = {}
    for d in folders:
        for f in sorted(pathlib.Path(d).iterdir()):
            if f.is_file():
                out[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def _run_notebook(nb):
    """Execute a notebook's code cells in one namespace, SMOKE=False after the config cell."""
    cells = [''.join(c['source']) for c in json.loads(pathlib.Path(nb).read_text())['cells']
             if c['cell_type'] == 'code']
    ns = {'__name__': '__main__'}
    for i, src in enumerate(cells):
        exec(compile(src, f'{pathlib.Path(nb).name}:cell{i}', 'exec'), ns)
        if i == 0:
            ns['SMOKE'] = False
            print('[validate] SMOKE = False')


def _step(name, cmd, cwd, env, log):
    print(f'[validate] {name} ...', flush=True)
    with open(log, 'w') as fh:
        rc = subprocess.run(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    tail = pathlib.Path(log).read_text().splitlines()
    for line in tail:
        if any(k in line for k in ('[roots]', 'receipts rule', '[ledger]', 'held', 'horizon:',
                                   'Written', 'Saved', '!!')):
            print(f'           {line[:150]}')
    if rc:
        print(textwrap.indent('\n'.join(tail[-15:]), '           '))
        raise SystemExit(f'[validate] {name} FAILED (exit {rc}) -- full log: {log}')


def _compare(label, prod, val, key):
    if not prod.exists():
        print(f'[validate] {label}: no production ledger to compare with')
        return True
    P = pd.read_csv(prod, dtype=str, keep_default_na=False).set_index(key)
    V = pd.read_csv(val, dtype=str, keep_default_na=False).set_index(key)
    common = P.index.intersection(V.index)
    same = (P.loc[common] == V.loc[common][P.columns]).all(axis=1)
    dropped, new = P.index.difference(V.index), V.index.difference(P.index)
    ok = len(dropped) == 0 and list(P.columns) == list(V.columns)
    msg = (f'[validate] {label} ledger: {len(P):,} rows in production | {int(same.sum()):,} '
           f'byte-identical | {int((~same).sum()):,} refreshed | {len(new):,} new | '
           f'{len(dropped):,} DROPPED')
    if 'status' in P.columns:
        fin = (P.loc[common, 'status'] == 'finished').values
        bad = int((fin & ~same.values).sum())
        msg += f' | frozen rows changed: {bad} of {int(fin.sum()):,}'
        ok &= bad == 0
    print(msg + ('' if ok else '   <-- FAIL'))
    return ok


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--exec-notebook':
        return _run_notebook(sys.argv[2])
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--at', help="replay the run at this Berlin time, e.g. '2026-10-10 18:00'")
    ap.add_argument('--scratch', help='scratch folder (default: a new temp folder)')
    args = ap.parse_args()

    prod = runroots.load_roots(env={})
    scratch = pathlib.Path(args.scratch or tempfile.mkdtemp(prefix='sfm_validate_'))
    state = SFM_STATE + _state_files(MMO_060) + _state_files(MMO_061)
    before = _hashes(prod.data / W, prod.publish)
    seeded = runroots.seed_validation_state(scratch, state)
    print(f'[validate] scratch {scratch} | seeded {sum(seeded.values())} of {len(seeded)} state files'
          + (f' | ABSENT in production: {[k for k, v in seeded.items() if not v]}'
             if not all(seeded.values()) else ''))

    env = dict(os.environ, SFM_VALIDATION_DIR=str(scratch), SFM_REPO=str(REPO))
    if args.at:
        env['SFM_VALIDATION_NOW'] = args.at
    py = sys.executable
    _step('006_040 scoring probabilities', [py, __file__, '--exec-notebook', str(SFM_NB)],
          SFM_NB.parent, env, scratch / '006_040.log')
    _step('006_041 SAR/PAR', [py, str(SFM_SAR)], SFM_SAR.parent, env, scratch / '006_041.log')
    _step('006_060 match outcomes', [py, str(MMO_060)], MMO_060.parent, env, scratch / '006_060.log')
    _step('006_061 season odds', [py, str(MMO_061)], MMO_061.parent, env, scratch / '006_061.log')

    after = _hashes(prod.data / W, prod.publish)
    touched = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    ok = not touched
    print(f'[validate] production files hashed: {len(before)} | changed: {len(touched)}'
          + ('' if ok else f'   <-- FAIL {touched[:5]}'))
    for label, (rel, key) in LEDGERS.items():
        ok &= _compare(label, prod.data / rel, scratch / rel, key)
    print(f'[validate] {"PASS" if ok else "FAIL"} -- logs and outputs in {scratch}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
