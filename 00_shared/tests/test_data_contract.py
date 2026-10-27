"""The training-data fingerprint: what it ignores, what it catches, and that the Colab export's
inline copy stays identical to the shared module."""

import ast
import json
import pathlib
import sys

import pandas as pd
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import data_contract as dc  # noqa: E402

ROOT = HERE.parents[1]
DEV_NB = ROOT / '01_SFM' / '00_code' / '003__SFM_II' / 'SFM_II__dev_EW.ipynb'


def _rows():
    base = {c: '1' for c in dc.FINGERPRINT_COLS}
    rows = []
    for season, n in (('2024/25', 3), ('2025/26', 3), ('2026/27', 2)):
        for i in range(n):
            r = dict(base, season=season, id_match=f'M-{season}-{i}', name_player=f'p{i}',
                     name_team='A', name_opp='B', name_league='L', kick_off='2025-01-01',
                     position_player='Sturm', goalsscored_share_player_team='0.5')
            rows.append(r)
    return pd.DataFrame(rows)


def _fp(df, tmp_path, name='d.csv'):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return dc.training_fingerprint(p, '2025/26')


def test_counts_only_training_seasons(tmp_path):
    fp = _fp(_rows(), tmp_path)
    assert fp['n_rows'] == 6 and sorted(fp['by_season']) == ['2024/25', '2025/26']


def test_row_order_number_format_extra_columns_and_current_season_are_ignored(tmp_path):
    ref = _fp(_rows(), tmp_path, 'a.csv')
    d = _rows().sample(frac=1, random_state=3)                            # row order
    d['goals_in_match'] = '1.0'                                          # '1' vs '1.0'
    d['goalsscored_share_player_team'] = '0.50000000000001'              # float noise
    d['odds_home'] = '2.1'                                               # an unrelated column
    d.loc[d.season == '2026/27', 'goals_in_match'] = '3'                 # the current season
    assert _fp(d, tmp_path, 'b.csv')['sha256'] == ref['sha256']


def test_a_changed_value_is_caught_and_its_season_named(tmp_path):
    ref = _fp(_rows(), tmp_path, 'a.csv')
    d = _rows()
    d.loc[(d.season == '2024/25') & (d.name_player == 'p1'), 'goals_in_match'] = '2'   # a brace repaired
    new = _fp(d, tmp_path, 'b.csv')
    assert new['sha256'] != ref['sha256'] and dc.changed_seasons(ref, new) == ['2024/25']


def test_an_in_memory_frame_hashes_like_the_file(tmp_path):
    # a writer with its own model inputs passes a frame; numeric dtypes and NaN must not matter
    d = _rows()
    f = d.copy()
    f['goals_in_match'] = f['goals_in_match'].astype(float)            # numeric dtype
    f['gameday'] = f['gameday'].astype(int)
    assert dc.fingerprint_frame(f, '2025/26') == _fp(d, tmp_path)
    m = dc.fingerprint_frame(f, '2025/26', columns=['season', 'id_match', 'goals_in_match'], keys=['id_match'])
    assert m['columns'] == ['season', 'id_match', 'goals_in_match'] and m['n_rows'] == 6


def test_check_raises_on_a_mismatch_unless_overridden(tmp_path):
    good = tmp_path / 'good.csv'
    _rows().to_csv(good, index=False)
    bundle = tmp_path / 'bundle.pkl'
    contract = {'fingerprint': dc.training_fingerprint(good, '2025/26')}
    assert 'OK' in dc.check(good, contract, bundle)
    d = _rows()
    d.loc[0, 'points_diff'] = '7'
    bad = tmp_path / 'bad.csv'
    d.to_csv(bad, index=False)
    with pytest.raises(RuntimeError, match='MISMATCH'):
        dc.check(bad, contract, bundle)
    assert 'MISMATCH' in dc.check(bad, contract, bundle, allow=True)


def test_baseline_file_is_used_when_the_bundle_has_no_fingerprint(tmp_path):
    csv = tmp_path / 'd.csv'
    _rows().to_csv(csv, index=False)
    bundle = tmp_path / 'old_bundle__LIGHT.pkl'
    assert 'NONE' in dc.check(csv, {}, bundle)                           # no fingerprint anywhere
    with pytest.raises(ValueError, match='reason'):
        dc.record_baseline(csv, '2025/26', bundle, reason='')
    dc.record_baseline(csv, '2025/26', bundle, reason='bundle predates fingerprints')
    assert (tmp_path / 'old_bundle__LIGHT__fingerprint.json').exists()
    assert 'old_bundle__LIGHT__fingerprint.json' in dc.check(csv, {}, bundle)


def _counts_rows():
    # 20 training rows (one share > 1, so 19 served) + 2 current-season rows; one hat-trick+
    d = _rows().iloc[:0]
    rows = []
    for i in range(20):
        rows.append(dict(_rows().iloc[0], id_match=f'M{i}', name_player=f'p{i}', season='2025/26',
                         goals_in_match='0' if i < 15 else '1', goalsscored_share_player_team='0.5'))
    rows[15]['goals_in_match'] = '4'                                     # capped at 3
    rows[19]['goalsscored_share_player_team'] = '2'                      # dropped, as served
    for i in range(2):
        rows.append(dict(rows[0], id_match=f'N{i}', season='2026/27', goals_in_match='2'))
    return pd.concat([d, pd.DataFrame(rows)], ignore_index=True)


def test_training_counts_filter_like_the_serving_scripts(tmp_path):
    csv = tmp_path / 'd.csv'
    _counts_rows().to_csv(csv, index=False)
    c = dc.training_counts(csv, '2025/26')
    assert c['n_rows'] == 19
    assert c['goal_cats_share'] == {0: round(15 / 19, 6), 1: round(3 / 19, 6), 3: round(1 / 19, 6)}


def test_counts_come_from_the_bundle_until_a_reviewed_baseline_accepts_a_repair(tmp_path):
    csv = tmp_path / 'd.csv'
    _counts_rows().to_csv(csv, index=False)
    bundle = tmp_path / 'b__LIGHT.pkl'
    contract = dc.training_counts(csv, '2025/26')                       # what the bundle stored at fit
    n, mix = contract['n_rows'], contract['goal_cats_share']
    assert 'OK' in dc.check_counts(n, mix, contract, bundle)
    # a repair flips one target 1 -> 0 (an own goal): the mix moves by 1/19, far past 1e-4
    d = _counts_rows()
    d.loc[16, 'goals_in_match'] = '0'
    d.to_csv(csv, index=False)
    rep = dc.training_counts(csv, '2025/26')
    with pytest.raises(RuntimeError, match='category mix MISMATCH'):
        dc.check_counts(rep['n_rows'], rep['goal_cats_share'], contract, bundle)
    assert 'FAILED' in dc.check_counts(rep['n_rows'], rep['goal_cats_share'], contract, bundle, allow=True)
    # a baseline WITHOUT counts (the pre-2026-10-27 format) leaves the bundle's counts in charge
    dc.record_baseline(csv, '2025/26', bundle, reason='fingerprint only')
    with pytest.raises(RuntimeError, match='MISMATCH'):
        dc.check_counts(rep['n_rows'], rep['goal_cats_share'], contract, bundle)
    # the reviewed baseline accepts the repaired counts too (through a JSON round trip: int keys)
    dc.record_baseline(csv, '2025/26', bundle, reason='own-goal repair', counts=rep)
    status = dc.check_counts(rep['n_rows'], rep['goal_cats_share'], contract, bundle)
    assert 'OK' in status and 'b__LIGHT__fingerprint.json' in status
    assert 'OK' in dc.check(csv, {k: v for k, v in contract.items()}, bundle)
    # ...and the pre-repair data now fails against it
    with pytest.raises(RuntimeError, match='MISMATCH'):
        dc.check_counts(n, mix, contract, bundle)


def _defs(src):
    tree = ast.parse(src)
    out = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in ('_canon', 'fingerprint_frame', 'training_fingerprint'):
            out[n.name] = ast.dump(n)
        if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', None) == 'FINGERPRINT_COLS':
            out['FINGERPRINT_COLS'] = ast.dump(n)
    return out


def test_the_colab_export_carries_an_identical_copy():
    cells = json.loads(DEV_NB.read_text())['cells']
    export = next(''.join(c['source']) for c in cells
                  if c['cell_type'] == 'code' and "'data_contract'" in ''.join(c['source'])
                  and 'def training_fingerprint' in ''.join(c['source']))
    assert "'fingerprint': training_fingerprint(" in export
    mod = _defs((HERE.parent / 'data_contract.py').read_text())
    assert _defs(export) == mod and len(mod) == 4, 'the export\'s inline copy drifted from data_contract.py'
