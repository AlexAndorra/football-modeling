"""Production vs validation roots, and the seeding that keeps validation honest."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import runroots  # noqa: E402

LEDGER = '10_data/106_Website/SFM_predictions__frozen.csv'


@pytest.fixture
def prod(tmp_path):
    data, pub = tmp_path / 'prod', tmp_path / 'consumer'
    (data / '10_data/106_Website').mkdir(parents=True)
    (data / LEDGER).write_text('fixture_key\nreal-ledger\n')
    pub.mkdir()
    cfg = tmp_path / 'sfm_local.toml'
    cfg.write_text(f'data_root = "{data}"\npublish_dir = "{pub}"\n')
    return data, pub, cfg


def test_production_roots(prod):
    data, pub, cfg = prod
    r = runroots.load_roots(cfg, env={})
    assert (r.data, r.state, r.publish, r.validation) == (data, data, pub, False)


def test_validation_reads_production_writes_scratch(prod, tmp_path):
    data, pub, cfg = prod
    scratch = tmp_path / 'scratch'
    runroots.seed_validation_state(scratch, [LEDGER], cfg)
    r = runroots.load_roots(cfg, env={'SFM_VALIDATION_DIR': str(scratch)})
    assert r.validation and r.data == data and r.state == scratch
    assert r.publish == scratch / 'publish' and r.publish != pub
    assert (r.state / LEDGER).read_text() == (data / LEDGER).read_text()


def test_unseeded_validation_is_refused(prod, tmp_path):
    _, _, cfg = prod
    with pytest.raises(RuntimeError, match='never seeded'):
        runroots.load_roots(cfg, env={'SFM_VALIDATION_DIR': str(tmp_path / 'empty')})


def test_scratch_inside_production_is_refused(prod):
    data, _, cfg = prod
    with pytest.raises(ValueError, match='inside the production'):
        runroots.load_roots(cfg, env={'SFM_VALIDATION_DIR': str(data / 'tmp')})


def test_absent_state_is_reported_not_invented(prod, tmp_path):
    _, _, cfg = prod
    got = runroots.seed_validation_state(tmp_path / 's', [LEDGER, 'no/such/cache.pkl'], cfg)
    assert got == {LEDGER: True, 'no/such/cache.pkl': False}
    assert not (tmp_path / 's/no/such/cache.pkl').exists()


def test_a_replayed_clock_only_on_validation(prod, tmp_path):
    _, _, cfg = prod
    scratch = tmp_path / 'scratch'
    runroots.seed_validation_state(scratch, [LEDGER], cfg)
    v = runroots.load_roots(cfg, env={'SFM_VALIDATION_DIR': str(scratch)})
    t = runroots.run_clock(v, env={'SFM_VALIDATION_NOW': '2026-10-10 18:00'})
    assert str(t) == '2026-10-10 18:00:00+02:00'
    with pytest.raises(RuntimeError, match='PRODUCTION'):
        runroots.run_clock(runroots.load_roots(cfg, env={}), env={'SFM_VALIDATION_NOW': '2026-10-10'})
    assert runroots.run_clock(runroots.load_roots(cfg, env={}), env={}).tzinfo is not None


def test_missing_settings_file_says_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match='sfm_local.example.toml'):
        runroots.load_roots(tmp_path / 'nope.toml', env={})
