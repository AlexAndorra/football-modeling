"""Where a forecast script reads, keeps its state, and publishes -- in production or in validation.

Three kinds of file, three roots:
  data     READ-ONLY inputs: OOS fixtures, history, the kick-off feed, the model bundle.
  state    files the NEXT run reads back: the frozen ledger, the train-predictions cache, the
           previous board, the tracker. Scripts read AND write these here.
  publish  the published artifacts (board pickles) that downstream consumers pick up.

Production: state = data, and publish = the consumer's folder from the local settings file.
Validation (env SFM_VALIDATION_DIR=/some/scratch): data stays production (real inputs), while state
and publish move into the scratch folder. A validation run reads production, writes scratch, never
the reverse.

STATE MUST BE SEEDED. An empty scratch state behaves like a first-ever run: no ledger, so nothing is
held; no train cache, so 006_040 would ship a board without its train block. That exercises a
different code path from production and passes while proving nothing. `seed_validation_state` copies
production's state files into scratch first, and `load_roots` refuses validation until it has run.

Cross-script reads: importing another script's CODE is fine. Reading another script's OUTPUT in a
validation run must come from the same run's scratch, never from production, or a new script gets
validated against an old one's artifact.

The machine-specific paths live in `sfm_local.toml` at the repo root, which is git-ignored (see
sfm_local.example.toml):
    data_root   = "/path/to/51_SoccerAnalytics"
    publish_dir = "/path/to/consumer/data/folder"
"""

import dataclasses
import os
import pathlib
import shutil
import tomllib

REPO = pathlib.Path(__file__).resolve().parents[1]
SETTINGS = REPO / 'sfm_local.toml'
SEED_MARKER = '.seeded_from_production'


@dataclasses.dataclass(frozen=True)
class Roots:
    data: pathlib.Path
    state: pathlib.Path
    publish: pathlib.Path
    validation: bool

    def describe(self):
        mode = 'VALIDATION (writes go to scratch)' if self.validation else 'PRODUCTION'
        return (f'[roots] {mode}\n   data    {self.data}\n   state   {self.state}\n'
                f'   publish {self.publish}')


def _settings(path=SETTINGS):
    if not path.exists():
        raise FileNotFoundError(f'{path} missing -- copy sfm_local.example.toml and set data_root '
                                f'and publish_dir for this machine')
    with path.open('rb') as f:
        s = tomllib.load(f)
    for k in ('data_root', 'publish_dir'):
        if k not in s:
            raise KeyError(f'{path}: `{k}` not set')
    return s


def load_roots(settings_path=SETTINGS, env=os.environ):
    """The roots for this run. Validation when SFM_VALIDATION_DIR is set, production otherwise."""
    s = _settings(settings_path)
    data = pathlib.Path(s['data_root'])
    scratch = env.get('SFM_VALIDATION_DIR')
    if not scratch:
        return Roots(data=data, state=data, publish=pathlib.Path(s['publish_dir']), validation=False)
    scratch = pathlib.Path(scratch)
    if scratch.resolve() == data.resolve() or data.resolve() in scratch.resolve().parents:
        raise ValueError(f'SFM_VALIDATION_DIR={scratch} lies inside the production data root')
    if not (scratch / SEED_MARKER).exists():
        raise RuntimeError(f'{scratch} was never seeded -- run seed_validation_state first; an empty '
                           f'state is a first-ever run and proves nothing about production')
    return Roots(data=data, state=scratch, publish=scratch / 'publish', validation=True)


def run_clock(roots, env=os.environ):
    """The run's clock, tz-aware Berlin. A validation run may replay another moment through
    SFM_VALIDATION_NOW (e.g. '2026-10-10 18:00', read as Berlin wall time), so the kick-off holds
    can be exercised mid-matchday on real inputs. Production refuses it: a fake clock there would
    decide which receipts are written."""
    import pandas as pd

    fake = env.get('SFM_VALIDATION_NOW')
    if not fake:
        return pd.Timestamp.now(tz='Europe/Berlin')
    if not roots.validation:
        raise RuntimeError('SFM_VALIDATION_NOW is set on a PRODUCTION run -- refused; a fake clock '
                           'may only replay a validation run')
    t = pd.Timestamp(fake)
    t = t.tz_localize('Europe/Berlin') if t.tzinfo is None else t.tz_convert('Europe/Berlin')
    print(f'[roots] VALIDATION clock replayed at {t}')
    return t


def seed_validation_state(scratch, state_files, settings_path=SETTINGS):
    """Copy production's state files (paths relative to data_root) into `scratch`, keeping their
    relative paths, and mark the folder seeded. Returns {relative path: copied?}. A state file that
    production does not have is reported, not invented."""
    data = pathlib.Path(_settings(settings_path)['data_root'])
    scratch = pathlib.Path(scratch)
    copied = {}
    for rel in state_files:
        src, dst = data / rel, scratch / rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied[str(rel)] = src.exists()
    (scratch / 'publish').mkdir(parents=True, exist_ok=True)
    (scratch / SEED_MARKER).write_text(
        '\n'.join(f'{"copied " if ok else "ABSENT "} {rel}' for rel, ok in copied.items()) + '\n')
    return copied
