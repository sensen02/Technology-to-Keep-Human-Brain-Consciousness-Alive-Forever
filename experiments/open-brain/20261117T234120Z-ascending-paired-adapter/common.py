"""P1 only: immutable inputs for paired ascending sensory calibration."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE.parent
FOUNDATION = BASE / '20261117T045901Z-foundation'
BIDIR = BASE / '20261117T051249Z-bidirectional'
FIXED = BASE / '20261117T164207Z-fixed-neck'
BODY = BASE / '20261117T211320Z-body-command-calibration'
sys.path[:0] = [str(FIXED), str(BIDIR), str(FOUNDATION)]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def pin(paths):
    return {str(Path(p).resolve()): sha(p) for p in paths}


def verify(pins):
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError(f'input changed: {path}')


def imported_sources():
    paths = set(HERE.glob('*.py'))
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).is_file() and Path(path).resolve().is_relative_to(ROOT):
            paths.add(Path(path).resolve())
    return paths
