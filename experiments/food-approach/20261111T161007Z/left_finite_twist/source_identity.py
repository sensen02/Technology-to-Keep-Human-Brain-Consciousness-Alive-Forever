"""Read-only source freeze checks, intentionally no physics dependencies."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
EXPECTED_ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_finite_twist')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_sources():
    if HERE != EXPECTED_ROOT:
        raise RuntimeError(f'Unexpected ROOT {HERE}; expected {EXPECTED_ROOT}')
    manifest = json.loads((HERE/'SOURCEHASH.json').read_text())
    bad = [p for p, h in manifest['files'].items() if not Path(p).is_file() or sha256(p) != h]
    if bad:
        raise RuntimeError(f'Frozen source/input drift: {bad}')
    prereg = json.loads((HERE/'PREREG.json').read_text())
    if prereg['root'] != str(HERE) or prereg['conditions'] != [dict(name='left_yaw_p060', speed_scale=1., yaw_rate=.06)]:
        raise RuntimeError('Only preregistered single LEFT +.06 condition allowed')
    return manifest, prereg
