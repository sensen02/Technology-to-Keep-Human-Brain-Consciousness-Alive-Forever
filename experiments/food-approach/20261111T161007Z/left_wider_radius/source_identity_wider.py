"""Strict new-root source checks; no physics imports and no automatic re-freeze."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
EXPECTED_ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_wider_radius')
FREEZE=HERE/'SOURCEHASH_WIDER_RADIUS.json'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify_sources():
    if HERE!=EXPECTED_ROOT:raise RuntimeError(f'Unexpected ROOT: {HERE}')
    manifest=json.loads(FREEZE.read_text())
    if manifest['root']!=str(HERE):raise RuntimeError('Freeze ROOT mismatch')
    bad=[p for p,h in manifest['files'].items() if not Path(p).is_file() or sha256(p)!=h]
    if bad:raise RuntimeError(f'Frozen input drift, no automatic reanchoring: {bad[:8]}')
    prereg=json.loads((HERE/'PREREG.json').read_text())
    if prereg['root']!=str(HERE) or prereg['conditions']!=[dict(name='left_yaw_p060',speed_scale=1.,yaw_rate=.06)]:
        raise RuntimeError('Only preregistered single LEFT allowed')
    return manifest,prereg


def write_unique_audit(obj):
    """New candidate root only; exclusive output creation, no old VERIFY writes."""
    import time
    import os
    suffix=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+str(os.getpid())+'_'+str(time.time_ns())
    out=HERE/('AUDIT_WIDER_RADIUS_'+suffix+'.json')
    with out.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
    return out
