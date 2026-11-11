"""Strict new-root actual-guard source identity and exclusive reports; stdlib."""
from pathlib import Path
import hashlib
import json
import os
import time

HERE=Path(__file__).resolve().parent
EXPECTED_ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_tarsal_actual_guard')
FREEZE=HERE/'SOURCEHASH_ACTUAL_GUARD.json'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def verify_sources():
    if HERE!=EXPECTED_ROOT:raise RuntimeError('Actual guard ROOT mismatch')
    m=json.loads(FREEZE.read_text())
    if m['root']!=str(HERE):raise RuntimeError('Freeze ROOT mismatch')
    bad=[p for p,h in m['files'].items() if not Path(p).is_file() or sha256(p)!=h]
    if bad:raise RuntimeError(f'Frozen drift, no automatic reanchor: {bad[:8]}')
    p=json.loads((HERE/'PREREG.json').read_text())
    if p['root']!=str(HERE) or p['conditions']!=[dict(name='left_yaw_p060',speed_scale=1.,yaw_rate=.06)] or p['feedback']['guard_native']!=.012:raise RuntimeError('Fixed candidate contract changed')
    return m,p


def exclusive(prefix,obj):
    suffix=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+str(os.getpid())+'_'+str(time.time_ns())
    out=HERE/(prefix+'_'+suffix+'.json')
    with out.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
    return out
