"""Static-probe identity/default helpers; Python stdlib only."""
from pathlib import Path
import hashlib
import json
import os
import time

HERE=Path(__file__).resolve().parent
EXPECTED_ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_tarsal_clearance_probe')
FREEZE=HERE/'SOURCEHASH_STATIC_PROBE.json'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def verify_sources():
    if HERE!=EXPECTED_ROOT:raise RuntimeError('Static probe ROOT mismatch')
    m=json.loads(FREEZE.read_text())
    if m['root']!=str(HERE):raise RuntimeError('Freeze ROOT mismatch')
    bad=[p for p,h in m['files'].items() if not Path(p).is_file() or sha256(p)!=h]
    if bad:raise RuntimeError(f'Frozen input drift, no automatic reanchoring: {bad[:8]}')
    p=json.loads((HERE/'PREREG.json').read_text())
    if p['root']!=str(HERE):raise RuntimeError('Prereg ROOT mismatch')
    return m,p


def exclusive_result(prefix,obj):
    suffix=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+str(os.getpid())+'_'+str(time.time_ns())
    out=HERE/(prefix+'_'+suffix+'.json')
    with out.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
    return out


def cpu1():
    allowed=os.sched_getaffinity(0);os.sched_setaffinity(0,{1 if 1 in allowed else min(allowed)})
