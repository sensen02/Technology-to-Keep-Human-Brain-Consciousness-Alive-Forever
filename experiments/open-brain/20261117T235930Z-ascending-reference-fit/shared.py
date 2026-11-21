"""P1 engineering reference; old teacher-missing evidence remains immutable."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE.parent
PREP = BASE/'20261117T234120Z-ascending-paired-adapter'
BIDIR = BASE/'20261117T051249Z-bidirectional'
FIXED = BASE/'20261117T164207Z-fixed-neck'
FOUNDATION = BASE/'20261117T045901Z-foundation'
sys.path[:0] = [str(FIXED), str(BIDIR), str(FOUNDATION)]
from circuit import eon
import numpy as np
import torch

DT = .0001
CAP_HZ = 1000.
UPDATES = 250
TRAIN_CASES = ('train_left', 'train_right')
HELD_CASES = ('held_left', 'held_right')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sources(extra=()):
    paths=set(HERE.glob('*.py'))|{HERE/'PROTOCOL.zh-CN.md',HERE/'PUBLIC_SOURCES.json'}|set(map(Path,extra))
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and Path(f).is_file() and Path(f).resolve().is_relative_to(ROOT):
            paths.add(Path(f).resolve())
    return {str(p.resolve()):sha(p) for p in paths}


def verify(pins):
    for path,expected in pins.items():
        if sha(path)!=expected:
            raise ValueError(f'input identity changed: {path}')


def load_data(case):
    folder=HERE/'DATA'/case
    meta=read(folder/'RESULT.json')
    verify(meta['source_sha256'])
    assert sha(folder/'PAIRED.npz')==meta['trace_sha256']
    with np.load(folder/'PAIRED.npz',allow_pickle=False) as a:
        arrays={k:a[k].copy() for k in a.files}
    return arrays,meta
