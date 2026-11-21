"""Frozen P1 causal diagnosis; imports the exact prior physical/runtime code."""
from pathlib import Path
import sys,json,hashlib,time,copy
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2];BASE=HERE.parent
PREVIOUS=BASE/'20261118T002600Z-ascending-behavior-comparison'
sys.path.insert(0,str(PREVIOUS))
from live import (Body,feature,observation,advance,phase_state,setup,extra_live,load_actor,
                  BODY_WEIGHT,FixedCircuit,CalibratedReceiver,Ground,GROUND,DT,mujoco)
HERE=Path(__file__).resolve().parent
BIDIR=BASE/'20261117T051249Z-bidirectional'
FIXED=BASE/'20261117T164207Z-fixed-neck'
SEEDS=(10901,11311)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def sources(extra=()):
    paths=set(HERE.glob('*.py'))|{HERE/'PROTOCOL.zh-CN.md'}|set(map(Path,extra))
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and Path(f).is_file() and Path(f).resolve().is_relative_to(ROOT):paths.add(Path(f).resolve())
    assets=read(BASE/'20261117T184709Z-body-terrain-repair/PREPARE_INPUTS.json')['source_sha256']
    verify(assets)
    result={str(p.resolve()):sha(p) for p in paths};result.update(assets);return result


def verify(pins):
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('source changed: '+p)


def load_trace(seed,mode):
    folder=PREVIOUS/'FORMAL'/f'seed{seed}'/mode
    meta=read(folder/'RESULT.json');assert sha(folder/'TRACE.npz')==meta['trace_sha256']
    with np.load(folder/'TRACE.npz',allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


import numpy as np
import torch
