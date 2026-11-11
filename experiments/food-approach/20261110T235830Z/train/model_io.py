"""Public immutable-lineage slow ResidualMotor storage and final1500 loader."""
from pathlib import Path
import importlib.util
import hashlib
import sys
import torch
import numpy as np
HERE=Path(__file__).resolve().parent
OLD=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/residual_candidate')
PARENT=OLD/'TRAIN_RESIDUAL_500/checkpoint_000500.pt'
PARENT_SHA='040b26b7ff0e48329c6f6f8f3461f5ced783f108db3d48c264cdf78c5e35e7e3'
spec=importlib.util.spec_from_file_location('_controlled_slow_residual_old',OLD/'residual_motor.py')
legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
ResidualMotor=legacy.ResidualMotor
BASE=legacy.BASE
REFERENCE=BASE.parent/'20261110T152754Z/NATIVE_ROLE_2500/neural_decisions.npz'

def sha256(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def verify_hashes(hashes):
 for p,h in hashes.items():
  if not Path(p).is_absolute() or sha256(p)!=h:raise ValueError('artifact drift '+p)

def exact(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return type(a)==type(b) and a==b

def load_parent():
 if sha256(PARENT)!=PARENT_SHA:raise ValueError('parent checkpoint changed')
 model,payload=legacy.load_candidate(PARENT)
 verify_hashes(payload['config']['source_hashes']);verify_hashes(payload['config']['data_hashes'])
 if payload['config']['curve'][-1]['step']!=500 or len(payload['config']['curve'])!=501:raise ValueError('parent curve requires500')
 return model,payload

def save(path,model,opt,config,curve,rng):
 path=Path(path).resolve()
 if not path.is_relative_to(HERE) or path.exists():raise ValueError('fresh scoped path required')
 torch.save(dict(format_version=1,identity=model.identity(),state_dict=model.state_dict(),optimizer=opt.state_dict(),config=config,curve=curve,rng=rng),path)
 return sha256(path)

def load_checked(checkpoint,sha,require_final=True):
 if sha256(checkpoint)!=sha:raise ValueError('checkpoint digest mismatch')
 p=torch.load(checkpoint,map_location='cpu',weights_only=False)
 if p.get('format_version')!=1:raise ValueError('new controlled continuation format required')
 verify_hashes(p['config']['source_hashes']);verify_hashes(p['config']['data_hashes'])
 model=ResidualMotor(legacy.load_baseline_actor())
 if p['identity']!=model.identity():raise ValueError('original architecture identity changed')
 model.load_state_dict(p['state_dict'],strict=True)
 if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('nonfinite weights')
 baseline=legacy.load_baseline_actor()
 if not exact(model.base.state_dict(),baseline.state_dict()):raise ValueError('frozen base changed')
 if p['config']['physical_eval_only_step']!=1500 or p['curve'][-1]['step'] not in (1000,1500):raise ValueError('fixed budget mismatch')
 if require_final and p['curve'][-1]['step']!=1500:raise ValueError('physical evaluation only1500')
 return model.eval(),p
