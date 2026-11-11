"""Own scoped storage; original ResidualMotor equations and legacy identity unchanged."""
from pathlib import Path
import importlib.util,hashlib,sys
import torch
import numpy as np
HERE=Path(__file__).resolve().parent
OLD=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/residual_candidate')
PARENT=OLD/'TRAIN_RESIDUAL_500/checkpoint_000500.pt'
PARENT_SHA='040b26b7ff0e48329c6f6f8f3461f5ced783f108db3d48c264cdf78c5e35e7e3'
spec=importlib.util.spec_from_file_location('_two_only_control_legacy_residual',OLD/'residual_motor.py')
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
 model,p=legacy.load_candidate(PARENT)
 verify_hashes(p['config']['source_hashes']);verify_hashes(p['config']['data_hashes'])
 if len(p['config']['curve'])!=501 or p['config']['curve'][-1]['step']!=500:raise ValueError('requires actualparent500')
 return model,p

def save(path,model,opt,config,curve,rng):
 path=Path(path).resolve()
 if not path.is_relative_to(HERE) or path.exists():raise ValueError('fresh control-scoped output required')
 torch.save(dict(format_version=1,identity=model.identity(),state_dict=model.state_dict(),optimizer=opt.state_dict(),config=config,curve=curve,rng=rng),path)
 return sha256(path)

def load_checked(checkpoint,sha,require_final=True):
 if sha256(checkpoint)!=sha:raise ValueError('checkpoint digest mismatch')
 p=torch.load(checkpoint,map_location='cpu',weights_only=False)
 if p.get('format_version')!=1 or p['config'].get('kind')!='original_two_only_budget_matched_control':raise ValueError('not newtwo-onlycontrol format')
 verify_hashes(p['config']['source_hashes']);verify_hashes(p['config']['data_hashes'])
 model=ResidualMotor(legacy.load_baseline_actor())
 if p['identity']!=model.identity():raise ValueError('original architecture identity changed')
 model.load_state_dict(p['state_dict'],strict=True)
 if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('nonfinite weights')
 if not exact(model.base.state_dict(),legacy.load_baseline_actor().state_dict()):raise ValueError('frozen base changed')
 if p['config']['physical_eval_only_step']!=1500 or p['curve'][-1]['step'] not in (1000,1500):raise ValueError('fixed budget mismatch')
 if p['config']['episode_names']!=['left_full','right_full'] or p['config']['loss_episode_count']!=2:raise ValueError('loss includes unregistered data')
 if require_final and p['curve'][-1]['step']!=1500:raise ValueError('physical evaluation only final1500')
 return model.eval(),p
