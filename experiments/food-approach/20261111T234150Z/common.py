"""Behavior-only long-history experiment; physical imports forbidden."""
from pathlib import Path
import importlib.util,sys,json,hashlib,os
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;EXP=HERE.parent
PARENT=EXP/'20261111T035953Z/TRAIN/checkpoint_003000.pt'
PARENT_SHA='bfd65a3896d259496114695d7f7f3550272cf4c369895a4ddb34f3ddc3352c06'
GEOMETRY=EXP/'20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json'
MOTOR=EXP/'20261111T161007Z/combined_turning/COMBINED.pt'
MOTOR_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
FORMAT='ODOR_LONG_HISTORY_RECOVERY_V1'
def guard():
 bad=[k for k in sys.modules if k in ('physics','student_wave','flygym','mujoco') or k.startswith('mujoco.')]
 if bad:raise RuntimeError('Behavior-only physical backend forbidden '+str(bad))
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def jsonable(v):
 if isinstance(v,dict):return {str(k):jsonable(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [jsonable(x) for x in v]
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,Path):return str(v)
 return v
def save(p,v,exclusive=True):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x' if exclusive else 'w',encoding='utf8') as f:json.dump(jsonable(v),f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def exact(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return type(a)==type(b) and a==b
def cpu1():
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});torch.set_num_threads(1)
 for var in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
  if os.environ.get(var)!='1':raise ValueError(var+'=1 required')
 guard()
def verify_hashes(h):
 for p,v in h.items():
  if not Path(p).is_absolute() or sha(p)!=v:raise ValueError('identity drift '+p)
def parent_io():
 # Existing loader learning.py imports `common` by basename. Bind old module only while loading.
 name='common';current=sys.modules.get(name);oldpath=EXP/'20261111T034309Z/common.py';old=module('_recovery_original_sensor_common',oldpath);sys.modules[name]=old
 try:loader=module('_recovery_balanced_parent_loader',EXP/'20261111T035953Z/common.py')
 finally:
  if current is None:sys.modules.pop(name,None)
  else:sys.modules[name]=current
 return loader
def initialize():
 guard();io=parent_io();m,p=io.load_behavior(PARENT,PARENT_SHA);m.requires_grad_(True)
 if not exact(m.state_dict(),p['state_dict']):raise ValueError('parent initial weights not exact')
 return m,p
def source_paths():return [HERE/n for n in ('PREREG.json','common.py','data.py','engine.py','generate_data.py','preflight.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py')]+[GEOMETRY,PARENT,EXP/'20261111T035953Z/common.py',EXP/'20261111T034309Z/common.py',EXP/'20261111T034309Z/learning.py',EXP/'20261110T191449Z/steering_behavior.py']
def freeze_proof():
 f=read(HERE/'FROZEN.json');verify_hashes(f['source_hashes']);return f

def load_checked(path,digest,require_final=True):
 guard()
 if sha(path)!=digest:raise ValueError('long-history checkpoint SHA mismatch')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p.get('format_version')!=FORMAT or p.get('kind')!='food_steering_behavior_long_history' or p.get('deployment_qualified') is not False:raise ValueError('new longhistory checkpoint format required')
 verify_hashes(p['config']['source_hashes']);verify_hashes(p['config']['data_hashes'])
 m,parent=initialize()
 if p['identity']!=m.identity() or p['parent_sha256']!=PARENT_SHA:raise ValueError('architecture/parent identity differs')
 m.load_state_dict(p['state_dict'],strict=True);step=p['curve'][-1]['step']
 if step not in (0,500,1000) or (require_final and step!=1000):raise ValueError('physical/validation onlyfinal1000')
 if step and len(p['curve'])!=step+1:raise ValueError('curve count must include actualstep0')
 if not all(torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('nonfinite model')
 opt=torch.optim.Adam(m.parameters(),lr=.0001);opt.load_state_dict(p['optimizer'])
 if step and (len(opt.state)!=8 or any(float(v['step'])!=step for v in opt.state.values())):raise ValueError('ALL8 freshAdam exact counters required')
 return m.eval(),p
