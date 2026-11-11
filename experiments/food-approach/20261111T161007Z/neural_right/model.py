"""Independent RIGHT_ONLY .06 domain; no teacher or physical imports."""
from pathlib import Path
import importlib.util, sys, hashlib
import torch
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
OLD=ROOT/'20261110T191449Z/residual_candidate/residual_motor.py'
TRANSFER=ROOT/'20261110T235830Z/train_two_only_control/TRAIN/checkpoint_001500.pt'
TRANSFER_SHA='c6b395f29676ef933e2f55ca5a92b7c5b55183a2d432226487b39e916b51acdb'
ENGINE=ROOT/'20261110T235830Z/train_two_only_control/engine.py'
DATA=HERE.parent/'attempt02/right_yaw_m060/SAMPLES_5MS.npz'
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
legacy=module('_right_only_legacy',OLD)
sha256=legacy.sha256
BASE=legacy.BASE
REFERENCE=BASE.parent/'20261110T152754Z/NATIVE_ROLE_2500/neural_decisions.npz'
def exact(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return type(a)==type(b) and a==b
def verify_hashes(h):
 for p,d in h.items():
  if not Path(p).is_absolute() or sha256(p)!=d:raise ValueError('identity drift '+p)
class RightMotor(legacy.ResidualMotor):
 VERSION='RIGHT_ONLY_relu_negative_turn_bound3_yaw006_v1'
 RESIDUAL_BOUND=3.
 def step(self,obs,command,state):
  if obs.ndim!=2 or obs.shape[-1]!=238 or command.shape!=(len(obs),2):raise ValueError('obs238 command2 required')
  if len(state)!=2 or state[0].shape!=(len(obs),128) or state[1].shape!=(len(obs),64):raise ValueError('explicit128+64 state required')
  if not torch.isfinite(obs).all() or not torch.isfinite(command).all() or not all(torch.isfinite(s).all() for s in state):raise ValueError('nonfinite input')
  if (command[:,0]!=0).any() or (command[:,1]>0).any() or (command[:,1]<-1).any():raise ValueError('RIGHT_ONLY forward0 turn[-1,0]')
  mean,bh=self.base.step(obs,state[0])
  f=torch.cat((obs[:,self.base.physical_mask.bool()],bh,command),-1)
  candidate=self.gru(torch.tanh(self.encoder(f)),state[1]);rh=(1-self.ALPHA)*state[1]+self.ALPHA*candidate
  correction=torch.relu(-command[:,1:2])*self.RESIDUAL_BOUND*torch.tanh(self.head(rh))
  return mean+correction,(bh,rh)
 def identity(self):
  i=super().identity();i.update(version=self.VERSION,residual_bound=3.,gate='relu(-command[:,1:2])',command_domain='forward0; turn[-1,0]; new yaw/.06, NOT yaw/.008',transfer=str(TRANSFER),transfer_sha256=TRANSFER_SHA,head_initialization='allzero',kind='independent_RIGHT_ONLY');return i

def initialize():
 if sha256(TRANSFER)!=TRANSFER_SHA:raise ValueError('fixed transfer SHA mismatch')
 m=RightMotor(legacy.load_baseline_actor());p=torch.load(TRANSFER,map_location='cpu',weights_only=False)
 verify_hashes(p['config']['source_hashes']);verify_hashes(p['config']['data_hashes'])
 if p.get('format_version')!=1 or p['curve'][-1]['step']!=1500:raise ValueError('transfer must trained1500')
 for name in ('encoder','gru'):
  getattr(m,name).load_state_dict({k[len(name)+1:]:v for k,v in p['state_dict'].items() if k.startswith(name+'.')},strict=True)
 with torch.no_grad():m.head.weight.zero_();m.head.bias.zero_()
 return m

def cached(m,x,u):
 bh=m.base.initial_state(x.shape[1]);means=[];states=[]
 with torch.no_grad():
  for a in x:
   mean,bh=m.base.step(a,bh);means.append(mean);states.append(bh)
 means=torch.stack(means);states=torch.stack(states)
 return dict(features=torch.cat((x[:,:,m.base.physical_mask.bool()],states,u),-1),gate=torch.relu(-u[:,:,1:2]),means=means,valid=torch.ones((*x.shape[:2],1),dtype=torch.bool),base_states=states)
def make(m):return module('_right_only_engine',ENGINE).make(m)
def dataset():
 with np.load(DATA,allow_pickle=False) as z:
  x=torch.from_numpy(z['X238'].copy())[:,None];u=torch.from_numpy(z['command2'].copy())[:,None];y=torch.from_numpy(z['Y42'].copy())[:,None]
 if x.shape!=(2880,1,238) or u.shape!=(2880,1,2) or y.shape!=(2880,1,42):raise ValueError('full trajectory dimensions')
 return x,u,y

def load_checked(path,digest,require_final=True):
 if sha256(path)!=digest:raise ValueError('checkpoint digest mismatch')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p.get('format_version')!='RIGHT_ONLY_V1' or p['config']['physical_eval_only_step']!=2000:raise ValueError('new strict RIGHT_ONLY format required')
 verify_hashes(p['config']['source_hashes']);verify_hashes(p['config']['data_hashes'])
 m=RightMotor(legacy.load_baseline_actor())
 if p['identity']!=m.identity():raise ValueError('new RIGHT_ONLY identity mismatch')
 m.load_state_dict(p['state_dict'],strict=True)
 if not all(torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('nonfinite checkpoint')
 if not exact(m.base.state_dict(),legacy.load_baseline_actor().state_dict()):raise ValueError('frozen base mismatch')
 if p['curve'][-1]['step'] not in (1000,2000) or (require_final and p['curve'][-1]['step']!=2000):raise ValueError('invalid checkpoint step')
 opt=torch.optim.Adam(m.adapter_parameters(),lr=.001);opt.load_state_dict(p['optimizer'])
 if len(opt.state)!=8 or any(float(v['step'])!=p['curve'][-1]['step'] for v in opt.state.values()):raise ValueError('Adam counters')
 return m.eval(),p
