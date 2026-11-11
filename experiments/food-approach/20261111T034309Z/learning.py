"""Shared-parameter scripted sensory BPTT and public behavior loader."""
import torch
from torch import nn
from typing import List,Tuple
from pathlib import Path
from common import SteeringBehavior,digest,verify_hashes,guard

class Scan(nn.Module):
 def __init__(self,model):
  super().__init__();self.encoder=model.encoder;self.gru=model.gru;self.head=model.head;self.alpha=float(model.ALPHA)
 def forward(self,x:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
  encoded=torch.tanh(self.encoder(x));state=x.new_zeros((x.shape[1],32));states=torch.jit.annotate(List[torch.Tensor],[])
  for k in range(x.shape[0]):
   candidate=self.gru(encoded[k],state);state=(1.-self.alpha)*state+self.alpha*candidate;states.append(state)
  history=torch.stack(states);return torch.tanh(self.head(history)).squeeze(-1),history

def make_scan(model):
 scan=torch.jit.script(Scan(model))
 for name,module in [('encoder',model.encoder),('gru',model.gru),('head',model.head)]:
  for key,p in module.named_parameters():
   if dict(getattr(scan,name).named_parameters())[key] is not p:raise ValueError('script copied parameter')
 return scan

def production(model,x):
 state=model.initial_state(x.shape[1]);out=[];history=[]
 for row in x:
  command,state=model.step(row,state);out.append(command[:,1]);history.append(state)
 return torch.stack(out),torch.stack(history)

def exact(a,b):
 import numpy as np
 if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return type(a)==type(b) and a==b

def load_behavior(path,sha):
 guard();path=Path(path).resolve()
 if digest(path)!=sha:raise ValueError('behavior digest mismatch')
 p=torch.load(path,map_location='cpu',weights_only=False);m=SteeringBehavior()
 if p.get('format_version')!=1 or p.get('kind')!='food_steering_behavior' or p.get('scope_tag')!='SENSOR_PRETRAIN_ONLY' or p.get('deployment_qualified') is not False:raise ValueError('sensor-only behavior schema mismatch')
 if p['identity']!=m.identity():raise ValueError('frozen behavior identity mismatch')
 verify_hashes(p['config']['source_hashes']);verify_hashes(p['config']['data_hashes'])
 m.load_state_dict(p['state_dict'],strict=True)
 if not all(torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('nonfinite weights')
 guard();return m.eval(),p
