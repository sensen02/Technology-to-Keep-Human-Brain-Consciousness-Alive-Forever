"""Originaltwo-only actual teacher data. No corrective file reads or labels."""
import json
import numpy as np
import torch
from model_io import OLD,REFERENCE,PARENT,sha256

def prepare(model):
 hashes={str(PARENT):sha256(PARENT),str(REFERENCE):sha256(REFERENCE)};episodes=[]
 for name in ('left_full','right_full'):
  folder=OLD.parent/'teacher'/name;rp=folder/'RESULT.json';sp=folder/'SAMPLES_5MS.npz'
  r=json.loads(rp.read_text())
  if not r['all_gates'] or not r['strict_nonfoot_zero'] or not r['cycles']['all_legs_completed'] or not r['requested_sign_agreement']:raise ValueError('qualified teacher failed '+name)
  if r['samples']['sha256']!=sha256(sp):raise ValueError('actual teacher samples changed')
  for p in (rp,sp):hashes[str(p)]=sha256(p)
  with np.load(sp,allow_pickle=False) as z:
   x=torch.tensor(z['X238'].copy(),dtype=torch.float64);u=torch.tensor(z['command2'].copy(),dtype=torch.float64);y=torch.tensor(z['Y42'].copy(),dtype=torch.float64)
  if x.shape!=(2880,238) or u.shape!=(2880,2) or y.shape!=(2880,42) or not torch.equal(u[:960],torch.zeros_like(u[:960])) or not torch.equal(u[:,0],torch.zeros(2880,dtype=torch.float64)):raise ValueError('actualfullsequence/turnscope incorrect')
  with torch.no_grad():
   state=model.base.initial_state(1);means=[];hidden=[]
   for row in x:
    mean,state=model.base.step(row[None],state);means.append(mean[0]);hidden.append(state[0])
   means=torch.stack(means);hidden=torch.stack(hidden)
   features=torch.cat((x[:,model.base.physical_mask.bool()],hidden,u),-1);gate=torch.tanh(u.square().sum(-1,keepdim=True))
  episodes.append(dict(name=name,x=x,u=u,y=y,means=means,hidden=hidden,features=features,gate=gate,length=2880))
 return episodes,hashes

def padded(episodes):
 if [e['name'] for e in episodes]!=['left_full','right_full']:raise ValueError('exactlyoriginaltwo required')
 t,b=2880,2
 f=torch.stack([e['features'] for e in episodes],1);g=torch.stack([e['gate'] for e in episodes],1)
 m=torch.stack([e['means'] for e in episodes],1);y=torch.stack([e['y'] for e in episodes],1)
 valid=torch.ones((t,b,1),dtype=torch.bool);weight=torch.zeros((t,b,1),dtype=torch.float64)
 weight[960:]=1./(1920*42*b)
 return dict(features=f,gate=g,means=m,targets=y,valid=valid,weights=weight)

def sequential(model,e):
 rh=model.initial_state(1)[1];out=[];states=[]
 for f,g,m in zip(e['features'],e['gate'],e['means']):
  candidate=model.gru(torch.tanh(model.encoder(f[None])),rh)
  rh=(1-model.ALPHA)*rh+model.ALPHA*candidate
  out.append(m+g*model.RESIDUAL_BOUND*torch.tanh(model.head(rh))[0]);states.append(rh[0])
 return torch.stack(out),torch.stack(states)

def batch(model,p):
 rh=model.initial_state(2)[1];out=[];states=[]
 for f,g,m,valid in zip(p['features'],p['gate'],p['means'],p['valid']):
  candidate=model.gru(torch.tanh(model.encoder(f)),rh);candidate=(1-model.ALPHA)*rh+model.ALPHA*candidate
  rh=torch.where(valid,candidate,rh)
  out.append(m+g*model.RESIDUAL_BOUND*torch.tanh(model.head(rh)));states.append(rh)
 return torch.stack(out),torch.stack(states)

def neutral(model):
 with np.load(REFERENCE,allow_pickle=False) as z,torch.no_grad():
  x=torch.tensor(z['X238'].copy(),dtype=torch.float64)[:,None]
  mean,h=model.sequence(x,x.new_zeros((2880,1,2)))
  checks=dict(mean=torch.equal(mean[:,0],torch.tensor(z['mean'])),action=torch.equal(mean[:,0].tanh(),torch.tensor(z['policy_action'])),base_hidden=torch.equal(h[0][:,0],torch.tensor(z['hidden_after'])))
 if not all(checks.values()):raise ValueError('neutral invariance lost')
 return checks
