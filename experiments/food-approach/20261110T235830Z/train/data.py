"""Four authenticated trajectories, failed-state labels remain explicitly failed observations."""
import json
from pathlib import Path
import numpy as np
import torch
from model_io import HERE,OLD,REFERENCE,PARENT,sha256,verify_hashes
CORRECTION=OLD.parent.parent/'20261110T203537Z/recovery_data'

def prepare(model):
 hashes={str(PARENT):sha256(PARENT),str(REFERENCE):sha256(REFERENCE)};episodes=[]
 for name in ('left_full','right_full','corrective_left','corrective_right'):
  corrective=name.startswith('corrective')
  folder=CORRECTION/('COLLECT_'+name.split('_')[-1]) if corrective else OLD.parent/'teacher'/name
  rp=folder/'RESULT.json';sp=folder/('SAMPLES_5MS.npz');r=json.loads(rp.read_text())
  if corrective:
   ap=folder/'VERIFIED_SAMPLES.json';lp=folder/'LABEL_REPLAY.json'
   a=json.loads(ap.read_text());l=json.loads(lp.read_text())
   if not a['integrity_all_pass'] or a['dataset_safety_admitted'] is not False or r['dataset_safety_admitted'] is not False:raise ValueError('corrective failure tag or integrity changed')
   if a['samples_sha256']!=sha256(sp) or a['result_sha256']!=sha256(rp) or a['trace_sha256']!=sha256(folder/'trace_1ms.npz'):raise ValueError('corrective artifact mismatch')
   if not l['all_teacher_labels_bitexact'] or l['max_label_error']!=0 or not l['root_balance_residuals_bitexact'] or l['label_rows']!=1920 or l['root_balance_max']>.02 or l['root_balance_above_point02_count']!=0:raise ValueError('label validity not established')
   verify_hashes(l['source_hashes'])
   summary=r['collection']
   if not r['complete'] or r['early_stop'] is not None or summary['nonfoot_force_max_native']!=0 or summary['foot_support_mean']<.9 or summary['com_height_min']<1.2 or summary['slip_ratio_max']<=.5:raise ValueError('corrective evidence scope mismatch')
   for p in (ap,lp,folder/'IDENTITY.json',folder/'trace_1ms.npz'):hashes[str(p)]=sha256(p)
   hashes.update(l['source_hashes']);ykey='teacher_action';ukey='command2';length=1920
  else:
   if not r['all_gates'] or not r['strict_nonfoot_zero'] or not r['cycles']['all_legs_completed'] or not r['requested_sign_agreement']:raise ValueError('qualified teacher failed')
   if r['samples']['sha256']!=sha256(sp):raise ValueError('original teacher samples changed')
   ykey='Y42';ukey='command2';length=2880
  for p in (rp,sp):hashes[str(p)]=sha256(p)
  with np.load(sp,allow_pickle=False) as z:
   x=torch.tensor(z['X238'].copy(),dtype=torch.float64);u=torch.tensor(z[ukey].copy(),dtype=torch.float64);y=torch.tensor(z[ykey].copy(),dtype=torch.float64)
  if x.shape!=(length,238) or u.shape!=(length,2) or y.shape!=(length,42) or not torch.equal(u[:960],torch.zeros_like(u[:960])) or not torch.equal(u[:,0],torch.zeros(length,dtype=torch.float64)):raise ValueError('actual sequence or turn-only command scope')
  with torch.no_grad():
   state=model.base.initial_state(1);means=[];hidden=[]
   for row in x:
    mean,state=model.base.step(row[None],state);means.append(mean[0]);hidden.append(state[0])
   means=torch.stack(means);hidden=torch.stack(hidden)
   features=torch.cat((x[:,model.base.physical_mask.bool()],hidden,u),-1);gate=torch.tanh(u.square().sum(-1,keepdim=True))
  episodes.append(dict(name=name,x=x,u=u,y=y,means=means,hidden=hidden,features=features,gate=gate,length=length,corrective=corrective))
 return episodes,hashes

def padded(episodes):
 t=max(e['length'] for e in episodes);b=len(episodes)
 f=torch.zeros((t,b,242),dtype=torch.float64);g=torch.zeros((t,b,1),dtype=torch.float64);m=torch.zeros((t,b,42),dtype=torch.float64);y=m.clone();valid=torch.zeros((t,b,1),dtype=torch.bool);weight=torch.zeros((t,b,1),dtype=torch.float64)
 for j,e in enumerate(episodes):
  n=e['length'];f[:n,j]=e['features'];g[:n,j]=e['gate'];m[:n,j]=e['means'];y[:n,j]=e['y'];valid[:n,j]=True
  weight[960:n,j]=1./((n-960)*42*b)
 return dict(features=f,gate=g,means=m,targets=y,valid=valid,weights=weight)

def sequential(model,e):
 rh=model.initial_state(1)[1];out=[];states=[]
 for f,g,m in zip(e['features'],e['gate'],e['means']):
  candidate=model.gru(torch.tanh(model.encoder(f[None])),rh)
  rh=(1-model.ALPHA)*rh+model.ALPHA*candidate
  out.append(m+g*model.RESIDUAL_BOUND*torch.tanh(model.head(rh))[0]);states.append(rh[0])
 return torch.stack(out),torch.stack(states)

def batch(model,p):
 rh=model.initial_state(p['features'].shape[1])[1];out=[];states=[]
 for f,g,m,valid in zip(p['features'],p['gate'],p['means'],p['valid']):
  candidate=model.gru(torch.tanh(model.encoder(f)),rh)
  candidate=(1-model.ALPHA)*rh+model.ALPHA*candidate
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
