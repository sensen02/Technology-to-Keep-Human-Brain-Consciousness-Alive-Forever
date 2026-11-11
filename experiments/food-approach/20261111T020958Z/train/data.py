"""Authenticated original left teacher, strictly causal frozen parent features."""
import json
from pathlib import Path
import numpy as np
import torch
from model_io import ROOT,OLD,PARENT,PARENT_SHA,BASE,REFERENCE,sha256,verify_hashes
TEACHER=ROOT.parent/'20261110T191449Z/teacher/left_full'
EVAL=ROOT.parent/'20261111T010218Z/evaluation'
CORRECTION=OLD/'PARENT_METADATA_CORRECTION.json'

def prepare(model):
 parent_payload=torch.load(PARENT,map_location='cpu',weights_only=False)
 hashes=dict(parent_payload['config']['data_hashes']);hashes[str(PARENT)]=PARENT_SHA;hashes[str(CORRECTION)]=sha256(CORRECTION)
 verify_hashes(hashes)
 for condition in ('neutral','left','right'):
  rp=EVAL/f'EVAL_{condition}/RESULT.json';ap=EVAL/f'VERIFY_{condition}.json'
  r=json.loads(rp.read_text());a=json.loads(ap.read_text())
  if r['checkpoint_sha256']!=PARENT_SHA or not a['integrity_all_pass'] or not all(a['checks'].values()):raise ValueError('parent actual audit binding')
  if condition in ('neutral','right') and not r['engineering_pass']:raise ValueError('registered parentfunction missing')
  if condition=='left' and (r['engineering_pass'] or r['autonomous']['gates']['slip']):raise ValueError('parentleft failure changed')
  hashes[str(rp)]=sha256(rp);hashes[str(ap)]=sha256(ap)
 rp=TEACHER/'RESULT.json';sp=TEACHER/'SAMPLES_5MS.npz';r=json.loads(rp.read_text())
 if not r['all_gates'] or not r['cycles']['all_legs_completed'] or not r['strict_nonfoot_zero'] or sha256(sp)!=r['samples']['sha256']:raise ValueError('left teacher not qualified')
 for p in (rp,sp,REFERENCE):hashes[str(p)]=sha256(p)
 with np.load(sp,allow_pickle=False) as z:x=torch.tensor(z['X238'].copy(),dtype=torch.float64);u=torch.tensor(z['command2'].copy(),dtype=torch.float64);y=torch.tensor(z['Y42'].copy(),dtype=torch.float64)
 if x.shape!=(2880,238) or u.shape!=(2880,2) or y.shape!=(2880,42) or not torch.equal(u[:960],torch.zeros_like(u[:960])):raise ValueError('full actual teacher required')
 with torch.no_grad():
  mean,state=model.parent.sequence(x[:,None],u[:,None]);bh=state[0][:,0];rh=state[1][:,0]
  features=model.features(x,(bh,rh),u)
 return dict(x=x,u=u,y=y,mean=mean[:,0],bh=bh,rh=rh,features=features),hashes

def cached(model,e):return e['mean']+model.correction(e['features'],e['u'])

def invariance(model,conditions=('neutral','right')):
 checks={}
 with torch.no_grad():
  for condition in conditions:
   with np.load(EVAL/f'EVAL_{condition}/neural_decisions.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in ('X238','command','mean','policy_action','hidden_after')}
   mean,state=model.sequence(torch.tensor(d['X238'],dtype=torch.float64)[:,None],torch.tensor(d['command'],dtype=torch.float64)[:,None])
   checks[condition+'_mean']=np.array_equal(mean[:,0].numpy(),d['mean']);checks[condition+'_action']=np.array_equal(mean[:,0].tanh().numpy(),d['policy_action']);checks[condition+'_dual_hidden']=np.array_equal(torch.cat(state,dim=-1)[:,0].numpy(),d['hidden_after'])
 return checks
