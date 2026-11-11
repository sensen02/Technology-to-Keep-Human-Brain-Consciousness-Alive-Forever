"""Frozen head2000 parent plus bounded nonlinear positive-turn ANN."""
from pathlib import Path
import importlib.util,sys,hashlib
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD=ROOT.parent/'20261111T010218Z/train'
PARENT=OLD/'TRAIN/checkpoint_002000.pt'
PARENT_SHA='eeced49c5bdd50a7e5bce2cfd63d56d9929b3bf2d2cb00c1985019d91321e6df'
spec=importlib.util.spec_from_file_location('_nonlinear_frozen_readout_io',OLD/'model_io.py');old_io=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old_io;spec.loader.exec_module(old_io)
BASE=old_io.BASE
REFERENCE=old_io.REFERENCE
sha256=old_io.sha256
verify_hashes=old_io.verify_hashes
exact=old_io.exact

def load_parent():return old_io.load_checked(PARENT,PARENT_SHA,require_final=True)

class NonlinearMotor(nn.Module):
 VERSION='frozen_positive_readout_parent_nonlinear306_64_42_positive_bound010_v1'
 def __init__(self,parent):
  super().__init__();self.parent=parent;self.parent.requires_grad_(False)
  self.encoder=nn.Linear(306,64,dtype=torch.float64);self.output=nn.Linear(64,42,dtype=torch.float64)
  nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 @property
 def base(self):return self.parent.base
 def initial_state(self,batch_size):return self.parent.initial_state(batch_size)
 def adapter_parameters(self):return list(self.encoder.parameters())+list(self.output.parameters())
 def features(self,obs,state,command):return torch.cat((obs[...,self.base.physical_mask.bool()],state[0],state[1],command),dim=-1)
 def correction(self,features,command):return torch.relu(command[...,1:2])*.10*torch.tanh(self.output(torch.tanh(self.encoder(features))))
 def step(self,obs,command,state):
  if obs.ndim!=2 or obs.shape[-1]!=238 or command.shape!=(len(obs),2):raise ValueError('actualobs238/command2 required')
  if not torch.isfinite(command).all() or (command[:,0]!=0).any() or (command[:,1].abs()>1).any():raise ValueError('forward0 turn[-1,1] required')
  mean,state=self.parent.step(obs,command,state)
  return mean+self.correction(self.features(obs,state,command),command),state
 def sequence(self,obs,command,state=None):
  state=self.initial_state(obs.shape[1]) if state is None else state
  means=[];bh=[];rh=[]
  for x,u in zip(obs,command):
   mean,state=self.step(x,u,state);means.append(mean);bh.append(state[0]);rh.append(state[1])
  return torch.stack(means),(torch.stack(bh),torch.stack(rh))
 def identity(self):return dict(version=self.VERSION,parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,parent_identity=self.parent.identity(),frozen_parent_parameters=197608,new_parameters=sum(p.numel() for p in self.adapter_parameters()),features='112permittedphysical+currentbase128+currentres64+command2',state_dims=[128,64],latent_bound=.10,gate='relu(actual_neural_turn)',forward_domain=[0.])

def save(path,model,opt,config,curve,rng):
 path=Path(path).resolve()
 if not path.is_relative_to(HERE) or path.exists():raise ValueError('fresh train-scoped output required')
 torch.save(dict(format_version=1,identity=model.identity(),state_dict=model.state_dict(),optimizer=opt.state_dict(),config=config,curve=curve,rng=rng),path)
 return sha256(path)

def load_checked(checkpoint,sha,require_final=True):
 if sha256(checkpoint)!=sha:raise ValueError('checkpoint digest mismatch')
 p=torch.load(checkpoint,map_location='cpu',weights_only=False)
 if p.get('format_version')!=1 or p['config']['kind']!='fixed_nonlinear_positive_readout5000':raise ValueError('format mismatch')
 verify_hashes(p['config']['source_hashes']);verify_hashes(p['config']['data_hashes'])
 parent,_=load_parent();model=NonlinearMotor(parent)
 if p['identity']!=model.identity():raise ValueError('architecture identity mismatch')
 model.load_state_dict(p['state_dict'],strict=True)
 frozen,_=load_parent()
 if not exact(model.parent.state_dict(),frozen.state_dict()):raise ValueError('frozen197608parent changed')
 if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('nonfinite saved weights')
 step=p['curve'][-1]['step']
 if p['config']['physical_eval_only_step']!=5000 or step not in (2500,5000):raise ValueError('fixed budget mismatch')
 if require_final and step!=5000:raise ValueError('only final5000 physical')
 return model.eval(),p
