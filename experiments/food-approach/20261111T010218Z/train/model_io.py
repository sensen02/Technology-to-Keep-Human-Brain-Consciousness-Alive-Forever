"""Frozen qualified-right parent plus a zero-initial positive-turn neural readout."""
from pathlib import Path
import importlib.util
import hashlib
import sys
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD=ROOT.parent/'20261110T235830Z/train_two_only_control'
PARENT=OLD/'TRAIN/checkpoint_001500.pt'
PARENT_SHA='c6b395f29676ef933e2f55ca5a92b7c5b55183a2d432226487b39e916b51acdb'
spec=importlib.util.spec_from_file_location('_readout_fixed_parent_io',OLD/'model_io.py')
parent_io=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent_io;spec.loader.exec_module(parent_io)
BASE=parent_io.BASE
REFERENCE=parent_io.REFERENCE


def sha256(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_hashes(values):
    for filename,digest in values.items():
        if not Path(filename).is_absolute() or sha256(filename)!=digest:raise ValueError('artifact drift '+filename)


def exact(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(exact(v,w) for v,w in zip(a,b))
    return type(a)==type(b) and a==b


def load_parent():return parent_io.load_checked(PARENT,PARENT_SHA,require_final=True)


class ReadoutMotor(nn.Module):
    VERSION='frozen_slow_parent192_positive_command_linear64to42_bound025_v1'
    def __init__(self,parent):
        super().__init__();self.parent=parent;self.parent.requires_grad_(False)
        self.readout=nn.Linear(64,42,dtype=torch.float64)
        nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)

    @property
    def base(self):return self.parent.base

    def initial_state(self,batch_size):return self.parent.initial_state(batch_size)

    def adapter_parameters(self):return list(self.readout.parameters())

    def correction(self,residual_hidden,command):
        return torch.relu(command[...,1:2])*.25*torch.tanh(self.readout(residual_hidden))

    def step(self,obs,command,state):
        if obs.ndim!=2 or obs.shape[-1]!=238 or command.shape!=(len(obs),2):raise ValueError('obs238/command2 required')
        if not torch.isfinite(command).all() or torch.any(command[:,0]!=0) or torch.any(command[:,1].abs()>1):raise ValueError('forward0 and turn[-1,1] only')
        mean,state=self.parent.step(obs,command,state)
        return mean+self.correction(state[1],command),state

    def sequence(self,obs,command,state=None):
        state=self.initial_state(obs.shape[1]) if state is None else state
        means=[];base_states=[];residual_states=[]
        for x,u in zip(obs,command):
            mean,state=self.step(x,u,state);means.append(mean);base_states.append(state[0]);residual_states.append(state[1])
        return torch.stack(means),(torch.stack(base_states),torch.stack(residual_states))

    def identity(self):
        return dict(version=self.VERSION,parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,parent_identity=self.parent.identity(),
                    frozen_parent_parameters=sum(p.numel() for p in self.parent.parameters()),readout_parameters=2730,
                    readout='Linear64to42 zero init',gate='relu(actual_neural_turn_command)',latent_bound=.25,state_dims=[128,64],
                    forward_domain=[0.],actor_inputs='actualobs238+neuralcommand2 only',parent_frozen=True)


def save(path,model,opt,config,curve,rng):
    path=Path(path).resolve()
    if not path.is_relative_to(HERE) or path.exists():raise ValueError('fresh train-scoped output required')
    torch.save(dict(format_version=1,identity=model.identity(),state_dict=model.state_dict(),optimizer=opt.state_dict(),config=config,curve=curve,rng=rng),path)
    return sha256(path)


def load_checked(checkpoint,sha,require_final=True):
    if sha256(checkpoint)!=sha:raise ValueError('checkpoint digest mismatch')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload.get('format_version')!=1 or payload['config']['kind']!='fixed_positive_neural_readout2000':raise ValueError('readout format mismatch')
    verify_hashes(payload['config']['source_hashes']);verify_hashes(payload['config']['data_hashes'])
    parent,_=load_parent();model=ReadoutMotor(parent)
    if model.identity()!=payload['identity']:raise ValueError('readout identity mismatch')
    model.load_state_dict(payload['state_dict'],strict=True)
    verified,_=load_parent()
    if not exact(model.parent.state_dict(),verified.state_dict()):raise ValueError('frozen parent changed')
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('nonfinite model')
    step=payload['curve'][-1]['step']
    if payload['config']['physical_eval_only_step']!=2000 or step not in (1000,2000):raise ValueError('fixed budget mismatch')
    if require_final and step!=2000:raise ValueError('physical evaluation only final2000')
    return model.eval(),payload
