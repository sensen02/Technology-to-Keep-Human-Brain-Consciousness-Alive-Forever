"""Parent-only Torch backend: artificial brain, body adapters and explicit two delayed ports.
Import only after an executable has checked --execute-parent-calibration.
"""
from typing import List,Tuple
import torch
from torch import nn
RAW_INDICES=list(range(0,42))+list(range(84,126))+list(range(168,195))
class NeuralBridge(nn.Module):
 def __init__(self,mean,std):
  super().__init__();self.register_buffer('observation_mean',mean.clone());self.register_buffer('observation_std',std.clone());self.register_buffer('allowed_raw_indices',torch.tensor(RAW_INDICES,dtype=torch.int64))
  self.feedback=nn.Sequential(nn.Linear(111,32,dtype=torch.float64),nn.Tanh(),nn.Linear(32,1,dtype=torch.float64),nn.Tanh())
  self.brain=nn.GRUCell(2,16,dtype=torch.float64);self.brain_out=nn.Linear(16,2,dtype=torch.float64)
  self.decoder=nn.Sequential(nn.Linear(2,16,dtype=torch.float64),nn.Tanh(),nn.Linear(16,1,dtype=torch.float64),nn.Tanh())
  self.reconstruction=nn.Sequential(nn.Linear(1,32,dtype=torch.float64),nn.Tanh(),nn.Linear(32,111,dtype=torch.float64))
 def normalize(self,body111):return (body111-self.observation_mean)/self.observation_std
 def brain_step(self,delivered_an,cue,h):
  after=self.brain(torch.cat((delivered_an,cue),dim=-1),h);return self.brain_out(after).tanh(),after
 def identity(self):return dict(format='ENGINEERING_ARTIFICIAL_BRAIN_REAL_ID_NECK_PORT_BRIDGE_V1',brain_hidden=16,brain_update='DISCRETE_GRU_5MS_NOT_OLD_SLOW_RELAXATION',brain_input=['deliveredAN1','experimental_instruction1'],DN=2,AN=1,decoder=[2,16,1],feedback=[111,32,1],reconstruction_training_only=[1,32,111],allowed_raw_indices=RAW_INDICES,neural_dt_s=.001,neck_delay_s=.002,decision_s=.005,decision_delay_steps=1,full_neck_model=False,native_DN_function=False,native_AN_tuning=False,body_physics_loaded=False)
class Scan(nn.Module):
 def __init__(self,m):
  super().__init__();self.feedback=m.feedback;self.brain=m.brain;self.brain_out=m.brain_out;self.decoder=m.decoder;self.reconstruction=m.reconstruction;self.register_buffer('observation_mean',m.observation_mean);self.register_buffer('observation_std',m.observation_std)
 def forward(self,body111:torch.Tensor,cue:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
  normalized=(body111-self.observation_mean)/self.observation_std;an=self.feedback(normalized);recon=self.reconstruction(an);h=body111.new_zeros((body111.shape[1],16));delivered_an=body111.new_zeros((body111.shape[1],1));delivered_dn=body111.new_zeros((body111.shape[1],2));turns=torch.jit.annotate(List[torch.Tensor],[]);dns=torch.jit.annotate(List[torch.Tensor],[]);states=torch.jit.annotate(List[torch.Tensor],[])
  for k in range(1920):
   h=self.brain(torch.cat((delivered_an,cue[k]),-1),h);dn=self.brain_out(h).tanh();turn=self.decoder(delivered_dn);turns.append(turn);dns.append(dn);states.append(h);delivered_an=an[k];delivered_dn=dn
  return torch.stack(turns),torch.stack(dns),an,recon,torch.stack(states)
def make_scan(m):
 s=torch.jit.script(Scan(m));a=dict(m.named_parameters());b=dict(s.named_parameters());assert a.keys()==b.keys() and all(a[k] is b[k] for k in a);return s
def python_rollout(m,body111,cue):
 h=body111.new_zeros((body111.shape[1],16));old_an=body111.new_zeros((body111.shape[1],1));old_dn=body111.new_zeros((body111.shape[1],2));rows=[[],[],[],[],[]]
 for k in range(1920):
  an=m.feedback(m.normalize(body111[k]));dn,h=m.brain_step(old_an,cue[k],h);turn=m.decoder(old_dn);recon=m.reconstruction(an)
  for row,value in zip(rows,(turn,dn,an,recon,h)):row.append(value)
  old_an=an;old_dn=dn
 return tuple(torch.stack(r) for r in rows)
def objective(out,teacher,normalized):
 turn,dn,an,recon,states=out;action=(turn-teacher).square().mean();feedback=(recon-normalized).square().mean();activity=dn.square().mean();return action+.1*feedback+.001*activity,dict(turn_MSE=action,feedback_MSE=feedback,DN_activity_MSE=activity)
