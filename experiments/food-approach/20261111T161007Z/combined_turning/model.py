"""Frozen composed ANN: one walking base128, independent RIGHT64 and LEFT64.
Only actual sensory112/shared basehidden/neurally split signed command enter branches.
This file must not be imported during CODE_ONLY preparation.
"""
from copy import deepcopy
import math
import torch
from torch import nn
FORMAT='COMBINED_TURN_V1'
STATE_DIMS=(128,64,64)

def equal_state(a,b):
 x=a.state_dict();y=b.state_dict();return x.keys()==y.keys() and all(torch.equal(x[k],y[k]) for k in x)
class FrozenBranch(nn.Module):
 def __init__(self,component):
  super().__init__();self.encoder=deepcopy(component.encoder);self.gru=deepcopy(component.gru);self.head=nn.Linear(64,42,dtype=torch.float64);self.alpha=float(component.ALPHA);self.bound=float(component.RESIDUAL_BOUND)
  with torch.no_grad():self.head.weight.copy_(component.head.weight);self.head.bias.copy_(component.head.bias)
  self.requires_grad_(False)
 def advance(self,physical,basehidden,command,previous):
  features=torch.cat((physical,basehidden,command),dim=-1);candidate=self.gru(torch.tanh(self.encoder(features)),previous);hidden=(1.-self.alpha)*previous+self.alpha*candidate;return torch.tanh(self.head(hidden)),hidden
class CombinedTurning(nn.Module):
 VERSION=FORMAT
 def __init__(self,right,left,component_bindings):
  super().__init__()
  if not equal_state(right.base,left.base):raise ValueError('RIGHT and LEFT must share bitexact validated base')
  if right.base.transform_identity()!=left.base.transform_identity():raise ValueError('base transform identity mismatch')
  if float(right.RESIDUAL_BOUND)!=3. or float(left.RESIDUAL_BOUND)!=6.:raise ValueError('declared RIGHT3/LEFT6 bounds required')
  alpha=-math.expm1(-.005/.1)
  if float(right.ALPHA)!=alpha or float(left.ALPHA)!=alpha:raise ValueError('same declared slow alpha required')
  if int(right.base.physical_mask.bool().sum())!=112:raise ValueError('permitted112 actual sensory channels required')
  self.base=deepcopy(right.base);self.right_branch=FrozenBranch(right);self.left_branch=FrozenBranch(left);self.bindings=deepcopy(component_bindings);self.base_identity=deepcopy(right.base.transform_identity());self.requires_grad_(False)
 def initial_state(self,batch_size):
  return (self.base.initial_state(batch_size),self.right_branch.head.weight.new_zeros((batch_size,64)),self.left_branch.head.weight.new_zeros((batch_size,64)))
 def step(self,obs,command,state):
  if obs.ndim!=2 or obs.shape[-1]!=238 or command.shape!=(len(obs),2):raise ValueError('actual obs(B,238),coupledcommand(B,2) required')
  if not isinstance(state,(tuple,list)) or len(state)!=3 or any(s.shape!=(len(obs),n) for s,n in zip(state,STATE_DIMS)):raise ValueError('explicit sharedbase128/RIGHT64/LEFT64 states required')
  if not torch.isfinite(obs).all() or not torch.isfinite(command).all() or not all(torch.isfinite(s).all() for s in state):raise ValueError('finite actual sensory/command/all256 state required')
  turn=command[:,1:2];positive=torch.relu(turn);negative=torch.relu(-turn)
  if (turn.abs()>1.).any() or ((command[:,0:1]-positive/3.).abs()>1e-12).any():raise ValueError('coupled f=relu(turn)/3, signedturn[-1,1] only')
  # Exactly one shared frozen base operation per decision; both neural branches advance.
  base_mean,basehidden=self.base.step(obs,state[0]);physical=obs[:,self.base.physical_mask.bool()]
  cmd_right=torch.cat((torch.zeros_like(negative),-negative),dim=-1);cmd_left=torch.cat((positive/torch.full_like(positive,3.),positive),dim=-1)
  correction_right,hidden_right=self.right_branch.advance(physical,basehidden,cmd_right,state[1]);correction_left,hidden_left=self.left_branch.advance(physical,basehidden,cmd_left,state[2])
  mean=base_mean+negative*self.right_branch.bound*correction_right+positive*self.left_branch.bound*correction_left
  return mean,(basehidden,hidden_right,hidden_left)
 def sequence(self,obs,command,state=None):
  if obs.ndim!=3 or obs.shape[-1]!=238 or command.shape!=obs.shape[:2]+(2,) or len(obs)==0:raise ValueError('nonempty actual(T,B,238),coupled(T,B,2) required')
  state=self.initial_state(obs.shape[1]) if state is None else state;means=[];base=[];right=[];left=[]
  for x,u in zip(obs,command):
   mean,state=self.step(x,u,state);means.append(mean);base.append(state[0]);right.append(state[1]);left.append(state[2])
  return torch.stack(means),(torch.stack(base),torch.stack(right),torch.stack(left))
 def identity(self):
  return dict(format_version=FORMAT,kind='frozen composition of separately trained neural components',component_bindings=self.bindings,base_identity=self.base_identity,inputs=238,physical_inputs=112,command_dim=2,command_domain='signedturn[-1,1]; forward=relu(turn)/3 tolerance1e-12',internal_commands=dict(RIGHT='[0,-relu(-turn)]',LEFT='[relu(turn)/3,relu(turn)]'),state_dims=list(STATE_DIMS),state_names=['shared_walking_base128','RIGHT_residual64','LEFT_residual64'],state_total=256,RIGHT_alpha=self.right_branch.alpha,LEFT_alpha=self.left_branch.alpha,RIGHT_bound=3.,LEFT_bound=6.,mean='base_mean+relu(-turn)*3*tanh(Rhead(hR))+relu(turn)*6*tanh(Lhead(hL))',action='tanh(mean)',both_branches_advance_every_step=True,base_step_calls_per_decision=1,no_command_or_withdrawal_reset=True,weights_frozen=True,training_performed=False,natural_joint_training=False,teacher_clock_phase_foodcoord_inference=False)
