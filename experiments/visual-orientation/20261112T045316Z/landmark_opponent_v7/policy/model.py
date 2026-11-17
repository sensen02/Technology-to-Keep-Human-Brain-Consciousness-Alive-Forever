"""RGB-only engineering fixed1x1neuralopponent receptor then learnedCNNGRU; no steeringrules."""
import math
from typing import List,Tuple
import torch
from torch import nn
class VisualPolicy(nn.Module):
 def __init__(self):
  super().__init__();self.register_buffer('opponent_weight',torch.tensor([1.,-.5,-.5],dtype=torch.float64).reshape(1,3,1,1));self.register_buffer('opponent_bias',torch.zeros(1,dtype=torch.float64));self.conv1=nn.Conv2d(1,8,3,stride=2,padding=1,dtype=torch.float64);self.conv2=nn.Conv2d(8,8,3,stride=2,padding=1,dtype=torch.float64);self.eye_head=nn.Linear(512,8,dtype=torch.float64);self.adapter=nn.Linear(16,32,dtype=torch.float64);self.gru=nn.GRUCell(32,32,dtype=torch.float64);self.head=nn.Linear(32,1,dtype=torch.float64);self.alpha=-math.expm1(-.005/.1)
 def initial_state(self,batch):return self.head.weight.new_zeros((batch,32))
 def encode(self,rgb):
  f,b=rgb.shape[:2];x=rgb.permute(0,1,2,5,3,4).reshape(f*b*2,3,32,32).to(dtype=torch.float64)/255.;x=torch.relu(torch.nn.functional.conv2d(x,self.opponent_weight,self.opponent_bias));x=torch.tanh(self.conv1(x));x=torch.tanh(self.conv2(x));x=torch.tanh(self.eye_head(x.reshape(f*b*2,512)));return x.reshape(f,b,16)
 def step(self,features,state):
  h=(1.-self.alpha)*state+self.alpha*self.gru(torch.tanh(self.adapter(features)),state);latent=self.head(h)[:,0];return latent.tanh(),latent,h
 def identity(self):return dict(format='VISION_FIXED_NEURAL_OPPONENT_ARCH_V7',input='actualRGB2x32x32x3only',eye_order=['LEFT','RIGHT'],fixed_receptor='Conv1x1RGBweights[1,-.5,-.5]bias0thenReLU',receptor_trainable=False,biological_retina_validity=False,shared_eye_CNN=[1,8,8],eye_feature=8,ordered_feature=16,hidden=32,alpha=self.alpha,camera_capture_s=.02,release_delay_s=.02,decision_s=.005,output='tanh(turnlatent)',normalization='uint8/255 then fixedneuralopponent',runtime_deadzone=False,hidden_reset_at_stimulus=False,olfaction_connected=False)
class Scan(nn.Module):
 def __init__(self,m):
  super().__init__();self.register_buffer('opponent_weight',m.opponent_weight);self.register_buffer('opponent_bias',m.opponent_bias);self.conv1=m.conv1;self.conv2=m.conv2;self.eye_head=m.eye_head;self.adapter=m.adapter;self.gru=m.gru;self.head=m.head;self.alpha=float(m.alpha)
 def forward(self,rgb:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
  f=rgb.shape[0];b=rgb.shape[1];x=rgb.permute(0,1,2,5,3,4).reshape(f*b*2,3,32,32).to(dtype=torch.float64)/255.;x=torch.relu(torch.nn.functional.conv2d(x,self.opponent_weight,self.opponent_bias));x=torch.tanh(self.conv1(x));x=torch.tanh(self.conv2(x));features=torch.tanh(self.eye_head(x.reshape(f*b*2,512))).reshape(f,b,16);h=features.new_zeros((b,32));history=torch.jit.annotate(List[torch.Tensor],[])
  for k in range(1920):
   h=(1.-self.alpha)*h+self.alpha*self.gru(torch.tanh(self.adapter(features[k//4])),h);history.append(h)
  states=torch.stack(history);latent=self.head(states).squeeze(-1);return latent.tanh(),latent,states
def make(m):
 scan=torch.jit.script(Scan(m))
 for name,p in m.named_parameters():
  if dict(scan.named_parameters())[name] is not p:raise ValueError('sharedall14parameters')
 return scan
def production(m,rgb):
 features=torch.cat([m.encode(rgb[i:i+1]) for i in range(len(rgb))],0);h=m.initial_state(rgb.shape[1]);turn=[];lat=[];states=[]
 for k in range(1920):
  t,l,h=m.step(features[k//4],h);turn.append(t);lat.append(l);states.append(h)
 return torch.stack(turn),torch.stack(lat),torch.stack(states)
def objective(turn,latent,target,weights,episode_scale):
 return (episode_scale*(((turn-target).square()+.01*(latent-torch.atanh(target.clamp(-.95,.95))).square())*weights).sum(0)).mean()
