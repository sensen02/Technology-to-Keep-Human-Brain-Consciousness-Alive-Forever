"""Same scripted scan as primary; original parameters shared, batch2 validalltrue."""
from typing import List,Tuple
import torch
from torch import nn

class Scan(nn.Module):
 def __init__(self,model):
  super().__init__();self.encoder=model.encoder;self.gru=model.gru;self.head=model.head
  self.alpha=float(model.ALPHA);self.bound=float(model.RESIDUAL_BOUND)
 def forward(self,features:torch.Tensor,gate:torch.Tensor,means:torch.Tensor,valid:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
  encoded=torch.tanh(self.encoder(features));rh=features.new_zeros((features.shape[1],64));states=torch.jit.annotate(List[torch.Tensor],[])
  for k in range(features.shape[0]):
   candidate=self.gru(encoded[k],rh)
   candidate=(1.-self.alpha)*rh+self.alpha*candidate
   rh=torch.where(valid[k],candidate,rh);states.append(rh)
  history=torch.stack(states)
  return means+gate*self.bound*torch.tanh(self.head(history)),history

def make(model):
 engine=torch.jit.script(Scan(model))
 for name,module in [('encoder',model.encoder),('gru',model.gru),('head',model.head)]:
  target=dict(getattr(engine,name).named_parameters())
  for key,p in module.named_parameters():
   if target[key] is not p:raise ValueError('script engine copied a parameter')
 return engine
