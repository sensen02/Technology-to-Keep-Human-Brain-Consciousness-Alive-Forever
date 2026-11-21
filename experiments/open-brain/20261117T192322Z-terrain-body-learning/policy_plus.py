"""Same stateless drive/readouts, with explicit current foot proprioception."""
from policy import Policy as OriginalPolicy
from shared import *
from torch import nn


class Policy(OriginalPolicy):
    def __init__(self):
        super().__init__();self.reflex[0]=nn.Linear(296,192)
        self.center=torch.zeros(296);self.scale=torch.ones(296)

    def forward(self,x,base):
        raw=self.reflex(self.inputs(x,base))
        gate=(x[...,224:230].abs().amax(-1,keepdim=True)/.05).clamp(0,1)
        drive=(raw*self.lambda_scale*gate).clamp(0,80)
        target=base+drive.to(base.dtype)@self.synergy.T
        return torch.maximum(torch.minimum(target,self.maximum),self.minimum),raw,gate


def migrate(state):
    model=Policy();target=model.state_dict()
    for name,value in state.items():
        if name not in ('reflex.0.weight','center','scale'):target[name]=value.clone()
    target['reflex.0.weight'].zero_();target['reflex.0.weight'][:,:230]=state['reflex.0.weight'][:,:230]
    target['reflex.0.weight'][:,254:]=state['reflex.0.weight'][:,230:]
    target['center'][:230]=state['center'][:230];target['center'][254:]=state['center'][230:]
    target['scale'][:230]=state['scale'][:230];target['scale'][254:]=state['scale'][230:]
    model.load_state_dict(target);return model


def read(path):
    value=torch.load(path,map_location='cpu',weights_only=True);model=Policy();model.load_state_dict(value['state']);model.eval();return model
