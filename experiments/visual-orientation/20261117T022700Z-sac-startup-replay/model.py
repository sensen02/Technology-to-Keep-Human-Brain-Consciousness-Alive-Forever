"""One source-connected artificial brain with a hybrid SAC readout."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from course import VIS_DIM, CHECKPOINT, new_brain
from unified import NavigationBrain

MODES=4
HIDDEN=32
STATE=HIDDEN+VIS_DIM


class SACBrain(nn.Module):
    def __init__(self,initialized=False):
        super().__init__()
        self.brain=new_brain() if initialized else NavigationBrain()
        self.actor=nn.Sequential(nn.Linear(STATE,96),nn.SiLU(),nn.Linear(96,MODES+2))
        self.intent=nn.Linear(5,HIDDEN,bias=False)
        nn.init.zeros_(self.intent.weight)
        normal=torch.load(CHECKPOINT,map_location="cpu",weights_only=True)
        self.register_buffer("body_mean",normal["mean"].float())
        self.register_buffer("body_scale",normal["scale"].float())

    def sequence(self,x,vision,memory=None):
        batch,steps=x.shape[:2]
        encoded,memory=self.brain.state(x.reshape(batch,steps*4,7),memory)
        return torch.cat((encoded[:,3::4],vision),-1),memory

    def distribution(self,features):
        pred=self.actor(features)
        logits=pred[...,:MODES]
        return logits.log_softmax(-1),pred[...,4],pred[...,5].clamp(-4.5,.6)

    def pace_sample(self,mean,log_std):
        std=log_std.exp()
        gaussian=mean+std*torch.randn_like(mean)
        pace=(gaussian.tanh()+1.)*.5
        log_normal=-.5*((gaussian-mean)/std).square()-log_std-.5*math.log(2.*math.pi)
        tanh_log_jacobian=2.*(math.log(2.)-gaussian-F.softplus(-2.*gaussian))
        return pace,log_normal-tanh_log_jacobian+math.log(2.)

    def predict(self,features,mode,pace):
        command=(mode!=0).float()
        action=torch.cat((F.one_hot(mode,MODES).float(),(pace*(mode==1)).unsqueeze(-1)),-1)
        hidden=features[...,:HIDDEN]+self.intent(action)
        return self.brain.predict(hidden,command)

    @torch.inference_mode()
    def choose(self,env,obs,*,deterministic=True):
        last=env.brain.memory
        h=last[0,0] if last is not None else torch.zeros(HIDDEN)
        feature=torch.cat((h,torch.from_numpy(obs[1])))
        logs,mu,log_std=self.distribution(feature)
        if deterministic:
            m=int(logs.argmax().item())
            p=float(((mu.tanh()+1.)*.5).item())
        else:
            m=int(torch.multinomial(logs.exp(),1).item())
            p=float(self.pace_sample(mu,log_std)[0].item())
        return m,p if m==1 else .5


class CriticPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.nets=nn.ModuleList([nn.Sequential(nn.Linear(STATE+MODES+1,128),nn.SiLU(),
                                    nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1)) for _ in range(2)])

    def chosen(self,state,mode,pace):
        code=F.one_hot(mode,MODES).float()
        x=torch.cat((state,code,(pace*(mode==1)).unsqueeze(-1)),-1)
        return tuple(net(x).squeeze(-1) for net in self.nets)

    def all(self,state,pace):
        # Each mode owns one row. Only forward uses the conditional speed.
        shape=state.shape[:-1]
        mode=torch.arange(MODES,device=state.device).expand(*shape,MODES)
        return self.chosen(state.unsqueeze(-2).expand(*shape,MODES,STATE),mode,
                           pace.unsqueeze(-1).expand(*shape,MODES))
