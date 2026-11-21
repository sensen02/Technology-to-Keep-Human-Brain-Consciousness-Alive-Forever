"""Five-millisecond externally conditioned body transition and two front-touch spike trains."""
import torch
from torch import nn

class BodyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion = nn.Sequential(nn.Linear(63*2+12+17*4,160),nn.SiLU(),nn.Linear(160,128),
                                    nn.SiLU(),nn.Linear(128,63))
        # The two forelegs share local touch/kinematic response parameters.
        self.feedback = nn.Sequential(nn.Linear(12+6+4+1,48),nn.SiLU(),nn.Linear(48,32),
                                      nn.SiLU(),nn.Linear(32,5))
        nn.init.zeros_(self.motion[-1].weight)
        nn.init.zeros_(self.motion[-1].bias)

    def forward(self, prev, now, touch, dn):
        """prev/now [B,21,3] mm; touch [B,2,6] µN; dn [B,4,17] past-bin counts."""
        ahead = now[:,1,:2]-now[:,0,:2]
        c,s = nn.functional.normalize(ahead,dim=1).unbind(1)
        zero=torch.zeros_like(c)
        rot=torch.stack((c,-s,zero,s,c,zero,zero,zero,torch.ones_like(c)),dim=-1).reshape(-1,3,3)
        rel=(now-now[:,:1])@rot
        height=now[:,0,2]
        vel=(now-prev)@rot
        pose=rel.clone()
        pose[:,0,2]=height
        inputs=torch.cat((pose.flatten(1), (vel/.05).flatten(1),
                          touch.flatten(1)/6,dn.flatten(1)),dim=1)
        delta=self.motion(inputs).reshape(-1,21,3)*.05
        next_pos=now+delta@rot.transpose(1,2)
        logits=[]
        for leg in (0,3):
            foot=3+3*leg
            local=torch.cat((rel[:,foot+1:foot+3].flatten(1),vel[:,foot+1:foot+3].flatten(1)/.05),1)
            # Reflect right limb into a common local frame.
            mirror=local.new_tensor([1,-1 if leg else 1,1]*4)
            ipsi=touch[:,:,leg:leg+3].reshape(-1,6)/6
            x=torch.cat((local*mirror,ipsi,dn.sum(dim=2)/4,height[:,None]),1)
            logits.append(self.feedback(x))
        return next_pos,torch.stack(logits,dim=-1)


class FeedbackSampler:
    """Convert 1 ms marginal probabilities to events with a 3-bin minimum spacing."""
    def __init__(self,lanes):
        self.last = torch.zeros((2,lanes,2),dtype=torch.bool)
        self.marginal = torch.zeros((2,lanes,2))

    def sample(self,prob,uniform):
        events=[]
        for p,u in zip(prob.unbind(1),uniform.unbind(1),strict=True):
            ready=(1-self.marginal.sum(0)).clamp(0,1)
            marginal=torch.minimum(p,ready)
            fired=(~self.last.any(0)) & (u<marginal/ready.clamp_min(1e-8))
            self.last=torch.stack((self.last[-1],fired))
            self.marginal=torch.stack((self.marginal[-1],marginal))
            events.append(fired)
        return torch.stack(events,1)
