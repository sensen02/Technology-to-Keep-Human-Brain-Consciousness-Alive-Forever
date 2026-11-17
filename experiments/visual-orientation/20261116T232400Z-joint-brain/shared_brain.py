"""One recurrent brain for propulsion decisions and action-conditioned predictions."""
import torch
from torch import nn

from walk_pilot import Brain, WalkHead

HORIZONS = (20, 50, 100, 200)
# Six tibia angles, six angular velocities, six loads, forward speed,
# and the upright projection. Support and moving are additional binary targets.
STATE_DIM = 20


class SharedBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = Brain()
        self.visual_head = WalkHead()
        self.state = nn.GRU(7,32,batch_first=True)
        self.action = nn.Linear(32,1)
        self.tactile_adjustment = nn.Linear(2,1,bias=False)
        self.auxiliary = nn.Sequential(nn.Linear(33,40),nn.SiLU())
        self.future_contacts = nn.Linear(40,8)
        self.future_load = nn.Linear(40,8)
        self.future_body = nn.Linear(40,STATE_DIM)
        self.future_motion = nn.Linear(40,7)
        self.vision.requires_grad_(False)
        self.visual_head.requires_grad_(False)

    @torch.no_grad()
    def visual_feature(self, image):
        return self.visual_head(self.vision(image)[0]).clamp(-4.,4.)

    def advance(self, samples, memory=None):
        latent, memory = self.state(samples,memory)
        action = (samples[...,0]+self.action(latent).squeeze(-1)
                  +self.tactile_adjustment(samples[...,1:3]).squeeze(-1))
        return action,latent,memory

    def predict(self, latent, selected):
        # Sensory losses update the shared state and head, not the action
        # selected at this time. A teacher command and actual policy command
        # have the same interface.
        z = self.auxiliary(torch.cat((latent,selected.detach().unsqueeze(-1)),dim=-1))
        shape = (*z.shape[:-1],len(HORIZONS),2)
        return (self.future_contacts(z).reshape(shape), self.future_load(z).reshape(shape),
                self.future_body(z),self.future_motion(z))
