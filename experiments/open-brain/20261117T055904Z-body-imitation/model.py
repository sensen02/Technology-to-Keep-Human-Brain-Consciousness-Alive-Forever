"""Feed-forward local motor student and saved-checkpoint adapter."""
import numpy as np
import torch
from torch import nn


class BodyPolicy(nn.Module):
    def __init__(self, n_input, width=192, envelope=False, no_control=False):
        super().__init__()
        self.envelope = envelope
        self.no_control = no_control
        self.register_buffer('center', torch.zeros(n_input))
        self.register_buffer('scale', torch.ones(n_input))
        self.register_buffer('target_scale', torch.ones(42))
        if envelope:
            self.register_buffer('target_bound', torch.full((42,), .4))
        self.layers = nn.Sequential(nn.Linear(n_input, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                    nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 42))

    def forward(self, observation):
        x = ((observation - self.center) / self.scale).clamp(-10, 10)
        if self.no_control:
            x = torch.cat((x[..., :86], torch.zeros_like(x[..., 86:128]), x[..., 128:]), dim=-1)
        result = self.layers(x) * self.target_scale
        if self.envelope:
            # Only a measured VNC drive can permit active deviations from the
            # known mechanical neutral. The network predicts the gait itself.
            readiness = (observation[..., :2].abs().amax(-1, keepdim=True) / .30).clamp(0,1)
            result = result.clamp(-self.target_bound, self.target_bound) * readiness
        return result


def load_policy(path):
    ckpt = torch.load(path, weights_only=True, map_location='cpu')
    policy = BodyPolicy(ckpt['n_input'], ckpt['width'], ckpt.get('envelope', False), ckpt.get('no_control',False))
    policy.load_state_dict(ckpt['state'])
    policy.eval()
    return policy


@torch.inference_mode()
def action(policy, obs):
    return policy(torch.from_numpy(obs)).numpy().astype(np.float64)
