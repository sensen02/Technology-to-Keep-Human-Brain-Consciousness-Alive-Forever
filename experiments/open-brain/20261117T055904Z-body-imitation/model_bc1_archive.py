"""Feed-forward local motor student and saved-checkpoint adapter."""
import numpy as np
import torch
from torch import nn


class BodyPolicy(nn.Module):
    def __init__(self, n_input, width=192):
        super().__init__()
        self.register_buffer('center', torch.zeros(n_input))
        self.register_buffer('scale', torch.ones(n_input))
        self.register_buffer('target_scale', torch.ones(42))
        self.layers = nn.Sequential(nn.Linear(n_input, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                    nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 42))

    def forward(self, observation):
        x = ((observation - self.center) / self.scale).clamp(-10, 10)
        return self.layers(x) * self.target_scale


def load_policy(path):
    ckpt = torch.load(path, weights_only=True, map_location='cpu')
    policy = BodyPolicy(ckpt['n_input'], ckpt['width'])
    policy.load_state_dict(ckpt['state'])
    policy.eval()
    return policy


@torch.inference_mode()
def action(policy, obs):
    return policy(torch.from_numpy(obs)).numpy().astype(np.float64)
