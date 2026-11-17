"""Small binocular network and learned visual steering decisions."""
import numpy as np
import torch
from torch import nn


class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.eye = nn.Sequential(nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(),
                                 nn.Conv2d(6, 6, 3, padding=1), nn.ReLU(),
                                 nn.AvgPool2d((8, 2)))
        self.perception = nn.Sequential(nn.Linear(768, 48), nn.ReLU(), nn.Linear(48, 2))
        self.start = nn.Parameter(torch.tensor(.11))
        self.stop = nn.Parameter(torch.tensor(-.043))

    def forward(self, rgb):
        x = rgb.float() / 255.
        opponent = (x[..., 0] - .5 * (x[..., 1] + x[..., 2])).clamp_min(0.)
        shape = opponent.shape
        features = self.eye(opponent.reshape(shape[0]*2, 1, 32, 32))
        return self.perception(features.reshape(shape[0], -1))

    def action_logits(self, estimate, previous):
        angle, presence = estimate.unbind(-1)
        threshold = torch.where(previous == 0, self.start.clamp(.04, .27), self.stop.clamp(-.12, .10))
        visible = presence.tanh() * .004
        right = (-angle - threshold + visible).masked_fill(previous == 1, -25.)
        left = (angle - threshold + visible).masked_fill(previous == -1, -25.)
        return torch.stack((right, torch.zeros_like(angle), left), dim=-1) * 40.


class VisualPolicy:
    def __init__(self, checkpoint):
        self.model = Brain().eval()
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.previous = 0

    @torch.inference_mode()
    def action(self, image):
        estimate = self.model(torch.from_numpy(np.asarray(image)[None]))
        bearing, visible = estimate[0].tolist()
        logits = self.model.action_logits(estimate, torch.tensor([self.previous]))[0]
        action = int(logits.argmax().item()) - 1 if visible > 0. else 0
        self.previous = action
        return action, float(bearing * 90.), float(visible)
