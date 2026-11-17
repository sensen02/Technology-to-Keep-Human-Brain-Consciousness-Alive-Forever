"""Current-observation motor correction for a missed stance contact."""
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
STUMBLE = HERE.parents[1] / "20261116T060654Z" / "student"
sys.path.insert(0, str(STUMBLE))
from stumble_policy import LEG_ORDER, base_sources, checksum, prototype, read, write

FORMAT = "CURRENT_STATE_GAP_1"
WALK_INPUTS = 213


def features(body, local, indices):
    body = np.asarray(body)
    local = np.asarray(local)
    if body.shape[-1] != WALK_INPUTS or body.shape[:-1] != local.shape[:-1]:
        raise ValueError("Incompatible current-state observations")
    return np.concatenate((body[..., indices], local), axis=-1)


def body_indices(legs):
    dofs = np.concatenate([np.arange(7 * LEG_ORDER.index(leg), 7 * (LEG_ORDER.index(leg) + 1)) for leg in legs])
    return np.r_[dofs, dofs + 42, dofs + 84, dofs + 126, np.arange(168, 213)]


class Correction(nn.Module):
    def __init__(self, mean, scale, bounds):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("bounds", torch.as_tensor(bounds, dtype=torch.float32))
        self.encoder = nn.Sequential(nn.Linear(len(self.mean), 80), nn.Tanh(), nn.Linear(80, 64), nn.Tanh())
        self.detect = nn.Linear(64, 1)
        self.motor = nn.Linear(64, len(self.bounds))
        nn.init.zeros_(self.motor.weight)
        nn.init.zeros_(self.motor.bias)

    def forward(self, observation):
        normalized = ((observation - self.mean) / self.scale).clamp(-12., 12.)
        signal = self.encoder(normalized)
        return self.detect(signal).squeeze(-1), self.motor(signal).tanh() * self.bounds


class Policy:
    """Apply a bounded residual selected from measured current state."""
    def __init__(self, model, channels, indices, sensor_fields, threshold):
        self.nominal = prototype.policy()
        self.model = model.eval()
        self.channels = np.asarray(channels, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.sensor_fields = tuple(sensor_fields)
        self.threshold = float(threshold)

    @torch.inference_mode()
    def correction(self, body, sensors, intent):
        if not intent:
            return None
        if np.shape(sensors) != (len(self.sensor_fields),):
            raise ValueError("Incompatible local sensor identity")
        observation = features(body, sensors, self.indices)
        logit, residual = self.model(torch.as_tensor(observation, dtype=torch.float32))
        if float(logit) <= self.threshold:
            return None
        return residual.numpy()

    def apply(self, nominal, body, sensors, intent, passive=None):
        residual = self.correction(body, sensors, intent)
        if residual is None:
            return nominal if passive is None else passive
        result = nominal.copy()
        result[self.channels] = np.clip(result[self.channels] + residual, -1., 1.)
        return result

    def __call__(self, body, intent, local):
        return self.apply(self.nominal(body, intent), body, local, intent)


def load(checkpoint):
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if saved["format"] != FORMAT or saved["base_sources"] != base_sources():
        raise ValueError("Frozen policy identity differs from training")
    state = saved["model"]
    net = Correction(state["mean"], state["scale"], state["bounds"])
    net.load_state_dict(state)
    return Policy(net, saved["channels"], saved["body_indices"], saved["sensor_fields"], saved["threshold"])
