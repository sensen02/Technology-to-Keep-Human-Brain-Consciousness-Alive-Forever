"""Feedforward bounded correction on the frozen standing and walking prototype."""
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / "20261116T000536Z"
TURNS = BASE.parent / "20261116T004235Z"
sys.path[:0] = [str(BASE), str(TURNS)]
from walking_bench import INPUTS, read, stand
from prototype import BANK, CHECKPOINT, policy as baseline_policy


class Residual(nn.Module):
    def __init__(self, mean, scale, bound):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("bound", torch.tensor(bound, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(INPUTS, 96), nn.Tanh(), nn.Linear(96, 96), nn.Tanh(),
                                 nn.Linear(96, 42))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, observation):
        normalized = ((observation - self.mean) / self.scale).clamp(-8., 8.)
        return self.bound * self.net(normalized).tanh()


def base_sources():
    manifest = read(BASE / "PROTOTYPES.json")
    assert manifest["archive_sha256"] == stand.sha(BANK)
    assert manifest["stop_sha256"] == stand.sha(CHECKPOINT)
    assert manifest["code_sha256"] == stand.sha(BASE / "prototype.py")
    return {str(p): stand.sha(p) for p in (BANK, CHECKPOINT, BASE / "prototype.py", BASE / "PROTOTYPES.json",
                                           BASE / "walking_bench.py", BASE / "fit_walk.py")}


def wrap(model):
    base = baseline_policy()
    model.eval()

    @torch.inference_mode()
    def apply(observation, intent):
        command = base(observation, intent)
        if not intent:
            return command
        x = torch.as_tensor(observation, dtype=torch.float32)
        return np.clip(command + model(x).numpy(), -1., 1.)

    return apply


def load(checkpoint):
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert state["baseline_sources"] == base_sources()
    params = state["model"]
    model = Residual(params["mean"], params["scale"], float(params["bound"]))
    model.load_state_dict(params)
    return wrap(model)
