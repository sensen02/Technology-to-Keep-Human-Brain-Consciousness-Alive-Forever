"""Current-state, single-leg stumble correction on the frozen walking policy."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
WALK = HERE.parents[1] / "20261116T000536Z"
sys.path.insert(0, str(WALK))
import prototype

LEG_ORDER = ("LF", "LH", "LM", "RF", "RH", "RM")
FORMAT = "SINGLE_LEG_STUMBLE_1"
WALK_INPUTS = 213


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def base_sources():
    files = (WALK / "prototype.py", WALK / "PROTOTYPES.json", prototype.BANK, prototype.CHECKPOINT,
             WALK / "fit_walk.py", WALK / "walking_bench.py")
    return {str(path): checksum(path) for path in files}


def local_features(x, local, leg):
    x = np.asarray(x)
    local = np.asarray(local)
    if x.shape[-1] != WALK_INPUTS or local.shape[:-1] != x.shape[:-1]:
        raise ValueError("Incompatible current bodily features")
    leg_id = LEG_ORDER.index(leg)
    j = slice(7*leg_id, 7*(leg_id+1))
    return np.concatenate((
        x[..., j],
        x[..., 42+j.start:42+j.stop],
        x[..., 84+j.start:84+j.stop],
        x[..., 126+j.start:126+j.stop],
        x[..., 168:183],
        x[..., 183+3*leg_id:186+3*leg_id],
        x[..., 201+leg_id:202+leg_id],
        x[..., 207+leg_id:208+leg_id],
        local,
    ), axis=-1)


class Correction(nn.Module):
    def __init__(self, mean, scale, bound):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("bound", torch.as_tensor(bound, dtype=torch.float32))
        self.body = nn.Sequential(nn.Linear(len(self.mean), 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.classifier = nn.Linear(64, 1)
        self.regression = nn.Linear(64, 7)
        nn.init.zeros_(self.regression.weight)
        nn.init.zeros_(self.regression.bias)

    def forward(self, observation):
        local = ((observation - self.mean) / self.scale).clamp(-12., 12.)
        shared = self.body(local)
        return self.classifier(shared).squeeze(-1), self.regression(shared).tanh() * self.bound


class Policy:
    """Apply only when the current observations call for correction."""
    def __init__(self, model, leg, threshold, force_floor=None, force_index=None, force_sign=1.,
                 support_isolation=False):
        self.base = prototype.policy()
        self.model = model.eval()
        self.leg = leg
        j = LEG_ORDER.index(leg) * 7
        self.joints = slice(j, j + 7)
        self.threshold = threshold
        self.force_floor = force_floor
        self.force_index = force_index
        self.force_sign = force_sign
        self.support_isolation = support_isolation
        if support_isolation:
            with np.load(prototype.BANK, allow_pickle=False) as z:
                self.halfspan = z["halfspan"][self.joints]

    def support_base(self, x, intent, local):
        current = x.copy()
        targets = slice(126 + self.joints.start, 126 + self.joints.stop)
        current[targets] -= local[12:19] / self.halfspan
        return self.base(current, intent)

    @torch.inference_mode()
    def __call__(self, x, intent, local):
        nominal = self.base(x, intent)
        if not intent:
            return nominal
        if self.force_floor is not None and self.force_sign * local[self.force_index] < self.force_floor:
            return nominal
        obs = local_features(x, local, self.leg)
        logits, correction = self.model(torch.as_tensor(obs, dtype=torch.float32))
        if float(logits) <= self.threshold:
            return nominal
        result = self.support_base(x, intent, local) if self.support_isolation else nominal.copy()
        result[self.joints] = np.clip(nominal[self.joints] + correction.numpy(), -1., 1.)
        return result


def load(checkpoint):
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if saved["format"] != FORMAT or saved["base_sources"] != base_sources():
        raise ValueError("Unexpected checkpoint source identity")
    state = saved["model"]
    model = Correction(state["mean"], state["scale"], state["bound"])
    model.load_state_dict(state)
    return Policy(model, saved["leg"], saved["threshold"],
                  force_floor=saved.get("force_floor"), force_index=saved.get("force_index"),
                  force_sign=saved.get("force_sign", 1.), support_isolation=saved.get("support_isolation", False))
