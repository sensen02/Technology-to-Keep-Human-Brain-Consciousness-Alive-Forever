"""Trainable endpoints separated by a fixed, source-qualified DNa02 projection."""
import json

import numpy as np
import torch
from torch import nn

from official_graph import BINDING, CONTRACT, ROOTS, digest


PROFILES = ("clean", "attenuation", "crosstalk", "noise", "combined", "stress")


def graph():
    data = json.loads(CONTRACT.read_text())
    assert tuple(data["roots"]) == ROOTS and data["materialization"] == 888 and data["synapse_release"] == "v2"
    assert next(x["sha256"] for x in data["sources"] if x["path"] == str(BINDING)) == digest(BINDING).hexdigest()
    targets = [row["root_888"] for row in data["targets"]]
    matrix = np.zeros((2, len(targets)), dtype=np.float32)
    column = {value: i for i, value in enumerate(targets)}
    for edge in data["connections"]:
        key = ROOTS.index(edge["pre"]), column[edge["post"]]
        assert not matrix[key] and edge["synapses"] > 0
        matrix[key] = edge["synapses"]
    assert np.linalg.matrix_rank(matrix) == 2 and np.all(matrix.max(axis=0) > 0)
    return torch.from_numpy(matrix / matrix.max(axis=0, keepdims=True))


def measurement(activity, projection, profile, rng):
    """Engineering model of recording distortion after signal delivery."""
    if profile not in PROFILES:
        raise ValueError(profile)
    shape = activity.shape
    clean = activity @ projection
    if profile == "clean":
        return clean
    strong = profile == "stress"
    loss = .45 if strong else .25
    mix = .33 if strong else .18
    noise = .18 if strong else .075
    measured = clean
    if profile in ("attenuation", "combined", "stress"):
        gains = 1. - loss + torch.rand(shape, generator=rng) * (loss + .13)
        measured = (activity * gains) @ projection
    if profile in ("crosstalk", "combined", "stress"):
        bleed = torch.rand(shape, generator=rng) * mix
        measured = measured + (activity.flip(-1) * bleed) @ projection
    if profile in ("noise", "combined", "stress"):
        common = torch.randn((*shape[:-1], 1), generator=rng) * noise * .38
        measured = measured + common + torch.randn(clean.shape, generator=rng) * noise
    return measured


class Codec(nn.Module):
    def __init__(self, n_targets):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 12), nn.Tanh(), nn.Linear(12, 2), nn.Sigmoid())
        self.decoder = nn.Sequential(nn.Linear(n_targets, 24), nn.Tanh(), nn.Linear(24, 3))

    def encode(self, labels):
        return self.encoder(torch.nn.functional.one_hot(labels, 3).float())

    def decode(self, measured):
        return self.decoder(measured)


class RuntimeCodec:
    def __init__(self, checkpoint, profile, seed):
        self.projection = graph()
        record = torch.load(checkpoint, weights_only=True, map_location="cpu")
        if record["graph_sha256"] != digest(CONTRACT).hexdigest():
            raise ValueError("Codec trained on another connection graph")
        self.model = Codec(self.projection.shape[1]).eval()
        self.model.load_state_dict(record["weights"])
        self.rng = torch.Generator().manual_seed(seed)
        self.profile = profile
        self.packet_index = 0

    @torch.inference_mode()
    def send(self, yaw):
        return self.model.encode(torch.tensor(yaw + 1)).tolist()

    @torch.inference_mode()
    def read(self, delivered):
        self.packet_index += 1
        drop_rate = {"combined": .10, "stress": .24}.get(self.profile, 0.)
        if drop_rate and torch.rand((), generator=self.rng).item() < drop_rate:
            return None, None
        measured = measurement(torch.tensor(delivered, dtype=torch.float32), self.projection, self.profile, self.rng)
        return int(self.model.decode(measured).argmax().item()) - 1, measured.tolist()
