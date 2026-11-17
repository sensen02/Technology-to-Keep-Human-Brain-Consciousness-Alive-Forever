"""Observation-indexed, command-conditioned imitation policy for the frozen body."""
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn

from course import CENTER_SPEED
from physical import BASE, BANK, CHECKPOINT, HERE, bench, sha, stock_policy

YAW_SCALE = .115
COMMAND_SCALE = np.array([.075/CENTER_SPEED, .12/YAW_SCALE])
PARAMETERS = "pace_local_v1"
RESIDUAL_BOUND = .10


def feature_rows(observation, command):
    q = np.asarray(observation)
    cmd = np.asarray(command)
    return np.concatenate((q[...,126:168], q[...,:42]*np.sqrt(.001), cmd*COMMAND_SCALE),axis=-1)


def base_hashes():
    return {str(p):sha(p) for p in [BASE/"prototype.py",BANK,CHECKPOINT,BASE/"PROTOTYPES.json",
                                  BASE/"walking_bench.py",HERE/"course.py",Path(__file__)]}


class Residual(nn.Module):
    def __init__(self, center, scale):
        super().__init__()
        self.register_buffer("center", torch.as_tensor(center,dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale,dtype=torch.float32))
        self.layers=nn.Sequential(nn.Linear(215,96),nn.Tanh(),nn.Linear(96,96),nn.Tanh(),nn.Linear(96,42))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, value):
        return RESIDUAL_BOUND * self.layers(((value-self.center)/self.scale).clamp(-6.,6.)).tanh()


class LocalBank:
    def __init__(self, keys, commands, actions, half, anchors, labels):
        self.keys=keys
        self.commands=commands
        self.actions=actions
        self.half=half
        self.anchors=anchors
        self.labels=labels
        self.index=cKDTree(keys)
        self.base=stock_policy()

    def indices(self, vector, *, exclude=None):
        if exclude is None:
            distance,index=self.index.query(vector,k=1)
            return distance,index
        distance,index=self.index.query(vector,k=2)
        use_second=index[:,0]==exclude
        i=np.arange(len(index))
        return distance[i,use_second.astype(int)],index[i,use_second.astype(int)]

    def reference(self, x, command):
        feature=feature_rows(x, command)
        distance,index=self.indices(feature)
        return index, distance

    def predict_rows(self, rows, index):
        feedback=(rows[...,:42]-self.keys[index,...,:42])*self.half*45.
        return np.clip(self.actions[index]-feedback,-1.,1.)

    def action(self, observation, command, residual=None):
        x=np.asarray(observation)
        intent=np.asarray(command)
        if x.shape!=(bench.INPUTS,) or intent.shape!=(2,) or not np.isfinite(np.r_[x,intent]).all():
            raise ValueError("Invalid current observation or engineering intent")
        if intent[0] == 0.:
            if intent[1] != 0.:
                raise ValueError("Moving policy expects zero yaw on a stop command")
            return self.base(x,0)
        keys=feature_rows(x,intent)
        distance, idx=self.indices(keys)
        # Keep exact stock gait when its own measured state is the chosen anchor.
        if self.anchors[idx] and abs(intent[0]-CENTER_SPEED)<1e-8 and intent[1]==0.:
            return self.base(x,1)
        action=self.predict_rows(keys,idx)
        if residual is not None:
            gate=float(np.clip(distance/.02,0.,1.))
            z=np.r_[x,intent/[CENTER_SPEED,YAW_SCALE]]
            with torch.inference_mode():
                action=np.clip(action+gate*residual(torch.as_tensor(z,dtype=torch.float32)).numpy(),-1.,1.)
        return action


def read_bank(path):
    model_path=Path(path).resolve()
    if model_path.suffix==".pt":
        saved=torch.load(model_path,weights_only=True,map_location="cpu")
        folder=model_path.parent
        library=folder/"LIBRARY.npz"
        if sha(library)!=saved["library_sha256"] or saved["base"]!=base_hashes():
            raise RuntimeError("Policy archive does not match checkpoint provenance")
        values=saved["state"]
        residual=Residual(values["center"],values["scale"])
        residual.load_state_dict(values)
        residual.eval()
    else:
        library=model_path
        residual=None
    info=bench.read(library.parent/"LIBRARY.json")
    if (info["format"]!=PARAMETERS or sha(library)!=info["archive_sha256"] or info["base"]!=base_hashes()):
        raise ValueError("Local imitation library mismatches its format, archive hash or fixed policy")
    with np.load(library,allow_pickle=False) as z:
        bank=LocalBank(*(z[k] for k in ("keys","commands","actions","half","anchors","labels")))
    return bank,residual


def load_policy(checkpoint):
    bank,residual=read_bank(checkpoint)
    return lambda x, command: bank.action(x,command,residual)
