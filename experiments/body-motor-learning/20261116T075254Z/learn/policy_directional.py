"""Separate signed steering reference branches without retaining motion history."""
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
import torch

from course import CENTER_SPEED
from physical import HERE
from policy import YAW_SCALE,feature_rows,read_bank


def make(checkpoint, use_primitives=False):
    bank,residual=read_bank(checkpoint)
    straight=lambda x,u:bank.action(x,u,residual)
    if use_primitives:
        from turning import load_proto
        capture={side:load_proto("arc_"+name) for side,name in ((-1,"right"),(1,"left"))}
    else:
        capture={}
    wanted={i:np.flatnonzero(np.sign(bank.commands[:,1])==i) for i in (-1,1)}
    if any(not len(v) for v in wanted.values()):
        raise ValueError("Both physically verified steering directions are required")
    indexes={i:cKDTree(bank.keys[v]) for i,v in wanted.items()}

    def action(observation, command):
        v,w=command
        if abs(w)<1e-12 or v<=0.:
            return straight(observation,command)
        sign=1 if w>0 else -1
        row=feature_rows(observation,command)
        distance,i=indexes[sign].query(row)
        idx=wanted[sign][i]
        if capture and .10<=v and .060<=abs(w)<=.14 and distance>.065:
            return capture[sign](observation)
        output=bank.predict_rows(row,idx)
        if residual is not None:
            z=np.r_[observation, np.asarray(command)/[CENTER_SPEED,YAW_SCALE]]
            with torch.inference_mode():
                delta=residual(torch.as_tensor(z,dtype=torch.float32)).numpy()
            output=np.clip(output+min(distance/.02,1.)*delta,-1.,1.)
        return output
    return action


def load_policy(checkpoint):
    return make(checkpoint)
