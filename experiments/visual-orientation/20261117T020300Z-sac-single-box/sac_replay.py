"""Episode-bound recurrent replay of qualified physical transitions."""
import json
from pathlib import Path

import numpy as np
import torch

from course import sha

BURN=50
LEARN=16
LENGTH=BURN+LEARN
HORIZONS=(20,50,100,200)


def physical_targets(x,physical):
    """World supervision and arrived tactile data stay on the loss side."""
    n=len(physical)
    i=np.arange(n)
    t=(x[1:,:,1:3]>.008).reshape(n*4,2)
    load=[]
    touch=[]
    masks=[]
    for h in HORIZONS:
        elapsed=h//5
        valid=4*i+elapsed<=len(t)
        future=np.lib.stride_tricks.sliding_window_view(t,elapsed,axis=0)
        where=np.minimum(4*i,len(future)-1)
        touch.append(future[where].any(-1))
        lo=int(h//20)-1
        hi=int(np.ceil(h/20))-1
        idx=np.clip(i+lo,0,n-1)
        far=np.clip(i+hi,0,n-1)
        load.append(((physical[idx,12:18]+physical[far,12:18])*.5)[:,[0,3]]/.45)
        masks.append(valid)
    # Physical state at 50 ms interpolates the two surrounding 20 ms samples.
    future=(physical[np.minimum(i+1,n-1)]+physical[np.minimum(i+2,n-1)])*.5
    motion=np.c_[future[:,12:18]>.02,future[:,18]>.025]
    return dict(aux_touch=np.stack(touch,1).astype(np.float32),aux_load=np.stack(load,1).astype(np.float32),
                aux_valid=np.stack(masks,1).astype(np.float32),aux_body=future.astype(np.float32),
                aux_motion=motion.astype(np.float32))


def read_episode(path: Path):
    receipt=json.loads((path/"RESULT.json").read_text())
    src=path/"EPISODE.npz"
    assert sha(src)==receipt["files"]["EPISODE.npz"]
    assert sha(path/"TRANSPORT.json")==receipt["files"]["TRANSPORT.json"]
    assert receipt["source"]["source_files_unchanged"]
    assert receipt["native_steps"]==receipt["actor_ticks"]*1600
    with np.load(src,allow_pickle=False) as z:
        row={k:z[k] for k in ("x","vision","mode","pace","reward","terminal","truncated","physical")}
    n=len(row["reward"])
    assert n==receipt["actor_ticks"] and len(row["x"])==n+1
    assert n>=LENGTH and not np.any(row["terminal"][:-1] | row["truncated"][:-1])
    assert bool(row["terminal"][-1]) != bool(row["truncated"][-1])
    assert all(np.isfinite(v).all() for v in row.values())
    row.update(physical_targets(row["x"],row.pop("physical")))
    pin=dict(path=str(src),sha256=sha(src),receipt_sha256=sha(path/"RESULT.json"),
             policy=receipt["policy"],case=receipt["case"],steps=n,native_steps=receipt["native_steps"],
             ending=receipt["reason"],truncated=receipt["truncated"])
    return row,pin


class Replay:
    def __init__(self,paths,generator):
        examples=[read_episode(path) for path in paths]
        self.rows=[row for row,_ in examples]
        self.manifest=[pin for _,pin in examples]
        self.rng=generator

    def sample(self,count):
        draws=[]
        for example in self.rng.integers(len(self.rows),size=count):
            row=self.rows[example]
            last=len(row["reward"])-LENGTH
            start=last if self.rng.random()<.30 else int(self.rng.integers(last+1))
            end=start+LENGTH
            values={}
            for k,v in row.items():
                if k in ("x","vision"):
                    values[k]=v[start:end+1]
                else:
                    values[k]=v[start+BURN:end]
            draws.append(values)
        return {k:torch.from_numpy(np.stack([row[k] for row in draws])) for k in draws[0]}
