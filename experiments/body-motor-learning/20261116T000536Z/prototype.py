"""A stateless prototype-imitated gait policy and physically anchored stand branch."""
import argparse
from pathlib import Path

import numpy as np
import torch

from fit_walk import OLD, archived, load_policy
from walking_bench import HERE, rollout, stand

CHECKPOINT = HERE/"FIT_INITIAL/step_1200.pt"
BANK = HERE/"PROTOTYPES.npz"


def build():
    sources={}
    x, _, a = archived(OLD/"EVAL_ROLE_2500",False,sources)
    core = stand.physics.Core(adhesion=False)
    h=np.maximum(np.ptp(core.actuator.target_bounds_rad,axis=1)*.5,.1)
    with BANK.open("xb") as f:
        np.savez_compressed(f,x=x[:,np.r_[126:168,0:42]],a=a,halfspan=h)
    stand.put(HERE/"PROTOTYPES.json",dict(sources=sources, proprio_samples=len(x), archive_sha256=stand.sha(BANK),
                                       code_sha256=stand.sha(Path(__file__)),stop_sha256=stand.sha(CHECKPOINT)))


def policy():
    paths=HERE/"PROTOTYPES.json"
    import json
    manifest=json.loads(paths.read_text())
    assert stand.sha(BANK)==manifest["archive_sha256"]
    with np.load(BANK,allow_pickle=False) as z:
        x,a,h = z["x"],z["a"],z["halfspan"]
    center=x[:,:42]
    positions=x[:,42:]
    stop = load_policy(CHECKPOINT)

    def step(feature,intent):
        if not intent:
            return stop(feature,intent)
        dt=feature[126:168]-center
        dp=feature[:42]-positions
        distance=np.square(dt).sum(1) + .001*np.square(dp).sum(1)
        i=int(distance.argmin())
        return np.clip(a[i] - 45*h*dt[i],-1.,1.)

    return step


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build",action="store_true")
    p.add_argument("--start")
    p.add_argument("--out",type=Path)
    p.add_argument("--walk-ms",type=int,default=9600)
    p.add_argument("--light",action="store_true")
    p.add_argument("--native",action="store_true")
    a=p.parse_args()
    torch.set_num_threads(1)
    if a.build:
        build()
    else:
        rollout(a.out.resolve(),a.start,a.walk_ms,policy(),light=a.light,native=a.native)
