"""Check deployment with learned moving-state outputs, not only initial standing."""
from pathlib import Path
import argparse

import numpy as np
import torch

from context import HERE,OLD_WEIGHT,first,phase,hash_file
from train import read_policy
from tasks import TRAIN


def main(checkpoint):
    model=read_policy(checkpoint).eval()
    deployed=phase.read_policy(checkpoint)
    old=phase.read_policy(OLD_WEIGHT)
    x,states=[],[]
    for task in TRAIN:
        archive=HERE/'TEACHER'/task.name/'TRACE.npz'
        with np.load(archive,allow_pickle=False) as record:
            move=np.flatnonzero(np.abs(record['x'][:,:2]).max(1)>.20)
            if move.size:
                rows=move[np.linspace(0,len(move)-1,8,dtype=int)]
                x.extend(record['x'][rows])
                states.extend(record['phase'][rows])
    local=np.asarray(x)
    states=np.asarray(states)
    assert len(local)>200
    with torch.no_grad():
        reference=model(torch.from_numpy(local),torch.from_numpy(states)).numpy()
    actual=np.array([deployed.act(a,b,True) for a,b in zip(local,states,strict=True)])
    prior=np.array([old.act(a,b,True) for a,b in zip(local,states,strict=True)])
    maximum=float(np.max(np.abs(reference-actual)))
    delta=float(np.sqrt(np.mean((prior-actual)**2)))
    assert maximum<1e-6 and delta>1e-7,(maximum,delta)
    first.write(HERE/'MOVING_SAVE_RELOAD.json',dict(checkpoint=str(checkpoint.resolve()),
          weight_sha256=hash_file(checkpoint),moving_samples=len(local),max_reload_error_rad=maximum,
          old_to_new_output_rms_rad=delta))
    print(len(local),'moving samples','reload error',maximum,'changed action RMS',delta,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path)
    a=p.parse_args()
    main(a.checkpoint)
