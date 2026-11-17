"""Warm-start on physical single-box prefixes and a successful flat control."""
import argparse

import numpy as np
import torch
from torch.nn import functional as F

from course import HERE, sha
from model import SACBrain
from sac_replay import read_episode

UPDATES=520


def encoded(model,path):
    row,pin=read_episode(path)
    with torch.inference_mode():
        x=torch.from_numpy(row["x"])[None]
        eye=torch.from_numpy(row["vision"])[None]
        _,memory=model.brain.state(torch.from_numpy(row["prefix"]).reshape(1,-1,7))
        state,_=model.sequence(x,eye,memory)
    use=row["demo"]>0.
    assert use.any(), path
    return state[0,:-1][use],torch.from_numpy(row["mode"][use]),torch.from_numpy(row["pace"][use]).float(),pin


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base",type=type(HERE))
    cli.add_argument("--output",type=type(HERE),default=HERE/"WARM_BOX.pt")
    cli.add_argument("episodes",type=type(HERE),nargs="+")
    args=cli.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(29486)
    state=torch.load(args.base,map_location="cpu",weights_only=True)
    agent=SACBrain()
    agent.load_state_dict(state["model"])
    data=[encoded(agent,path) for path in args.episodes]
    x,m,v=(torch.cat([row[i] for row in data]) for i in range(3))
    group=np.concatenate([np.full(len(row[0]),i) for i,row in enumerate(data)])
    startup=np.concatenate([1.+5.*(np.arange(len(row[0]))<30) for row in data])
    weights=torch.tensor(startup/np.bincount(group)[group],dtype=torch.float64)
    # Soften the endpoint labels to keep a useful continuous exploration scale.
    v=v.clamp(.12,.88)
    opt=torch.optim.AdamW(agent.actor.parameters(),lr=.0012,weight_decay=.0004)
    for i in range(UPDATES):
        batch=torch.multinomial(weights,192,replacement=True)
        logp,mu,ls=agent.distribution(x[batch])
        forward=m[batch]==1
        assert forward.any()
        degree=(mu.tanh()+1.)*.5
        loss=F.nll_loss(logp,m[batch])+F.smooth_l1_loss(degree[forward],v[batch][forward],beta=.1)
        loss+=.10*(ls+.95).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.inference_mode():
        logp,mu,ls=agent.distribution(x)
        counts=torch.bincount(m,minlength=4).tolist()
        acc=float((logp.argmax(-1)==m).float().mean())
    destination=args.output
    torch.save(dict(model=agent.state_dict(),bc_updates=UPDATES,optimizer=opt.state_dict(),
                    source=[row[-1] for row in data],parent=dict(path=str(args.base.resolve()),sha256=sha(args.base)),
                    frames=len(x),mode_counts=counts,accuracy=acc),destination.with_suffix(".tmp"))
    destination.with_suffix(".tmp").rename(destination)
    print(dict(updates=UPDATES,training_frames=len(x),mode_counts=counts,accuracy=acc,sha256=sha(destination)),flush=True)


if __name__=="__main__":
    main()
