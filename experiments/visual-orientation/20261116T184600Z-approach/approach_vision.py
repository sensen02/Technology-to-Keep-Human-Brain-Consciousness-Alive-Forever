"""Fit apparent bearing and range of the fixed-size red landmark from binocular images."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
import mujoco as mj

from approach_runtime import HERE, Eyes, prepare, FORWARD, sha

WIDTH=.15
COLOR=(1.,.045,.035,1.)
CHECKPOINT=HERE/"APPROACH_EYES.pt"


def features(rgb):
    v=rgb.float()/255.
    red=(v[...,0]-.5*(v[...,1]+v[...,2])).clamp_min(0.)
    return torch.cat((red.mean(dim=-2).flatten(1), red.mean(dim=-1).flatten(1),
                      red.amax(dim=-2).flatten(1)),dim=1)


def network():
    return nn.Sequential(nn.Linear(192,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU(),nn.Linear(64,3))


def collect():
    directory = HERE/"VISION"
    body,_=prepare(directory,"nominal")
    original=body.d.qpos.copy()
    with np.load(FORWARD/"WALK_HELD_normal/TRACE.npz",allow_pickle=False) as saved:
        poses=np.vstack((original,saved["qpos"][1000:9000:1000]))
    rng=np.random.default_rng(781)
    images, labels=[],[]
    eyes=Eyes(body)
    try:
        for i in range(1600):
            body.d.qpos[:]=poses[i%len(poses)]
            mj.mj_forward(body.plant,body.d)
            a = float(rng.uniform(-42.,42.))
            d = float(rng.uniform(1.65,3.55))
            seen = i%9 != 0
            target=eyes.place(a,d) if seen else None
            bg=(eyes.pose()[0].copy(),rng.uniform(-.8,.8),rng.uniform(.18,.34))
            images.append(eyes.image(target,WIDTH,COLOR,bg))
            labels.append((a/40.,d-2.5,float(seen)))
    finally:
        eyes.close()
    np.savez_compressed(directory/"OPTICAL.npz",images=images,labels=labels)


def fit():
    torch.set_num_threads(1)
    torch.manual_seed(44)
    path=HERE/"VISION/OPTICAL.npz"
    with np.load(path,allow_pickle=False) as source:
        x=features(torch.from_numpy(source["images"]))
        y=torch.tensor(source["labels"],dtype=torch.float32)
    model=network()
    opt=torch.optim.AdamW(model.parameters(),lr=.002)
    for _ in range(1900):
        pick=torch.randint(0,1380,(100,))
        pred=model(x[pick])
        valid=y[pick,2]>0
        loss=torch.nn.functional.smooth_l1_loss(pred[valid,:2],y[pick][valid,:2],beta=.08) \
             + .22*torch.nn.functional.binary_cross_entropy_with_logits(pred[:,2],y[pick,2])
        opt.zero_grad()
        loss.backward()
        opt.step()
    out=model(x[1380:]).detach()
    target=y[1380:]
    valid=target[:,2]>0
    errors=(out[valid,:2]-target[valid,:2]).abs().mean(0)
    result=dict(bearing_mae_deg=float(errors[0]*40),range_mae_native=float(errors[1]),
                  presence=float(((out[:,2]>0)==valid).float().mean()),held=220,train=1380,
                  updates=1900,weight_domain="fixed-size named landmark",optical_sha256=sha(path))
    torch.save(model.state_dict(),CHECKPOINT)
    from approach_runtime import turn
    turn.stand.put(HERE/"VISION_RESULT.json",result)
    print(result,flush=True)


if __name__=="__main__":
    torch.set_num_threads(1)
    collect()
    fit()
