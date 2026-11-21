"""Fit a brain-side search preference using body energy as privileged supervision."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from engine.embodied.feeding import EMPTY, FeedingBody, FeedingState

# Keep a 0.06 reserve after a reference 12 s at the rest cost.
DESIRED_RESERVE = .06+.018*12
REFERENCE = HERE.parent/'20261119T042050Z-orchard-survival/GROUND_EXTENSION/FRESH/MEAL.npy'
SCALE = .1


def simulate(seed, episodes):
    rng = np.random.default_rng(seed)
    body = FeedingBody()
    rows = []
    for _ in range(episodes):
        reserve = rng.uniform(.08,.55)
        circulating = np.clip(reserve+rng.uniform(-.055,.055),0.,1.)
        state = FeedingState(EMPTY,reserve=reserve,circulating=circulating)
        activity = rng.choice((0.,.012))
        for t in range(75):
            if t%3==0:
                rows.append((state.circulating,state.reserve))
            state = body.advance(state,.04,feed=False,contact=False,activity_use=activity).state
    return np.array(rows)


def logit(value):
    return 1./(1.+np.exp(-np.clip(value,-50.,50.)))


def measure(x,weights):
    p=logit(weights[0]+weights[1]*(x[:,0]-DESIRED_RESERVE)/SCALE)
    y=x[:,1]<DESIRED_RESERVE
    certain=np.abs(x[:,1]-DESIRED_RESERVE)>.04
    return dict(n=len(y),accuracy=float(np.mean((p>=.5)==y)),
                accuracy_outside_reserve_margin=float(np.mean((p[certain]>=.5)==y[certain])),
                margin_samples=int(certain.sum()))


def fit():
    samples=simulate(821,160)
    reference=np.load(REFERENCE,allow_pickle=False)[::50][:,[7,3]]
    samples=np.concatenate((samples,reference))
    x=np.column_stack((np.ones(len(samples)),(samples[:,0]-DESIRED_RESERVE)/SCALE))
    y=(samples[:,1]<DESIRED_RESERVE).astype(float)
    w=np.zeros(2)
    stages=[]
    for k in range(800):
        p=logit(x@w)
        loss=-np.mean(y*np.log(np.maximum(p,1e-15))+(1.-y)*np.log(np.maximum(1.-p,1e-15)))
        if k in (0,799):
            stages.append(float(loss))
        w -= .25*(x.T@(p-y)/len(y)+np.array([0.,.006*w[1]]))
    assert w[1]<0.
    heldout=measure(simulate(1927,48),w)
    assert heldout['accuracy_outside_reserve_margin']>=.94
    report=dict(weights=w.tolist(),offset=DESIRED_RESERVE,scale=SCALE,
                features=['body_circulating_nutrient_proxy'],teacher='body reserve below required reserve',
                teacher_reserve=DESIRED_RESERVE,fit_steps=800,cross_entropy_first_last=stages,
                train=measure(samples,w),heldout=heldout,archive_samples=len(reference),
                search_on=.60,search_off=.40,eon_weight_updates=0)
    (HERE/'GATE_FIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    fit()
