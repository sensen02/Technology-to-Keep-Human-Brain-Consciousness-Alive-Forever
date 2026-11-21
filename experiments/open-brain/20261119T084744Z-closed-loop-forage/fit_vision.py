"""Train the danger cue on actual P1 eye frames and unseen scene seeds."""
from dataclasses import asdict
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from common import (HERE, TRAIN_SEEDS, VISION_HOLDOUT, init_course, start_pose, OrchardEyes,
                    optic_features, desired_hazard)


def capture(seeds):
    frames=[]
    rows=[]
    metadata=[]
    for seed in seeds:
        for side in (-1,1):
            difficulty='long_bridge' if (seed+side)%4==0 else 'bridge'
            env,cfg,_=init_course(seed,difficulty,side)
            env.reset()
            start=start_pose(env)
            phy=env.physics
            eye=OrchardEyes(phy)
            for x in (0.,.16,.30,.45,.58,.72,.88,1.08):
                for y,yaw in ((0.,0.),(.19*side,.25*side),(-.17*side,-.18*side)):
                    phy.data.qpos[:]=start
                    phy.data.qpos[:3]=(x,y,.1378)
                    phy.data.qpos[3:7]=(np.cos(yaw*.5),0.,0.,np.sin(yaw*.5))
                    phy.forward()
                    eyes=eye.sample()
                    origin,head,_=eye.head_pose()
                    heading=np.arctan2(head[1],head[0])
                    measured=optic_features(*eyes[:2])
                    actual=desired_hazard(cfg,(x,y),heading)
                    rows.append((measured,actual))
                    metadata.append(dict(seed=seed,side=side,x=x,y=y,yaw=yaw,t=asdict(cfg)))
                    if len(frames)<3 and seed==seeds[0] and y==0. and x in (.16,.45,.72):
                        frames.append(eyes[0].copy())
            env.close()
    x,y=map(np.stack,zip(*rows))
    return x,y,frames,metadata


def design(x):
    return np.c_[x,np.sqrt(np.maximum(x,0.)),np.ones(len(x))]


def run(out):
    out.mkdir()
    x,y,_,aux=capture(TRAIN_SEEDS)
    q,z,frames,metadata=capture(VISION_HOLDOUT)
    a=design(x)
    weights=np.linalg.solve(a.T@a+np.eye(a.shape[1])*.10,a.T@y)
    predicted=np.clip(design(q)@weights,0.,1.)
    mask=np.max(q[:,:30].reshape(-1,6,5),axis=2)>0.
    predicted*=mask
    baseline=np.clip(q[:,:30].reshape(-1,6,5)[:,:,:4].mean(-1)*5.,0.,1.)
    one=dict(train_views=len(x),heldout_views=len(q),
             original_mae=float(np.abs(baseline-z).mean()),
             fitted_mae=float(np.abs(predicted-z).mean()),
             original_positive_recall=float((baseline[z>.4]>.30).mean()),
             fitted_positive_recall=float((predicted[z>.4]>.30).mean()),
             seed_train=TRAIN_SEEDS,seed_heldout=VISION_HOLDOUT)
    np.savez_compressed(out/'VISION.npz',weights=weights,train_x=x,train_y=y,
                       held_x=q,held_y=z,predicted=predicted,baseline=baseline)
    (out/'VIEWS.json').write_text(json.dumps(dict(train=aux,held=metadata),allow_nan=False)+'\n')
    (out/'RESULT.json').write_text(json.dumps(one,indent=2)+'\n')
    if frames:
        Image.fromarray(np.concatenate(frames,axis=1)).save(out/'EYES.png')
    print(one,flush=True)


if __name__=='__main__':
    run(Path(sys.argv[1]).resolve())
