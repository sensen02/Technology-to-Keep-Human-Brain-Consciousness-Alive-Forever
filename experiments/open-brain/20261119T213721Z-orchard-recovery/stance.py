"""Fit a local fall warning and supported leg postures from development rollouts."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'20261119T205558Z-orchard-curriculum'
LEGS=[(f'T{i}',side) for i in (1,2,3) for side in ('left','right')]

def features(feet,qpos,vz):
    up=Rotation.from_quat(qpos[..., [4,5,6,3]]).as_matrix()[...,2,:]
    return np.concatenate((feet,up,np.expand_dims(vz,axis=-1)),axis=-1)

def main():
    xs=[]; ys=[]; good=[]; masks=[]; episodes={}
    for name in ('R1_CLEAR','R1_DEBRIS','R2_CLEAR','R2_DEBRIS','R3_CLEAR','R3_DEBRIS','FRAME_CLEAR','FRAME_DEBRIS'):
        p=PREV/name
        r=json.loads((p/'RESULT.json').read_text())
        fall=r['first_fall_s']
        with np.load(p/'BODY.npz') as b, np.load(p/'TRAIN_MEASURES.npz') as z:
            t=b['trace']; q=b['poses']; feet=z['feet']
        use=(t[:,0]>=.16)&(t[:,0]<(fall if fall else r['duration_s']-.08))
        near=(t[:,0]>=fall-.08) if fall else np.zeros(len(t),bool)
        xs.append(features(feet,q,t[:,9])[use]);ys.append(near[use])
        stable=use & ~near & (t[:,6]>.95) & (feet.sum(1)>=4) & (np.linalg.norm(t[:,7:9],axis=1)>.12)
        good.extend(q[stable]);masks.extend(feet[stable])
        episodes[name]=dict(rows=int(use.sum()),prefall=int(near[use].sum()),stance=int(stable.sum()),
                            last_feet=feet[use][-1].astype(int).tolist())
    X=np.vstack(xs); y=np.hstack(ys); mean=X.mean(0); scale=np.maximum(X.std(0),.035)
    design=np.column_stack((np.ones(len(X)),(X-mean)/scale))
    weight=np.where(y,.5/y.sum(),.5/(~y).sum())
    def loss(x):
        a=design@x; delta=expit(a)-y
        return float(weight@(np.logaddexp(0.,a)-a*y)+.025*(x[1:]@x[1:])),design.T@(weight*delta)+.05*np.r_[0.,x[1:]]
    res=minimize(loss,np.zeros(design.shape[1]),jac=True,method='L-BFGS-B');assert res.success
    prob=expit(design@res.x);q=np.array(good);m=np.array(masks,dtype=bool)
    poses=np.stack([np.median(q[m[:,i]],axis=0) for i in range(6)])
    np.savez(HERE/'STANCE.npz',mean=mean,scale=scale,coef=res.x,stance=poses)
    row=dict(episodes=episodes,frames=len(y),positives=int(y.sum()),loss=float(res.fun),
             mean_score_prefall=float(prob[y].mean()),mean_score_other=float(prob[~y].mean()),moving_stances=len(q))
    (HERE/'STANCE.json').write_text(json.dumps(row,indent=2)+'\n')
    print(row)
if __name__=='__main__':main()
