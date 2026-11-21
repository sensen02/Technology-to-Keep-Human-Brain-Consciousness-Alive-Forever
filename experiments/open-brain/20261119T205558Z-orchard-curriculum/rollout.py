"""Collect actual odor-policy transitions in the P1 orchard, with measured oral distance."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
SCENE=HERE.parent/'20261119T194948Z-orchard-odor-transfer'
sys.path.insert(0,str(SCENE))
import transfer as tr


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('site',choices=['CLEAR','DEBRIS'])
    p.add_argument('out',type=Path)
    p.add_argument('--policy',type=Path,default=tr.POLICY)
    p.add_argument('--duration',type=float,default=2.)
    p.add_argument('--start-x',type=float,default=0.,help='training start offset in cm')
    p.add_argument('--noise',type=float,default=0.)
    p.add_argument('--seed',type=int,default=29)
    p.add_argument('--exploration-seed',type=int,default=701)
    a=p.parse_args()
    a.out=a.out.resolve()
    a.support='mesh'
    a.odor=True
    tr.POLICY=a.policy.resolve()
    base_run=tr.docking.run
    def run(cfg):
        cfg.neural_seed=a.seed
        return base_run(cfg)
    tr.docking.run=run
    old_fit=tr.fit_spawn
    def fit(crop,env,base_start):
        if not a.start_x:
            return old_fit(crop,env,base_start)
        # Evaluate the original stand registration on the terrain under the new origin.
        original=crop.floor
        crop.floor=lambda yx: original(np.asarray(yx)+[0.,a.start_x])
        try:
            q=old_fit(crop,env,base_start)
        finally:
            crop.floor=original
        q[0]+=a.start_x
        with env.physics.reset_context():
            env.physics.data.qpos[:]=q
        return q
    tr.fit_spawn=fit
    rng=np.random.default_rng(a.exploration_seed)
    def action(policy,obs,profile):
        u=tr.actor_mean(policy,obs)
        if a.noise:
            u=np.clip(u+rng.normal(0,a.noise,2),-1.,1.)
        return u,tr.intent(obs,u)
    tr.infer=action
    # Record geometry for scoring only; the actor still receives the 14 baseline features.
    old_contact=tr.SkinFood.contact
    rows=[]
    calls=0
    def contact(bank,mouth,supported):
        nonlocal calls
        bite=old_contact(bank,mouth,supported)
        if calls%25==24:
            rows.append([bank.phy.time(),*mouth,10*np.linalg.norm(mouth-bank.crop.goal),bool(bite)])
        calls+=1
        return bite
    tr.SkinFood.contact=contact
    scan=tr.docking.Contacts
    masks=[]
    class Feet(scan):
        def scan(self,phy):
            feet,n,oral=super().scan(phy)
            if calls%25==24:
                masks.append([int((leg,side) in feet) for leg in ('T1','T2','T3') for side in ('left','right')])
            return feet,n,oral
    tr.docking.Contacts=Feet
    tr.run(a)
    np.savez_compressed(a.out/'TRAIN_MEASURES.npz',oral=rows,feet=masks)
    (a.out/'CONFIG.json').write_text(json.dumps(vars(a),default=str,indent=2)+'\n')


if __name__=='__main__':
    main()
