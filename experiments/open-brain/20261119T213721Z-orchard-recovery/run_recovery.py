"""Run the P1 recovery controller and the local feeding reflex together on a chosen source mesh."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from context import HERE,rollout,tr
from flank import Flank,active_build,place,field
from crouch import leg_targets
from reflex import controller

def run():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--start-on-fruit',action='store_true')
    options,remaining=parser.parse_known_args();sys.argv[1:]=remaining
    output=Path(remaining[1])
    params=json.loads((HERE/'SELECTED.json').read_text())['params']
    cfg=json.loads((HERE/'MEAL_CONFIG.json').read_text())
    tr.docking.ReversedYawWalk=controller(params)
    if options.start_on_fruit:
        class UpperPatch(Flank):
            def __init__(self,site,support):
                super().__init__(site,support);self.set_face(cfg['face'])
        tr.Crop=UpperPatch;tr.odor=field
    normal_start=tr.fit_spawn
    state={'ingested':0.};log=[]
    def build(crop):
        env,bank=active_build(crop);state['bank']=bank
        advance=bank.advance
        def measure(*args,**kwargs):
            receipt=advance(*args,**kwargs);state['ingested']+=receipt.ingested.volume
            return receipt
        bank.advance=measure
        return env,bank
    def start(crop,env,base):
        bank=state['bank']
        q=place(env,bank,crop,base,cfg['yaw']) if options.start_on_fruit else normal_start(crop,env,base)
        phy=env.physics;m=phy.model.ptr
        feet={i for i in range(m.ngeom) if any(s in m.body(m.geom_bodyid[i]).name for s in ('tarsus','claw'))}
        names=env.action_spec().name.split('\t');parts=('rostrum','haustellum','head')
        mouth=[names.index(k) for k in parts]
        indices=desired=None
        step=env.task.before_step
        ramp=0.;dt=env.control_timestep()
        def before(phy,action,random_state):
            nonlocal indices,desired,ramp
            contact=any(c.efc_address>=0 and ((c.geom1 in feet and c.geom2 in bank.skin_ids) or
                (c.geom2 in feet and c.geom1 in bank.skin_ids)) for c in phy.data.contact[:phy.data.ncon])
            if contact and indices is None:
                # Compute the flexion goal where contact actually occurred.
                li,lt=leg_targets(phy,names,cfg['lift'])
                indices=mouth+li;spec=env.action_spec()
                desired=np.clip([*cfg['target'],*lt],spec.minimum[indices],spec.maximum[indices])
            satisfied=state['ingested']>=tr.docking.MEAL
            ramp=0. if satisfied else np.clip(ramp+np.clip(float(contact)-ramp,-dt/.025,dt/.08),0.,1.)
            u=action.copy()
            if indices is not None and ramp:
                u[indices]+=ramp*(desired-u[indices])
            if round(phy.time()/dt)%25==0:log.append((phy.time(),contact,ramp,state['ingested']))
            return step(phy,u,random_state)
        env.task.before_step=before
        return q
    tr.build=build;tr.fit_spawn=start;rollout.main()
    np.savez_compressed(output/'LOCAL_REFLEX.npz',samples=log)
    conf=json.loads((output/'CONFIG.json').read_text())
    conf.update(start_on_fruit=options.start_on_fruit,body_params=params,feeding_params=cfg,
                site='FLANK' if options.start_on_fruit else conf['site'])
    (output/'CONFIG.json').write_text(json.dumps(conf,indent=2)+'\n')
if __name__=='__main__':run()
