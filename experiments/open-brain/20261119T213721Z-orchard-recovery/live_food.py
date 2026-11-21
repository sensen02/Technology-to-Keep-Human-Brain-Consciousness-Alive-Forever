"""Paired localized feeding assay; sugar-bearing skin contact at the tarsi starts a body reflex."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from context import HERE,rollout,tr
from crouch import leg_targets
from flank import Flank,active_build,place,field

def run():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--reflex',choices=['on','off'],required=True)
    parser.add_argument('--assay-config',type=Path,required=True)
    a,rest=parser.parse_known_args()
    sys.argv[1:]=rest;out=Path(rest[1])
    cfg=json.loads(a.assay_config.read_text())
    class Geometry(Flank):
        def __init__(self,*args):
            super().__init__(*args);self.set_face(cfg['face'])
    context={};records=[]
    def build(crop):
        env,bank=active_build(crop);context['bank']=bank;return env,bank
    def spawn(crop,env,start):
        bank=context['bank'];q=place(env,bank,crop,start,cfg['yaw'])
        phy=env.physics;m=phy.model.ptr
        feet={i for i in range(m.ngeom) if any(n in m.body(m.geom_bodyid[i]).name for n in ('tarsus','claw'))}
        names=env.action_spec().name.split('\t');mi=[names.index(k) for k in ('rostrum','haustellum','head')]
        li,lt=leg_targets(phy,names,cfg['lift']) if cfg['lift'] else ([],[])
        indices=mi+li
        desired=np.clip(np.array([*cfg['target'],*lt]),env.action_spec().minimum[indices],env.action_spec().maximum[indices])
        base=env.task.before_step;ramp=0.;dt=env.control_timestep()
        def reflex(phy,action,random_state):
            nonlocal ramp
            taste=any(c.efc_address>=0 and ((c.geom1 in feet and c.geom2 in bank.skin_ids) or
                (c.geom2 in feet and c.geom1 in bank.skin_ids)) for c in phy.data.contact[:phy.data.ncon])
            value=float(taste and a.reflex=='on')
            ramp=np.clip(ramp+np.clip(value-ramp,-dt/.025,dt/.08),0.,1.)
            u=action.copy();u[indices]+=ramp*(desired-u[indices])
            if round(phy.time()/dt)%25==0:
                actual=[float(phy.data.qpos[m.joint('walker/'+k).qposadr[0]]) for k in ('rostrum','haustellum','head')]
                records.append([phy.time(),int(taste),ramp,*u[mi],*actual])
            return base(phy,u,random_state)
        env.task.before_step=reflex
        return q
    tr.Crop=Geometry;tr.build=build;tr.fit_spawn=spawn;tr.odor=field
    rollout.main()
    np.savez_compressed(out/'TARSAL_REFLEX.npz',samples=records)
    info=json.loads((out/'CONFIG.json').read_text())
    info.update(site='FLANK',source_fruit='ground-003',mode=a.reflex,geometry=cfg)
    (out/'CONFIG.json').write_text(json.dumps(info,indent=2)+'\n')
if __name__=='__main__':run()
