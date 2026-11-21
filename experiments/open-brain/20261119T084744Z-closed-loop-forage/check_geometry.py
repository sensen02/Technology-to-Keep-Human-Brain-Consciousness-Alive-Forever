"""Verify the blocked direct lane and usable physical side corridor."""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from common import init_course, start_pose
from live_course import Contacts


def probe(seed,stage,side,food):
    env,cfg,bank=init_course(seed,stage,side,food=food)
    env.reset()
    phy=env.physics
    pose=start_pose(env)
    m,d=phy.model.ptr,phy.data.ptr
    ids=np.zeros(1,dtype=np.int32)

    def surface(x,y):
        dist=mujoco.mj_ray(m,d,np.array((x,y,.35)),np.array((0.,0.,-1.)),
                           None,1,mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'walker/'),ids)
        assert dist>=0.
        return .35-dist

    ys=(0.,side*(cfg.edge+.26))
    xs=np.linspace(cfg.near+.045,cfg.far-.045,5)
    heights=np.array([[surface(float(x),float(y)) for x in xs] for y in ys])
    assert np.max(heights[0])<-.18 and np.max(np.abs(heights[1]))<1e-7
    scan=Contacts(phy)
    pose[:2]=(float(xs[2]),ys[1])
    with phy.reset_context():
        phy.data.qpos[:]=pose
        phy.data.qvel[:]=0.
        phy.data.ctrl[:]=0.
        phy.data.act[:]=0.
    feet=[]
    impacts=[]
    for tick in range(1100):
        phy.step()
        if tick>=900:
            support,hit,_=scan.scan(phy)
            feet.append(len(support))
            impacts.append(hit)
    assert min(feet)>=4 and max(impacts)==0
    report=dict(seed=seed,stage=stage,side=side,midpoint_x=float(xs[2]),
                direct_floor_z_cm=heights[0].tolist(),side_floor_z_cm=heights[1].tolist(),
                side_stance_min_feet=int(min(feet)),side_nonfoot=int(max(impacts)))
    env.close()
    return report


if __name__=='__main__':
    output=Path(sys.argv[1])
    measured=[probe(seed,s,side,False) for seed,s in [
        (15121,'wide_bridge'),(15131,'bridge'),(15131,'long_bridge')] for side in (-1,1)]
    output.write_text(json.dumps(measured,indent=2,allow_nan=False)+'\n')
    print('collision geometry and physical side stance',len(measured),'pass',flush=True)
