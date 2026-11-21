"""Measure the six-foot body footprint to bound useful bridge margins."""
import json
import sys
from pathlib import Path

import mujoco as mj
import numpy as np

from common import init_course, start_pose
from live_course import Contacts


def main(output):
    env,_,_=init_course(15121,'flat',1,food=False)
    env.reset()
    start_pose(env)
    phy=env.physics
    sampler=Contacts(phy)
    m,d=phy.model.ptr,phy.data.ptr
    data=[]
    for tick in range(2500):
        phy.step()
        if tick<2100 or tick%10:
            continue
        q=d.qpos
        w,x,y,z=q[3:7]
        yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        side=np.array([-np.sin(yaw),np.cos(yaw)])
        f=np.zeros(6)
        measured={}
        for c,con in enumerate(d.contact[:d.ncon]):
            a,b=int(con.geom1),int(con.geom2)
            if sampler.walker[b]:
                a,b=b,a
            leg=sampler.geom_foot.get(a)
            if leg is None or sampler.walker[b] or con.efc_address<0:
                continue
            # Some task engines refresh after stepping. Only positional
            # contacts are used, so no stale force quantity enters.
            value=float((con.pos[:2]-q[:2])@side)
            measured.setdefault(str(leg),[]).append(value*10.)
        if len(measured)==6:
            data.append(measured)
    assert data
    by_leg={name:float(np.median([max(abs(v) for v in row[name]) for row in data]))
            for name in data[0]}
    evidence=dict(grounded_samples=len(data),median_transverse_reach_mm=by_leg,
                  median_maximum_transverse_reach_mm=max(by_leg.values()))
    output.write_text(json.dumps(evidence,indent=2)+'\n')
    print(evidence,flush=True)
    env.close()


if __name__=='__main__':
    main(Path(sys.argv[1]))
