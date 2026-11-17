"""Identify contact geometries around the largest saved native non-pad force."""
import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from rollout import build_scene, OBSTACLE, stan, W


def run(output, trial):
    result=json.loads((trial/'RESULT.json').read_text())
    with np.load(trial/'NATIVE.npz',allow_pickle=False) as z:
        forces=z['forces'][:,12]
        idx=int(np.argmax(forces))
        exact_peak=float(forces[idx]/W)
    with np.load(trial/'TRACE.npz',allow_pickle=False) as z:
        traces={name:z[name] for name in ('qpos','qvel','ctrl')}
    c,_,_=build_scene(result['plan'])
    ob=c.plant.geom(OBSTACLE).id
    t=idx/80.
    near=[]
    for ms in range(max(0, int(t)-3),min(len(traces['qpos']),int(t)+5)):
        c.d.qpos[:]=traces['qpos'][ms]
        c.d.qvel[:]=traces['qvel'][ms]
        c.d.ctrl[:]=traces['ctrl'][ms]
        mj.mj_forward(c.plant,c.d)
        actual=[]
        for j in range(c.d.ncon):
            contact=c.d.contact[j]
            a,b=int(contact.geom1),int(contact.geom2)
            if contact.efc_address<0 or ob not in (a,b):
                continue
            fly=b if a==ob else a
            if c.geom_leg[fly]>=0:
                continue
            sign=1. if a==ob else -1.
            f=np.zeros(6)
            mj.mj_contactForce(c.plant,c.d,j,f)
            if f[0]<=0.:
                continue
            actual.append(dict(geom=c.plant.geom(fly).name, normal_into_fly_z=float(sign*contact.frame[2]),
                               normal_load_W=float(f[0]/W),point_z_native=float(contact.pos[2])))
        if actual:
            near.append(dict(retained_ms=ms,reconstructed=actual))
    report=dict(native_peak_nonpad_vertical_force_W=exact_peak,native_peak_ms=t,
                stored_ms_pose_contacts_near_native_peak=near,
                scope='Contact geometry reconstruction from retained 1 ms frames only. Native peak remains separately measured. No physical advancement.')
    stan.put(output,report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('trial',type=Path)
    a=p.parse_args()
    run(a.output.resolve(),a.trial.resolve())
