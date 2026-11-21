"""Check the imported soil, registered skin, odor and neutral physical support."""
import json
from pathlib import Path
import sys

import numpy as np

from transfer import (Crop, FRUIT_MASK, bind_physics, build, cast, docking, fit_spawn, odor, sample, scene_ids)

HERE=Path(__file__).resolve().parent


def check(site,support):
    crop=Crop(site,support)
    env,bank=build(crop)
    env.reset()
    phy=bind_physics(env,bank)
    start=fit_spawn(crop,env,docking.start_pose)
    soil=[i for i,name in scene_ids(phy).items() if name.startswith('crop_soil_')]
    previous_groups=phy.model.geom_group.copy()
    phy.model.geom_group[:]=4
    phy.model.geom_group[soil]=5
    group=np.array([0,0,0,0,0,1],dtype=np.uint8)
    errors=[]
    for i in (6,34,50,70,93):
        for j in (8,32,40,52,84,111):
            x,y=crop.x[j],crop.y[i]
            d,geom=cast(phy,(x,y,1.),(0.,0.,-1.),group)
            assert geom in soil,(d,geom,soil,x,y)
            errors.append(abs((1.-d)-crop.terrain[i,j]))
    phy.model.geom_group[:]=previous_groups
    if support!='plane':
        assert max(errors)<.0015,errors
    d,geom=cast(phy,crop.goal-np.array([.17,0.,0.]),(1.,0.,0.),FRUIT_MASK)
    assert geom in bank.skin_ids and abs(d-.17)<.009,(d,geom)
    cfg=env.task
    cfg._ref_qpos=np.pad(cfg._ref_qpos,((0,2000),(0,0)),mode='edge')
    cfg._ref_qvel=np.pad(cfg._ref_qvel,((0,2000),(0,0)),mode='edge')
    cfg.check_termination=lambda _:False
    docking.fold_until_launch(cfg,phy)
    env._observation_updater.reset(env._physics_proxy,env.random_state)
    contacts=docking.Contacts(phy)
    native=phy.timestep()
    th=phy.model.name2id('walker/thorax','body')
    traces=[]
    motor=np.zeros(env.action_spec().shape)
    for step in range(round(.32/env.control_timestep())):
        last=env.step(motor.copy())
        assert not last.last()
        if step%10==0:
            feet,nonfoot,mouth=contacts.scan(phy)
            traces.append((float(phy.time()),len(feet),nonfoot,float(phy.data.xmat[th,8]),
                           float(np.linalg.norm(phy.data.qvel[:2]))))
    tail=np.array(traces)[-40:]
    stance=bool(np.all((tail[:,1]>=4)&(tail[:,3]>.94)&(tail[:,4]<=.08)))
    field,meta=odor(crop)
    ant=[phy.model.name2id('walker/antenna_'+side,'body') for side in ('left','right')]
    smell=sample(field,phy.data.xpos[ant,:2])
    r=dict(site=site,support=support,start_xyz_cm=start[:3].tolist(),start_quat=start[3:7].tolist(),
           plane_corridor_error_mm=crop.plane_corridor_error_mm,
           max_soil_error_mm=10*max(errors),
           skin_registration_error_mm=10*abs(d-.17),skin_colliders=len(bank.skin_ids),
           final_feet_min=int(tail[:,1].min()),final_nonfoot_max=int(tail[:,2].max()),
           final_upright_min=float(tail[:,3].min()),final_speed_max_mm_s=10*float(tail[:,4].max()),
           support_pass=stance,odor_at_settled_antennae=smell,odor=meta,
           steps=round(.32/native))
    env.close()
    return r


if __name__=='__main__':
    support=sys.argv[1] if len(sys.argv)>1 else 'mesh'
    rows=[check(site,support) for site in ('CLEAR','DEBRIS')]
    filename='PREFLIGHT.json' if support=='mesh' else f'PREFLIGHT_{support.upper()}.json'
    (HERE/filename).write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    print(json.dumps(rows,indent=2,allow_nan=False))
