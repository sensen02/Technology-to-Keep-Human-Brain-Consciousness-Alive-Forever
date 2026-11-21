"""Match the orchard and flat body at stand and prescribed low-speed walk."""
import json
from pathlib import Path
import argparse
from collections import Counter

import numpy as np
from run_case import Contacts, OrchardWalk, SEED, build_orchard, fold_until_launch, land


def run(name, flat, mode, out):
    env, _ = build_orchard()
    if flat:
        arena = env.task._arena.mjcf_model
        patch = arena.find('geom','orchard_terrain')
        patch.contype = patch.conaffinity = 0
        arena.find('geom','groundplane').pos = (0,0,0)
    env.random_state.seed(SEED)
    env.task.check_termination = lambda _:False
    step = env.reset()
    task,phy = env.task,env.physics
    padding = ((0,task._future_steps+2),(0,0))
    task._ref_qpos = np.pad(task._ref_qpos,padding,mode='edge')
    task._ref_qvel = np.pad(task._ref_qvel,padding,mode='edge')
    contacts = Contacts(phy)
    initial = np.r_[phy.data.qpos,phy.data.qvel]
    walk = OrchardWalk(env,.30) if mode != 'neutral' else None
    if mode != 'ungated_hold':
        fold_until_launch(task,phy)
    values=[]
    geom_pairs=Counter()
    action=np.zeros(env.action_spec().shape)
    prior=task.after_substep

    def sample(physics,rng):
        old_floor,_,feet,hit,_ = contacts.sample()
        center=phy.named.data.subtree_com['walker/']
        _,q=task._walker.get_pose(phy)
        yaw,roll,pitch=land.angles(q)
        values.append((phy.time(),*center,len(old_floor),len(feet),hit,yaw,roll,pitch))
        if hit:
            for c in phy.data.contact:
                ids = c.geom1,c.geom2
                walker = [contacts.bodies[j] for j in ids if contacts.bodies[j][0]]
                if len(walker)==1 and walker[0][1] is None and c.efc_address>=0:
                    geom_pairs[' | '.join(sorted(phy.model.id2name(j,'geom') for j in ids))]+=1
        prior(physics,rng)

    task.after_substep=sample
    dt=env.control_timestep()
    for i in range(round(.30/dt)):
        if walk is not None and i%10==0:
            speed = .85 if mode=='slow' and phy.time()>=.08 else 0.
            action=walk.action(phy,speed,0.,step.observation)
        step=env.step(action.copy())
        assert not step.last()
    values=np.asarray(values)
    np.save(out/(name+'_TRACE.npy'),values,allow_pickle=False)
    np.save(out/(name+'_INITIAL.npy'),initial,allow_pickle=False)
    recover = values[:,0] > .10
    summary=dict(name=name,samples=len(values),
                 early_80ms_six_feet_s=float(np.sum((values[:,0]<=.08)&(values[:,4]==6))*phy.timestep()),
                 early_80ms_nonfoot_s=float(np.sum((values[:,0]<=.08)&(values[:,6]>0))*phy.timestep()),
                 late_min_floor_feet=int(values[recover,4].min()),
                 total_nonfoot_steps=int(np.count_nonzero(values[:,6])),
                 nonfoot_pairs=geom_pairs.most_common(9),
                 early_max_abs_yaw_deg=float(np.rad2deg(np.abs(values[values[:,0]<=.08,7]).max())),
                 early_height_span_mm=float(np.ptp(values[values[:,0]<=.08,3])*10.),
                 end_xyz_mm=(10.*values[-1,1:4]).tolist())
    if walk is not None:
        walk.env.close()
    env.close()
    return summary


def main(out):
    out.mkdir()
    rows={}
    for label,flat in (('orchard',False),('flat',True)):
        for mode in ('neutral','hold','slow','ungated_hold'):
            key=label+'_'+mode
            rows[key]=run(key,flat,mode,out)
            print(rows[key],flush=True)
    (out/'RESULT.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    main(p.parse_args().output.resolve())
