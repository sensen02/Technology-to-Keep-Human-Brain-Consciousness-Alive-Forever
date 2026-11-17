"""Short physical probes initialized by an independently integrated AN prefix.

Outputs are development only. Full tests always start with the frozen stand snapshot.
"""
import argparse
from pathlib import Path
import sys

import mujoco as mj
import numpy as np
import torch

from physical import bench,build_scene,Local,frozen_policy,read,stan,W,RF,RM,RF_JOINTS,RM_JOINTS
from gap_demo import Teacher

STEM_MS=2750


def snapshot(c):
    spec=mj.mjtState.mjSTATE_INTEGRATION
    state=np.zeros(mj.mj_stateSize(c.plant,spec))
    mj.mj_getState(c.plant,c.d,state,spec)
    fields={k:getattr(c,k).copy() for k in ('buf','_padz_prev','_pad_rate','requested_rate')}
    act=c.actuator
    motors={k:getattr(act,k).copy() for k in ('target_rad','last_input_rate_rad_s','last_limited_rate_rad_s',
             'last_effective_rate_rad_s','rate_clip_mask','angle_clip_mask')}
    return state,fields,motors,c.t,act.command_ticks


def restore(c,snap):
    q,fields,motors,step,clock=snap
    mj.mj_setState(c.plant,c.d,q,mj.mjtState.mjSTATE_INTEGRATION)
    mj.mj_forward(c.plant,c.d)
    c.t=step
    c.actuator.command_ticks=clock
    for key,v in fields.items():
        setattr(c,key,v.copy())
    for key,v in motors.items():
        setattr(c.actuator,key,v.copy())
    return c.measure()


def main(scenario,output,plans,stop):
    torch.set_num_threads(1)
    shape=read(scenario)
    sha=read(bench.STARTS/'BANK.json')['nominal']['sha256']
    c,_=build_scene(shape,bench.STARTS/'nominal.npz',sha)
    local=Local(c,9600)
    base=frozen_policy()
    m=c.measure()
    for t in range(STEM_MS):
        if t%5==0:
            a=base(bench.features(c,m),int(t>=500))
        m=c.tick(a)
    stem=snapshot(c)
    p=np.asarray(c.pads['RF'])
    output.mkdir()
    for path in plans:
        m=restore(c,stem)
        phase=Teacher(local,read(path))
        trace=[]
        count=0
        for t in range(STEM_MS,stop):
            if t%5==0:
                x=bench.features(c,m)
                b=base(x,int(t>=500))
                a=phase(t,x,1,local.sense(m),b,m)
                count+=np.count_nonzero(a!=b)
            trace.append(np.r_[c.d.geom_xpos[p].mean(0),c.d.subtree_com[c.th],
                              m['loads']/W,m['nonfoot']/W,m['padz'][RF],m['speed']])
            m=c.tick(a)
        z=np.asarray(trace)
        ax,bx,ay,by=shape['opening']
        outside=(z[:,0]<ax)|(z[:,0]>bx)|(z[:,1]<ay)|(z[:,1]>by)
        reload=(z[:,6+RF]>.02)&outside
        sustained=np.flatnonzero(np.convolve(reload,np.ones(25,dtype=int),'valid')==25)
        support=z[:,6:12].sum(1)
        dd=z[:,6:12].sum(0)
        slide=(z[:,6:12]*z[:,14:20]).sum(0)/np.maximum(dd,1e-9)
        result=dict(config=str(path),onset=phase.onset,first_reload_ms=(int(sustained[0])+STEM_MS if len(sustained) else None),
                    peak_clearance=float(z[:,13].max()),mean_support=float(support.mean()),min_com=float(z[:,5].min()),
                    physical_move_native=(z[-1,3:6]-z[0,3:6]).tolist(),max_ground_slide_native=float(slide.max()),
                    named_absolute_ground_slide=slide.tolist(),max_nonfoot_W=float(z[:,12].max()),changed_commands=int(count))
        np.save(output/(path.stem+'.npy'),z)
        stan.put(output/(path.stem+'.json'),result)
        print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('scenario',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('plans',type=Path,nargs='+')
    p.add_argument('--stop',type=int,default=3990)
    a=p.parse_args()
    main(a.scenario.resolve(),a.output.resolve(),a.plans,a.stop)
