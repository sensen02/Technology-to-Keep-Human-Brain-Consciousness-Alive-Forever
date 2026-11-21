"""Immutable qualified body/teacher and source-qualified descending replay."""
from pathlib import Path
import sys
import json
import numpy as np
import torch
import mujoco

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE=HERE.parent
TEACHER=BASE/'20261117T184709Z-body-terrain-repair'
sys.path.insert(0,str(TEACHER))
from body_model import Body,DT,sha,write
from run_teacher_final import setup,COMMAND_PATH
from flygym_demo.complex_terrain.hybrid_controller import HybridControllerObservation
from flygym_demo.complex_terrain.common import dof_spec_to_jointdof

TRAIN_SEEDS=(119,227,383,941)
DEV_SEEDS=(181,263)
HELD_SEEDS=(1151,1723)
HEIGHTS=(0.,.15,.30)
STEPS=17000
HOLD=10


def rotation(quat):
    w,x,y,z=np.asarray(quat).T
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                     2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                     2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),axis=-1).reshape(*np.asarray(quat).shape[:-1],3,3)


def feature(qpos,qvel,link_force,command,phase,weight):
    r=rotation(qpos[...,3:7])
    local=np.einsum('...ij,...lj->...li',np.swapaxes(r,-1,-2),link_force.reshape(*link_force.shape[:-3],24,3))
    gravity=r[...,2,:]
    return np.concatenate((command,qpos[...,7:],qvel[...,6:]/35.,
       np.arcsinh(local.reshape(*local.shape[:-2],72)/(weight/6)),gravity,qvel[...,3:6]/20.,
       np.sin(phase[...,:6]),np.cos(phase[...,:6]),phase[...,6:]),axis=-1).astype(np.float32)


def command_trace():
    with np.load(COMMAND_PATH,allow_pickle=False) as a:return a['trace'][:,6:8].copy()


def observation(body):
    return HybridControllerObservation.from_sim(body.sim,body.name,legs=body.legs,
                         stumbling_links=('tibia','tarsus1','tarsus2','tarsus5'))


def phase_state(cpg):
    return np.r_[cpg.curr_phases%(2*np.pi),cpg.curr_magnitudes].copy()


def advance(cpg,frequencies,command):
    cpg.intrinsic_amps=np.repeat(np.abs(command),3)
    cpg.intrinsic_freqs=frequencies*np.repeat(np.where(command>=0,1.,-1.),3)
    cpg.step()


def assess(a,b,height):
    pos=a['position'];j=np.where(b.m.jnt_limited)[0];q=a['qpos'][:,b.m.jnt_qposadr[j]]
    excess=np.maximum(np.maximum(b.m.jnt_range[j,0]-q,q-b.m.jnt_range[j,1]),0)
    rms=float(np.sqrt(np.mean(np.sum((np.diff(pos[-2000:,:2],axis=0)/DT)**2,axis=1))))
    gates=dict(completed=len(pos)==STEPS,finite=bool(all(np.isfinite(v).all() for v in a.values())),
       no_warning=not b.d.warning.number.any(),upright=bool(a['upright'].min()>=.8),
       height=bool(pos[:,2].min()>=.7*a['initial_position'][2]),nonfoot=bool(a['nonfoot'].max()/b.weight<=.2),
       joint_limit=bool(excess.max()<=.02),target_limits=bool((a['control']>=b.minimum-1e-12).all() and
                            (a['control']<=b.maximum+1e-12).all()),stand=rms<=.30)
    if height:gates.update(block_contact=bool(a['block_force'].max()>1e-6),cross=bool(pos[-1,0]>=3.15))
    else:gates['advance']=bool(pos[-1,0]-a['initial_position'][0]>=2.)
    return dict(pass_=all(gates.values()),gates=gates,dx=float(pos[-1,0]-a['initial_position'][0]),
        final_x=float(pos[-1,0]),stand_rms=rms,min_height=float(pos[:,2].min()),min_upright=float(a['upright'].min()),
        nonfoot_peak_bodyweight=float(a['nonfoot'].max()/b.weight),joint_excess=float(excess.max()))
