"""Causal brain-side intention with a bounded learned residual."""
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
PAIR=HERE.parent/'20261119T073818Z-sac-terrain-brain-pair'
sys.path.insert(0,str(PAIR))
from forage import Terrain
from policy_runtime import actor_mean


def terrain(seed,challenge,side):
    from dataclasses import replace
    base=Terrain.from_seed(seed,int(challenge!='flat'),side)
    rng=np.random.default_rng(seed+42357)
    near=float(rng.uniform(.59,.77))
    if challenge=='flat':
        return replace(base,goal_x=float(rng.uniform(1.48,1.63)),goal_y=float(rng.uniform(-.12,.12)))
    if challenge=='wide_bridge':
        near+=.16
        return replace(base,near=near,far=near+float(rng.uniform(.23,.28)),
                       edge=float(rng.uniform(.11,.165)),goal_x=float(rng.uniform(1.75,1.87)),
                       goal_y=float(side*rng.uniform(.16,.235)))
    if challenge=='bridge':
        return replace(base,near=near,far=near+float(rng.uniform(.25,.31)),
                       edge=float(rng.uniform(.155,.245)),goal_x=float(rng.uniform(1.63,1.82)),
                       goal_y=float(side*rng.uniform(.12,.23)))
    return replace(base,near=near,far=near+float(rng.uniform(.32,.41)),
                   edge=float(rng.uniform(.245,.35)),goal_x=float(rng.uniform(1.95,2.13)),
                   goal_y=float(side*rng.uniform(.035,.18)),roughness=.15)


def proposal(obs,action,profile="small"):
    # T-positive rotates head's world-yaw negative in the established adapter.
    angle=np.arctan2(obs[1],obs[2])
    h=float(np.max(obs[4:10]))
    if not obs[0] and obs[3]>=.999:
        return 0.,0.
    nominal=np.clip(-.32*angle/np.deg2rad(35.),-.19,.19)
    brake=np.clip((1.-float(action[0]))/2.,0.,1.)
    if profile=='strong':
        weight=np.clip(h/.52,0.,1.)
        pace=.55-.39*weight*brake
        steer=np.clip(nominal*(1.-.66*weight)-.34*weight*float(action[1]),-.24,.24)
    else:
        pace=.55-.13*h*brake
        steer=np.clip(nominal-.21*h*float(action[1]),-.24,.24)
    return float(pace),float(steer)


def decode_levels(left,right):
    return .92*np.clip((left+right)/.96,0.,1.), np.clip(1.12*(left-right),-.76,.76)


def infer(policy,obs,profile="small"):
    a=np.array([1.,0.],dtype=np.float32) if policy is None else actor_mean(policy,obs)
    return a,proposal(obs,a,profile)
