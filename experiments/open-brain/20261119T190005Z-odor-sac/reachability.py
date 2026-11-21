"""Use private geometry to test physical feasibility and supply exploration rollouts."""
import json
import math

import numpy as np

from odor_env import HORIZON, course, odor_steering


def route_action(env):
    cfg = env.cfg
    if cfg.stage == 'open':
        return odor_steering(env.obs)
    edge = cfg.high if cfg.side == 1 else -cfg.low
    lane = cfg.side*(edge+.28)
    if env.x < cfg.near+.17:
        target_y = lane
        target_x = max(env.x+.16,cfg.near-.18)
    else:
        target_x,target_y = cfg.goal
    desired = math.atan2(target_y-env.y,target_x-env.x)
    error = math.atan2(math.sin(desired-env.yaw),math.cos(desired-env.yaw))
    return np.array([-.72 if env.x<cfg.near+.20 else .25,
                     np.clip(-3.2*error,-1.,1.)],dtype=np.float32)


def run(motor, factory, stage, seeds):
    results = []
    for seed in seeds:
        for side in (1,-1):
            env = factory(motor)
            env.reset(course(seed,stage,side),seed+887,fixed_pose=True)
            for _ in range(HORIZON):
                _,_,done,timeout,info=env.step(route_action(env))
                if done or timeout:
                    break
            results.append(info | dict(seed=seed,side=side,path=env.path,stage=stage))
    return results


if __name__=='__main__':
    from train import Motor, CAL, DEV, summarize
    from proximity import AssistedProxy
    motor=Motor(json.loads(CAL.read_text()))
    for mode in ('wall','detour'):
        print(mode,summarize(run(motor,AssistedProxy,mode,DEV)))
