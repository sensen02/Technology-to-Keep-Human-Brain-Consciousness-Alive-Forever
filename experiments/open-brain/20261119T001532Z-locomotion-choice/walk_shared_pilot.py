import argparse
from pathlib import Path

import numpy as np

from walk_adapter import SharedWalk, HERE, land
from perching import build_scene


def main(duration):
    env=build_scene(8.03)
    task=env.task
    setup=task.initialize_episode
    env._time_limit=duration+.001

    def start(phy,rng):
        setup(phy,rng)
        land.set_ground_pose(env)
        task._ref_qpos[:,:3]=task._walker.get_pose(phy)[0]
        task._ref_qpos[:,3:7]=(1,0,0,0)
        task._ref_qvel[:]=0.

    task.initialize_episode=start
    task.check_termination=lambda _:False
    t=env.reset()
    n=round(duration/env.control_timestep())
    task._ref_qpos=np.pad(task._ref_qpos,((0,n),(0,0)),mode='edge')
    task._ref_qvel=np.pad(task._ref_qvel,((0,n),(0,0)),mode='edge')
    stock=land.make_env()
    obs=stock.reset().observation
    mover=land.LaunchMotor(task,env.physics.bind(task._wing_joints).qpos.copy(),
                           land.LeggedFlightTeacher(stock,obs,env),env.control_timestep(),land.launch_settings(29,duration))
    walker=SharedWalk(env,duration)
    for i in range(n):
        if i%10==0:
            action=walker.action(env.physics,2.,t.observation)
        t=env.step(action.copy())
        if (i+1)%1000==0:
            print((i+1)*env.control_timestep(),env.physics.named.data.subtree_com['walker/'].tolist(),t.last(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--duration',type=float,default=1.0)
    main(p.parse_args().duration)
