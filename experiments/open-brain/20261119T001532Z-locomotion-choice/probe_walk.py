from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'20261118T234622Z-flight-fruit'))
from perching import land
from flybody.fly_envs import walk_imitation
from flybody.tasks.trajectory_loaders import constant_speed_trajectory
from fly import teacher_action

if __name__ == '__main__':
    env=walk_imitation()
    ref,vel=constant_speed_trajectory(n_steps=3900,speed=2.0,init_pos=(0,0,.1278),
                                        control_timestep=env.control_timestep())
    env.task._traj_generator.set_next_trajectory(ref,vel)
    step=env.reset()
    walking=land.load_policy(HERE/'POLICY/walking')
    print(env.control_timestep(),list(env.observation_spec()),env.action_spec().name,flush=True)
    for i in range(800):
        action=teacher_action(walking,step.observation,env.action_spec())
        step=env.step(action)
        if i%100==99:
            print(i,env.physics.named.data.subtree_com['walker/'].tolist(),step.last(),flush=True)
        if step.last(): break
