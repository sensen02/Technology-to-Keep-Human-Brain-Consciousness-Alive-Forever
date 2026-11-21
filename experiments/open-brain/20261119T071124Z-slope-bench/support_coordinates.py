"""Use a foot-estimated tangent frame for the frozen walking policy's sensor copy."""
import argparse

import numpy as np
from scipy.spatial.transform import Rotation

import slope_solved as bench


class SupportCoordinatesWalk(bench.ReversedYawWalk):
    def __init__(self, env, duration):
        super().__init__(env,duration)
        self.initial_reference_z = self.ref[0,2]

    def action(self, phy, speed, yaw_rate, live_observation):
        surfaces = phy.step.__self__
        normal, point = surfaces.plane_normal, surfaces.plane_point
        axis = np.cross(normal,[0.,0.,1.])
        magnitude = np.linalg.norm(axis)
        flatten = Rotation.identity() if magnitude < 1e-12 else Rotation.from_rotvec(
            axis*(np.arctan2(magnitude,normal[2])/magnitude))
        ph = self.env.physics
        ph.data.qpos[self.qcopy[0]] = phy.data.qpos[self.qcopy[1]]
        ph.data.qvel[self.vcopy[0]] = phy.data.qvel[self.vcopy[1]]
        origin = (point@normal)*normal
        ph.data.qpos[:3] = flatten.apply(phy.data.qpos[:3]-origin)
        quat = flatten*Rotation.from_quat(phy.data.qpos[[4,5,6,3]])
        ph.data.qpos[3:7] = quat.as_quat()[[3,0,1,2]]
        ph.data.qvel[:3] = flatten.apply(phy.data.qvel[:3])
        ph.forward()

        i = self.step_index
        dt = self.env.control_timestep()
        if i == 0 or np.all(self.filtered == 0.):
            self.position[:] = ph.data.qpos[:2]
        self.heading -= yaw_rate*dt
        ahead = flatten.apply([np.cos(self.heading),np.sin(self.heading),0.])[:2]
        ahead /= np.linalg.norm(ahead)
        self.position += speed*dt*ahead
        ref = self.ref[i:i+self.task._future_steps+2]
        ref[:,:2] = self.position+np.arange(len(ref))[:,None]*dt*speed*ahead
        ref[:,2] = self.initial_reference_z
        ref[:,3:7] = (np.cos(.5*np.arctan2(ahead[1],ahead[0])),0.,0.,
                      np.sin(.5*np.arctan2(ahead[1],ahead[0])))

        self.task._step_counter = i
        updater = self.env._observation_updater
        updater.reset(self.env._physics_proxy,self.env.random_state)
        observation = updater.get_observation()
        for name in ('accelerometer','appendages_pos','force','gyro','touch','velocimeter'):
            key='walker/'+name
            if key in live_observation and observation[key].shape == live_observation[key].shape:
                observation[key] = live_observation[key]
        target = bench.teacher_action(self.policy,{k:observation[k] for k in self.keys},self.spec)
        self.filtered += -np.expm1(-dt/.01)*(target-self.filtered)
        self.last[self.columns] = self.filtered
        for act,j in self.action_for_act:
            if act>=0:
                ph.data.act[act]=self.filtered[j]
        self.step_index += 1
        return self.last.copy()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=bench.Path)
    parser.add_argument('--angles',type=int,nargs='+',default=bench.ANGLES)
    parser.add_argument('--yaws',type=int,nargs='+',default=[0])
    bench.ReversedYawWalk=SupportCoordinatesWalk
    bench.main(parser.parse_args())
