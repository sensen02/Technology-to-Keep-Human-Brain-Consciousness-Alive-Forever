"""Pair the frozen policy with a trajectory reference estimated from loaded feet."""
import argparse
import numpy as np
from scipy.spatial.transform import Rotation

import slope_solved as bench


class ContactFrameWalk(bench.ReversedYawWalk):
    def __init__(self, env, duration):
        super().__init__(env, duration)
        self.live = env.physics
        self.nominal_height = float(self.ref[0,2])
        self.updater = self.env._observation_updater
        self.default_observation = self.updater.get_observation
        self.updater.get_observation = self.observation

    def observation(self):
        # Parent advances its XY/yaw reference before querying these observations.
        # The physical body is never moved here.
        contacts = self.live.step.__self__
        normal, point = contacts.plane_normal, contacts.plane_point
        i = self.step_index
        if normal[2] < .35:
            return self.default_observation()
        look = self.ref[i:i+self.task._future_steps+2]
        look[:,2] = (self.nominal_height + normal@point - look[:,:2]@normal[:2])/normal[2]
        forward = np.array([np.cos(self.heading),np.sin(self.heading),0.])
        forward -= (forward@normal)*normal
        forward /= np.linalg.norm(forward)
        side = np.cross(normal,forward)
        orient = Rotation.from_matrix(np.column_stack((forward,side,normal))).as_quat()
        look[:,3:7] = orient[[3,0,1,2]]
        return self.default_observation()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=bench.Path)
    parser.add_argument('--angles',type=int,nargs='+',default=bench.ANGLES)
    parser.add_argument('--yaws',type=int,nargs='+',default=[0])
    bench.ReversedYawWalk=ContactFrameWalk
    bench.main(parser.parse_args())
