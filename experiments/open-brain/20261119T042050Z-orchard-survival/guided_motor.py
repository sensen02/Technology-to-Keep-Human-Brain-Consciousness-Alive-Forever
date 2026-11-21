"""Engineering reachability control for the specified outer branch fruit."""
import numpy as np

from run_case import BRANCH, OrchardFlight, WingReadout, land, yaw_rotation


class CurvedBranchApproach(OrchardFlight):
    """Give the frozen wing policy a continuous bearing and braking reference."""

    def set_reference(self, physics):
        task = self.task
        anchor = physics.named.data.subtree_com['walker/'][:2].copy()
        _, measured = task._walker.get_pose(physics)
        initial_heading = land.angles(measured)[0]
        self.heading = np.arctan2(BRANCH.y-anchor[1],BRANCH.x-anchor[0])
        # LaunchMotor provides the tested speed, climb and pitch.
        land.LaunchMotor.set_reference(self,physics)
        raw = task._ref_qpos.copy()
        centers = raw[:,:3] - land.com2root(np.zeros_like(raw[:,:3]),raw[:,3:7])
        along = centers[:,0]-anchor[0]
        tangent = np.array([np.cos(self.heading),np.sin(self.heading)])
        normal = np.array([-tangent[1],tangent[0]])
        slope = np.tan(np.clip(initial_heading-self.heading,-.55,.55))
        length = 4.5
        u = np.clip(along/length,0.,1.)
        lateral = slope*along*(1.-u)**2
        derivative = slope*(1.-u)*(1.-3.*u)
        centers[:,:2] = anchor + along[:,None]*tangent + lateral[:,None]*normal
        headings = self.heading + np.arctan(derivative)
        orient = np.array([WingReadout.multiply(yaw_rotation(angle),q)
                           for angle,q in zip(headings,raw[:,3:7],strict=True)])
        task._ref_qpos[:] = np.column_stack((land.com2root(centers,orient),orient))
        task._ref_qvel[:,:3] = np.gradient(centers,self.dt,axis=0)
        task._ref_qvel[:,5] = np.gradient(headings,self.dt)

    def set_approach(self, physics):
        super().set_approach(physics)
        start = round(self.landing_started/self.dt)
        task = self.task
        refs = task._ref_qpos[start:]
        com = refs[:,:3] - land.com2root(np.zeros_like(refs[:,:3]),refs[:,3:7])
        initial = physics.named.data.subtree_com['walker/'][:2].copy()
        velocity = physics.bind(task._root_joint).qvel[:2].copy()
        duration = self.approach.brake_s
        tangent = np.array([np.cos(self.heading),np.sin(self.heading)])
        target = np.array([BRANCH.x,BRANCH.y])-.15*tangent
        tau = np.clip(np.arange(len(refs))*self.dt/duration,0.,1.)
        com[:,:2] = initial + (3*tau**2-2*tau**3)[:,None]*(target-initial)
        com[:,:2] += (tau*(1.-tau)**2)[:,None]*duration*velocity
        task._ref_qpos[start:,:3] = land.com2root(com,refs[:,3:])
        task._ref_qvel[start:,:2] = np.gradient(com[:,:2],self.dt,axis=0)
