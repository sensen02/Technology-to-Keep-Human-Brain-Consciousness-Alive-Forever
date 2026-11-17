"""Short support-target offset on the independently audited braking teacher."""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np

from gap_stop import Brake
from gap_servo import Servo
from physical import RM,RM_JOINTS,read,run,stan


def smooth(v):
    u=np.clip(v,0.,1.)
    return u*u*(3.-2.*u)


class Weight(Brake):
    def __init__(self, local, plan):
        self.brake_plan=plan.copy()
        plan=plan.copy()
        plan['support_native']=0.
        super().__init__(local,plan)

    def __call__(self,t,x,intent,local,base,m):
        was=self.onset is not None
        a=super().__call__(t,x,intent,local,base,m)
        if self.onset is None:
            return a
        c=self.s.body
        age=t-self.onset
        if not was:
            self.helper_target=c.actuator.target_rad[RM_JOINTS].copy()
            padids=self.pads[RM]
            pos=c.d.geom_xpos[padids].mean(0)
            mj.mj_jac(c.plant,c.d,self.jac,None,pos,int(c.plant.geom_bodyid[padids[0]]))
            j=self.jac[:,c.a_dof[RM_JOINTS]]
            self.helper_offset=j.T@np.linalg.solve(j@j.T+.006*np.eye(3),np.array([0.,0.,-.055]))
        if age<460:
            gain=smooth((age+5)/95.)*(1.-smooth((age+5-280.)/180.))
            goal=self.helper_target+self.helper_offset*gain
            desired=(goal-c.actuator.target_rad[RM_JOINTS])/(.005*stan.physics.MAX_RATE)
            a[RM_JOINTS]=np.clip(desired,-1.,1.)
        return a


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    import gap_demo
    gap_demo.Teacher=Weight
    run(a.output,'teacher',Path('AS.json'),native=True,probe_ms=6450,
        teacher_config=Path('PLAN_BRAKE_RF_ONLY.json'),policy_source=Path(__file__))
