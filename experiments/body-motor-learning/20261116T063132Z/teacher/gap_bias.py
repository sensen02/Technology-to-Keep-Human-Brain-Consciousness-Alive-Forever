"""Small tactile-triggered foot bias superposed on physically accepted stance timing."""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np

from physical import RF,RM,RF_JOINTS,RM_JOINTS,read,run,stan
from gap_servo import Servo


def smooth(x):
    u=np.clip(x,0.,1.)
    return u*u*(3.-2.*u)


class Bias(Servo):
    def offset(self, t):
        p,c=self.p,self.s.body
        age=t-self.onset
        rise=smooth(age/p['rise_ms'])
        height=rise*(1.-smooth((age-p['lower_ms'])/p['fall_duration_ms']))
        lateral=smooth((age-p['lateral_start_ms'])/p['lateral_duration_ms'])
        remove=1.-smooth((t-p['second_swing_ms'])/p['second_release_ms'])
        mid=smooth(age/85.)*(1.-smooth((age-195.)/240.))
        self.ref.qpos[:]=self.s.nominal_qpos[t]
        mj.mj_fwdPosition(c.plant,self.ref)
        out=np.zeros(42)
        for leg,goal in [(RF,[p['shift_x']*lateral*remove,p['shift_y']*lateral*remove,p['lift_native']*height]),
                         (RM,[0.,0.,-p['support_native']*mid])]:
            ids=self.pads[leg]
            pos=self.ref.geom_xpos[ids].mean(0)
            mj.mj_jac(c.plant,self.ref,self.jac,None,pos,int(c.plant.geom_bodyid[ids[0]]))
            j=self.jac[:,c.a_dof[self.groups[leg]]]
            out[self.groups[leg]]=j.T@np.linalg.solve(j@j.T+.008*np.eye(3),goal)
        return out

    def __call__(self,t,x,intent,local,base,measured):
        if not intent:
            return base
        p=self.p
        missing=(local[0]>=p['expected_load_W'] and local[1]<=p['unloaded_W']
                 and measured['padz'][RF]<-.012 and local[2]<-p['depression_native'])
        if self.onset is None:
            if not missing:
                return base
            self.onset=t
        if t>=p['second_swing_ms']+p['second_release_ms']:
            return base
        current=self.offset(t)
        follow=self.offset(t+5)
        # Remove only planned actuator offset while evaluating the frozen nearest
        # gait pattern in consistent joint coordinates. No root changes occur.
        neutralized=x.copy()
        scales=self.unscaled_halfspan
        neutralized[:42]-=current/scales
        neutralized[126:168]-=current/scales
        command=self.base(neutralized,1)+(follow-current)/(.005*stan.physics.MAX_RATE)
        return np.clip(command,-1.,1.)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('scenario',type=Path)
    p.add_argument('plan',type=Path)
    p.add_argument('--native',action='store_true')
    p.add_argument('--probe-ms',type=int)
    a=p.parse_args()
    import gap_demo
    gap_demo.Teacher=Bias
    run(a.output,'teacher',a.scenario,native=a.native,probe_ms=a.probe_ms,
        teacher_config=a.plan,policy_source=Path(__file__))


if __name__=='__main__':
    main()
