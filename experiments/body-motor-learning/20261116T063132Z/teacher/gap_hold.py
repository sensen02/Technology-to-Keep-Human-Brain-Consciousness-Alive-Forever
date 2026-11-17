"""Finite-rate localized recovery while pausing the gait reference."""
import argparse
from pathlib import Path
import time

import mujoco as mj
import numpy as np

from physical import RM,RF,RM_JOINTS,RF_JOINTS,Local,read,run,stan
from gap_servo import Servo


def smooth(x):
    u=np.clip(x,0.,1.)
    return u*u*(3.-2.*u)


class Hold(Servo):
    def __init__(self,sensor,plan):
        super().__init__(sensor,plan)
        self.reloaded_at=None
        self.resume=None
        self.resume_motor=None

    def __call__(self,t,x,intent,local,base,m):
        if not intent:
            return base
        p,c=self.p,self.s.body
        if self.onset is None:
            if not (local[0]>=p['expected_load_W'] and local[1]<=p['unloaded_W']
                    and m['padz'][RF]<-.012):
                return base
            self.onset=t
            self.anchor=self.position(RF).copy()
            self.anchor[0]+=p['shift_x']
            self.anchor[1]=p['landing_y']
            self.rm_anchor=self.position(RM).copy()
            self.hold=c.actuator.target_rad.copy()
            self.ref.qpos[:]=c.d.qpos
            mj.mj_fwdPosition(c.plant,self.ref)
        age=t-self.onset
        if self.resume is not None:
            effective=t-self.resume+self.onset
            target=self.s.nominal[effective+5].copy()
            phase=1.-smooth((effective-p['next_swing_ms'])/p['swing_release_ms'])
            target[RF_JOINTS]+=phase*self.resume_motor[RF_JOINTS]
            return np.clip((target-c.actuator.target_rad)/(.005*stan.physics.MAX_RATE),-1.,1.)
        action=(self.hold-c.actuator.target_rad)/(.005*stan.physics.MAX_RATE)
        rf=self.position(RF)
        if self.stage==0 and rf[2]>=p['lift_z']-.008:
            self.stage=1
        if self.stage==1 and np.linalg.norm(rf[:2]-self.anchor[:2])<.012:
            self.stage=2
        if self.stage==2 and m['loads'][RF]>.027*stan.physics.W and self.reloaded_at is None:
            self.reloaded_at=t
        if self.reloaded_at is not None and t-self.reloaded_at>=p['stand_before_resume_ms']:
            self.resume=t
            self.resume_motor=c.actuator.target_rad-self.s.nominal[self.onset]
        if self.stage==0:
            goal=np.r_[rf[:2],p['lift_z']]
        else:
            goal=self.anchor.copy()
            goal[2]=p['lift_z'] if self.stage==1 else p['stance_z']
        action[RF_JOINTS]=self.joint_rate(RF,rf,goal,cap=p.get('vmax',.57))
        if p['support_native']>0:
            rm=self.position(RM)
            helper=self.rm_anchor.copy()
            helper[2]-=p['support_native']*smooth(age/100.)
            action[RM_JOINTS]=self.joint_rate(RM,rm,helper,cap=.20)
        return np.clip(action,-1.,1.)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('scenario',type=Path)
    p.add_argument('plan',type=Path)
    p.add_argument('--native',action='store_true')
    p.add_argument('--probe-ms',type=int)
    a=p.parse_args()
    # Import-gap_demo.py keeps previous variants immutable. This process replaces
    # the class binding only to pass the normal teacher policy call.
    import gap_demo
    gap_demo.Teacher=Hold
    run(a.output,'teacher',a.scenario,native=a.native,probe_ms=a.probe_ms,
        teacher_config=a.plan,policy_source=Path(__file__))


if __name__=='__main__':
    main()
