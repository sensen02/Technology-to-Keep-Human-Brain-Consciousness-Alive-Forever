"""Diagnostic safe stop and actual right front foot replant from a missing foothold."""
import argparse
from pathlib import Path

import numpy as np

from physical import RF,RM,RF_JOINTS,RM_JOINTS,read,run,stan
from gap_servo import Servo


class Brake(Servo):
    def __init__(self,sensor,plan):
        super().__init__(sensor,plan)
        self.reload_seen=None

    def __call__(self,t,x,intent,local,base,m):
        if self.onset is None:
            if not intent or not (local[0]>=.004 and local[1]<=.004 and m['padz'][RF]<-.012):
                return base
            self.onset=t
            self.anchor=self.position(RF).copy()
            self.anchor[0]-=self.p['back_native']
            self.anchor[2]=self.p['stance_z']
            self.rm_anchor=self.position(RM).copy()
        age=t-self.onset
        action=self.base(x,0).copy()
        p=self.p
        if self.reload_seen is not None and t-self.reload_seen>p['postcontact_ms']:
            return action
        rf=self.position(RF)
        if self.stage==0 and m['padz'][RF]>.003:
            self.stage=1
        if self.stage==1 and np.linalg.norm(rf[:2]-self.anchor[:2])<.018:
            self.stage=2
        if self.stage==2 and m['loads'][RF]>.03*stan.physics.W and self.reload_seen is None:
            self.reload_seen=t
        if self.stage==0:
            # Constrain x/y in the upward phase enough to avoid grazing the lip.
            goal=np.r_[rf[:2],p['lift_z']]
        else:
            goal=self.anchor.copy()
            if self.stage==1:
                goal[2]=p['lift_z']
        action[RF_JOINTS]=self.joint_rate(RF,rf,goal,cap=.65)
        if age<p['support_hold_ms'] and p['support_native']>0:
            rm=self.position(RM)
            helper=rm.copy()
            helper[2]=self.rm_anchor[2]-p['support_native']*min(1.,age/75.,(p['support_hold_ms']-age)/90.)
            # Add a small force-generating target rate atop normal load-bearing stop,
            # while its other leg joints retain the same stopping target.
            action[RM_JOINTS]+=self.joint_rate(RM,rm,helper,cap=.18)
        return np.clip(action,-1.,1.)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('scenario',type=Path)
    p.add_argument('plan',type=Path)
    p.add_argument('--native',action='store_true')
    p.add_argument('--probe-ms',type=int,default=6450)
    a=p.parse_args()
    import gap_demo
    gap_demo.Teacher=Brake
    run(a.output,'teacher',a.scenario,native=a.native,probe_ms=a.probe_ms,
        teacher_config=a.plan,policy_source=Path(__file__))


if __name__=='__main__':
    main()
