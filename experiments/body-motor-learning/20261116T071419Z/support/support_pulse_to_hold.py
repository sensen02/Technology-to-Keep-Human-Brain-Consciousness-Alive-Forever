"""Early alternate landing pulse followed by the stable mild helper hold."""
import argparse
from pathlib import Path

import numpy as np

import support_experiment as rig
from support_experiment import ARCHIVE, HERE, group
from support_pulse import Pulse


class ReturnToHold(Pulse):
    def __call__(self,t,x,measured,nominal,intended):
        a = super().__call__(t,x,measured,nominal,intended)
        if self.return_begin is None:
            return a
        h = group(self.ids[1])
        plan = self.plan
        mild = self.trigger_joint_targets[h]+(self.helper_ref-self.trigger_joint_targets[h])*(
            plan["return_down_native"]/plan["helper_target_down_native"])
        u = np.clip((t+5-self.return_begin)/plan["pulse_return_ms"],0.,1.)
        smooth = u*u*(3.-2.*u)
        a[h] = self.setpoint_rate(h,(1.-smooth)*self.return_target+smooth*mild)
        return a


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("on","off","control"))
    p.add_argument("output",type=Path)
    p.add_argument("--case",type=Path,default=HERE/"CASE_C.json")
    p.add_argument("--plan",type=Path,default=HERE/"PLAN_PULSE_TO_50.json")
    p.add_argument("--reference",type=Path,default=ARCHIVE/"TRACE.npz")
    a=p.parse_args()
    rig.Teacher=ReturnToHold
    rig.run(a.mode,a.output.resolve(),a.case.resolve(),a.plan.resolve(),a.reference.resolve())
