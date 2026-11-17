"""Release the previously unloaded helper after a verified early load pulse."""
import argparse
from pathlib import Path

import numpy as np

import support_experiment as rig
from support_experiment import ARCHIVE, HERE, W, group
from support_hold import Hold


class Pulse(Hold):
    def __init__(self,*args):
        super().__init__(*args)
        self.contact_frames = 0
        self.return_begin = None

    def __call__(self,t,x,measured,nominal,intended):
        action = super().__call__(t,x,measured,nominal,intended)
        if self.helper_first is None:
            return action
        p = self.plan
        h = group(self.ids[1])
        if self.return_begin is None:
            self.contact_frames = self.contact_frames+1 if measured["loads"][self.ids[1]]>=.02*W else 0
            # Same recovery policy, short intervention only. Capturing the current
            # target avoids any jump in the commanded reset of the helper.
            if self.contact_frames >= p["pulse_contact_samples"] or t-self.helper_first >= p["pulse_max_ms"]:
                self.return_begin = t
                self.return_target = self.core.actuator.target_rad[h].copy()
        if self.return_begin is not None:
            fraction = np.clip((t+5-self.return_begin)/p["pulse_return_ms"],0.,1.)
            eased = fraction*fraction*(3.-2.*fraction)
            goal = (1.-eased)*self.return_target+eased*self.trigger_joint_targets[h]
            action[h] = self.setpoint_rate(h,goal)
        return action


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("on","off","control"))
    p.add_argument("output",type=Path)
    p.add_argument("--case",type=Path,default=HERE/"CASE_C.json")
    p.add_argument("--plan",type=Path,default=HERE/"PLAN_PULSE.json")
    p.add_argument("--reference",type=Path,default=ARCHIVE/"TRACE.npz")
    a=p.parse_args()
    rig.Teacher=Pulse
    rig.run(a.mode,a.output.resolve(),a.case.resolve(),a.plan.resolve(),a.reference.resolve())
