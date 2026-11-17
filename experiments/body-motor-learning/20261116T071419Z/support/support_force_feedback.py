"""Feedback on the unloaded helper's actual acquired ground force."""
import argparse
from pathlib import Path

import numpy as np

import support_experiment as rig
from support_experiment import ARCHIVE, HERE, W, group
from support_hold import Hold


class ForceFeedback(Hold):
    def __init__(self,*a):
        super().__init__(*a)
        self.feedback_age = 0
        self.bearing = 0
        self.release = None

    def __call__(self,t,x,measured,nominal,intended):
        a = super().__call__(t,x,measured,nominal,intended)
        if self.helper_first is None:
            return a
        i = self.ids[1]
        j = group(i)
        p = self.plan
        init = self.trigger_joint_targets[j]
        if self.feedback_age==0:
            self.feedback_depth = p["helper_target_down_native"]
            self.direction = (self.helper_ref-init)/self.feedback_depth
        self.feedback_age += 1
        measured_load = measured["loads"][i]/W
        if self.release is None:
            self.bearing = self.bearing+1 if measured_load>=.0205 else 0
            if self.bearing>=p["hold_bearing_ticks"] or t-self.helper_first>=p["max_feedback_ms"]:
                self.release = t
                self.depth_release = self.feedback_depth
            else:
                step = p["feedback_gain_native_per_W"]*(p["lf_load_goal_W"]-measured_load)
                self.feedback_depth = np.clip(self.feedback_depth+step,.04,p["feedback_max_depth"])
        if self.release is not None:
            ptime = np.clip((t+5-self.release)/p["fade_ms"],0.,1.)
            delta = ptime*ptime*(3.-2.*ptime)
            self.feedback_depth = (1.-delta)*self.depth_release+delta*p["return_down_native"]
        a[j] = self.setpoint_rate(j,init+self.feedback_depth*self.direction)
        return a


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("on","off","control"))
    p.add_argument("output",type=Path)
    p.add_argument("--case",type=Path,default=HERE/"CASE_C.json")
    p.add_argument("--plan",type=Path,default=HERE/"PLAN_FORCE_FEEDBACK.json")
    p.add_argument("--reference",type=Path,default=ARCHIVE/"TRACE.npz")
    a=p.parse_args()
    rig.Teacher=ForceFeedback
    rig.run(a.mode,a.output.resolve(),a.case.resolve(),a.plan.resolve(),a.reference.resolve())
