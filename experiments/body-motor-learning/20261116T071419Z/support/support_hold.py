"""Hold already loaded feet during one missing-foothold response."""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np

import support_experiment as rig
from support_experiment import ARCHIVE, HERE, W, group, stan


class Hold(rig.Teacher):
    def theta_difference(self,i,offset):
        joint = group(self.ids[i])
        c = self.core
        point = self.pose(i)
        geom = self.pads[i][0]
        mj.mj_jac(c.plant,c.d,self.jac,None,point,int(c.plant.geom_bodyid[geom]))
        jac = self.jac[:,c.a_dof[joint]]
        return jac.T@np.linalg.solve(jac@jac.T+.004*np.eye(3),offset)

    def setpoint_rate(self,j,target,max_channel=1.0):
        return np.clip((target-self.core.actuator.target_rad[j])/(.005*stan.physics.MAX_RATE),-max_channel,max_channel)

    def __call__(self,t,x,measured,nominal,intended):
        p = self.plan
        missing, available = self.ids
        h, d = group(available),group(missing)
        if self.first is None:
            if not intended:
                return nominal
            site = self.pose(0)
            x0,x1,y0,y1 = self.opening
            if not (x0<site[0]<x1 and y0<site[1]<y1 and self.expected[t]>=p["expected_load_W"]
                    and measured["loads"][missing]<=p["target_unloaded_W"]*W
                    and measured["padz"][missing]<-p["below_rim_native"]):
                return nominal
            self.first = t
            self.target_anchor = site.copy()
            self.target_anchor[1] = y1+.028
            self.target_anchor[2] = p["target_stance_z"]
            self.helper_available = bool(measured["loads"][available]==0 and measured["padz"][available]>0)
            self.available_at_trigger = self.helper_available
            self.trigger_joint_targets = self.core.actuator.target_rad.copy()
            self.lift_joint_target = self.trigger_joint_targets[d]+self.theta_difference(0,np.array([0.,0.,.100]))
            self.helper_ref = self.trigger_joint_targets[h]+self.theta_difference(1,np.array([0.,0.,-p["helper_target_down_native"]]))
        # Holding current targets under finite actuators keeps the support polygon
        # in place while the two active joint groups are tested.
        action = np.zeros_like(nominal)
        lost = self.pose(0)
        if self.stage==0 and measured["padz"][missing]>.006:
            self.stage=1
        if self.stage==1 and abs(lost[1]-self.target_anchor[1])<.010:
            self.stage=2
        if self.stage==2 and measured["loads"][missing]>.02*W and self.replanted is None:
            self.replanted=t
        if self.replanted is None or t-self.replanted<=p["postcontact_ms"]:
            if self.stage==0:
                action[d]=self.setpoint_rate(d,self.lift_joint_target)
            else:
                goal=self.target_anchor.copy()
                if self.stage==1:
                    goal[2]=p["target_lift_z"]
                action[d]=self.rate(0,goal,p["target_gain"],.8)
        if self.enabled and self.helper_available and t-self.first<p["helper_hold_ms"]:
            if self.helper_first is None:
                self.helper_first = t
            action[h]=self.setpoint_rate(h,self.helper_ref)
        return np.clip(action,-1.,1.)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("on","off","control"))
    p.add_argument("output",type=Path)
    p.add_argument("--case",type=Path,default=HERE/"CASE_C.json")
    p.add_argument("--plan",type=Path,default=HERE/"PLAN_HOLD.json")
    p.add_argument("--reference",type=Path,default=ARCHIVE/"TRACE.npz")
    a=p.parse_args()
    rig.Teacher=Hold
    rig.run(a.mode,a.output.resolve(),a.case.resolve(),a.plan.resolve(),a.reference.resolve())
