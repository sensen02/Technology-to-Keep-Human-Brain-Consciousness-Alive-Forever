"""Teacher-only phase-preserving target feedback with true retraction and placement."""
import mujoco as mj
import numpy as np

from gap_servo import Servo
from physical import RF,RM,RF_JOINTS,RM_JOINTS,stan


def smooth(x):
    t=np.clip(x,0.,1.)
    return t*t*(3.-2.*t)


class Anchored(Servo):
    def __call__(self,t,x,intent,local,base,measured):
        if not intent:
            return base
        p,c=self.p,self.s.body
        missing=(local[0]>=p['expected_load_W'] and local[1]<=p['unloaded_W']
                 and local[2]<-p['depression_native'] and measured['padz'][RF]<-.012)
        if self.onset is None:
            if not missing:
                return base
            self.onset=t
            self.anchor=self.position(RF).copy()
            self.anchor[0]+=p.get('shift_x',0.)
            self.anchor[1]=p['landing_y']
            self.rm_anchor=self.position(RM).copy()
            self.old_offset=None
            self.reload_time=None
        age=t-self.onset
        nominal=self.s.nominal[t+5]
        raw=(nominal-c.actuator.target_rad)/(.005*stan.physics.MAX_RATE)
        if t>=p['phase_end_ms']:
            return base
        if self.stage==3:
            blend=smooth((t-self.reload_time)/p.get('stance_blend_ms',120.))
            release=1.-smooth((t-p['next_swing_ms'])/p.get('swing_release_ms',280.))
            posture=nominal[RF_JOINTS]+self.old_offset*release
            keeper=(posture-c.actuator.target_rad[RF_JOINTS])/(.005*stan.physics.MAX_RATE)
            anchor=self.anchor.copy()
            anchor[2]=p['stance_z']
            raw[RF_JOINTS]=blend*keeper+(1.-blend)*self.joint_rate(RF,self.position(RF),anchor)
        else:
            rf=self.position(RF)
            if self.stage==0 and rf[2]>=p['lift_z']-.009:
                self.stage=1
            if self.stage==1 and np.linalg.norm(rf[:2]-self.anchor[:2])<.014:
                self.stage=2
            if self.stage==2 and measured['loads'][RF]>.036*stan.physics.W and rf[2]>.001:
                self.stage=3
                self.reload_time=t
                self.old_offset=c.actuator.target_rad[RF_JOINTS]-self.s.nominal[t,RF_JOINTS]
            if self.stage==0:
                goal=np.r_[rf[:2],p['lift_z']]
            else:
                goal=self.anchor.copy()
                goal[2]=p['lift_z'] if self.stage==1 else p['stance_z']
            raw[RF_JOINTS]=self.joint_rate(RF,rf,goal,cap=p.get('vmax',.42))
        if p['support_native']>0 and age<p['support_hold_ms']:
            rm=self.position(RM)
            # Preserve ordinary foot horizontal target while selecting a calibrated
            # small downward offset to transfer actual load.
            self.ref.qpos[:]=self.s.nominal_qpos[t+5]
            mj.mj_fwdPosition(c.plant,self.ref)
            ids=self.pads[RM]
            pm=self.ref.geom_xpos[ids].mean(0)
            mj.mj_jac(c.plant,self.ref,self.jac,None,pm,int(c.plant.geom_bodyid[ids[0]]))
            j=self.jac[:,c.a_dof[RM_JOINTS]]
            ramp=smooth(age/85.)*(1.-smooth((age-p['support_hold_ms']+160.)/160.))
            d=j.T@np.linalg.solve(j@j.T+.014*np.eye(3),np.array([0.,0.,-p['support_native']*ramp]))
            raw[RM_JOINTS]=(nominal[RM_JOINTS]+d-c.actuator.target_rad[RM_JOINTS])/(.005*stan.physics.MAX_RATE)
        return np.clip(raw,-1.,1.)
