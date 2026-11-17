"""Physical retraction, alternative landing, and mid-leg support teacher."""
import mujoco as mj
import numpy as np

from physical import RF,RM,RF_JOINTS,RM_JOINTS,LEGS,stan


def smooth(progress):
    t=np.clip(progress,0.,1.)
    return t*t*(3.-2.*t)


class Teacher:
    def __new__(cls,sensor,plan):
        if plan.get('anchored'):
            from gap_anchored import Anchored
            return Anchored(sensor,plan)
        if plan.get('servo'):
            from gap_servo import Servo
            return Servo(sensor,plan)
        return super().__new__(cls)

    def __init__(self, sensor, plan):
        self.s=sensor
        self.plan=plan
        self.onset=None
        self.reference=mj.MjData(sensor.body.plant)
        self.jac=np.zeros((3,sensor.body.plant.nv))
        self.channels={RF:RF_JOINTS,RM:RM_JOINTS}
        self.pads={i:np.array(sensor.body.pads[LEGS[i]]) for i in self.channels}

    def joint_offset(self,leg,linear):
        c=self.s.body
        ids=self.pads[leg]
        x=self.reference.geom_xpos[ids].mean(0)
        mj.mj_jac(c.plant,self.reference,self.jac,None,x,int(c.plant.geom_bodyid[ids[0]]))
        block=self.jac[:,c.a_dof[self.channels[leg]]]
        return block.T @ np.linalg.solve(block@block.T+.006*np.eye(3),linear)

    def __call__(self,t,x,intent,local,base,measured):
        if not intent:
            return base
        p,c=self.plan,self.s.body
        missing=(local[0]>=p['expected_load_W'] and local[1]<=p['unloaded_W']
                 and local[2]<-p['depression_native'])
        if self.onset is None and missing:
            self.onset=t
        if self.onset is None:
            return base
        age=t+5-self.onset
        rise=smooth(age/p['rise_ms'])
        lower=1.-smooth((age-p['lower_start_ms'])/p['lower_ms'])
        lateral=smooth((age-p['lateral_start_ms'])/p['lateral_ms'])
        remove=1.-smooth((t+5-p['next_swing_ms'])/p['next_swing_release_ms'])
        if remove<=0.:
            return base
        support=smooth(age/p['support_rise_ms'])*(1.-smooth((age-p['support_hold_ms'])/p['support_release_ms']))
        nominal=self.s.nominal[t+5]
        self.reference.qpos[:]=self.s.nominal_qpos[t+5]
        mj.mj_fwdPosition(c.plant,self.reference)
        target_rf=nominal[RF_JOINTS]+self.joint_offset(RF,
            np.array([lateral*remove*p['shift_x'],lateral*remove*p['shift_y'],rise*lower*p['lift_native']]))
        action=base.copy()
        action[RF_JOINTS]=(target_rf-c.actuator.target_rad[RF_JOINTS])/(.005*stan.physics.MAX_RATE)
        if p['support_native']>0 and support>0:
            target_rm=nominal[RM_JOINTS]+self.joint_offset(RM,np.array([0.,0.,-support*p['support_native']]))
            action[RM_JOINTS]=(target_rm-c.actuator.target_rad[RM_JOINTS])/(.005*stan.physics.MAX_RATE)
        return np.clip(action,-1.,1.)
