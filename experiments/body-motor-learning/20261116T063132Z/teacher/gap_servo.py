"""Teacher with upper-floor reach and observed stance-foot anchoring."""
import mujoco as mj
import numpy as np

from physical import RF,RM,RF_JOINTS,RM_JOINTS,LEGS,stan
from prototype import policy as frozen_policy


class Servo:
    def __init__(self,sensor,plan):
        self.s=sensor
        self.p=plan
        self.onset=None
        self.anchor=None
        self.stage=0
        self.ref=mj.MjData(sensor.body.plant)
        self.jac=np.zeros((3,sensor.body.plant.nv))
        self.pads={i:np.array(sensor.body.pads[LEGS[i]]) for i in (RF,RM)}
        self.groups={RF:RF_JOINTS,RM:RM_JOINTS}
        self.base=frozen_policy()
        c=sensor.body
        self.halfspan=np.maximum(np.ptp(c.actuator.target_bounds_rad,axis=1)*.5,.1)
        self.unscaled_halfspan=np.maximum(np.ptp(c.plant.jnt_range[c.actuator.joint_ids],axis=1)*.5,.1)

    def position(self,leg):
        return self.s.body.d.geom_xpos[self.pads[leg]].mean(0)

    def joint_rate(self,leg,site,goal,cap=.55):
        c=self.s.body
        group=self.groups[leg]
        ids=self.pads[leg]
        mj.mj_jac(c.plant,c.d,self.jac,None,site,int(c.plant.geom_bodyid[ids[0]]))
        j=self.jac[:,c.a_dof[group]]
        error=goal-site
        v=np.clip(self.p.get('gain',12.)*error,-cap,cap)
        joint=j.T@np.linalg.solve(j@j.T+.012*np.eye(3),v)
        return joint/stan.physics.MAX_RATE

    def nominalize(self,x,base,t,use_support):
        if not self.p['isolate']:
            return base.copy()
        c=self.s.body
        self.ref.qpos[:]=self.s.nominal_qpos[t]
        temp=x.copy()
        groups=(RF_JOINTS,RM_JOINTS) if use_support else (RF_JOINTS,)
        for g in groups:
            target_delta=c.actuator.target_rad[g]-self.s.nominal[t,g]
            pos_delta=c.d.qpos[c.a_qpos[g]]-self.ref.qpos[c.a_qpos[g]]
            temp[126+g.start:126+g.stop]-=target_delta/self.halfspan[g]
            temp[g]-=pos_delta/self.unscaled_halfspan[g]
        return self.base(temp,1)

    def __call__(self,t,x,intent,local,base,measured):
        if not intent:
            return base
        p=self.p
        missing=(local[0]>=p['expected_load_W'] and local[1]<=p['unloaded_W']
                 and local[2]<-p['depression_native'])
        if self.onset is None:
            if not missing:
                return base
            self.onset=t
            self.anchor=self.position(RF).copy()
            self.anchor[0]+=p['shift_x']
            self.anchor[1]=p['landing_y']
            self.rm_anchor=self.position(RM).copy()
        age=t-self.onset
        if t>=p['next_swing_ms']:
            return base
        rf=self.position(RF)
        if self.stage==0 and rf[2]>=p['lift_z']-.013:
            self.stage=1
        if self.stage==1 and abs(rf[1]-self.anchor[1])<.017:
            self.stage=2
        helper=p['support_native']>0 and age<p['support_hold_ms']
        a=self.nominalize(x,base,t,helper)
        if self.stage==0:
            goal=np.r_[rf[:2],p['lift_z']]
        else:
            goal=self.anchor.copy()
            goal[2]=p['lift_z'] if self.stage==1 else p['stance_z']
        a[RF_JOINTS]=self.joint_rate(RF,rf,goal)
        if helper:
            rm=self.position(RM)
            goal_rm=self.rm_anchor.copy()
            env=min(1.,age/65.,(p['support_hold_ms']-age)/100.)
            goal_rm[2]-=p['support_native']*env
            a[RM_JOINTS]=self.joint_rate(RM,rm,goal_rm,cap=.21)
        return np.clip(a,-1.,1.)
