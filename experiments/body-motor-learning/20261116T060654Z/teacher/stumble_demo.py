"""A FlyGym-style contact-triggered leg lift demonstration for the limited plant."""
import mujoco as mj
import numpy as np

from physical import RF, RF_JOINTS, stan


def smooth(progress):
    x=np.clip(progress,0.,1.)
    return x*x*(3.-2.*x)


class Teacher:
    def __init__(self, sensor, plan, baseline):
        self.s = sensor
        self.plan = plan
        self.baseline = baseline
        self.onset = None
        self.reference = mj.MjData(sensor.body.plant)
        self.jac = np.zeros((3,sensor.body.plant.nv))
        self.halfspan=np.maximum(np.ptp(sensor.body.actuator.target_bounds_rad,axis=1)*.5,.1)

    def __call__(self, t, x, intent, local, base, measured):
        if not intent:
            return base
        p,c=self.plan,self.s.body
        opposing=max(0.,float(-local[:9].reshape(3,3)[:,0].min()))
        if self.onset is None and opposing>p['trigger_W'] and measured['loads'][RF]<.08*stan.physics.W:
            self.onset=t
        if self.onset is None:
            return base
        age=t-self.onset
        if age>p['rise_ms']+p['hold_ms']+p['fade_ms']+80:
            return base
        until=t+5
        age=until-self.onset
        lift=smooth(age/p['rise_ms'])*(1.-smooth((age-p['rise_ms']-p['hold_ms'])/p['fade_ms']))
        reference=self.s.nominal[until]
        self.reference.qpos[:]=self.s.nominal_qpos[until]
        m=c.plant
        mj.mj_fwdPosition(m,self.reference)
        toe_geom=self.s.pads[0]
        position=self.reference.geom_xpos[self.s.pads].mean(0)
        mj.mj_jac(m,self.reference,self.jac,None,position,int(m.geom_bodyid[toe_geom]))
        jac=self.jac[:,c.a_dof[RF_JOINTS]]
        delta=lift*np.array([p.get('shift_x',0.),p.get('shift_y',0.),p['lift_native']])
        desired=reference[RF_JOINTS]+jac.T @ np.linalg.solve(jac@jac.T+.006*np.eye(3),delta)
        # Keep the other five legs on their original observation-based support
        # coordination while the impacted leg runs the corrective target path.
        coordination=p.get('coordination','archived_support')
        a=base.copy()
        if coordination!='live_support':
            isolated=x.copy()
            ix=np.arange(126+RF_JOINTS.start,126+RF_JOINTS.stop)
            departure=(local[12:19] if coordination=='observable_support'
                       else c.actuator.target_rad[RF_JOINTS]-self.s.nominal[t,RF_JOINTS])
            isolated[ix]-=departure/self.halfspan[RF_JOINTS]
            a=self.baseline(isolated,intent).copy()
        a[RF_JOINTS]=(desired-c.actuator.target_rad[RF_JOINTS])/(.005*stan.physics.MAX_RATE)
        return np.clip(a,-1.,1.)
