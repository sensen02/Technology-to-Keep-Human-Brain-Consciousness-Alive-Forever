"""Explicit finite joint-PD variant for incremental targets; no gait or root control.

Input remains a bounded42-vector target increment rate. The derivative reference
is the actual rate of the clipped target integrator, not a clock/reference gait.
All native symmetric torque caps and joint bounds come from the original class.
"""
import sys
from pathlib import Path
import numpy as np
import mujoco as mj
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-incremental/20261107T010000Z')
sys.path.insert(0,str(BASE))
from incremental_actuator import IncrementalActuator


class RateTrackingServo(IncrementalActuator):
    gain_scale=32.
    damping_mode='same'
    damping_ratio=.7

    def __init__(self,model,data,**kwargs):
        # Runtime supplies the historical scale2 gains. Record the change at the
        # actuator interface, keeping model/gear/native torque caps untouched.
        kwargs['kp']=np.asarray(kwargs['kp'])*(self.gain_scale/2.)
        if self.damping_mode=='same':
            kwargs['kd']=np.asarray(kwargs['kd'])*np.sqrt(self.gain_scale/2.)
            self.initial_joint_inertia=None
        elif self.damping_mode=='critical':
            inertia=np.empty((model.nv,model.nv))
            mj.mj_fullM(model,inertia,data.qM)
            jids=model.actuator_trnid[:,0]
            addresses=model.jnt_dofadr[jids]
            self.initial_joint_inertia=np.diag(inertia)[addresses].copy()
            assert np.all(self.initial_joint_inertia>0) and np.isfinite(inertia).all()
            kwargs['kd']=2*self.damping_ratio*np.sqrt(kwargs['kp']*self.initial_joint_inertia)
        else:raise ValueError(self.damping_mode)
        super().__init__(model,data,**kwargs)

    def torque(self,data):
        q,v=self._pose(data)
        desired_velocity=self.last_effective_rate_rad_s
        raw=self.kp*(self.target_rad-q)+self.kd*(desired_velocity-v)
        assert np.isfinite(raw).all()
        return np.clip(raw,-self.torque_limit,self.torque_limit)
