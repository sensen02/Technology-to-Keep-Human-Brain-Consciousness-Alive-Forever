"""Wider-radius LEFT teacher, horizontal targets only; no physics on import.

Reuse frozen finite SE2 mathematics by absolute path. Original stride.36 base
reference/time/step/IK stay unchanged. Ramp stride by scaling horizontal delta,
including qdot in diagnostic velocity. No root or physical pose writes exist.
"""
from pathlib import Path
import importlib.util
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
FINITE_SOURCE=HERE.parent/'left_finite_twist'/'turning_teacher.py'
_spec=importlib.util.spec_from_file_location('_wider_radius_finite_math',FINITE_SOURCE)
_finite=importlib.util.module_from_spec(_spec)
sys.modules[_spec.name]=_finite
_spec.loader.exec_module(_finite)
BASELINE=_finite.BASELINE
pin_baseline=_finite.pin_baseline
finite_xy_velocity=_finite.finite_xy_velocity


class WiderRadiusReference:
    """Teacher-only stride ramp; command scale is nominal v/.075, not clock scale."""
    def __init__(self,base,speed_scale=1.,yaw_rate=.06,prime_s=4.8,ramp_s=1.):
        if speed_scale!=1. or yaw_rate!=.06 or prime_s!=4.8 or ramp_s!=1.:
            raise ValueError('Only preregistered LEFT+.06, speed_scale1, prime4.8/ramp1 allowed')
        if base.stride_native!=.36 or base.step_s!=.8 or base.lift_native!=.12:
            raise ValueError('Base .36 stride/.8 step/.12 lift must remain unchanged')
        self.base=base
        self.rest=np.array(base.rest_body_feet[:,:2],copy=True)
        self.speed_scale,self.yaw_rate,self.prime_s,self.ramp_s=1.,.06,4.8,1.
        self.metadata=dict(label='algorithm teacher wider-radius finite SE2 horizontal targets',
            prime_stride=.36,final_stride=.48,step_s=.8,prime_s=4.8,ramp_s=1.,
            nominal_v_initial=.075,nominal_v_final=.1,requested_yaw_final=.06,curvature_final=.6,
            blend='b=u^2*(3-2u),u=clip(t-4.8,0,1)',
            stride_scale='q=1+b/3; horizontal delta=q*(base.target_xy-rest)',
            scaled_velocity='q*base.velocity_xy+qdot*(base.target_xy-rest)',
            commands='v=.075+.025*b; omega=.06*b; k=omega/v; kdot=.0045*bdot/v^2',
            phase='t exactly; no time warping; step .8 unchanged',
            velocity='analytic qdot and kdot chain derivative using stable finite SE2 helper',
            changed_fields=['target_xy','velocity_xy'],nonhorizontal_fields_unchanged=True,
            physical_root_writes=False,physical_pose_writes=False,extra_forces=False,
            base_obs_target_speed_constant=.075,base_obs_note='historical constant feature only, not current nominal desired speed',
            future_ANN_forward_domain='new diagnostic normalizedforward v/.075-1 reaches1/3; not deployed/qualified')

    def kinematics(self,t):
        t=float(t)
        if not np.isfinite(t):raise ValueError('Finite elapsed time required')
        u=float(np.clip((t-self.prime_s)/self.ramp_s,0.,1.))
        b=u*u*(3.-2.*u)
        bdot=6.*u*(1.-u)/self.ramp_s
        q=1.+b/3.;qdot=bdot/3.
        v=.075+.025*b;omega=.06*b
        k=omega/v;kdot=.0045*bdot/(v*v)
        return dict(blend=b,blend_dot=bdot,q=q,qdot=qdot,stride=.36+.12*b,
                    nominal_v=v,nominal_omega=omega,k=k,kdot=kdot)

    def command(self,t):
        z=self.kinematics(t)
        return float(t),z['q'],z['k']

    def sample(self,t):
        z=self.kinematics(t)
        sample=self.base.sample(float(t))
        if z['blend']==0.:
            return sample  # whole original straight sample, bit-for-bit prefix
        delta=sample['target_xy']-self.rest
        xy=self.rest+z['q']*delta
        velocity=z['q']*sample['velocity_xy']+z['qdot']*delta
        sample['target_xy'],sample['velocity_xy']=finite_xy_velocity(self.rest,xy,velocity,z['k'],z['kdot'])
        return sample


def make_teacher(core,*,speed_scale=1.,yaw_rate=.06,prime_s=4.8):
    pin_baseline()
    from static_teacher import StaticTeacher
    from student_wave import teacher_config
    teacher=StaticTeacher(core,teacher_config(100))
    teacher.reference=WiderRadiusReference(teacher.reference,speed_scale,yaw_rate,prime_s)
    teacher.metadata['conditional_twist']=teacher.reference.metadata
    return teacher
