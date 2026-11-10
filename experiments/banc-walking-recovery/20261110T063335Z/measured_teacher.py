"""Measured-foot-reference training teacher. No root/pose writes or auxiliary forces.

This algorithm is a declared teacher only. All motion is produced through bounded
42-joint rate commands, existing target integration, finite PD and physical contact.
"""
from dataclasses import asdict,dataclass
import numpy as np
import mujoco as mj
from scipy.optimize import lsq_linear
from physics import LEGS,MAX_RATE
from reference import FootReference


@dataclass(frozen=True)
class TeacherConfig:
    scale: float=10.
    vertical_gain: float=.25
    time_warp: float=10.
    source_duration_s: float=.24
    blend_s: float=.5
    tau_xy_s: float=.08
    tau_z_s: float=.08
    height_gain: float=2.
    height_damping: float=.2
    orientation_gain: float=1.
    orientation_damping: float=.1
    vertical_velocity_limit: float=2.
    body_velocity_limit: float=.5
    ridge: float=.01
    posture_gain: float=.05
    command_dt_s: float=.005


def skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def leg_indices(c):
    ids=[np.array([i for i,j in enumerate(c.actuator.joint_ids)
                  if c.plant.joint(int(j)).name.startswith('joint_'+leg)],int) for leg in LEGS]
    if any(len(a)!=7 for a in ids) or sorted(np.concatenate(ids).tolist())!=list(range(42)):
        raise ValueError('seven distinct actuated joints per leg required')
    return ids


def foot_jacobians(c,leg):
    jp,jr=np.zeros((3,c.plant.nv)),np.zeros((3,c.plant.nv))
    mj.mj_jac(c.plant,c.d,jp,jr,c.d.xpos[c.th],c.th)
    pads=c.pads[LEGS[leg]]
    point=np.mean([c.d.geom_xpos[g] for g in pads],axis=0)
    world=np.zeros_like(jp)
    for g in pads:
        j=np.zeros_like(jp)
        mj.mj_jac(c.plant,c.d,j,None,c.d.geom_xpos[g],int(c.plant.geom_bodyid[g]))
        world+=j/len(pads)
    R=c.d.xmat[c.th].reshape(3,3)
    relative=R.T@(world-jp+skew(point-c.d.xpos[c.th])@jr)
    # Horizontal metric is the pad-centre mean; clearance is the lowest sphere
    # bottom. Use its active-geometry vertical derivative, not the centroid's.
    lowest=min(pads,key=lambda g:float(c.d.geom_xpos[g,2]-c.plant.geom_size[g,0]))
    bottom_jac=np.zeros_like(jp)
    mj.mj_jac(c.plant,c.d,bottom_jac,None,c.d.geom_xpos[lowest],int(c.plant.geom_bodyid[lowest]))
    world_bottom=world.copy();world_bottom[2]=bottom_jac[2]
    return relative,world_bottom,point


def bounded_rate(A,b,bias,lower,upper,ridge):
    augmented=np.vstack([A,np.sqrt(ridge)*np.eye(len(bias))])
    target=np.r_[b,np.sqrt(ridge)*bias]
    unconstrained=np.linalg.lstsq(augmented,target,rcond=None)[0]
    if np.all(unconstrained>=lower) and np.all(unconstrained<=upper):
        return unconstrained,unconstrained,True
    result=lsq_linear(augmented,target,bounds=(lower,upper),method='bvls',tol=1e-8,max_iter=100)
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('bounded teacher rate solve failed: '+result.message)
    return result.x,unconstrained,False


class MeasuredTeacher:
    def __init__(self,core,config=None):
        self.config=config or TeacherConfig()
        self.ids=leg_indices(core)
        self.rest=np.array(core.measure()['foot'],copy=True)
        self.rest_target=core.actuator.target_rad.copy()
        self.height=float(core.d.subtree_com[core.th,2])
        self.rotation=core.d.xmat[core.th].reshape(3,3).copy()
        cfg=self.config
        self.reference=FootReference(self.rest,scale=cfg.scale,vertical_gain=cfg.vertical_gain,
             time_warp=cfg.time_warp,blend_s=cfg.blend_s,source_duration_s=cfg.source_duration_s)

    def action(self,c,elapsed_s):
        cfg=self.config
        ref=self.reference.sample(elapsed_s)
        sensed=c.measure()
        R=c.d.xmat[c.th].reshape(3,3)
        jr=np.zeros((3,c.plant.nv));jp=np.zeros_like(jr)
        mj.mj_jac(c.plant,c.d,jp,jr,c.d.xpos[c.th],c.th)
        omega=jr@c.d.qvel
        orientation_error=.5*sum(np.cross(R[:,i],self.rotation[:,i]) for i in range(3))
        omega_des=cfg.orientation_gain*orientation_error-cfg.orientation_damping*omega
        velocity=c.com_velocity()
        vz_des=np.clip(cfg.height_gain*(self.height-float(c.d.subtree_com[c.th,2]))
                       -cfg.height_damping*velocity[2],-cfg.body_velocity_limit,cfg.body_velocity_limit)
        root_velocity=c.d.qvel.copy();root_velocity[c.a_dof]=0.
        rates=np.zeros(42);unlimited=np.zeros(42);errors=np.zeros((6,3));residuals=np.zeros((6,3))
        bounds_hit=np.zeros(42,dtype=bool)
        for leg,ids in enumerate(self.ids):
            relative,world,point=foot_jacobians(c,leg)
            columns=c.a_dof[ids]
            xy_error=ref['target_xy'][leg]-sensed['foot'][leg,:2]
            z_error=float(ref['target_z'][leg]-sensed['padz'][leg])
            desired_xy=ref['velocity_xy'][leg]+xy_error/cfg.tau_xy_s
            desired_z=float(ref['velocity_z'][leg]+z_error/cfg.tau_z_s)
            if bool(ref['stance'][leg]):
                # Ground compliance determines stance pad depth. Do not pull a
                # loaded pad toward a geometric z=0 anchor and remove its support.
                # Common body regulation is delivered ONLY by stance-joint rates.
                desired_z=-vz_des-np.cross(omega_des,point-c.d.xpos[c.th])[2]
            desired_z=float(np.clip(desired_z,-cfg.vertical_velocity_limit,cfg.vertical_velocity_limit))
            full=np.vstack([relative[0],relative[1],world[2]])
            A=full[:,columns];b=np.r_[desired_xy,desired_z]-full@root_velocity
            target=c.actuator.target_rad[ids]
            limits=c.actuator.target_bounds_rad[ids]
            lower=np.maximum(-MAX_RATE,(limits[:,0]-target)/cfg.command_dt_s)
            upper=np.minimum(MAX_RATE,(limits[:,1]-target)/cfg.command_dt_s)
            if np.any(lower>=upper):raise RuntimeError('empty actuator rate bounds')
            bias=-cfg.posture_gain*(target-self.rest_target[ids])
            dq,raw,_=bounded_rate(A,b,bias,lower,upper,cfg.ridge)
            rates[ids]=dq;unlimited[ids]=raw
            bounds_hit[ids]=(np.abs(dq-lower)<1e-6)|(np.abs(dq-upper)<1e-6)
            errors[leg]=np.r_[xy_error,z_error]
            residuals[leg]=A@dq-b
        if not np.isfinite(rates).all() or np.max(np.abs(rates))>MAX_RATE+1e-8:
            raise FloatingPointError('invalid teacher rates')
        return np.clip(rates/MAX_RATE,-1.,1.),dict(reference=ref,tracking_error=errors,
             velocity_residual=residuals,unlimited_rate=unlimited,bounds_hit=bounds_hit,
             body_height_target=self.height,body_height=float(c.d.subtree_com[c.th,2]),
             body_vertical_velocity_command=float(vz_des),orientation_error=orientation_error)

    def metadata(self):
        return dict(config=asdict(self.config),reference=self.reference.metadata,
             rest_body_feet=self.rest.tolist(),rest_target=self.rest_target.tolist(),
             regulated_height=self.height,control_source='declared measured-reference algorithm teacher only',
             physical_interface='42bounded rate outputs through unchanged finite PD; no root writes or external force')
