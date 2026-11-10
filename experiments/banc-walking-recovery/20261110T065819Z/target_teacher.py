"""Reference teacher using private kinematic target evaluation, never state forcing.

A separate MjData estimates foot geometry at the actuator's current target joints.
The physical MjData is read only by this controller. Its only output is a bounded
joint target rate consumed by the existing finite-torque incremental actuator.
"""
from dataclasses import dataclass,asdict
import numpy as np
import mujoco as mj
from physics import LEGS,MAX_RATE
from reference import FootReference


@dataclass(frozen=True)
class TargetConfig:
    time_warp: float=10.
    vertical_gain: float=.25
    source_duration_s: float=.24
    blend_s: float=.5
    tau_s: float=.10
    height_gain: float=0.
    orientation_gain: float=0.
    reference_mode: str='measured'
    shadow_root_mode: str='initial'
    feedback_space: str='position'
    horizontal_gain: float=.25
    scale: float=10.
    command_dt: float=.005
    ridge: float=.01


class TargetTeacher:
    def __init__(self,c,config=None):
        self.config=config or TargetConfig()
        self.shadow=mj.MjData(c.plant)
        self.shadow_root=c.d.qpos[:7].copy()
        self.ids=[np.array([i for i,j in enumerate(c.actuator.joint_ids)
                  if c.plant.joint(int(j)).name.startswith('joint_'+leg)],int) for leg in LEGS]
        assert all(len(ids)==7 for ids in self.ids)
        self.rest=np.array(c.measure()['foot'],copy=True)
        self.height=float(c.d.subtree_com[c.th,2])
        self.orientation=c.d.xmat[c.th].reshape(3,3).copy()
        self.reference=FootReference(self.rest,time_warp=self.config.time_warp,vertical_gain=self.config.vertical_gain,
            source_duration_s=self.config.source_duration_s,blend_s=self.config.blend_s,scale=self.config.scale)
        self.ground_pad=c.measure()['padz'].copy()
        self.target_com0=None
        self._update(c)
        self.target_rest,self.target_pad0,self.target_com0=self._geometry(c)

    def _update(self,c):
        self.shadow.qpos[:]=c.d.qpos
        if self.config.shadow_root_mode=='initial':
            self.shadow.qpos[:7]=self.shadow_root
        elif self.config.shadow_root_mode!='current':
            raise ValueError('unknown private kinematic frame')
        self.shadow.qpos[c.a_qpos]=c.actuator.target_rad
        self.shadow.qvel[:]=0.
        mj.mj_forward(c.plant,self.shadow)

    def _geometry(self,c):
        feet=[];pad=[];R=self.shadow.xmat[c.th].reshape(3,3)
        for leg in LEGS:
            ps=[self.shadow.geom_xpos[g] for g in c.pads[leg]]
            feet.append(R.T@(np.mean(ps,axis=0)-self.shadow.xpos[c.th]))
            pad.append(min(float(self.shadow.geom_xpos[g,2]-c.plant.geom_size[g,0]) for g in c.pads[leg]))
        return np.asarray(feet),np.asarray(pad),self.shadow.subtree_com[c.th].copy()

    def action(self,c,t):
        self._update(c)
        feet,padz,shadow_com=self._geometry(c)
        cfg=self.config;ref=self.reference.sample(t)
        measured=c.measure()
        if cfg.reference_mode=='hold':
            desired_xy=self.target_rest[:,:2].copy();desired_z=self.target_pad0.copy()
            stance=np.ones(6,dtype=bool);feed_xy=np.zeros((6,2));feed_z=np.zeros(6)
        else:
            desired_xy=self.target_rest[:,:2]+cfg.horizontal_gain*(ref['target_xy']-self.rest[:,:2])
            # Preserve the initial target-vs-actual elastic geometry. A geometric
            # z0 pad anchor would otherwise demand removal of contact compression.
            desired_z=self.target_pad0+ref['target_z']
            stance=ref['stance'];feed_xy=cfg.horizontal_gain*ref['velocity_xy'];feed_z=ref['velocity_z']
        height_error=self.height-float(c.d.subtree_com[c.th,2])
        R=c.d.xmat[c.th].reshape(3,3)
        orient=.5*sum(np.cross(R[:,i],self.orientation[:,i]) for i in range(3))
        rates=np.zeros(42);raw=np.zeros(42);error=np.zeros((6,3));hits=np.zeros(42,dtype=bool)
        for leg,ids in enumerate(self.ids):
            pads=c.pads[LEGS[leg]];J=np.zeros((3,c.plant.nv))
            for g in pads:
                j=np.zeros_like(J)
                mj.mj_jac(c.plant,self.shadow,j,None,self.shadow.geom_xpos[g],int(c.plant.geom_bodyid[g]))
                J+=j/len(pads)
            active=min(pads,key=lambda g:float(self.shadow.geom_xpos[g,2]-c.plant.geom_size[g,0]))
            Z=np.zeros_like(J)
            mj.mj_jac(c.plant,self.shadow,Z,None,self.shadow.geom_xpos[active],int(c.plant.geom_bodyid[active]))
            shadow_R=self.shadow.xmat[c.th].reshape(3,3)
            A=np.vstack([(shadow_R.T@J)[:2,c.a_dof[ids]],Z[2,c.a_dof[ids]]])
            xy=desired_xy[leg]-feet[leg,:2]
            dz=desired_z[leg]-padz[leg]
            body_correction=0.
            if stance[leg]:
                point=np.mean([self.shadow.geom_xpos[g] for g in pads],axis=0)
                body_correction=-cfg.height_gain*height_error-np.cross(cfg.orientation_gain*orient,point-self.shadow.xpos[c.th])[2]
            if cfg.feedback_space=='position':
                dz+=body_correction
                body_correction=0.
            elif cfg.feedback_space!='velocity':
                raise ValueError('unknown body feedback space')
            b=np.r_[feed_xy[leg]+xy/cfg.tau_s,feed_z[leg]+dz/cfg.tau_s+body_correction]
            dq=A.T@np.linalg.solve(A@A.T+cfg.ridge*np.eye(3),b)
            target=c.actuator.target_rad[ids];limits=c.actuator.target_bounds_rad[ids]
            lower=np.maximum(-MAX_RATE,(limits[:,0]-target)/cfg.command_dt)
            upper=np.minimum(MAX_RATE,(limits[:,1]-target)/cfg.command_dt)
            bounded=np.clip(dq,lower,upper)
            rates[ids]=bounded;raw[ids]=dq;hits[ids]=np.abs(dq-bounded)>1e-8
            error[leg]=np.r_[xy,dz]
        return rates/MAX_RATE,dict(raw_rate=raw,tracking_error=error,bounds_hit=hits,
           actual_pad=measured['padz'].copy(),shadow_pad=padz,reference_z=ref['target_z'],stance=stance,
           height_error=height_error,orientation_error=orient)

    def metadata(self):
        return dict(config=asdict(self.config),reference=self.reference.metadata,
          target_geometry_rest=self.target_rest.tolist(),target_pad_reference=self.target_pad0.tolist(),
          actual_pad_rest=self.ground_pad.tolist(),label='private kinematic prediction only; no physical state/force writes')
