"""Mechanical forward model. Native length mm, mass g, time s, force µN."""
from pathlib import Path
import sys
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
ARCH=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain')
TRACK=ARCH/'20261117T192322Z-terrain-body-learning/HELD/flat_1151/TRACE.npz'
sys.path.insert(0,str(ARCH/'20261117T184709Z-body-terrain-repair'))
from body_model import Body

with np.load(HERE/'geometry.npz') as geo:
    X,Y,H,ORIGIN=(geo[k] for k in ('x_edges_mm','y_edges_mm','height_mm','world_origin_m'))
CX,CY=np.meshgrid((X[1:]+X[:-1])/2,(Y[1:]+Y[:-1])/2)
SHAPE=CX.shape
DT=.0001
K=1500.
EDGE_K=75.
MASS=.000035
DAMP=.40

def lap(a):
    out=np.zeros_like(a)
    diff=np.diff(a,axis=-1)
    out[...,:-1]-=diff
    out[...,1:]+=diff
    diff=np.diff(a,axis=-2)
    out[...,:-1,:]-=diff
    out[...,1:,:]+=diff
    return out

class Floor:
    def __init__(self,kind,probe_xy=None):
        self.moving=kind!='sand'
        self.fly=None
        if kind in ('bench','hover','probes'):
            s=mj.MjSpec()
            s.option.timestep=DT
            s.option.gravity=[0,0,-9810]
        else:
            self.fly=Body()
            self.fly.restore(1151)
            s=self.fly.sim.world.mjcf_root.copy()
            plane=s.geom('ground_plane')
            plane.pos=[0,0,-3]
            for g in np.unique(self.fly.m.pair_geom2):
                s.geom(self.fly.m.geom(int(g)).name).contype=2
        s.option.jacobian=mj.mjtJacobian.mjJAC_SPARSE
        if kind=='sand':
            field=s.add_hfield(name='original_sand',nrow=len(Y),ncol=len(X),
                               size=[(X[-1]-X[0])/2,(Y[-1]-Y[0])/2,float(np.ptp(H)),.2])
            field.userdata=((H-H.min())/np.ptp(H)).ravel().tolist()
            s.worldbody.add_geom(name='sandfield',type=mj.mjtGeom.mjGEOM_HFIELD,hfieldname=field.name,
                                  pos=[(X[0]+X[-1])/2,(Y[0]+Y[-1])/2,H.min()],contype=0,conaffinity=2,
                                  priority=2,condim=3,solref=[.0002,1],solimp=[.98,.99,1e-5,.5,3],
                                  friction=[1,.02,.0001])
        for j,i in (np.ndindex(SHAPE) if self.moving else ()):
            name=f'cell_{j}_{i}'
            patch=H[j:j+2,i:i+2]
            z=patch.mean()
            slope_x=np.diff(patch,axis=1).mean()/.25
            slope_y=np.diff(patch,axis=0).mean()/.25
            quat=Rotation.from_rotvec([slope_y,-slope_x,0]).as_quat()[[3,0,1,2]]
            # Top plate is the tangent plane of its sampled original sand facet.
            body=s.worldbody.add_body(name=name,pos=[CX[j,i],CY[j,i],z])
            if self.moving:
                body.gravcomp=1
                body.add_joint(name=name,type=mj.mjtJoint.mjJNT_SLIDE,axis=[0,0,1],
                               stiffness=K,damping=DAMP)
            normal=Rotation.from_quat(quat[[1,2,3,0]]).apply([0,0,1])
            body.add_geom(name=name,type=mj.mjtGeom.mjGEOM_BOX,size=[.125,.125,.025],
                          pos=-.025*normal,quat=quat,mass=MASS,contype=0,conaffinity=2,
                          priority=2,condim=3,solref=[.0002,1],solimp=[.98,.99,1e-5,.5,3],
                          friction=[1,.02,.0001])
        if kind=='probes':
            if probe_xy is None:
                probe_xy=[(CX[7,col]+.070,CY[7,col]-.045) for col in (17,19)]
            for i,(px,py) in enumerate(probe_xy):
                row=np.searchsorted(Y,py)-1; col=np.searchsorted(X,px)-1
                height=H[row:row+2,col:col+2].mean()
                probe=s.worldbody.add_body(name=f'weight_{i}',gravcomp=1,
                    pos=[px,py,height+.064])
                probe.add_joint(name=f'weight_{i}',type=mj.mjtJoint.mjJNT_SLIDE,axis=[0,0,1],damping=.018)
                probe.add_geom(type=mj.mjtGeom.mjGEOM_SPHERE,size=[.045,0,0],mass=.000055,
                               contype=2,conaffinity=0)
        if kind=='hover':
            for index in range(36):
                body=s.worldbody.add_body(name=f'probe_{index}',mocap=True,pos=[0,0,5])
                rod=index<18
                body.add_geom(type=mj.mjtGeom.mjGEOM_CYLINDER if rod else mj.mjtGeom.mjGEOM_SPHERE,
                              size=[.038,.25,0] if rod else [.066,0,0],contype=2,conaffinity=0)
        self.m=s.compile()
        self.d=mj.MjData(self.m)
        names=[f'cell_{j}_{i}' for j,i in np.ndindex(SHAPE)]
        if kind=='sand':
            self.m.hfield_data[:]=((H-H.min())/np.ptp(H)).ravel()
            names=['sandfield']
        self.geoms=[self.m.geom(n).id for n in names]
        self.grid={g:i for i,g in enumerate(self.geoms)}
        if self.moving:
            self.q=np.array([self.m.joint(n).qposadr[0] for n in names])
            self.v=np.array([self.m.joint(n).dofadr[0] for n in names])
        if self.fly:
            b=self.fly
            self.d.qpos[:b.m.nq]=b.d.qpos
            self.d.qvel[:b.m.nv]=b.d.qvel
            self.d.qpos[2]+=.025
            self.d.ctrl[:]=b.d.ctrl
        mj.mj_forward(self.m,self.d)

    def step(self,external=None):
        if self.moving:
            pos=self.d.qpos[self.q].reshape(SHAPE)
            applied=-EDGE_K*lap(pos)
            if external is not None:
                applied-=external
            self.d.qfrc_applied[self.v]=applied.ravel()
        mj.mj_step(self.m,self.d)
        if self.d.warning.number.any() or not np.isfinite(self.d.qpos).all():
            raise RuntimeError(f'Unstable model at {self.d.time}')

    def depth(self):
        return -1000*self.d.qpos[self.q].reshape(SHAPE)
