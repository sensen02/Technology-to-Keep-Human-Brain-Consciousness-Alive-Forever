"""Source-labelled finite joint envelopes and canonical non-fused pad dynamics."""
from pathlib import Path
import json
import sys
import numpy as np
import mujoco
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE=HERE.parent
PREV=BASE/'20261117T181339Z-action-reachability'
sys.path.insert(0,str(PREV))
import context as prior
from flygym.compose.world import FlatGroundWorld
from flygym_demo.complex_terrain.common import make_locomotion_fly
from flygym.compose import ActuatorType
from flygym.simulation import Simulation
from flygym.utils.math import Rotation3D
from flygym.anatomy import BodySegment
from flygym_demo.complex_terrain.hybrid_controller import HybridControllerObservation
from flygym_demo.complex_terrain.turning_controller import HybridTurningController
sha=prior.sha
write=prior.write
DT=.0001
GAIN=5.02424055  # .5 original-body weight; native force, fixed engineering budget.


class Terrain(FlatGroundWorld):
    def __init__(self,height=0.):
        super().__init__()
        if height:
            geom=self.mjcf_root.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,name='reach_block',
                 size=(.20,3.,height/2),pos=(2.45,0,height/2),rgba=(.3,.48,.7,1),contype=0,conaffinity=0)
            self.ground_geoms.append(geom)


class Body:
    def __init__(self,height=0.,adhesion=True,limited=True,seed=119):
        self.fly=fly=make_locomotion_fly(add_adhesion=adhesion,adhesion_gain=GAIN)
        if limited:
            self.limits=json.loads((HERE/'JOINT_LIMITS.json').read_text())['joints']
            for dof,joint in fly.jointdof_to_mjcfjoint.items():
                config=self.limits['nmf/'+joint.name]
                joint.limited=True
                joint.range=config['joint_range_rad']
            for dof,act in fly.jointdof_to_mjcfactuator_by_type[ActuatorType.POSITION].items():
                config=self.limits['nmf/'+fly.jointdof_to_mjcfjoint[dof].name]
                act.ctrllimited=True;act.ctrlrange=config['control_range_rad']
        world=Terrain(height);world.add_fly(fly,(0,0,.5),Rotation3D('quat',(1,0,0,0)))
        # Stable canonical representation in both 42 and 48 actuator variants.
        world.mjcf_root.compiler.fusestatic=False
        self.sim=sim=Simulation(world,timestep=DT)
        self.m,self.d=sim.mj_model,sim.mj_data;self.name=fly.name
        self.dofs=fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        self.actuators=sim._intern_actuatorids_by_type_by_fly[ActuatorType.POSITION][fly.name]
        self.adhesion_ids=np.array([i for i in range(self.m.nu) if i not in self.actuators],dtype=int)
        self.legs=fly.get_legs_order()
        self.thorax=sim._internal_bodyids_by_fly[self.name][fly.get_bodysegs_order().index(BodySegment('c_thorax'))]
        self.feet=np.array([sim._internal_bodyids_by_fly[self.name][fly.get_bodysegs_order().index(BodySegment(f'{leg}_tarsus5'))] for leg in self.legs])
        self.jids=self.m.actuator_trnid[self.actuators,0];self.qadr=self.m.jnt_qposadr[self.jids];self.vadr=self.m.jnt_dofadr[self.jids]
        self.ground_geoms=set(int(i) for i in sim._internal_ground_geom_ids)
        self.blocks={g for g in self.ground_geoms if 'reach_block' in mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_GEOM,g)}
        self.geom_leg={}
        for segment,geoms in sim._internal_geomid_by_bodyseg_by_fly[self.name].items():
            if segment.pos in self.legs:
                for geom in np.atleast_1d(geoms):self.geom_leg[int(geom)]=self.legs.index(segment.pos)
        self.axis_bodies={i for i in range(1,self.m.nbody) if any(k in (mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_BODY,i) or '')
                              for k in ('c_head','c_thorax','c_abdomen','wing'))}
        self.weight=float(self.m.body_mass.sum()*abs(self.m.opt.gravity[2]))
        self.motor=HybridTurningController(timestep=DT,enable_adhesion=adhesion,output_dof_order=self.dofs)
        self.motor.reset(seed=seed)
        self.neutral=self.motor.preprogrammed_steps.default_pose_by_dof_order(self.dofs)
        self.minimum=self.m.actuator_ctrlrange[self.actuators,0];self.maximum=self.m.actuator_ctrlrange[self.actuators,1]
        self.adhesion=adhesion

    def teacher(self,command):
        obs=HybridControllerObservation.from_sim(self.sim,self.name,legs=self.motor.legs)
        action=self.motor.step(command,obs)
        return np.clip(action.joint_angles,self.minimum,self.maximum),action.adhesion_onoff

    def apply(self,control,pads=None):
        self.d.ctrl[self.actuators]=np.clip(control,self.minimum,self.maximum)
        if len(self.adhesion_ids):self.sim.set_leg_adhesion_states(self.name,np.zeros(6) if pads is None else pads)
        self.sim.step()

    def restore(self,seed):
        with np.load(HERE/'CANONICAL_INIT.npz',allow_pickle=False) as z:
            for name in ('qpos','qvel','act','qacc_warmstart','qfrc_applied','xfrc_applied','eq_active','userdata','mocap_pos','mocap_quat'):
                getattr(self.d,name)[:]=z[name]
            self.d.time=float(z['time']);self.d.ctrl[self.actuators]=z['ctrl_joint']
            if len(self.adhesion_ids):self.d.ctrl[self.adhesion_ids]=0.
        mujoco.mj_forward(self.m,self.d);self.motor.reset(seed=seed)

    def contact(self):
        load=np.zeros(6);force=np.zeros((6,3));nonfoot=0.;block=0.;w=np.zeros(6)
        for i,c in enumerate(self.d.contact[:self.d.ncon]):
            if c.exclude:continue
            if c.geom1 in self.ground_geoms:g,o,s=int(c.geom1),int(c.geom2),1.
            elif c.geom2 in self.ground_geoms:g,o,s=int(c.geom2),int(c.geom1),-1.
            else:continue
            mujoco.mj_contactForce(self.m,self.d,i,w);n=float(np.linalg.norm(w[:3]))
            if g in self.blocks:block+=n
            leg=self.geom_leg.get(o)
            if leg is not None:
                load[leg]+=w[0];force[leg]+=s*c.frame.reshape(3,3).T@w[:3]
            elif self.m.geom_bodyid[o] in self.axis_bodies:nonfoot+=n
        return load,force,nonfoot,block


def snapshot(b):
    arrays={name:getattr(b.d,name).copy() for name in ('qpos','qvel','act','qacc_warmstart','qfrc_applied','xfrc_applied','eq_active','userdata','mocap_pos','mocap_quat')}
    arrays.update(time=np.array(b.d.time),ctrl_joint=b.d.ctrl[b.actuators].copy())
    np.savez_compressed(HERE/'CANONICAL_INIT.npz',**arrays)
