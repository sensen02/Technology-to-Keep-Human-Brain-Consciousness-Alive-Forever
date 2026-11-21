"""Reuse the fruit orchard's articulated FlyBody plant for local recovery."""
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'20261119T052620Z-hunger-foraging'
sys.path.insert(0,str(PRIOR))
from orchard_trial import RESOURCES, Contacts, build_orchard, fold_until_launch, land
from scipy.spatial.transform import Rotation

LEG_NAMES=[(s,l) for s in ('left','right') for l in ('T1','T2','T3')]
INTEGRATION=mj.mjtState.mjSTATE_INTEGRATION

class Plant:
    def __init__(self,env=None):
        if env is None:
            env,_=build_orchard()
            env.reset()
        self.env=env
        self.phy=self.env.physics
        self.model=self.phy.model.ptr
        self.data=self.phy.data.ptr
        self.base=self.data.qpos.copy()
        self.scanner=Contacts(self.phy)
        self.root=self.phy.bind(self.env.task._root_joint)
        self.root_body=self.model.jnt_bodyid[0]
        self.th=self.model.body('walker/thorax').id
        self.wing_joint=np.asarray([self.model.joint('walker/'+e.name).id for e in self.env.task._wing_joints])
        self.names=[self.model.actuator(i).name.removeprefix('walker/') for i in range(self.model.nu)]
        self.min,self.max=self.model.actuator_ctrlrange.T.copy()
        self.foot_body=np.asarray([self.model.body(f'walker/claw_{l}_{s}').id for s,l in LEG_NAMES])
        self.act=np.asarray([[i for i,x in enumerate(self.names) if x.endswith(f'_{l}_{s}')
                             and not x.startswith('adhere')] for s,l in LEG_NAMES])
        self.dofs=self.model.jnt_dofadr[self.model.actuator_trnid[self.act[:,:7],0]]
        self.qa=self.model.jnt_qposadr[self.model.actuator_trnid[self.act[:,:7],0]]
        self.adh=np.asarray([self.names.index(f'adhere_claw_{l}_{s}') for s,l in LEG_NAMES])
        assert self.act.shape[0]==6
        self.ctrl_neutral=np.zeros(self.model.nu)
        self.wing_act=np.array([self.names.index(x) for x in
                     ('wing_yaw_left','wing_roll_left','wing_pitch_left',
                      'wing_yaw_right','wing_roll_right','wing_pitch_right')])
        self.zero=np.zeros(mj.mj_stateSize(self.model,INTEGRATION))
        self.geo_leg=np.full(self.model.ngeom,-2,dtype=np.int32)
        for g,(fly,foot,_) in enumerate(self.scanner.bodies):
            if fly:
                self.geo_leg[g]=LEG_NAMES.index((foot[1],foot[0])) if foot in [
                    (leg,side) for side,leg in LEG_NAMES] else -1
        self.normal_pose=self.data.qpos.copy()

    def state(self,data=None):
        data=self.data if data is None else data
        result=np.empty_like(self.zero)
        mj.mj_getState(self.model,data,result,INTEGRATION)
        return result

    def put(self,state,data=None):
        data=self.data if data is None else data
        mj.mj_setState(self.model,data,state,INTEGRATION)
        mj.mj_forward(self.model,data)

    def init(self,roll,pitch,yaw,xy,settle=.16,source=None):
        mj.mj_resetData(self.model,self.data)
        self.data.qpos[:]=self.base if source is None else source
        if source is None:
            self.data.qpos[:3]=(*xy,.32)
            q=Rotation.from_euler('xyz',[roll,pitch,yaw],degrees=True).as_quat()
            self.data.qpos[3:7]=q[[3,0,1,2]]
        self.data.ctrl[:]=self.ctrl_neutral
        self.data.act[:]=0.
        for i in self.wing_act:
            addr=self.model.actuator_actadr[i]
            if addr>=0:
                self.data.act[addr]=self.ctrl_neutral[i]
        mj.mj_forward(self.model,self.data)
        mj.mj_step(self.model,self.data,nstep=round(settle/self.model.opt.timestep))
        mj.mj_forward(self.model,self.data)
        return self.state()
