"""Local support-frame adaptation of the frozen FlyBody walking observation."""
import os
import numpy as np
from scipy.spatial.transform import Rotation
import rollout
from run_case import teacher_action

class FootFrame(rollout.tr.docking.ReversedYawWalk):
    def __init__(self,env,duration):
        super().__init__(env,duration)
        self.floor_height=0.
        self.normal=np.array([0.,0.,1.])
        self.foot=set()
        model=env.physics.model.ptr
        for i in range(model.ngeom):
            body=model.body(model.geom_bodyid[i]).name
            if body.startswith(('walker/tarsus','walker/claw_')):
                self.foot.add(i)
        self.soil={i for i,n in rollout.tr.scene_ids(env.physics).items() if n.startswith('crop_soil')}
        self.clearance=float(os.environ.get('P1_GAIT_CLEARANCE_CM','0.008'))

    def action(self,phy,speed,yaw_rate,live_observation):
        q=phy.data.qpos
        contacts=[c for c in phy.data.contact[:phy.data.ncon] if c.efc_address>=0 and
                  ((c.geom1 in self.foot and c.geom2 in self.soil) or
                   (c.geom2 in self.foot and c.geom1 in self.soil))]
        if contacts:
            normals=np.array([c.frame[:3].copy() for c in contacts])
            normals*=np.where(normals[:,2:3]<0,-1.,1.)
            valid=normals[:,2]>.65
            if valid.any():
                n=np.median(normals[valid],axis=0)
                n/=np.linalg.norm(n)
                points=np.array([c.pos.copy() for c in contacts])[valid]
                heights=points[:,2]+((points[:,:2]-q[:2])@n[:2])/n[2]
                self.floor_height+=.18*(np.median(heights)-self.floor_height)
                self.normal+=.18*(n-self.normal)
        n=self.normal/np.linalg.norm(self.normal)
        axis=np.cross(n,[0.,0.,1.])
        size=np.linalg.norm(axis)
        align=Rotation.from_rotvec(axis*(np.arctan2(size,n[2])/size)) if size>1e-12 else Rotation.identity()
        ph=self.env.physics
        ph.data.qpos[self.qcopy[0]]=q[self.qcopy[1]]
        ph.data.qvel[self.vcopy[0]]=phy.data.qvel[self.vcopy[1]]
        ph.data.qpos[2]=(q[2]-self.floor_height)*n[2]
        rot=align*Rotation.from_quat(q[[4,5,6,3]])
        ph.data.qpos[3:7]=rot.as_quat()[[3,0,1,2]]
        ph.data.qvel[:3]=align.apply(phy.data.qvel[:3])
        ph.forward()
        i=self.step_index
        dt=self.env.control_timestep()
        self.heading-=yaw_rate*dt
        direction=np.array([np.cos(self.heading),np.sin(self.heading)])
        self.position+=speed*dt*direction
        ref=self.ref[i:i+self.task._future_steps+2]
        ref[:,:2]=self.position+np.arange(len(ref))[:,None]*dt*speed*direction
        ref[:,2]=.1278+self.clearance
        ref[:,3:7]=[np.cos(self.heading/2),0.,0.,np.sin(self.heading/2)]
        self.task._step_counter=i
        updater=self.env._observation_updater
        updater.reset(self.env._physics_proxy,self.env.random_state)
        obs=updater.get_observation()
        for kind in ('accelerometer','appendages_pos','force','gyro','touch','velocimeter'):
            name='walker/'+kind
            if name in live_observation and obs[name].shape==live_observation[name].shape:
                obs[name]=live_observation[name]
        target=teacher_action(self.policy,{k:obs[k] for k in self.keys},self.spec)
        self.filtered+=-np.expm1(-dt/.01)*(target-self.filtered)
        self.last[self.columns]=self.filtered
        for act,j in self.action_for_act:
            if act>=0:
                ph.data.act[act]=self.filtered[j]
        self.step_index+=1
        return self.last.copy()

if __name__=='__main__':
    rollout.tr.docking.ReversedYawWalk=FootFrame
    rollout.main()
