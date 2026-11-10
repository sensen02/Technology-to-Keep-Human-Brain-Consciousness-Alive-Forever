"""Additional audit recorder: exact official claw feet and explicit adhesion support.
Frozen first short run remains intact. Policy and physics behavior are unchanged.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import run_teacher as base
ROOT=Path(__file__).resolve().parent

class AuditedRecorder(base.Recorder):
    def __init__(self,physics):
        super().__init__(physics)
        m=self.m
        root=int(m.body_jntadr[int(m.body_rootid[self.thorax])])
        self.root_qpos=int(m.jnt_qposadr[root]);self.root_dof=int(m.jnt_dofadr[root])
        # Original writer logs qpos[:7]; verify its ordering instead of silently assuming it.
        assert self.root_qpos==0 and self.root_dof==0
        self.feet={m.name2id('walker/tarsal_claw_'+leg+'_collision','geom'):i for i,leg in enumerate(base.LEGS)}
        self.real=np.array([n.startswith('walker/') for n in self.body_names])
        self.adhesion=[m.name2id('walker/adhere_claw_'+leg,'actuator') for leg in base.LEGS]
        self.environmental=[]
    def native(self,physics,random_state):
        begin=len(self.contacts)
        super().native(physics,random_state)
        force=np.zeros(6);slip=np.zeros(6);nonfoot=0.
        for c in self.contacts[begin:]:
            # cvel is referenced to root-subtree COM, not the individual body COM.
            # Compute material contact velocity independently through the point Jacobian.
            b1=int(self.m.geom_bodyid[int(c[1])]);b2=int(self.m.geom_bodyid[int(c[2])])
            body=b2 if b1==0 else b1
            jac=np.zeros((3,self.m.nv));jacr=np.zeros_like(jac)
            base.mujoco.mj_jac(self.m.ptr,self.scratch,jac,jacr,c[5:8],body)
            c[11:14]=jac@self.scratch.qvel
            leg=self.feet.get(int(c[1]),self.feet.get(int(c[2]),-1))
            c[3]=leg
            load=max(0.,float(c[10]))
            if leg>=0:force[leg]+=load;slip[leg]+=load*np.linalg.norm(c[11:13])
            else:nonfoot+=load
        self.rows[-1][1:7]=force;self.rows[-1][7]=nonfoot;self.rows[-1][18:24]=slip
        self.rows[-1][24]=np.max(np.abs(self.d.qfrc_applied[:6]))
        self.rows[-1][25]=np.max(np.abs(self.d.xfrc_applied[self.real]))
        d=self.scratch;r=self.root_dof
        moment=d.actuator_moment[self.adhesion,r:r+3]
        realized=moment*d.actuator_force[self.adhesion,None]
        actadr=self.m.actuator_actadr[self.adhesion]
        self.environmental.append(np.r_[d.time,d.qfrc_constraint[r:r+3],d.qfrc_actuator[r:r+3],realized.reshape(-1),d.actuator_force[self.adhesion],d.act[actadr]])

OriginalRecorder=base.Recorder
OriginalSave=np.savez_compressed
OriginalMake=base.make_env
live=[]
terminal=[]

def make_env(seconds=None):
    env=OriginalMake(seconds);original=env.step
    def step(a):
        ts=original(a)
        if ts.last():terminal.append(base.observation(ts))
        return ts
    env.step=step
    return env
class Registered(AuditedRecorder):
    def __init__(self,p):super().__init__(p);live.append(self)

def saving(path,*args,**kwargs):
    if Path(path).name=='NATIVE.npz':
        kwargs['environmental']=np.asarray(live[-1].environmental)
        kwargs['environmental_columns']=np.array(['time','constraint_root_x','constraint_root_y','constraint_root_z','actuator_root_x','actuator_root_y','actuator_root_z']+[f'adhesion_root_{leg}_{axis}' for leg in base.LEGS for axis in 'xyz']+[f'adhesion_force_{leg}' for leg in base.LEGS]+[f'adhesion_activation_{leg}' for leg in base.LEGS])
    return OriginalSave(path,*args,**kwargs)

if __name__=='__main__':
    import sys
    tag=sys.argv[sys.argv.index('--tag')+1]
    folder=base.OUT/tag
    assert not folder.exists()
    base.Recorder=Registered
    base.make_env=make_env
    np.savez_compressed=saving
    base.main()
    if terminal:OriginalSave(folder/'TERMINAL_OBS.npz',**terminal[-1])
    with (folder/'ADDITIONAL_PROVENANCE.json').open('x') as f:
        json.dump(dict(source=str(Path(__file__)),sha256=base.sha(__file__),base_source_sha256=base.sha(ROOT/'run_teacher.py'),
            foot_definition='exact six official tarsal_claw collision capsules; all other walker-ground geoms nonfoot',
            environmental='scratch recomputation: native constraint and actuator root translational forces, per-adhesion realized root wrench, forces and activations; contact forces alone overstate net support',
            root_qpos_adr=live[-1].root_qpos,root_dof_adr=live[-1].root_dof,physical_policy_unchanged=True),f,indent=2)
