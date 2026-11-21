"""A near-food assay on a source upper facet of the same apple, under gravity."""
import json
from copy import copy
import numpy as np
from context import HERE,tr
from mouth import active_build
from scipy.spatial.transform import Rotation
Crop=tr.Crop
class Flank(Crop):
    def __init__(self,site,support='mesh'):
        super().__init__(site,support)
        with np.load(HERE/'FLANK.npz') as f:
            self.soil=f['soil']; self.skin=f['skin']; self.source_ids=f['indices']
        self.debris=np.empty((0,3,3));self.labels=np.empty(0,dtype=int)
        self.terrain=np.full_like(self.terrain,self.soil[:,:,2].min())
        self.source=dict(self.source,objects=[])
        self.set_face(35)
    def set_face(self,index):
        self.face_index=index
        self.goal=self.skin[index].mean(axis=0)
        self.cfg=tr.Experiment('FLANK','ground-003',*self.goal)
def field(crop):
    exposed=copy(crop)
    exposed.skin=np.empty((0,3,3))
    return original_field(exposed)
original_field=tr.odor

def place(env,bank,crop,base_start,yaw):
    # Register the whole archived resting stance just at first physical surface contact.
    phy=tr.bind_physics(env,bank); q=base_start(env)
    with np.load(tr.SETTLED) as saved:q[:]=saved['poses'][-1]
    tri=crop.skin[crop.face_index]
    n=np.cross(tri[1]-tri[0],tri[2]-tri[0]); n/=np.linalg.norm(n)
    tilt=tr.plane_frame(-n[0]/n[2],-n[1]/n[2])
    base=Rotation.from_quat(q[[4,5,6,3]]);front=base.apply([1.,0.,0.])
    q[3:7]=(tilt*Rotation.from_rotvec([0.,0.,yaw-np.arctan2(front[1],front[0])])*base).as_quat()[[3,0,1,2]]
    q[:2]=crop.goal[:2]-.09*np.array([np.cos(yaw),np.sin(yaw)])
    phy.data.qvel[:]=0;phy.data.ctrl[:]=0;phy.data.act[:]=0
    low=crop.goal[2]-.20;high=crop.goal[2]+.4
    for _ in range(17):
        z=.5*(low+high);q[2]=z;phy.data.qpos[:]=q;phy.forward()
        touch=any(c.efc_address>=0 and (c.geom1 in bank.skin_ids or c.geom2 in bank.skin_ids)
                  for c in phy.data.contact[:phy.data.ncon])
        if touch:low=z
        else:high=z
    q[2]=high+.00005
    with phy.reset_context():
        phy.data.qpos[:]=q; phy.data.qvel[:]=0;phy.data.ctrl[:]=0;phy.data.act[:]=0
    return q.copy()

def set_mouth(env,targets):
    base=env.task.before_step
    names=env.action_spec().name.split('\t');j=[names.index(v) for v in ('rostrum','haustellum','head')]
    low,high=env.action_spec().minimum[j],env.action_spec().maximum[j]
    assert np.all((low<=targets)&(targets<=high))
    def motor(phy,action,random_state):
        u=action.copy()
        ramp=np.clip((phy.time()-.04)/.08,0.,1.)
        u[j]=(1-ramp)*u[j]+ramp*np.asarray(targets)
        return base(phy,u,random_state)
    env.task.before_step=motor
    return base

def search():
    crop=Flank('CLEAR');env,bank=active_build(crop); env._time_limit=1.
    env.task.check_termination=lambda _:False
    rows=[]
    trial_specs=[(f,y,m) for m in ((-1.15,-1.48,-.22),(-.6,-.85,.24),(-1.19,.12,.24),(-.8,-1.45,.28))
                 for f in (35,26,14,22) for y in (0.,1.56,3.13)]
    for face,yaw,target in trial_specs:
        crop.set_face(face);env.random_state.seed(29);env.reset();phy=env.physics
        q=place(env,bank,crop,tr.docking.start_pose,yaw)
        # Update the resource location as well as the geometric gate.
        tr.docking.fold_until_launch(env.task,phy)
        scan=tr.docking.Contacts(phy);th=phy.model.name2id('walker/thorax','body')
        base=set_mouth(env,np.asarray(target))
        zero=np.zeros(env.action_spec().shape);sustained=peak=0.;best=None;standing=0
        for k in range(1000):
            foot,_,lip=scan.scan(phy)
            support=len(foot)>=3 and phy.data.xmat[th,8]>=.88
            bite=bank.contact(lip,support)
            if k>=400:
                if support:standing+=1
                sustained=sustained+env.control_timestep() if bite else 0.
                peak=max(sustained,peak)
                if support:
                    hits=[tr.cast(phy,lip,(x,0.,0.),tr.FRUIT_MASK) for x in (-1.,1.)]
                    gap=min((d*10 for d,g in hits if g in bank.skin_ids and d>=0),default=None)
                    if gap is not None:best=gap if best is None else min(best,gap)
            env.step(zero.copy())
        env.task.before_step=base
        out=dict(face=face,source_face=int(crop.source_ids[face]),yaw=yaw,target=target,standing=standing,
                 continuous_contact_s=peak,best_gap_mm=best)
        rows.append(out)
        (HERE/'FLANK_SEARCH.json').write_text(json.dumps(rows,indent=2)+'\n')
        if peak>.085:
            break
    env.close();print(max(rows,key=lambda r:r['continuous_contact_s']),flush=True)
if __name__=='__main__':search()
