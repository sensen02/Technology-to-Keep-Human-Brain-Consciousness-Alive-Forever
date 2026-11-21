"""Calculate native leg flexion targets for a short supported feeding posture."""
import json
import numpy as np
import mujoco
from context import HERE,tr
from flank import Flank,active_build,place
from stance import LEGS

def leg_targets(phy,names,lift):
    model=phy.model.ptr
    shadow=mujoco.MjData(model)
    shadow.qpos[:]=phy.data.qpos
    indices=[]; targets=[]
    for leg,side in LEGS:
        suffix=f'_{leg}_{side}'
        keys=[k for k in names if k.endswith(suffix) and 'adhere' not in k]
        ids=np.array([model.joint('walker/'+k).id for k in keys])
        adr=model.jnt_qposadr[ids]; dof=model.jnt_dofadr[ids]
        geom=model.geom('walker/tarsal_claw'+suffix+'_collision').id
        mujoco.mj_forward(model,shadow)
        goal=shadow.geom_xpos[geom].copy()
        goal[2]+=lift*(1.15 if leg=='T1' else .85 if leg=='T3' else 1.)
        for _ in range(12):
            jac=np.zeros((3,model.nv))
            mujoco.mj_jacGeom(model,shadow,jac,None,geom); J=jac[:,dof]
            err=goal-shadow.geom_xpos[geom]
            dq=J.T@np.linalg.solve(J@J.T+.00003*np.eye(3),err)
            shadow.qpos[adr]=np.clip(shadow.qpos[adr]+np.clip(dq,-.18,.18),*model.jnt_range[ids].T)
            mujoco.mj_forward(model,shadow)
        indices.extend(names.index(k) for k in keys);targets.extend(shadow.qpos[adr])
    return indices,targets

def install(env,targets,lift):
    base=env.task.before_step
    names=env.action_spec().name.split('\t')
    parts=['rostrum','haustellum','head']
    mi=[names.index(k) for k in parts];li=[];lt=[]
    if lift:
        li,lt=leg_targets(env.physics,names,lift)
    indices=np.asarray(mi+li); desired=np.array([*targets,*lt])
    spec=env.action_spec();desired=np.clip(desired,spec.minimum[indices],spec.maximum[indices])
    def before(phy,action,random_state):
        u=action.copy()
        ramp=np.clip((phy.time()-.04)/.08,0.,1.)
        u[indices]+=ramp*(desired-u[indices])
        return base(phy,u,random_state)
    env.task.before_step=before
    return base

def search():
    crop=Flank('CLEAR');env,bank=active_build(crop)
    env.task.check_termination=lambda _:False;rows=[]
    for lift in (.04,.065,.08,.10):
        for face,yaw in ((22,3.13),(35,1.56),(35,3.13),(26,0.),(14,0.),(35,.75),(26,1.56)):
            for target in ((-1.2,-1.55,.28),(-1.15,-1.4,-.3)):
                env.random_state.seed(29);crop.set_face(face);env.reset();phy=env.physics
                place(env,bank,crop,tr.docking.start_pose,yaw)
                pattern,_=tr.docking.fold_until_launch(env.task,phy)
                base=install(env,target,lift)
                scan=tr.docking.Contacts(phy);th=phy.model.name2id('walker/thorax','body')
                zero=np.zeros(env.action_spec().shape);speeds=[];good=peak=sustained=0;gap=None
                for tick in range(1250):
                    foot,_,lip=scan.scan(phy); support=len(foot)>=3 and phy.data.xmat[th,8]>=.88
                    bite=bank.contact(lip,support)
                    if tick>=450 and support:
                        good+=1
                        hits=[tr.cast(phy,lip,(x,0.,0.),tr.FRUIT_MASK) for x in (-1.,1.)]
                        v=min((d*10 for d,g in hits if g in bank.skin_ids and d>=0),default=99.)
                        gap=v if gap is None else min(gap,v)
                    sustained=sustained+env.control_timestep() if tick>=450 and bite else 0.
                    peak=max(peak,sustained)
                    env.step(zero.copy())
                env.task.before_step=base;env.task._wbpg.step=pattern
                rows.append(dict(face=face,yaw=yaw,target=target,lift=lift,standing=good,
                    source_face=int(crop.source_ids[face]),continuous_contact_s=peak,best_gap_mm=gap))
                (HERE/'CROUCH_SEARCH.json').write_text(json.dumps(rows,indent=2)+'\n')
                if peak>.10:
                    env.close();print(rows[-1],flush=True);return
    env.close()
if __name__=='__main__':search()
