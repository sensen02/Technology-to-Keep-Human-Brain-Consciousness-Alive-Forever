"""Select a near patch for which contact followed by release permits the original standing gate."""
import json
import numpy as np
from context import HERE,tr
from flank import Flank,active_build,place
from crouch import install

def main():
    crop=Flank('CLEAR');env,bank=active_build(crop);env.task.check_termination=lambda _:False
    rows=[]
    for lift in (.055,.075,.095,.12):
        for face,yaw in ((26,0.),(14,3.13),(26,1.56),(14,0.),(35,3.13),(35,1.56),(14,1.56),(26,3.13)):
            crop.set_face(face);env.random_state.seed(29);env.reset();phy=env.physics
            for attr in ('_ref_qpos','_ref_qvel'):
                ref=getattr(env.task,attr)
                required=2300+env.task._future_steps+2
                if len(ref)<required:
                    setattr(env.task,attr,np.pad(ref,((0,required-len(ref)),(0,0)),mode='edge'))
            place(env,bank,crop,tr.docking.start_pose,yaw)
            pattern,_=tr.docking.fold_until_launch(env.task,phy)
            target=(-1.22,-1.55,.28);base=install(env,target,lift)
            scan=tr.docking.Contacts(phy);th=phy.model.name2id('walker/thorax','body')
            zero=np.zeros(env.action_spec().shape);best=None;peak=count=0;last=[]
            for k in range(2300):
                if k==1250:env.task.before_step=base
                foot,_,lip=scan.scan(phy);up=phy.data.xmat[th,8]
                if k<1250:
                    bite=bank.contact(lip,len(foot)>=3 and up>=.88)
                    count=count+1 if bite else 0;peak=max(peak,count)
                if k>=1600:last.append(len(foot)>=4 and up>.94 and np.linalg.norm(phy.data.qvel[:2])<=.08)
                env.step(zero.copy())
            env.task.before_step=base;env.task._wbpg.step=pattern
            row=dict(face=face,source_face=int(crop.source_ids[face]),yaw=yaw,lift=lift,target=target,
                     peak_contact_s=peak*env.control_timestep(),terminal_stand=bool(all(last)))
            rows.append(row)
            (HERE/'RELEASE_SEARCH.json').write_text(json.dumps(rows,indent=2)+'\n')
            if row['terminal_stand'] and row['peak_contact_s']>.07:
                (HERE/'MEAL_CONFIG.json').write_text(json.dumps(row,indent=2)+'\n');env.close();return
    env.close()
if __name__=='__main__': main()
