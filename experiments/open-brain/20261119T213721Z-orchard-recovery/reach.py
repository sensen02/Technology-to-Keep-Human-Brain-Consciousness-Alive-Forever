"""Search dynamically testable original-patch/secondary-patch poses with a native extended proboscis."""
import json,sys
import numpy as np
from context import HERE,tr
from mouth import active_build

def main(site):
    crop=tr.Crop(site);env,bank=active_build(crop);env.reset()
    phy=tr.bind_physics(env,bank);scan=tr.docking.Contacts(phy)
    base=tr.fit_spawn(crop,env,tr.docking.start_pose)
    actuators=env.action_spec().name.split('\t')
    parts=('rostrum','haustellum','head')
    assert all(n in actuators for n in parts)
    addr=[phy.model.jnt_qposadr[phy.model.name2id('walker/'+n,'joint')] for n in parts]
    rotation=tr.Rotation.from_quat(base[[4,5,6,3]])
    ground0=crop.ground(0,0)
    rng=np.random.default_rng(824)
    best=[]; best_deep=None; trials=0
    for i in range(21000):
        y=rng.uniform(-.48,.48) if i%2==0 else rng.uniform(-1.7,1.7)
        yaw=rng.uniform(-np.pi,np.pi); q=base.copy()
        q[3:7]=(tr.Rotation.from_rotvec([0.,0.,yaw])*rotation).as_quat()[[3,0,1,2]]
        # Angles are candidate motor targets only. Live starts always have folded joints.
        q[addr]=[rng.uniform(-1.24,-.2),rng.uniform(-1.58,.2),rng.uniform(-.47,.28)]
        q[:2]=[1.5,y]
        desired=rng.uniform(.014,.05)
        for _ in range(3):
            q[2]=base[2]+crop.ground(*q[:2])-ground0
            phy.data.qpos[:]=q; phy.forward()
            lip=phy.data.geom_xpos[scan.mouth].mean(axis=0)
            origin=np.array([-.75,lip[1],lip[2]])
            distance,g=tr.cast(phy,origin,(1.,0.,0.),tr.FRUIT_MASK)
            if distance<0 or g not in bank.skin_ids:break
            q[0]+=origin[0]+distance-desired-lip[0]
            if not -.7<q[0]<2.9:break
        else:
            phy.data.qpos[:]=q;phy.forward();foot,_,lip=scan.scan(phy)
            gap,g=tr.cast(phy,lip,(1.,0.,0.),tr.FRUIT_MASK)
            if g not in bank.skin_ids or not .001<=gap<=.055 or len(foot)<3:continue
            trials+=1
            target=q[addr].copy()
            point=lip+np.array([gap,0.,0.])
            original=bool(np.linalg.norm(lip-crop.goal)<tr.SKIN_PATCH_RADIUS)
            def deepest():
                return max((max(0.,-float(c.dist)) for c in phy.data.contact[:phy.data.ncon] if
                    c.efc_address>=0 and (c.geom1 in bank.skin_ids or c.geom2 in bank.skin_ids)),default=0.)
            penetration=deepest(); q[addr]=base[addr]
            phy.data.qpos[:]=q;phy.forward();start_pen=deepest()
            rank=max(penetration,start_pen)
            if best_deep is None or rank<best_deep:best_deep=rank
            if rank>.002: continue
            best.append(dict(qpos=q.tolist(),target=target.tolist(),original=original,skin_point=point.tolist(),
                gap_mm=gap*10,feet=len(foot),penetration_mm=10*rank,offset_mm=10*np.linalg.norm(point-crop.goal)))
            if len(best)>=90 and sum(s['original'] for s in best)>=12:break
    best.sort(key=lambda v:(not v['original'],v['penetration_mm'],-v['feet'],v['offset_mm']))
    result=dict(site=site,tested=i+1,hits=trials,min_penetration_mm=10*best_deep if best_deep is not None else None,
                valid=len(best),choices=best[:20])
    (HERE/f'EXTENDED_{site}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(site,trials,len(best),best_deep,flush=True)
    env.close()
if __name__=='__main__':main(sys.argv[1])
