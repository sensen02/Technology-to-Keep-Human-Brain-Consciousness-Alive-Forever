"""Locate a supported near-fruit start on the original mesh; verify by dynamic trial separately."""
import json
import sys
import numpy as np
from context import HERE,tr

def main(site):
    crop=tr.Crop(site); env,bank=tr.build(crop); env.reset()
    phy=tr.bind_physics(env,bank); scan=tr.docking.Contacts(phy)
    base=tr.fit_spawn(crop,env,tr.docking.start_pose)
    rotation=tr.Rotation.from_quat(base[[4,5,6,3]])
    # Search real points at mouth height. Probe the shell along the original world-x rays,
    # then offset the whole neutral stance so its lips approach from the outside.
    q=base.copy(); ground0=crop.ground(0.,0.)
    rng=np.random.default_rng(14267)
    candidates=[]; count=0; hits=0
    for i in range(14000):
        y=rng.uniform(-.45,.45) if i%2==0 else rng.uniform(-1.65,1.65)
        yaw=rng.uniform(-np.pi,np.pi)
        q[:2]=[1.4,y]; q[2]=base[2]+crop.ground(*q[:2])-ground0
        q[3:7]=(tr.Rotation.from_rotvec([0.,0.,yaw])*rotation).as_quat()[[3,0,1,2]]
        for _ in range(3):
            phy.data.qpos[:]=q; phy.forward()
            lip=phy.data.geom_xpos[scan.mouth].mean(axis=0)
            origin=np.array([-.7,lip[1],lip[2]])
            distance,g=tr.cast(phy,origin,(1.,0.,0.),tr.FRUIT_MASK)
            if g not in bank.skin_ids or distance<0: break
            hit=origin+np.array([distance,0.,0.])
            q[0]+=hit[0]-lip[0]-.047
            if q[0]<-.6 or q[0]>2.9:break
            q[2]=base[2]+crop.ground(*q[:2])-ground0
        else:
            phy.data.qpos[:]=q; phy.forward()
            foot,nonfoot,lip=scan.scan(phy)
            dist,g=tr.cast(phy,lip,(1.,0.,0.),tr.FRUIT_MASK)
            if g not in bank.skin_ids or not (.001<dist<=tr.SKIN_REACH) or len(foot)<3:continue
            count+=1
            c=phy.data.contact[:phy.data.ncon]
            penetration=max((max(0.,-float(r.dist)) for r in c if r.efc_address>=0 and
                (r.geom1 in bank.skin_ids or r.geom2 in bank.skin_ids)),default=0.)
            if penetration>.002:continue
            hits+=1
            hit=lip+np.array([dist,0.,0.]); original=np.linalg.norm(lip-crop.goal)<=tr.SKIN_PATCH_RADIUS
            candidates.append(dict(qpos=q.tolist(),skin_point=hit.tolist(),original=bool(original),
                gap_mm=float(dist*10),skin_penetration_mm=10*penetration,feet=len(foot),nonfoot=nonfoot,
                target_distance=float(np.linalg.norm(hit-crop.goal)),yaw=float(yaw)))
            if hits>=50 and any(v['original'] and v['nonfoot']==0 for v in candidates): break
    env.close()
    candidates.sort(key=lambda r:(not r['original'],r['nonfoot']>0,r['skin_penetration_mm'], -r['feet'],r['target_distance']))
    result=dict(site=site,searched=i+1,geometry_hits=count,admissible=hits,choices=candidates[:8])
    (HERE/f'SURFACE_{site}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(site,i+1,count,hits,[(s['original'],s['feet'],s['gap_mm']) for s in candidates[:3]],flush=True)

if __name__=='__main__': main(sys.argv[1])
