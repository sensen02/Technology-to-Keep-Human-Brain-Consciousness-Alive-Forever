"""Check feasible physical poses for sampling an orchard terminal transition."""
import json
from pathlib import Path
import numpy as np
from rollout import HERE,tr

def shifted(q,crop,x,y,angle):
    start=q.copy()
    start[2]+=crop.ground(x,y)-crop.ground(*q[:2])
    start[:2]=[x,y]
    rot=tr.Rotation.from_rotvec([0,0,angle])*tr.Rotation.from_quat(q[[4,5,6,3]])
    start[3:7]=rot.as_quat()[[3,0,1,2]]
    return start

def main():
    rows={}
    for site in ('CLEAR','DEBRIS'):
        crop=tr.Crop(site,'mesh')
        env,bank=tr.build(crop);env.reset()
        phy=tr.bind_physics(env,bank)
        scan=tr.docking.Contacts(phy)
        field,_=tr.odor(crop)
        antenna=[phy.model.name2id('walker/antenna_'+s,'body') for s in ('left','right')]
        with np.load(HERE/f'R2_{site}'/'BODY.npz') as data:
            ts=data['trace']; qs=data['poses']
            ok=(ts[:,6]>.90)&(ts[:,4]>=4)&(ts[:,0]>.24)
            idx=np.flatnonzero(ok)[::max(1,int(ok.sum()/8))][:8]
        best=None; nearest=None; count=0; feasible=0
        for k in idx:
            for angle in (0.,-.65,.65,1.57,-1.57):
                for x in np.arange(1.46,2.07,.035):
                    for y in np.arange(-.38,.39,.06):
                        q=shifted(qs[k],crop,x,y,angle)
                        phy.data.qpos[:]=q;phy.forward()
                        feet,_,mouth=scan.scan(phy)
                        if len(feet)<3 or np.linalg.norm(mouth-crop.goal)>.49:continue
                        count+=1
                        contacts=phy.data.contact[:phy.data.ncon]
                        # Permit surface tangency up to 20 microns; reject penetrating starts.
                        deepest=min((float(c.dist) for c in contacts if c.efc_address>=0 and
                            (c.geom1 in bank.skin_ids or c.geom2 in bank.skin_ids)),default=0.)
                        if deepest<-.002:continue
                        gap=min((v for v,i in (tr.cast(phy,mouth,(s,0,0),tr.FRUIT_MASK) for s in (-1.,1.))
                                 if i in bank.skin_ids and v>=0),default=100.)*10
                        feasible+=1
                        smell=tr.sample(field,phy.data.xpos[antenna,:2])
                        rec=dict(pose_index=int(k),angle=float(angle),x=float(x),y=float(y),gap_mm=gap,
                                 skin_penetration_mm=-10*deepest,feet=len(feet),odor=list(smell))
                        if nearest is None or gap<nearest['gap_mm']:nearest=rec
                        if .03<gap<=.55:
                            score=abs(gap-.24)+(.12 if min(smell)==0 else 0.)+.05*abs(angle)
                            if best is None or score<best[0]:
                                best=(score,rec,q.copy())
        if best:
            np.save(HERE/f'TERMINAL_{site}.npy',best[2])
        rows[site]=dict(tested_upright_patch_poses=count,nonpenetrating=feasible,nearest=nearest,
                        selected=best[1] if best else None)
        print(site,rows[site],flush=True)
        env.close()
    (HERE/'TERMINAL_SCAN.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
