"""Find curriculum spawning poses near actual reachable skin. No physics rollout or reward here."""
from pathlib import Path
import json
import numpy as np
from rollout import HERE,tr

def main():
    out={}
    for site in ('CLEAR','DEBRIS'):
        crop=tr.Crop(site,'mesh')
        env,bank=tr.build(crop)
        env.reset()
        phy=tr.bind_physics(env,bank)
        base=tr.fit_spawn(crop,env,tr.docking.start_pose)
        initial=tr.Rotation.from_quat(base[[4,5,6,3]])
        geom=phy.model.ptr
        walker=np.array([geom.body(geom.geom_bodyid[i]).name.startswith('walker/') for i in range(geom.ngeom)])
        feet=tr.docking.Contacts(phy)
        field,_=tr.odor(crop)
        antennas=[phy.model.name2id('walker/antenna_'+s,'body') for s in ('left','right')]
        rows=[]
        original=crop.floor
        for x in np.arange(1.52,2.13,.04):
            for y in np.arange(-.4,.41,.08):
                # Use the same local surface registration as a physical run.
                crop.floor=lambda yx: original(np.asarray(yx)+[y,x])
                q=tr.fit_spawn(crop,env,tr.docking.start_pose)
                crop.floor=original
                q[:2]+=[x,y]
                flat=tr.Rotation.from_quat(q[[4,5,6,3]])
                for yaw in (0.,-.8,.8,-1.55,1.55,3.14):
                    rot=tr.Rotation.from_rotvec([0.,0.,yaw])*flat
                    phy.data.qpos[:]=q
                    phy.data.qpos[3:7]=rot.as_quat()[[3,0,1,2]]
                    phy.forward()
                    foot,other,mouth=feet.scan(phy)
                    gaps=[tr.cast(phy,mouth,(s,0.,0.),tr.FRUIT_MASK) for s in (1.,-1.)]
                    gap=min((v for v,i in gaps if i in bank.skin_ids and v>=0),default=100.)
                    if gap>.16 or np.linalg.norm(mouth-crop.goal)>.48 or len(foot)<3:continue
                    skin_contacts=[c for c in phy.data.contact[:phy.data.ncon] if c.efc_address>=0 and
                                   ((c.geom1 in bank.skin_ids and walker[c.geom2]) or
                                    (c.geom2 in bank.skin_ids and walker[c.geom1]))]
                    if skin_contacts: continue
                    smell=tr.sample(field,phy.data.xpos[antennas,:2])
                    rows.append(dict(x=float(x),y=float(y),yaw=yaw,gap_mm=10*gap,odor=list(smell)))
        env.close()
        out[site]={}
        for key,lo,hi in [('approach',.55,1.6),('bite',0.,.55)]:
            group=[r for r in rows if lo<r['gap_mm']<=hi]
            group.sort(key=lambda r:(-int(min(r['odor'])>.0001),abs(r['gap_mm']-(.7 if key=='approach' else .4)),abs(r['yaw'])))
            out[site][key]=group[:3]
        print(site,out[site],flush=True)
    (HERE/'SPAWNS.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
