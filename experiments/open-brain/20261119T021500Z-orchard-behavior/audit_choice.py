"""Label selected eye regions after the run, using saved body poses only."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

from run_case import OrchardEyes, RESOURCES, SEED, build_orchard


def audit(out):
    env,_ = build_orchard()
    env.random_state.seed(SEED)
    env.reset()
    phy=env.physics
    eye=OrchardEyes(phy)
    geom_names={}
    for index in range(phy.model.ngeom):
        name = phy.model.id2name(index,'geom') or ''
        if name.startswith(('food_','rind_')):
            geom_names[index] = name.split('_',1)[1]
        if name.startswith('fruit_dimple_'):
            geom_names[index] = RESOURCES[int(name.rsplit('_',2)[1])].name
    rows=[]
    tiles=[]
    for case in ('FRESH','EMPTY_GROUND'):
        folder=out/case
        obs=[json.loads(line) for line in (folder/'EYES.jsonl').read_text().splitlines()]
        facts=json.loads((folder/'RESULT.json').read_text())
        poses=np.load(folder/'QPOS.npy',allow_pickle=False)
        first_pose=np.load(folder/'INITIAL.npy',allow_pickle=False)[:phy.model.nq]
        takeoff=facts['takeoff_s']
        if case=='FRESH':
            first_intake=min(facts['first_intake'].values(),default=2.)
            sample=[0, max(0,int(first_intake/.02)-1)]
        else:
            sample=[round(1.98/.02)]
            if takeoff is not None:
                launch=int(takeoff/.02)
                sample+=list(range(max(0,launch-3),launch+1))
        for i in sample:
            row=obs[i]
            choice=row['selection']
            if not choice or 'box' not in choice:
                continue
            t=row['time_s']
            j=round(t/.005)-1
            phy.data.qpos[:]=first_pose if j<0 else poses[j]
            phy.forward()
            left,_,eyes,_=eye.sample()
            identifiers=eye.render(segmentation=True)[0][...,0]
            x0,y0,x1,y1=choice['box']
            ids=identifiers[y0:y1,x0:x1]
            matches={}
            for h,name in geom_names.items():
                matches[name]=matches.get(name,0)+int((ids==h).sum())
            match=max(matches,key=matches.get)
            iou=matches[match]/ids.size
            assert any(tuple(choice['box'])==item.box for item in eyes), (case,t,choice)
            rows.append(dict(case=case,time_s=t,high=choice['high'],pixel_box=choice['box'],
                             fruit=match if iou>=.28 else None,box_fruit_fraction=iou))
            if i in (sample[0],sample[-1]):
                frame=Image.fromarray(left).resize((768,432),Image.Resampling.NEAREST)
                draw=ImageDraw.Draw(frame)
                draw.rectangle((2*x0,2*y0,2*x1,2*y1),outline='#fff395',width=3)
                caption=f"{case}   {t:.2f}s   {match if iou>=.28 else 'unmatched'}"
                draw.rectangle((8,8,365,35),fill='#20252d')
                draw.text((16,14),caption,fill='white',font_size=15)
                tiles.append(frame)
    (out/'TARGET_AUDIT.json').write_text(json.dumps(rows,indent=2)+'\n')
    montage=Image.new('RGB',(1536,432*((len(tiles)+1)//2)),'#22303b')
    for i,im in enumerate(tiles):
        montage.paste(im,((i%2)*768,(i//2)*432))
    montage.save(out/'SELECTED_FRUITS.png')
    env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    audit(p.parse_args().root)
