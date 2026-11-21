"""Recover fruit identity and oral geometry from saved full-body poses."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'20261119T052620Z-hunger-foraging'/'VERIFIED'
sys.path.insert(0,str(HERE.parent/'20261119T012530Z-survival-orchard'))
from orchard import build_orchard, RESOURCES
from orchard_eye import OrchardEyes
from validate import camera_frame


def audit():
    env,_=build_orchard()
    env.reset()
    phy=env.physics
    eyes=OrchardEyes(phy)
    fruit=RESOURCES[1]
    fruit_ids=[]
    for i in range(phy.model.ngeom):
        n=phy.model.id2name(i,'geom') or ''
        if n==fruit.geom or n.startswith('fruit_dimple_1_'):
            fruit_ids.append(i)
    mouth_ids=[phy.model.name2id('walker/labrum_'+side+'_lower_collision','geom') for side in ('left','right')]
    out=[]
    panel=[]
    for case,stamps in (('FADE',(4.42,4.66,4.72,4.76,4.82,4.84,4.86)),
                        ('HUNGRY',(2.48,2.62,2.66,2.69,3.10))):
        d=BASE/case
        qpos=np.load(d/'QPOS.npy',allow_pickle=False)
        inputs=[json.loads(line) for line in (d/'EYES.jsonl').read_text().splitlines()]
        for t in stamps:
            phy.data.qpos[:]=qpos[round(t/.005)-1]
            phy.forward()
            mouth=phy.data.geom_xpos[mouth_ids].mean(axis=0)
            center=phy.named.data.subtree_com['walker/'].copy()
            eye=inputs[round(t/.02)]
            chosen=eye['selection']
            eyes.pose(*eyes.head_pose())
            left=eyes.render(segmentation=True)[0][...,0]
            if chosen and chosen['box']:
                x0,y0,x1,y1=chosen['box']
                fruit_fraction=float(np.isin(left[y0:y1,x0:x1],fruit_ids).mean())
            else:
                fruit_fraction=None
            out.append(dict(case=case,time_s=t,ground_far_pixels=int(np.isin(left,fruit_ids).sum()),
                            selected_box_second_fruit_fraction=fruit_fraction,
                            mouth_planar_mm=float(10*np.hypot(mouth[0]-fruit.x,mouth[1]-fruit.y)),
                            mouth_height_over_surface_mm=float(10*(mouth[2]-fruit.top))))
            if (case,t) in [('FADE',4.66),('FADE',4.72),('FADE',4.84),('HUNGRY',2.69)]:
                external=Image.fromarray(camera_frame(phy,(center,1.35,132,-22),600,355))
                panel.append((case,t,external))
    (HERE/'POSE_REVIEW.json').write_text(json.dumps(dict(
        mouth_planar_limit_mm=10*(fruit.radius-.035),mouth_height_range_mm=(-.02,1.4),samples=out),
        indent=2)+'\n')
    canvas=Image.new('RGB',(1200,782),'#192328')
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',17)
    captions=('下降组 · 果缘初接触','下降组 · 支撑减少',
              '下降组 · 非足碰撞前','低饱足对照 · 实际摄入')
    for i,((_,t,im),caption) in enumerate(zip(panel,captions,strict=True)):
        x,y=i%2*600,i//2*391
        canvas.paste(im,(x,y+36))
        draw.text((x+10,y+7),f'{caption}  {t:.3f} s',font=font,fill='white')
    canvas.save(HERE/'FRUIT_EDGE_FRAMES.png')
    env.close()


if __name__=='__main__':
    audit()
