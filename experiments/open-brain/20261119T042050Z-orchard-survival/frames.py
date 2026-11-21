"""Inspect key events from the saved physical poses, without a second simulation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'20261119T012530Z-survival-orchard'))
from validate import camera_frame
from orchard import build_orchard


def frames(root, output, diagnostics, ground_extension):
    env,_ = build_orchard()
    env.reset()
    phy=env.physics
    size=(560,365)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18)
    conditions=[(root,'FRESH','有地果 · 首口'),
                (root,'EMPTY_GROUND','空地果 · 离地'),
                (root,'GUIDED','给定枝果 · 结束'),
                (root,'FRESH','有地果 · 耗竭时姿态')]
    if diagnostics and (diagnostics/'GUIDED'/'RESULT.json').exists():
        conditions += [(diagnostics,'GUIDED','平滑引导复测 · 接近最佳时刻')]
    ground=ground_extension or diagnostics
    if ground and (ground/'FRESH'/'RESULT.json').exists():
        conditions += [(ground,'FRESH','地面制动复测 · 停稳进食')]
    montage=Image.new('RGB',(size[0]*2,(size[1]+36)*((len(conditions)+1)//2)),'#19242a')
    draw=ImageDraw.Draw(montage)
    for pane,(run,name,caption) in enumerate(conditions):
        folder=run/name
        result=json.loads((folder/'RESULT.json').read_text())
        trace=np.load(folder/'TRACE.npy',allow_pickle=False)
        qpos=np.load(folder/'QPOS.npy',allow_pickle=False)
        stamp=min(result['first_intake'].values(),default=result['elapsed_s'])
        if pane==3:
            stamp=result['elapsed_s']
        if name=='FRESH' and run==ground:
            stamp=min(stamp+1.6,result['elapsed_s'])
        if name=='EMPTY_GROUND':
            stamp=(result['first_air_s'] or result['elapsed_s']-.06)+.065
        if name=='GUIDED' and run==diagnostics:
            nearest=np.argmin(np.hypot(trace[:,1]-8.45,trace[:,2]-3.85))
            stamp=trace[nearest,0]
        idx=np.clip(round(stamp/.005)-1,0,len(qpos)-1)
        phy.data.qpos[:]=qpos[idx]
        phy.forward()
        center=phy.named.data.subtree_com['walker/'].copy()
        frame=Image.fromarray(camera_frame(phy,(center,1.6,131,-22),*size))
        x,y=(pane%2)*size[0],(pane//2)*(size[1]+36)
        montage.paste(frame,(x,y+36))
        draw.text((x+13,y+4),f'{caption}   {(idx+1)*.005:.3f} s',
                  fill='#f4fafc',font=font)
    montage.save(output)
    env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--diagnostics',type=Path)
    parser.add_argument('--ground-extension',type=Path)
    args=parser.parse_args()
    frames(args.root,args.output,args.diagnostics,args.ground_extension)
