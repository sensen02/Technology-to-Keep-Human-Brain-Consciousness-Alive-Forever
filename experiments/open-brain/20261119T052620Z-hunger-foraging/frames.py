"""Render physical poses already produced by the foraging trials."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'20261119T012530Z-survival-orchard'))
from orchard import build_orchard
from validate import camera_frame

CASES=(('FULL','高饱足 · 仍在原处'),
       ('HUNGRY','低饱足 · 第二颗地果'),
       ('FADE','循环营养下降 · 第二颗地果'))


def run(root,out):
    env,_=build_orchard()
    env.reset()
    phy=env.physics
    width,height=545,360
    image=Image.new('RGB',(width*len(CASES),height+35),'#172427')
    draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18)
    for i,(case,label) in enumerate(CASES):
        directory=root/case
        result=json.loads((directory/'RESULT.json').read_text())
        poses=np.load(directory/'QPOS.npy',allow_pickle=False)
        t=result['first_intake_s']
        if t is None and case!='FULL':
            label=label.split(' · ')[0]+' · 回合结束'
        end=result['duration']
        if t is not None:
            ledger=np.load(directory/'MEAL.npy',allow_pickle=False)
            positive=ledger[ledger[:,5]>0,0]
            t=positive[np.abs(positive-(t+.45)).argmin()]
        else:
            t=end
        key=np.clip(round(t/.005)-1,0,len(poses)-1)
        phy.data.qpos[:]=poses[key]
        phy.forward()
        center=phy.named.data.subtree_com['walker/'].copy()
        frame=camera_frame(phy,(center,1.5,132,-23),width,height)
        offset=i*width
        image.paste(Image.fromarray(frame),(offset,35))
        draw.text((offset+10,5),f'{label}  {(key+1)*.005:.2f}s',fill='white',font=font)
    image.save(out)
    env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    run(args.root,args.output)
