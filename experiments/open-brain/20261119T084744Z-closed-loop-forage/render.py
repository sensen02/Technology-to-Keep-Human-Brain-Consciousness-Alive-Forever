"""Render first intake or first lip encounter from the saved physical rollouts."""
import json
import sys
from pathlib import Path

from PIL import Image,ImageDraw
import numpy as np
from dm_control.mujoco import engine,wrapper

from assess import CASES
from common import init_course


def run(root):
    planned=CASES+('DOCK_15103','DOCK_15107')
    cases=[name for name in planned if (root/name/'RESULT.json').is_file() and name!='NO_FOOD']
    assert cases,root
    if len(cases)>6:
        cases=[c for c in cases if c not in ('FINAL_FLAT_B','LONG_R','BRIDGE_L')]
    width,height=360,184
    sheet=Image.new('RGB',(width*3,(height+24)*len(cases)),'#19262c')
    draw=ImageDraw.Draw(sheet)
    for row,name in enumerate(cases):
        p=root/name
        meta=json.loads((p/'RESULT.json').read_text())
        env,*_=init_course(meta['seed'],meta['challenge'],meta['side'],food=meta['food'])
        env.reset()
        phy=env.physics
        with np.load(p/'BODY.npz',allow_pickle=False) as f:
            poses=f['poses']
            trace=f['trace']
            ledger=f['meal']
        now=(meta['first_intake_s'] if meta['first_intake_s'] is not None else
             (.69*meta['first_fall_s'] if meta['first_fall_s'] is not None else .62*meta['duration_s']))
        options=wrapper.MjvOption()
        options.sitegroup[:]=0
        for col,t in enumerate((.24,now,meta['duration_s']-.01)):
            i=int(np.clip(np.searchsorted(trace[:,0],t),0,len(poses)-1))
            phy.data.qpos[:]=poses[i]
            phy.forward()
            camera=engine.MovableCamera(phy,height=height,width=width)
            camera.set_pose((.90,0.,.03),3.75,90.,-62.)
            rgb=camera.render(scene_option=options)
            x,y=col*width,row*(height+24)
            sheet.paste(Image.fromarray(rgb),(x,y+24))
            meal_idx=max(0,int(np.searchsorted(ledger[:,0],trace[i,0],side='right')-1))
            draw.text((x+8,y+5),f'{name}  {trace[i,0]:.2f}s  intake {ledger[meal_idx,1]:.2f}',
                      fill='white')
        env.close()
    sheet.save(root/'COURSES.png')


if __name__=='__main__':
    run(Path(sys.argv[1]))
