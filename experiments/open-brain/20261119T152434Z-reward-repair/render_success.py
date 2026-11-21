"""Render saved MuJoCo poses from the successful held-out right-hand bridge."""
import json
from pathlib import Path
import sys

from PIL import Image,ImageDraw
import numpy as np
from dm_control.mujoco import engine,wrapper

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'20261119T084744Z-closed-loop-forage'))
from common import init_course


def main():
    folder=HERE/'HOLD_R'
    facts=json.loads((folder/'RESULT.json').read_text())
    assert facts['meal_pass']
    env,*_=init_course(facts['seed'],facts['challenge'],facts['side'],food=True)
    env.reset()
    physics=env.physics
    opt=wrapper.MjvOption()
    opt.sitegroup[:]=0
    physics.model.light_ambient[:]*=.73
    physics.model.light_diffuse[:]*=.76
    with np.load(folder/'BODY.npz',allow_pickle=False) as data:
        poses=data['poses']
        path=data['trace']
        meals=data['meal']
    times=(.32,1.65,facts['first_far_cross'][0],facts['first_intake_s'],facts['duration_s']-.015)
    width,height=320,216
    board=Image.new('RGB',(len(times)*width,height+25),'#1e2932')
    draw=ImageDraw.Draw(board)
    for i,t in enumerate(times):
        index=int(np.clip(np.searchsorted(path[:,0],t),0,len(path)-1))
        physics.data.qpos[:]=poses[index]
        physics.forward()
        view=engine.MovableCamera(physics,height=height,width=width)
        view.set_pose((max(.72,path[index,1]),-.09,.04),2.25,89.,-49.)
        scene=view.render(scene_option=opt)
        photos=Image.fromarray(scene)
        x=i*width
        board.paste(photos,(x,25))
        j=int(np.searchsorted(meals[:,0],path[index,0],side='right')-1)
        draw.text((x+9,6),f'{path[index,0]:.3f} s   intake {meals[max(j,0),1]:.3f}',fill='white')
    env.close()
    board.save(HERE/'LIVE_WIDE_MEAL.png')


if __name__=='__main__':
    main()
