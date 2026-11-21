"""Render stored MuJoCo poses and the matching physical run's optical frames."""
import argparse
import json
from pathlib import Path

from dm_control.mujoco import engine, wrapper
import numpy as np
from PIL import Image, ImageDraw

from live_compound import make_board
from visible_food_probe import board_with_cue


def render(folder, out, cue):
    record = json.loads((folder/'RESULT.json').read_text())
    factory = board_with_cue if cue else make_board
    env, _, _ = factory(record['seed'], record['challenge'], record['side'], food=True)
    env.reset()
    phys = env.physics
    with np.load(folder/'BODY.npz',allow_pickle=False) as z:
        trace = z['trace']
        poses = z['poses']
        meal = z['meal']
        optical = z['visual']
    option = wrapper.MjvOption()
    option.sitegroup[:] = 0
    phys.model.light_ambient[:] *= .75
    phys.model.light_diffuse[:] *= .75
    width, height = 320, 205
    graph = Image.new('RGB',(4*width,height+25+108),'#212d39')
    draw = ImageDraw.Draw(graph)
    times = np.linspace(max(.18,trace[0,0]),trace[-1,0],4)
    for i,t in enumerate(times):
        index=int(np.clip(np.searchsorted(trace[:,0],t),0,len(trace)-1))
        position=trace[index]
        phys.data.qpos[:]=poses[index]
        phys.forward()
        camera=engine.MovableCamera(phys,height=height,width=width)
        camera.set_pose((max(.65,position[1]-.05),position[2]*.60,.035),2.33,87.,-53.)
        graph.paste(Image.fromarray(camera.render(scene_option=option)),(width*i,25))
        last = max(0,np.searchsorted(meal[:,0],position[0],side='right')-1)
        draw.text((8+width*i,7),f"{position[0]:.2f} s   intake {meal[last,1]:.3f}", fill='white')
        # The physical runner keeps the first 3 camera frames, then one per
        # 0.4 s. Use the nearest stored frame to each replay pose.
        capture_times=np.r_[[.16,.24,.32],np.arange(.40,30.,.40)][:len(optical)]
        frame = Image.fromarray(optical[int(np.argmin(np.abs(capture_times-t)))])
        x=width*i+(width-192)//2
        graph.paste(frame.resize((192,108)),(x,25+height))
    env.close()
    graph.save(out)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--visible-cue',action='store_true')
    args=p.parse_args()
    render(args.folder,args.out,args.visible_cue)
