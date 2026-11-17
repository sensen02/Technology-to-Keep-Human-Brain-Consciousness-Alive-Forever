"""Render synchronized held Eon→BANC→body trajectories from saved physical states."""
import argparse
from pathlib import Path
import subprocess

from overlap import study
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from render_evidence import image


def render(formal,out):
    out.mkdir()
    names=[('271_cruise_open','oDN1 · cruise'),('271_fast_open','oDN1 · fast'),
           ('271_a01_left_open','DNa01 · left'),('271_a01_right_open','DNa01 · right'),
           ('271_a02_left_open','DNa02 · left'),('271_a02_right_open','DNa02 · right'),
           ('271_a01_left_steer','Left DNa01 cut'),('271_a02_left_steer','Left DNa02 cut')]
    model=mujoco.MjModel.from_binary_path(str(study.FIRST/'BODY.mjb'))
    data=mujoco.MjData(model)
    traces=[]
    for name,title in names:
        with np.load(formal/name/'TRACE.npz',allow_pickle=False) as saved:
            traces.append((title,saved['trace'],saved['qpos']))
    fig,axs=plt.subplots(2,4,figsize=(15,8),layout='constrained')
    for ax,(title,raw,_) in zip(axs.flat,traces,strict=True):
        local=raw[:,1:3]-raw[0,1:3]
        ax.plot(local[:,0],local[:,1],color='#1d587e')
        ax.set_aspect('equal')
        t = np.arange(2500,13000,2000)
        theta=raw[t,5]
        ax.quiver(local[t,0],local[t,1],np.cos(theta),np.sin(theta),color='#bb608e',scale=11)
        ax.set_xlim(-.5,12)
        ax.set_ylim(-4.,4.)
        ax.set_title(title)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
    fig.savefig(out/'TRAJECTORIES.png',dpi=135)
    font=ImageFont.load_default(size=15)
    chosen=(traces[2],traces[6],traces[4],traces[7])
    command=['ffmpeg','-nostdin','-loglevel','error','-f','rawvideo','-pixel_format','rgb24',
             '-video_size','720x528','-framerate','30','-i','pipe:0','-c:v','libx264',
             '-pix_fmt','yuv420p',str(out/'TURN_WITH_AND_WITHOUT_DN.mp4')]
    positions=(0,230,600,1030,1550)
    review=Image.new('RGB',(5*360,4*264),'white')
    with mujoco.Renderer(model,height=240,width=360) as renderer:
        def tile(title,raw,qpos,idx):
            im=Image.new('RGB',(360,264),'#19242d')
            im.paste(Image.fromarray(image(renderer,model,data,idx,qpos,raw)),(0,24))
            painter=ImageDraw.Draw(im)
            painter.text((7,3),f'{title} | {idx/1000:.3f} s | 0.60x',font=font,fill='white')
            return im
        for row,(title,raw,qpos) in enumerate(chosen):
            for col,idx in enumerate(positions):
                review.paste(tile(title,raw,qpos,idx),(col*360,row*264))
        review.save(out/'CUT_REVIEW.png')
        with subprocess.Popen(command,stdin=subprocess.PIPE) as encoder:
            for idx in range(0,1700,20):
                frame=Image.new('RGB',(720,528))
                for panel,(title,raw,qpos) in enumerate(chosen):
                    frame.paste(tile(title,raw,qpos,idx),((panel%2)*360,(panel//2)*264))
                encoder.stdin.write(np.asarray(frame,dtype=np.uint8).tobytes())
            encoder.stdin.close()
            assert encoder.wait()==0
    print('8 held trajectory panels; 85 frames, indexed native states',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('formal',type=Path)
    parser.add_argument('out',type=Path)
    arg=parser.parse_args()
    render(arg.formal,arg.out)
