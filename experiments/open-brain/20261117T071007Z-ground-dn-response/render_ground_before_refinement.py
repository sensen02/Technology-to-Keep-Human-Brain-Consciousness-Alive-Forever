"""Render measured ground responses and their same-stimulus cut controls."""
import argparse
import json
from pathlib import Path
import subprocess

from ground_response import DT, TICKS, TARGETS, first
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from render_evidence import image


def render(formal,out):
    out.mkdir()
    model=mujoco.MjModel.from_binary_path(str(first.HERE/'BODY.mjb'))
    data=mujoco.MjData(model)
    trials={}
    for name in TARGETS:
        trials[name]={}
        for arm in ('open','cut'):
            folder=formal/f'{name}_held0_{arm}'
            case=json.loads((folder/'RESULT.json').read_text())['case']
            with np.load(folder/'TRACE.npz',allow_pickle=False) as pack:
                trials[name][arm]=(case,pack['trace'],pack['qpos'],pack['feet'])
    fig, axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    titles={'stride_left':'DNg13 left','stride_right':'DNg13 right',
            'bluebell':'DNg60 stop','web_b':'DNg74_b stop'}
    for column,(name,arms) in enumerate(trials.items()):
        labelled, control=arms['open'],arms['cut']
        for top,(_,trace,_,feet) in ((True,labelled),(False,control)):
            label='Intact' if top else 'Target cut'
            time=np.arange(len(trace))*DT
            start=labelled[0]['action']*DT
            stop=labelled[0]['off']*DT
            c='#175b84' if top else '#b46534'
            if name.startswith('stride'):
                angle=np.rad2deg(np.unwrap(trace[:,5])-trace[labelled[0]['action'],5])
                axes[0,column].plot(time,angle,color=c,label=label)
                if top:
                    i,j=(0,3) if name.endswith('left') else (3,0)
                    axes[1,column].plot(time,feet[:,i],label='Inner front foot')
                    axes[1,column].plot(time,feet[:,j],label='Outer front foot')
                    axes[1,column].set_ylabel('Foot position in thorax x (mm)')
            else:
                step=np.linalg.norm(np.diff(trace[:,1:3],axis=0,prepend=trace[:1,1:3]),axis=1)/DT
                mean=np.convolve(step,np.full(401,1./401),mode='same')
                axes[0,column].plot(time,mean,color=c,label=label)
                axes[1,column].plot(time,np.abs(trace[:,6:8]).max(1),color=c,label=label)
                axes[1,column].axvspan(start+.26,start+.54,color='#999999',alpha=.12)
                axes[1,column].set_ylabel('Body drive amplitude')
            axes[0,column].axvspan(start,stop,color='#999999',alpha=.06)
        axes[0,column].set_title(titles[name])
        axes[0,column].set_ylabel('Yaw (°)' if name.startswith('stride') else 'Horizontal speed (mm/s)')
        for ax in axes[:,column]:
            ax.legend(loc='best',frameon=False,fontsize=8)
            ax.set_xlabel('Simulated time (s)')
    fig.savefig(out/'RESPONSE.png',dpi=142)

    selected=[(titles[name],trials[name]) for name in ('stride_left','bluebell','web_b')]
    collage=Image.new('RGB',(2*360,3*264),'white')
    font=ImageFont.load_default(size=15)
    command=['ffmpeg','-nostdin','-loglevel','error','-f','rawvideo','-pixel_format','rgb24',
             '-video_size','720x792','-framerate','30','-i','pipe:0','-c:v','libx264',
             '-pix_fmt','yuv420p',str(out/'GROUND_DN_CUT_PAIRS.mp4')]
    review=Image.new('RGB',(6*360,3*264),'white')
    with mujoco.Renderer(model,height=240,width=360) as renderer:
        def tile(title,mode,index,trace,qpos):
            block=Image.new('RGB',(360,264),'#19242d')
            picture=image(renderer,model,data,index,qpos,trace)
            block.paste(Image.fromarray(picture),(0,24))
            ImageDraw.Draw(block).text((7,4),f'{title} / {mode} | {index/1000:.3f} s',font=font,fill='white')
            return block

        for row,(title,arms) in enumerate(selected):
            for col,(arm,(_,trace,qpos,_)) in enumerate(arms.items()):
                for k,index in enumerate((180,720,1220)):
                    review.paste(tile(title,arm,index,trace,qpos),((col*3+k)*360,row*264))
        review.save(out/'PHYSICAL_REVIEW.png')
        with subprocess.Popen(command,stdin=subprocess.PIPE) as encoder:
            for i in range(0,TICKS//10,20):
                for row,(title,arms) in enumerate(selected):
                    for col,(arm,(_,trace,qpos,_)) in enumerate(arms.items()):
                        collage.paste(tile(title,arm,i,trace,qpos),(col*360,row*264))
                encoder.stdin.write(np.asarray(collage,dtype=np.uint8).tobytes())
            encoder.stdin.close()
            assert encoder.wait()==0
    print('Physical trace render and video complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('formal',type=Path)
    parser.add_argument('out',type=Path)
    a=parser.parse_args()
    render(a.formal,a.out)
