"""Render paired physical records and the experimental mode-selection contrasts."""
import argparse
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from dm_control.mujoco.engine import MovableCamera


HERE=Path(__file__).resolve().parent
PALETTE = {
    'HIGH': ('Elevated fruit, adapter', '#117d73'),
    'LOW': ('Low fruit, adapter', '#456fb0'),
    'NO_ADAPTER': ('Elevated fruit, adapter off', '#8a5837'),
    'CUT': ('Elevated fruit, DNp11 cut', '#646878'),
}


def physical(run):
    facts=json.loads((run/'RESULT.json').read_text())
    return facts,dict(zip(facts['columns'],np.load(run/'TRACE.npy',allow_pickle=False).T,strict=True))


def plot(output):
    fig,axes=plt.subplots(2,2,figsize=(11.1,6.7),layout='constrained')
    for key,(label,color) in PALETTE.items():
        facts,trace=physical(HERE/key)
        times=trace['time_s']
        axes[0,0].plot(times,10*trace['z_cm'],label=label,color=color)
        axes[0,1].plot(times,10*trace['x_cm'],color=color)
        axes[1,1].plot(times,trace['intake'],color=color)
        with np.load(HERE/key/'NEURAL.npz',allow_pickle=False) as syn:
            tau=np.arange(len(syn['values']))*.0001
            axes[1,0].plot(tau,np.min(syn['values'][:,:2],axis=1),color=color)
    axes[0,0].legend(loc='upper right',fontsize=9,frameon=False)
    axes[0,0].set(title='Takeoff and landing occur only for the elevated fruit',ylabel='Body height (mm)')
    axes[0,1].axhline(80.3,ls=':',lw=1.,color='#7f8690')
    axes[0,1].set(title='Forward approach with both motor modes',ylabel='Forward position (mm)')
    axes[1,0].axhline(.22,ls=':',lw=1.,color='#7f8690',label='Takeoff threshold')
    axes[1,0].set(title='Bilateral DNp11 after the cervical circuit',ylabel='VNC readout',xlim=(0,.16))
    axes[1,0].legend(loc='upper right',fontsize=9,frameon=False)
    axes[1,1].set(title='Food physically transferred from the fruit',ylabel='Cumulative intake (simulation units)')
    for ax in axes.ravel():
        ax.set_xlabel('Simulation time (s)')
        ax.grid(alpha=.20)
    fig.savefig(output,dpi=155)
    plt.close(fig)


def render(run, output):
    from common_eye import scene

    facts, trace=physical(run)
    selection=facts['mode']=='elevated'
    poses=np.load(run/'QPOS.npy',allow_pickle=False)
    geom=facts['geometry']
    env=scene(geom['x_cm'],selection)
    env.reset()
    phy=env.physics
    for i in range(phy.model.ngeom):
        if 'ghost' in (phy.model.id2name(i,'geom') or ''):
            phy.model.geom_rgba[i,3]=0.
    phy.model.site_rgba[:,3]=0.
    floor=phy.model.name2id('groundplane','geom')
    phy.model.geom_size[floor,:2]=25.
    camera=MovableCamera(phy,height=432,width=768)
    facts_takeoff=facts['takeoff_s']
    facts_touch=facts['first_fruit_contact_s']
    intake=facts['first_intake_s']
    resolution=tuple(camera.render().shape[:2][::-1])
    assert resolution==(768,432)
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24',
         '-video_size','768x432','-framerate','24','-i','-','-an','-threads','2',
         '-c:v','libx264','-pix_fmt','yuv420p',str(output)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as encoder:
        stride=54 if selection else 125
        for i in range(stride-1,len(poses),stride):
            k=4*i+3
            t=float(trace['time_s'][k])
            x,y,z=(float(trace[key][k]) for key in ('x_cm','y_cm','z_cm'))
            phy.data.qpos[:]=poses[i]
            phy.forward()
            close=abs(geom['x_cm']-x) < 2. or t<.21
            center=(x+.13,y,max(z,.22)) if close else ((x+geom['x_cm'])/2,y,.36)
            camera.set_pose(lookat=center,distance=1.60 if close else 5.1,azimuth=121,elevation=-31 if not close else -38)
            canvas=Image.fromarray(camera.render())
            draw=ImageDraw.Draw(canvas)
            draw.rectangle((0,0,767,35),fill='#142f43')
            phase='Feeding' if intake and t>=intake else 'Perched' if facts_touch and t>=facts_touch and selection else (
                'Flight' if facts_takeoff and t>=facts_takeoff else 'Ground')
            draw.text((15,11),f"{'Elevated' if selection else 'Low'} fruit   |   {t:5.3f} s   |   {phase}"
                      f"   |   Intake {trace['intake'][k]:.3f}",fill='white')
            encoder.stdin.write(canvas.tobytes())
        encoder.stdin.close()
        assert encoder.wait()==0
    env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    p.add_argument('--run',type=Path)
    a=p.parse_args()
    if a.run:
        render(a.run,a.output)
    else:
        plot(a.output)
