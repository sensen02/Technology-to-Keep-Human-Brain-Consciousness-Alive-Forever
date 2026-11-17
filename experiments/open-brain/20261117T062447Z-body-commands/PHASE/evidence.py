"""Plot held pose paths with left/right VNC drives and held physical postures."""
import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from phase import HERE, study
import mujoco
from render_evidence import image


def render(out):
    model = mujoco.MjModel.from_binary_path(str(study.FIRST/'BODY.mjb'))
    data = mujoco.MjData(model)
    cases = [('held_cruise_0','Cruise'),('held_fast_0','Fast'),('held_arc_left_0','Arc left'),
             ('held_arc_right_0','Arc right'),('held_pivot_left_0','Pivot left'),
             ('held_pivot_right_0','Pivot right'),('held_back_0','Reverse'),
             ('held_brake_0','Brake'),('held_cut_steer','Left DN cut')]
    fig, axes = plt.subplots(3,3,figsize=(12,10),layout='constrained')
    pose_fig, pose_axes = plt.subplots(3,3,figsize=(12,9),layout='constrained')
    with mujoco.Renderer(model,height=330,width=440) as renderer:
        for (name,title), ax, posax in zip(cases,axes.flat,pose_axes.flat,strict=True):
            with np.load(HERE/'HELD'/name/'TRACE.npz',allow_pickle=False) as archive:
                trace = archive['trace']
                poses = archive['qpos']
                drive = trace[:,6:8]
                time = trace[:,0]
                motion_idx = 750 if name!='held_brake_0' else 1150
                posax.imshow(image(renderer,model,data,motion_idx,poses,trace))
            posax.set_title(f'{title} · {motion_idx/1000:.2f} s')
            posax.axis('off')
            xy = trace[:,1:3]-trace[0,1:3]
            ax.plot(time,drive[:,0],label='Left drive',c='#1c7590')
            ax.plot(time,drive[:,1],label='Right drive',c='#bb497b',linestyle='--')
            ax.set_ylim(-1.05,1.05)
            yaw = np.rad2deg(np.unwrap(trace[:,5])-trace[0,5])
            other = ax.twinx()
            other.plot(time,yaw,color='#9c6907',alpha=.65)
            other.set_ylim(-105,105)
            ax.set_title(title)
            ax.set_xlabel('Simulated time (s)')
            if name=='held_cruise_0':
                ax.legend(loc='lower left',frameon=False,fontsize=8)
            if name in ('held_arc_left_0','held_pivot_right_0','held_cut_steer'):
                other.set_ylabel('Yaw (°)',color='#9c6907')
    out.mkdir()
    fig.savefig(out/'COMMANDS.png',dpi=145)
    pose_fig.savefig(out/'POSTURES.png',dpi=145)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    render(args.out)
