"""Render the exploratory wall transfer using the saved head observations and poses."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np


def run(p):
    fig, axes=plt.subplots(1,2,figsize=(11.2,4.9),layout='constrained')
    names=('WALL_PRIMARY','WALL_ALT','WALL_VIS','WALL_ODOR')
    colors=('#407f72','#395eac','#857e86','#ba743d')
    labels=('Both cues, initial 1','Both cues, initial 2','Vision only','Odor only')
    for key,color,label in zip(names,colors,labels,strict=True):
        case=p/key
        meta=json.loads((case/'RESULT.json').read_text())
        with np.load(case/'TRACE.npz',allow_pickle=False) as data:
            a=data['trace'][:,meta['qpos_columns']:]
            axes[0].plot(a[:,1],a[:,2],color=color,lw=1.7,label=label)
            contact=np.flatnonzero(a[:,7]>0.)
            if len(contact):
                axes[0].scatter(*a[contact[0],1:3],facecolors='white',edgecolors=color,zorder=10)
            axes[1].plot(a[:,0],a[:,8],color=color,lw=1.8)
    axes[0].add_patch(Rectangle((4.3,-2.55),.4,5.10,facecolor='#343945'))
    axes[0].add_patch(Circle((9.,0.),.6,facecolor='#eb846b',edgecolor='#a83d25',alpha=.7))
    axes[0].scatter([0.],[0.],color='black',s=21,zorder=5)
    axes[0].set(xlabel='World x (mm)',ylabel='World y (mm)',title='Open-field policy transferred to a wall')
    axes[0].set_aspect('equal')
    axes[0].legend(frameon=False,loc='upper left',fontsize=8)
    axes[1].set(xlabel='Time (s)',ylabel='Cumulative intake (units)',title='Actual food transfer')
    for a in axes:
        a.grid(alpha=.2)
        a.spines[['top','right']].set_visible(False)
    fig.savefig(p/'WALL_TRANSFER.png',dpi=151)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    run(parser.parse_args().run)
