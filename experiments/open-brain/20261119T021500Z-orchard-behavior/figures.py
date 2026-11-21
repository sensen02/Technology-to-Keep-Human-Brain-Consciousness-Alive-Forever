"""Plot physical paths and inventory from the digitized orchard runs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

HERE = Path(__file__).resolve().parent
CASES = ('FRESH','EMPTY_GROUND','GUIDED')
LABELS = ('Ground food available','Ground food empty','Guided branch flight')
COLORS = ('#007d73','#c66024','#6454aa')
FRUITS = (
    ('Near ground',42.5,-8.2,6.0),('Far ground',72.5,3.8,5.6),
    ('Outer branch',84.5,38.5,6.7),('Inner branch',116.8,75.5,7.3),
    ('Table L',123.5,-35.8,5.6),('Table M',135.7,-52.3,6.9),('Table R',153.6,-40.4,5.8),
)


def draw(root, reference):
    fig, ax = plt.subplot_mosaic([['map','height'],['map','energy']],figsize=(12.3,7.2),
                                 layout='constrained',width_ratios=(1.4,1))
    title = 'Yaw-sign recheck' if root != reference else 'First behavior test'
    fig.suptitle(f'P1   |   Orchard foraging · {title}',fontsize=18,x=.02,ha='left')
    for label,x,y,r in FRUITS:
        ax['map'].add_patch(Circle((x,y),r,facecolor='#f4c3b3',edgecolor='#ae6051',zorder=1))
        ax['map'].annotate(label,(x,y),xytext=(5,8),textcoords='offset points',fontsize=8,zorder=5)
    for case,label,color in zip(CASES,LABELS,COLORS,strict=True):
        folder = (reference if case == 'GUIDED' else root)/case
        line = '--' if case == 'EMPTY_GROUND' else '-'
        t = np.load(folder/'TRACE.npy')
        m = np.load(folder/'MEAL.npy')
        r = json.loads((folder/'RESULT.json').read_text())
        ax['map'].plot(10*t[::20,1],10*t[::20,2],color=color,lw=2,ls=line,label=label,zorder=3)
        ax['map'].scatter(10*t[-1,1],10*t[-1,2],s=30,color=color,zorder=4)
        ax['height'].plot(t[::20,0],10*t[::20,3],color=color,lw=1.6,ls=line)
        air = r['first_air_s']
        if air is not None:
            i=np.searchsorted(t[:,0],air)
            ax['map'].scatter(10*t[i,1],10*t[i,2],marker='^',s=70,
                              facecolor='white',edgecolor=color,zorder=6)
            ax['height'].axvline(air,color=color,lw=1,ls='--',alpha=.5)
        ax['energy'].plot(m[:,0],m[:,3],color=color,lw=1.6,ls=line)
        ax['energy'].plot(m[:,0],np.cumsum(m[:,2]),color=color,lw=1.2,ls=':')
    ax['map'].scatter(0,0,c='#222',marker='+',s=70,zorder=4)
    ax['map'].set(title='Actual trajectories and food surfaces',xlabel='x (mm)',ylabel='y (mm)')
    ax['map'].set_aspect('equal',adjustable='box')
    ax['map'].legend(loc='best',frameon=False,fontsize=9)
    ax['height'].set(title='Center-of-mass height',xlabel='Simulated time (s)',ylabel='Height (mm)')
    ax['height'].axhline(6.8,color='#aaa',zorder=0,lw=1)
    ax['height'].text(.98,6.8,'Outer branch fruit surface',fontsize=8,va='bottom',ha='right',
                      transform=ax['height'].get_yaxis_transform())
    ax['energy'].set(title='Reserve (solid) and total intake (dotted)',xlabel='Simulated time (s)',
                     ylabel='Engineering units')
    for plot in ax.values():
        plot.grid(alpha=.23)
        plot.spines[['top','right']].set_visible(False)
    fig.savefig(root/'ORCHARD_BEHAVIOR.png',dpi=160,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--reference',type=Path)
    args=p.parse_args()
    draw(args.root.resolve(),(args.reference or args.root).resolve())
