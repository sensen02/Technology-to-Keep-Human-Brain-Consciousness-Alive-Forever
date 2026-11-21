from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from evaluate import obs


def draw(out):
    v = obs(11701,'intact')
    fig, axes = plt.subplots(2,1, figsize=(8.6,5.1), sharex=True, layout='constrained')
    start,end=2492,2537
    for i, ax in enumerate(axes):
        a=v[start:end,i*8:i*8+8]
        extent=[(start-.5)/10,(end-.5)/10,.5,8.5]
        im=ax.imshow(a.T, aspect='auto',origin='lower',cmap='RdBu_r',vmin=-1.2,vmax=1.2,extent=extent)
        ax.set_ylabel('Contact number')
        ax.set_title(('Head bank   |   shared event at 251.4 ms','Thorax bank   |   separate events at 250.0 and 252.8 ms')[i],loc='left')
        for t in ([251.4] if i==0 else [250.,252.8]):
            ax.axvline(t,color='black',alpha=.55,ls=':',lw=1)
        ax.set_yticks([1,3,5,7,8])
    axes[1].set_xlabel('Simulation time (ms)')
    fig.colorbar(im,ax=axes,label='Electrode voltage (µV), clipped at ±1.2')
    fig.savefig(out/'collision.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    draw(Path(sys.argv[1]))
