"""Plot measured per-leg support, clearance, contact speed, and body trajectory."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

LEGS=('LF','LH','LM','RF','RH','RM')
W=.002494271478*9801.

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    run=Path(a.run).resolve();r=json.loads((run/'RESULT.json').read_text());ann=r['autonomous']
    with np.load(run/'trace_1ms.npz',allow_pickle=False) as z:
        t=z['time'].copy();load=z['loads'].copy()/W;pad=z['padz'].copy();speed=z['speed'].copy();com=z['com'].copy();source=z['control_source'].copy()
    dt=.001;boundary=r['transition']['time_s'];auto=source==3
    fig,axes=plt.subplots(5,1,figsize=(14,12),layout='constrained',sharex=True)
    for ax,data,title,cmap in [(axes[0],load,'Actual per-leg foot load / body weight','viridis'),(axes[1],pad,'Actual per-leg pad bottom height (native length)','coolwarm'),(axes[2],load*speed,'Actual per-leg load/W x contact speed (slip numerator integrand)','magma')]:
        image=ax.imshow(data.T,aspect='auto',origin='upper',extent=[t[0]-dt/2,t[-1]+dt/2,5.5,-.5],cmap=cmap,interpolation='nearest',norm=PowerNorm(.25) if ax is axes[2] else None)
        if ax is axes[2]:title += ' (power color scale .25, no values clipped)'
        ax.set_yticks(range(6),LEGS);ax.set_title(title,fontsize=11);fig.colorbar(image,ax=ax,pad=.01)
    axes[3].plot(t,com[:,2],label='COM height',color='#136f9b');axes[3].axhline(1.2,color='#a33b29',linestyle='--',label='Engineering floor1.20');axes[3].set_ylabel('Native height');axes[3].legend(loc='upper right')
    axes[4].plot(t,com[:,0]-com[0,0],label='World x displacement',color='#136f9b');axes[4].plot(t,com[:,1]-com[0,1],label='World y displacement',color='#a33b29');axes[4].set_ylabel('Native length');axes[4].set_xlabel('Actual physical time (seconds)');axes[4].legend(loc='upper left')
    for ax in axes:
        ax.axvline(1.,color='grey',linestyle=':');ax.axvline(boundary,color='black',linestyle='--');ax.set_xlim(0,t[-1])
    title='ENGINEERING PASS' if ann['all_gates'] else 'ENGINEERING FAIL'
    fig.suptitle(f'Actual full-body neural handover: {title}; max slip={ann["slip_ratio_max"]:.9f}\n0..1s zero settle |1..5.8s algorithm teacher |5.8..15.4s pure substitute ANN',fontsize=13)
    out=Path(a.out).resolve();fig.savefig(out,dpi=150);plt.close(fig)
    print(json.dumps(dict(plot=str(out),source=str(run),ANN_samples=int(auto.sum()),gates=ann['gates'],slip=ann['slip_ratio_max'])))

if __name__=='__main__':main()
