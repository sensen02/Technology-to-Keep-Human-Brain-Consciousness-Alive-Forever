"""Standalone plot from frozen actual saved traces; no physics."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
profiles=json.loads((HERE/'PREREG.json').read_text())['profiles']
fig,axes=plt.subplots(1,3,figsize=(17,5))
for name in profiles:
    r=json.loads((HERE/name/'RESULT.json').read_text())
    with np.load(HERE/name/'trace_1ms.npz',allow_pickle=False) as z:
        xy=z['com'][5799:,:2].copy();t=z['time'][5799:].copy()
    with np.load(HERE/name/'geometry_1ms.npz',allow_pickle=False) as z:
        yaw=np.unwrap(z['thorax_yaw'][5799:].copy())
    axes[0].plot(xy[:,0]-xy[0,0],xy[:,1]-xy[0,1],label=name)
    axes[1].plot(t-t[0],yaw-yaw[0],label=name)
    axes[2].bar(name,r['autonomous']['slip_ratio_max'])
axes[0].set(xlabel='Actual COM delta world X',ylabel='Actual COM delta world Y')
axes[1].set(xlabel='Actual seconds after withdrawal',ylabel='Actual thorax yaw change (rad)')
axes[1].axvline(4.8,c='gray',ls='--');axes[2].axhline(.5,c='red',ls='--',label='Frozen slip limit .5')
axes[2].set(ylabel='Original maximum per-leg slip ratio');axes[2].tick_params(axis='x',rotation=30)
for ax in axes:ax.grid(alpha=.2)
axes[0].legend();axes[1].legend();axes[2].legend()
fig.suptitle('Actual restricted-range neural diagnostic ONLY — no food/deployment qualification')
fig.tight_layout();fig.savefig(HERE/'ACTUAL_RANGE_DIAGNOSTIC.png',dpi=150);plt.close(fig)
