"""Plot the measured geometry, fixed kernels and saved single-pulse responses."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fixed_neck import HERE, RISE_MS, DECAY_MS, CHARGE_NS_MS, full_catalog, parameters

fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
catalog=full_catalog()['ports']
records=parameters()['ports']
directions=['DN','AN','SA']
colors={'DN':'#18587e','AN':'#ae4d67','SA':'#6b7444'}
for direction in directions:
    keys=[root for root in records if catalog[root]['direction']==direction]
    axes[0,0].scatter([records[k]['length_um'] for k in keys],
                    [records[k]['implemented_delay_ms'] for k in keys],
                    c=colors[direction],label=direction,s=28,alpha=.7)
axes[0,0].set(xlabel='Measured cervical path (µm)',ylabel='Implemented axonal delay (ms)',
              title='Geometry-based delays, assumed velocity 0.10 m/s')
axes[0,0].legend(frameon=False)
time=np.linspace(0,30,1000)
for rise,decay,q,name in zip(RISE_MS,DECAY_MS,CHARGE_NS_MS,['ACh','GABA','GluCl'],strict=True):
    g=q/(decay-rise)*(np.exp(-time/decay)-np.exp(-time/rise))
    axes[0,1].plot(time,g,label=name)
axes[0,1].set(xlabel='Time after synaptic release (ms)',ylabel='Unit conductance (nS)',
              title='Separate receptor kinetics')
axes[0,1].legend(frameon=False)
for sign,name in ((1,'Excitatory DN'),(-1,'Inhibitory DN')):
    with np.load(HERE/f'PULSE_{sign}.npz',allow_pickle=False) as trace:
        time=trace['time_ms']
        v=trace['values'][:,0]
        current=trace['values'][:,1:4].sum(1)
    axes[1,0].plot(time,v,label=name)
    axes[1,1].plot(time,current,label=name)
axes[1,0].set(xlabel='Simulation time (ms)',ylabel='Receiver membrane voltage (mV)',
              title='Single brain-side pulse → measured target')
axes[1,1].set(xlabel='Simulation time (ms)',ylabel='Instantaneous synaptic current (pA)',
              title='Voltage-dependent current')
for ax in axes[1]:
    ax.legend(frameon=False)
fig.savefig(HERE/'DYNAMICS.png',dpi=150)
print('Saved DYNAMICS.png',flush=True)
