"""Plot measured engineering bridge interventions, with explicit scope."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--json',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args();d=json.loads(a.json.read_text())
fig,axs=plt.subplots(2,2,figsize=(13,9))
for name,cond in d['conditions'].items():
    r=cond['trace'];t=np.array([x['time_s'] for x in r]);p=np.array([x['position_mm'] for x in r]);yaw=np.unwrap([x['yaw_rad'] for x in r])
    axs[0,0].plot(p[:,0],p[:,1],label=name)
    axs[0,1].plot(t,np.degrees(yaw),label=name)
    axs[1,0].plot(t,[x['intent']['turn'] for x in r],label=name)
    axs[1,1].plot(t,np.degrees([x['tilt_rad'] for x in r]),label=name)
axs[0,0].set(xlabel='World x (mm)',ylabel='World y (mm)',title='Measured body trajectories');axs[0,0].axis('equal')
axs[0,1].set(xlabel='Time (s)',ylabel='Yaw (deg)',title='Heading, not prerecorded motion')
if 'intact' in d['conditions']:axs[0,1].axhline(np.degrees(d['conditions']['intact']['target_absolute_yaw_rad']),ls='--',color='black',label='virtual target')
axs[1,0].set(xlabel='Time (s)',ylabel='Turn intent',title='Filtered simulated DNa01 spikes')
axs[1,1].set(xlabel='Time (s)',ylabel='Tilt from world up (deg)',title='Body orientation')
for ax in axs.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('Engineered BANC DNa01 bridge: virtual heading feedback + external walking permission\nShipped hybrid controller generates gait; this is NOT connectome-generated locomotion',fontsize=12)
fig.tight_layout(rect=(0,0,1,.93));fig.savefig(a.out,dpi=140)
print(a.out)
