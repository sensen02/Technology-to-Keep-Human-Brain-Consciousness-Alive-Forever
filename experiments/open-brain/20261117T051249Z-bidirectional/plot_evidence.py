"""Plot the measured ground/flight responses from saved physics traces."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def values(folder, case):
    return np.load(HERE / folder / case / 'TRACE.npy', allow_pickle=False)


fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})
for key,label in (('ground_forward','Forward'),('ground_speed','Add DNg97'),('ground_halt','Web halt'),
                  ('ground_back','Moonwalker'),('ground_cut','DN cut')):
    trace = values('PHYSICAL_SENSORY',key)
    start = trace[229, 1]
    axes[0,0].plot(trace[229:,0],trace[229:,1]-start,label=label)
axes[0,0].set(title='Ground advance, speed, stop and reverse', ylabel='Forward displacement (mm)', xlabel='Simulation time (s)')
axes[0,0].legend(frameon=False)
for key,folder,label in (('ground_left','PHYSICAL_SENSORY','Walking, left'),('ground_right','PHYSICAL_SENSORY','Walking, right'),
                         ('ground_pivot_left','PHYSICAL_PIVOT_GAIN','Standing, left'),('ground_pivot_right','PHYSICAL_PIVOT_GAIN','Standing, right')):
    trace=values(folder,key)
    axes[0,1].plot(trace[:,0],np.rad2deg(trace[:,4]-trace[0,4]),label=label)
axes[0,1].set(title='Left and right ground steering', ylabel='Thorax yaw (deg)', xlabel='Simulation time (s)')
axes[0,1].legend(frameon=False)
flight_cases = [('intact','Intact', '#116b83'),('cut','Cut from start','#777777'),
                ('cut_restore_short','Short cut + restore','#3a9a64'),('cut_restore_long','Long cut + restore','#cc603d')]
for key,label,color in flight_cases:
    trace=values('FREE_FLIGHT_QUAL',key)
    axes[1,0].plot(trace[:,0]*1e3,trace[:,3]*10,label=label,color=color)
    if key in ('intact','cut_restore_short'):
        axes[1,1].plot(trace[:,0]*1e3,trace[:,5],label=label,color=color,alpha=.8,linewidth=.8)
    off=trace[:,4]==0
    if key=='cut_restore_short':
        axes[1,1].axvspan(trace[off,0][0]*1e3,trace[off,0][-1]*1e3,color='#d9ece1',zorder=-1)
axes[1,0].set(title='Free flight from the same airborne start', ylabel='Centre-of-mass height (mm)', xlabel='Simulation time (ms)')
axes[1,0].legend(frameon=False)
axes[1,1].set(title='Wing actuator response during a short interruption', ylabel='Sum of absolute wing actuator forces', xlabel='Simulation time (ms)',xlim=(60,165))
axes[1,1].legend(frameon=False)
for ax in axes.ravel():
    ax.spines[['top','right']].set_visible(False)
fig.savefig(HERE/'RESPONSE.png',dpi=150)
