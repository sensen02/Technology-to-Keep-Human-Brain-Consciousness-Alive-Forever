from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent/'outputs/model_branches/fine_clock'
def render(rows):
 fig,axes=plt.subplots(1,3,figsize=(14,4.7))
 for name,r in rows.items():
  t=[v*1000 for v in r['time_s']]
  axes[0].plot(t,r['com_z'],label=name)
  axes[1].plot(t,r['act_max'],label=name)
  axes[2].plot(t,r['nonfoot'],label=name)
 for ax in axes:ax.set_xlabel('Time (ms)');ax.grid(alpha=.2);ax.legend(fontsize=7)
 axes[0].set(title='Actual neural closed loop',ylabel='COM height (mm)')
 axes[1].set(title='No activation clamp',ylabel='Maximum activation')
 axes[2].set(title='Non-foot floor contacts',ylabel='Contact entries')
 fig.text(.5,.01,'Only body integration timestep differs; neural 0.5ms / sensory-command 5ms fixed. Boundedness is not convergence.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.05,1,1));p=P/'fine_clock_comparison.png';fig.savefig(p,dpi=140,bbox_inches='tight');plt.close(fig);print(p)
if __name__=='__main__':
 with np.load(P/'fine_clock_traces.npz',allow_pickle=False) as z:
  rows={name:dict(time_s=z[name+'_body_time_s'],com_z=z[name+'_heights_mm'][:,2],act_max=z[name+'_body_act'].max(axis=1),nonfoot=z[name+'_body_nonfoot_contacts']) for name in ('100us_normal','50us_normal','25us_normal','25us_spike_output_silence')}
 render(rows)
