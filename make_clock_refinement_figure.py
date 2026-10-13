from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent/'outputs/model_branches/clock_refinement'
def render(rows):
 fig,axes=plt.subplots(1,2,figsize=(11,4.8))
 for label,r in rows.items():
  t=[x*1000 for x in r['time_s']]
  axes[0].plot(t,r['com_z'],label=label)
  axes[1].plot(t,r['root_z'],label=label)
 axes[0].set(ylabel='COM height (mm)',title='Physical timestep refinement')
 axes[1].set(ylabel='Root height (mm)',title='Root is not COM')
 for ax in axes:ax.set_xlabel('Time (ms)');ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.text(.5,.01,'Same zero-floor neural model; no extra drive. Similar early trajectories do not establish late-time convergence.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.05,1,1));p=P/'clock_refinement.png';fig.savefig(p,dpi=140,bbox_inches='tight');plt.close(fig);print(p)
if __name__=='__main__':
 with np.load(P/'clock_refinement_traces.npz',allow_pickle=False) as z:
  rows={name:dict(time_s=z[name+'_body_time_s'],com_z=z[name+'_body_com_xyz_mm'][:,2],root_z=z[name+'_body_qpos'][:,2]) for name in ('50us_normal','25us_normal','12.5us_normal','6.25us_normal')}
 render(rows)
