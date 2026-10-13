from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent/'outputs/model_branches/fixed_replay_refinement'
def render(rows):
 fig,axes=plt.subplots(1,2,figsize=(11,4.8))
 for label,r in rows.items():
  t=[v*1000 for v in r['time_s']]
  axes[0].plot(t,r['com_z'],label=label)
  axes[1].plot(t,r['distance_to_reference'],label=label)
 axes[0].set(title='Fixed-command diagnostic replay',ylabel='COM height (mm)')
 axes[1].set(title='Separation from 6.25us reference',ylabel='3D COM distance (mm)')
 for ax in axes:ax.set_xlabel('Time (ms)');ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.text(.5,.01,'Replayed controls are diagnostic only, not autonomous. Body discretization and feedback sensitivity are distinct.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.05,1,1));p=P/'fixed_replay_comparison.png';fig.savefig(p,dpi=140,bbox_inches='tight');plt.close(fig);print(p)
if __name__=='__main__':
 root=Path(__file__).resolve().parent
 with np.load(P/'fixed_replay_refinement_traces.npz',allow_pickle=False) as z,np.load(root/'outputs/model_branches/clock_refinement/clock_refinement_traces.npz',allow_pickle=False) as old:
  ref=z['6.25us_replay_body_com_xyz_mm'];rows={}
  for name in ('12.5us_replay','6.25us_replay'):
   com=z[name+'_body_com_xyz_mm'];rows[name]=dict(time_s=z[name+'_body_time_s'],com_z=com[:,2],distance_to_reference=np.linalg.norm(com-ref,axis=1))
  com=old['12.5us_normal_body_com_xyz_mm'];rows['12.5us_closed_feedback']=dict(time_s=old['12.5us_normal_body_time_s'],com_z=com[:,2],distance_to_reference=np.linalg.norm(com-ref,axis=1))
 render(rows)
