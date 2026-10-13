"""Plot adapted trace fields after reading verified neural pose artifact."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent/'outputs/neural_muscle_closure'
def render(traces):
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for label,r in traces.items():
        t=[v*1000 for v in r['time_s']]
        axes[0].plot(t,r['COM_z_mm'],label=label)
        axes[1].plot(t,r['joint_limit_count'],label=label)
    axes[0].set(title='Actual neural commands, zero initial activation',ylabel='Whole-body COM height (mm)')
    axes[1].set(title='Strict joint-range diagnostic (soft constraints)',ylabel='Number of out-of-range scalar joints')
    for ax in axes:ax.set_xlabel('Time (ms)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.text(.5,.01,'Explicit initial pose comparison, not hidden clipping. Proprioceptive reference changes with initial pose. No standing claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));p=P/'neural_pose_comparison.png';fig.savefig(p,dpi=140,bbox_inches='tight');plt.close(fig);print(p)

if __name__=='__main__':
    d=json.loads((P/'neural_pose_comparison.json').read_text())
    with np.load(P/'neural_pose_comparison.npz',allow_pickle=False) as z:
        traces={name:dict(time_s=z['body_time_s'],COM_z_mm=z[name+'_heights_mm'][:,2],joint_limit_count=z[name+'_body_metrics'][:,0]) for name in d['episodes']}
    render(traces)
