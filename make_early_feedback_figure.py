from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent/'outputs/neural_muscle_closure';d=json.loads((P/'early_feedback_probe.json').read_text());z=np.load(P/'early_feedback_probe.npz',allow_pickle=False)
fig,axes=plt.subplots(1,2,figsize=(12,5))
for label in ['baseline','push_normal','push_replay','push_sensory_cut']:
    t=z[label+'_time_s']*1000
    axes[0].plot(t,z[label+'_com_xyz'][:,2],label=label)
    axes[1].plot(t,z[label+'_com_xyz'][:,0]-z[label+'_com_xyz'][0,0],label=label)
for ax in axes:ax.axvspan(1,2,color='gray',alpha=.2);ax.set_xlabel('Time (ms)');ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[0].set(title='Whole-body COM height',ylabel='Height (mm)');axes[1].set(title='Whole-body COM horizontal displacement',ylabel='X displacement (mm)')
fig.suptitle('Early physical perturbation: actual feedback versus baseline-command replay')
fig.text(.5,.01,'1–2ms +x push at 0.1 model weight; replay is diagnostic only, not a neural controller. No standing claim.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.94));p=P/'early_feedback_probe.png';fig.savefig(p,dpi=140,bbox_inches='tight');plt.close(fig);print(p)
