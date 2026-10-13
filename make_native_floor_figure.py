from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent/'outputs/neural_muscle_closure'
with np.load(P/'native_floor_audit.npz',allow_pickle=False) as z:
 t=z['zero_ctrl_integration_time_s']*1000
 a=z['zero_ctrl_integration_act']
 f=z['zero_ctrl_integration_active_delta_actuator_force']
 fig,axes=plt.subplots(1,2,figsize=(10,4.7))
 axes[0].plot(t,a.min(axis=1),label='minimum');axes[0].plot(t,a.max(axis=1),ls='--',label='maximum');axes[0].set(ylabel='Native muscle activation',title='All decoded controls remain exactly zero')
 axes[1].plot(t,np.max(np.abs(f),axis=1));axes[1].set(ylabel='Maximum |active force| (model force)',title='Same-state total minus act=0 scratch')
 for ax in axes:ax.set_xlabel('Time (ms)');ax.grid(alpha=.2)
 axes[0].legend();fig.text(.5,.01,'Frozen native floor preserved. Nonzero is not evidence of meaningful movement or cause of collapse.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.05,1,1));p=P/'native_floor_audit.png';fig.savefig(p,dpi=140,bbox_inches='tight');plt.close(fig);print(p)
