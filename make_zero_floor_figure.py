from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
P=ROOT/'outputs/model_branches/zero_floor'
with np.load(P/'zero_floor_traces.npz',allow_pickle=False) as z, np.load(ROOT/'outputs/neural_muscle_closure/neural_duration_probe.npz',allow_pickle=False) as old:
 fig,axes=plt.subplots(1,2,figsize=(11,4.8))
 t=np.arange(2001)*.1
 axes[0].plot(t,old['original_normal_heights_mm'][:,2],label='canonical normal')
 for arm in ['normal','spike_output_silence','replay']:
  axes[0].plot(t,z[arm+'_heights_mm'][:,2],label='zero-floor '+arm,ls='--' if arm=='replay' else '-')
 axes[1].plot(t,np.max(np.abs(z['spike_output_silence_body_act']),axis=1),label='silenced activation')
 axes[1].plot(t,np.max(np.abs(z['spike_output_silence_body_ctrl']),axis=1),label='silenced command',ls='--')
 axes[0].set(title='Actual neural branch versus frozen canonical',ylabel='Whole-body COM height (mm)')
 axes[1].set(title='Zero-floor output-silence control',ylabel='Maximum absolute value (dimensionless)')
 for ax in axes:ax.set_xlabel('Time (ms)');ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.text(.5,.01,'Only muscle control lower bound changes; no gain tuning, background drive or stepwise activation reset.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.05,1,1));p=P/'zero_floor_comparison.png';fig.savefig(p,dpi=140,bbox_inches='tight');plt.close(fig);print(p)
