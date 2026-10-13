from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent/'outputs/neural_muscle_closure';d=json.loads((P/'connected_feedback_audit.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,5))
for arm in ['normal','cut_proprio','cut_campaniform']:
    h=d['height_samples'][arm];t=[1000*r['time_s'] for r in h]
    axes[0].plot(t,[r['whole_body_com_z_model'] for r in h],label=arm)
    axes[1].plot(t[1:],d['comparisons'][arm]['motor_counts_per_interval'],label=arm)
axes[0].set(title='True whole-body COM, not root height',xlabel='Time (ms)',ylabel='COM height (mm)')
axes[1].set(title='Mapped motor spikes per completed 5ms',xlabel='Interval end (ms)',ylabel='Spike count across 202 mapped cells')
for ax in axes:ax.legend();ax.grid(alpha=.2)
fig.suptitle('Selective sensory cuts: unchanged physical model and neural parameters')
fig.text(.5,.01,'Explicit diagnostic interventions only; different output establishes input dependence, not autonomous standing.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.94));p=P/'connected_feedback_audit.png';fig.savefig(p,dpi=140,bbox_inches='tight');plt.close(fig);print(p)
