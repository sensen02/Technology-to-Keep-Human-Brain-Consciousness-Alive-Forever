from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent/'outputs/neural_muscle_closure'
d=json.loads((P/'motor_state_audit.json').read_text()); rows=d['interval_summaries'];t=[r['interval_end_ms'] for r in rows]
fig,axes=plt.subplots(1,3,figsize=(14,4.5))
axes[0].plot(t,[r['motor_spikes'] for r in rows]);axes[0].set(title='Mapped motor spikes',ylabel='Spikes per completed 5ms')
for key,label in [('exc_drive_pre_nA','Excitatory'),('inh_drive_pre_nA','Inhibitory'),('adaptation_pre_nA','Adaptation magnitude')]:
 axes[1].plot(t,[r['time_population_means'][key] for r in rows],label=label)
axes[1].set(title='Pre-update signed synaptic drives',ylabel='Time-population mean (nA)');axes[1].legend(fontsize=8)
axes[2].plot(t,[r['time_population_means']['v_postreset_mV'] for r in rows]);axes[2].set(title='Post-reset voltage (not peak voltage)',ylabel='Mean voltage (mV)')
for ax in axes:ax.set_xlabel('Interval end (ms)');ax.grid(alpha=.2)
fig.suptitle('Exact saved-input neural replay: observations, not causal ablation')
fig.text(.5,.01,'Population means hide cell heterogeneity; no parameter tuning or biological calibration claimed.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,.93));p=P/'motor_state_audit.png';fig.savefig(p,dpi=140,bbox_inches='tight');plt.close(fig);print(p)
