from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent/'outputs/neural_muscle_closure'
def render(traces):
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    for label,row in traces.items():
        t=[v*1000 for v in row['time_s']]
        axes[0].plot(t,row['COM_z_mm'],label=label)
        axes[1].plot(t,row['nonfoot_contacts'],label=label)
        axes[2].plot(row['spike_interval_end_ms'],row['motor_spikes'],label=label)
    axes[0].set(title='Whole-body COM',ylabel='Height (mm)');axes[1].set(title='Non-foot floor contact entries',ylabel='Count');axes[2].set(title='Mapped motor discharge',ylabel='Spikes per 5ms')
    for ax in axes:ax.set_xlabel('Time (ms)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.text(.5,.01,'200ms actual neural closure, unchanged gains and zero initial activation; first50ms must reproduce prior trials.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));p=P/'neural_duration_probe.png';fig.savefig(p,dpi=140,bbox_inches='tight');plt.close(fig);print(p)
if __name__=='__main__':
    d=json.loads((P/'neural_duration_probe.json').read_text())
    with np.load(P/'neural_duration_probe.npz',allow_pickle=False) as z:
        traces={name:dict(time_s=z[name+'_body_time_s'],COM_z_mm=z[name+'_heights_mm'][:,2],nonfoot_contacts=z[name+'_body_metrics'][:,2],spike_interval_end_ms=np.arange(1,41)*5,motor_spikes=r['neural_spike_counts']['motor_neurons']['per_5ms']) for name,r in d['episodes'].items()}
    render(traces)
