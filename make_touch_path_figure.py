from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent/'outputs/neural_muscle_closure'
def render(trace):
    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(trace['time_ms'],trace['network_different_spikes'],label='All-tier differing spike cells')
    ax.plot(trace['time_ms'],trace['motor_different_spikes'],label='Motor differing spike cells')
    ax.axvspan(5,15,alpha=.12,color='gray',label='External LF tactile current')
    ax.set(xlabel='Neural time (ms)',ylabel='Number of different binary spike values',title='Same sensory replay: tactile timing effect is not motor-command effect')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();p=OUT/'touch_path_audit.png';fig.savefig(p,dpi=140);plt.close(fig);print(p)

if __name__=='__main__':
    d=json.loads((OUT/'touch_path_audit.json').read_text())
    with np.load(d['spike_npz'],allow_pickle=False) as z:
        diff=z['normal_spikes']!=z['no_external_touch_spikes']
        render(dict(time_ms=(np.arange(len(diff))+1)*.5,network_different_spikes=diff.sum(axis=1),motor_different_spikes=diff[:,z['motor_indices']].sum(axis=1)))
