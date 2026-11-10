"""Evidence plots from saved results, with failure thresholds explicit."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from identity import atomic_json
HERE=Path(__file__).resolve().parent


def main():
    parent=HERE.parent/'20261110T045410Z'
    names=[];metrics=[]
    old=json.loads((parent/'legacy_replay/EVALUATION.json').read_text())['evaluation']
    names.append('Legacy saved');metrics.append(old)
    b=json.loads((parent/'ff_baseline_3u/TRAINING.json').read_text())
    for row in b['updates']:
        names.append('Corrected FF '+str(row['update']+1));metrics.append(row['evaluation'])
    for name,label in [('bc_ff238','FF238 BC'),('bc_gru238','GRU238 BC')]:
        path=HERE/name/'BC_RESULT.json'
        if path.exists():names.append(label);metrics.append(json.loads(path.read_text())['evaluation'])
    for run,label in [('hold_ff238_3u','FF238 hold'),('hold_gru238_3u','GRU238 hold')]:
        path=HERE/run/'TRAINING.json'
        if path.exists():
            d=json.loads(path.read_text())
            if d['updates']:names.append(label);metrics.append(d['updates'][-1]['evaluation'])
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    panels=[('forward_signed_native_per_s','Signed forward (native/s)',.05),
            ('slip_ratio_max','Maximum per-leg slip ratio (lower is better)',.5),
            ('com_height_min','Minimum COM height (native)',1.20),
            ('nonfoot_force_max_native','Peak nonfoot force (native)',.2*24.446354755878005)]
    for ax,(key,label,gate) in zip(axs.flat,panels):
        ax.bar(range(len(names)),[m[key] for m in metrics],color='#497a9d')
        ax.axhline(gate,color='#b23434',linestyle='--',label='Derived engineering threshold')
        ax.set_xticks(range(len(names)),names,rotation=35,ha='right',fontsize=8)
        ax.set_title(label);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Replacement ANN / derived triad body: diagnostic results, NOT walking acceptance')
    output=HERE/'progress.png';fig.savefig(output,dpi=140);plt.close(fig)
    atomic_json(HERE/'PROGRESS_PLOT.json',dict(labels=names,metrics=metrics,figure=str(output),scope='mixed diagnostic stages; not controlled architecture attribution; independent final original acceptance absent'))
    print(str(output))

if __name__=='__main__':main()
