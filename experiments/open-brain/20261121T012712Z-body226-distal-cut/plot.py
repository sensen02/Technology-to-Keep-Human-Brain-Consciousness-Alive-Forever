import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
out=Path(__file__).resolve().parent
rows=json.loads((out/'evaluation.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(11,3.6),layout='constrained')
for ax,seed in zip(axes[:2],('11701','11903')):
    sample=[r for r in rows if r['name'].startswith(seed)]
    x=np.arange(3)
    for i,(key,color) in enumerate([('old','#73869c'),('new','#16765e')]):
        vals=[np.mean([r[key][k] for r in sample]) for k in ('step_mm','50ms_mm','200ms_mm')]
        ax.bar(x+(i-.5)*.34,vals,width=.32,color=color,label=('Prior weights','226-needle adaptation')[i])
    ax.set_xticks(x,['5','50','200']);ax.set_xlabel('Prediction horizon (ms)')
    ax.set_ylim(0,.56);ax.set_title(f'Start {seed} ({len(sample)} trials)')
axes[0].set_ylabel('Mean 3D marker error (mm)');axes[0].legend(fontsize=8)
ax=axes[2]
q=[r for r in rows if r['name'].endswith(('intact','clean_on'))]
for i,(label,col) in enumerate([('Physical teacher','#73869c'),('Recorded pressure','#16765e'),('Estimated pressure','#cc7744')]):
    ax.bar(np.arange(2)+(i-1)*.23,[r['teacher_last100ms_mm_s'] if i==0 else r['new'][['last100ms_mm_s','self_floor_last100ms_mm_s'][i-1]] for r in q],width=.22,color=col,label=label)
ax.set_xticks([0,1],['11701','11903']);ax.set_title('Final 100 ms, intact trials');ax.set_xlabel('Start');ax.set_ylabel('Net thorax speed (mm/s)');ax.legend(fontsize=8)
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
fig.savefig(out/'RESULT.png',dpi=145)
