from pathlib import Path
import argparse, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);r=ap.parse_args().run
s=json.loads((r/'SUMMARY.json').read_text());c=json.loads((r/'CANDIDATES.json').read_text())
fig,axs=plt.subplots(1,3,figsize=(19,6))
ax=axs[0]
for x in c:
    col='k' if 'BASELINE' in x['tag'] else ('tab:red' if not x['root_ok'] else 'tab:blue')
    ax.scatter(x['minimax'],max(x['root_norm'],1e-13),c=col,s=55)
ax.axhline(1e-8,color='k',ls='--',lw=.8);ax.axvline(3.3086,color='k',ls='--',lw=.8)
ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('42-row minimax');ax.set_ylabel('root |r6|inf (z,roll,pitch)')
ax.set_title('root residual vs joint minimax, two axes')
ax=axs[1]
ax.bar(range(len(c)),[x['minimax'] for x in c],color=['k' if 'BASELINE' in x['tag'] else 'tab:blue' for x in c])
ax.axhline(3.3086,color='r',ls='--',label='baseline 3.3086')
ax.set_xticks(range(len(c)));ax.set_xticklabels([x['tag'][:12] for x in c],rotation=80,fontsize=6)
ax.set_ylabel('minimax');ax.legend();ax.set_title('no coordinated move beats the baseline minimax')
ax=axs[2]
ax.bar(range(len(c)),[len(x['unreachable']) for x in c],color=['k' if 'BASELINE' in x['tag'] else 'tab:blue' for x in c])
ax.axhline(6,color='r',ls='--',label='baseline 6 rows')
ax.set_xticks(range(len(c)));ax.set_xticklabels([x['tag'][:12] for x in c],rotation=80,fontsize=6)
ax.set_ylabel('# unreachable rows');ax.legend();ax.set_title('5 rows is reachable only with a worse minimax')
fig.suptitle('S181 coordinated multi-row search: 18 candidates, forward=%d, valid=%d, best=baseline' % (s['calls']['forward'], s['n_valid']))
fig.tight_layout();fig.savefig(r/'S181_SELECTED.png',dpi=150)
print('saved S181_SELECTED.png')
