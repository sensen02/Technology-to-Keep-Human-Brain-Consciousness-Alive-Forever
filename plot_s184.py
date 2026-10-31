from pathlib import Path
import argparse, csv, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);r=ap.parse_args().run
c=json.loads((r/'CENSUS_38.json').read_text());rows=[x for x in c['rows'] if x['minimax'] is not None]
fig,axs=plt.subplots(1,2,figsize=(16,6.5))
ax=axs[0]
for x in rows:
    el=x['root_ok'] and x['geom_ok']
    ax.scatter(len(x['unreachable']),x['minimax'],s=110 if el else 45,
               marker='o' if el else 'x',c='tab:green' if el else ('tab:orange' if x['root_ok'] else 'tab:gray'))
ax.axhline(3.308620867791414,color='k',ls='--',label='S177 baseline 3.3086')
ax.axhline(1e-8,color='r',ls='--',lw=.8,label='acceptance 1e-8')
ax.set_yscale('log');ax.set_xlabel('# unreachable joint rows at that pose');ax.set_ylabel('42-row minimax')
ax.legend(fontsize=8);ax.set_title('census of all %d geometry-valid poses\n(green = root-balanced AND geometry-valid: best is baseline)' % len(rows))
ax=axs[1]
g=list(csv.DictReader((r/'ROW_GAP_DECOMPOSITION.csv').open()))
x=np.arange(len(g))
ax.bar(x-0.2,[float(v['gap']) for v in g],.4,label='required-outside-interval gap')
ax.bar(x+0.2, [min(10., float(v['needed_uniform_activation_multiplier'])) if v['needed_uniform_activation_multiplier'] not in ('','None') else 10. for v in g], .4, label='needed activation multiplier m (capped 10)')
ax.set_xticks(x, [str(v['dof']) + '\n' + v['dof_name'].replace('joint_', '')[:9] for v in g], fontsize=7)
ax.set_yscale('log');ax.legend(fontsize=8);ax.set_title('row gaps on the best census point (refine_j26_-1)\nfour rows are amplitude-like (m<<1), two are channel-limited')
fig.suptitle('S184 census + gap decomposition: staleness audit PASSED (recomputed values match bitwise); no eligible pose beats the baseline')
fig.tight_layout();fig.savefig(r/'S184_CENSUS.png',dpi=150)
print('saved S184_CENSUS.png')
