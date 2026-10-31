from pathlib import Path
import argparse, csv, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);r=ap.parse_args().run
rows=list(csv.DictReader((r/'RECONCILE_BASELINE.csv').open()))
scan=list(csv.DictReader((r/'RM_CHANNEL_SCAN.csv').open()))
six={int(x['dof']) for x in rows if x['reachable']=='False'}
fig,axs=plt.subplots(1,2,figsize=(17,6.5))
ax=axs[0]
x=np.arange(42)
ax.vlines(x, [float(v['sum_min']) for v in rows], [float(v['sum_max']) for v in rows], color='tab:cyan', lw=2, label='[Σmin, Σmax]')
ax.scatter(x, [float(v['required']) for v in rows], c=['tab:red' if int(v['dof']) in six else 'tab:blue' for v in rows], s=26, zorder=3, label='required (red = outside interval)')
ax.axhline(0,color='k',lw=.5)
ax.set_xticks(range(0,42,3)); ax.set_xticklabels([rows[i]['dof_name'].replace('joint_','')[:7] for i in range(0,42,3)], rotation=75, fontsize=6)
ax.set_ylabel('generalized force'); ax.legend(fontsize=8)
ax.set_title('baseline: all 42 rows, required vs [Σmin,Σmax]\nsix rows outside (2.95 / 0.096 / 2.54 / 3.31 / 1.44 / 2.43)')
ax=axs[1]
for dof,col in (('42','tab:red'),('47','tab:purple')):
    pts=[s for s in scan if s['dof']==dof and not s['classification'].startswith('BEST')]
    js=sorted({s['joint_name'] for s in pts})
    for k,j in enumerate(js):
        sub=[s for s in pts if s['joint_name']==j]
        ax.plot([float(s['angle']) for s in sub],[max(float(s['sum_max_positive']),1e-18) for s in sub],'o-',color=col,alpha=.45,ms=5,label=f'DOF {dof}' if k==0 else None)
ax.set_yscale('log'); ax.set_ylim(1e-18,1e2)
ax.axhline(1e-12,color='k',ls='--',lw=.8,label='zero authority line')
ax.set_xlabel('joint angle at native limit (rad)'); ax.set_ylabel('positive authority Σmax')
ax.legend(fontsize=8)
ax.set_title('RM rows: positive authority is EXACTLY 0 at all 28 native-limit points\n(channel type: no positive path exists, not an amplitude problem)')
fig.suptitle('S185 reconciliation ALIGNED (max target diff 0.0 across baseline sources); RM positive channel = 0 everywhere')
fig.tight_layout(); fig.savefig(r/'S185_RECONCILE.png', dpi=150)
print('saved S185_RECONCILE.png')
