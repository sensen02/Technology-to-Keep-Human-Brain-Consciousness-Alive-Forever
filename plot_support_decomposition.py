"""S167 saved-only four-condition comparison. Missing witnesses never fabricated."""
from pathlib import Path
import argparse,json,csv
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('/home/sensen/Desktop/engine修复验收');S165=BASE/'s165-static-support-corrected/20261024T231400000000Z'
def main(p):
 audit=json.loads((p/'INDEPENDENT_CHECK.json').read_text());legs=['LF','LM','LH','RF','RM','RH'];rows=list(csv.DictReader((p/'LEG_SUPPORT.csv').open()));fig,axs=plt.subplots(3,3,figsize=(18,12))
 for row,i in enumerate([0,15,83]):
  with np.load(S165/f'rigid_upper_{i:03d}_upper0.5.npz') as z:ind=z['x'][:90].copy()
  with np.load(p/f'CASE_{i:03d}_fill.npz') as z:act=z['activation'].copy();res=z['residual'].copy()
  axs[row,0].plot(ind,label='Both relaxed (S165 independent90)');axs[row,0].plot(act,label='Only missing-output relaxed (fill)');axs[row,0].set_title(f'State {i}: feasible activation witnesses');axs[row,0].legend(fontsize=7)
  legf=[float(next(r['Fz'] for r in rows if int(r['sample'])==i and r['condition']=='fill' and r['leg']==l)) for l in legs];axs[row,1].bar(range(6),legf);axs[row,1].set_xticks(range(6),legs);axs[row,1].set_title(f'Fill witness leg support; max|48res|={abs(res).max():.2e}')
  axs[row,2].axis('off');axs[row,2].text(.03,.9,'Four-condition result (all three states)\n\nBoth restrictions (S166): INFEASIBLE*\nOnly shared binding relaxed: INFEASIBLE*\nOnly missing outputs relaxed: FEASIBLE\nBoth relaxed (S165): FEASIBLE\n\n* independently certified; no fictitious witness\nNative pyramid min margin=0\nctrl=act compatible with act_dot=0\n\nMathematical rigid upper bounds ONLY\nNo missing neural sources established',va='top',fontsize=10)
 fig.suptitle('S167 missing outputs vs shared-pool binding — no new body/neural run, NOT standing');fig.tight_layout();fig.savefig(p/'FOUR_CONDITION_COMPARISON.png',dpi=150)
if __name__=='__main__':
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);main(pa.parse_args().output)
