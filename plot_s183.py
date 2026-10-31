from pathlib import Path
import argparse, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);r=ap.parse_args().run
s=json.loads((r/'SUMMARY.json').read_text());c=json.loads((r/'CANDIDATES.json').read_text())
fig,axs=plt.subplots(1,3,figsize=(19,6))
ax=axs[0]
gs=np.array([x['geom_score'] for x in c]);mv=np.array([x['max_violation_six'] for x in c]);ok=np.array([x['geom_ok'] for x in c])
ax.scatter(np.maximum(gs[~ok],1e-12),mv[~ok],s=10,c='tab:gray',label=f'geometry FAIL ({(~ok).sum()})')
ax.scatter(np.maximum(gs[ok],1e-12),mv[ok],s=45,c='tab:green',label=f'geometry PASS ({ok.sum()})')
ax.set_xscale('log');ax.set_xlabel('geometry violation score (0 = all gates pass)');ax.set_ylabel('max violation over the six rows')
ax.legend(fontsize=8);ax.set_title('geometry-targeted sweep: gates are reachable,\nbut the six-row violation stays >= 4.8')
ax=axs[1]
ch=s['gate_joint_checks']
labels=[x['tag'][:16] for x in ch];mm=[x['minimax'] for x in ch]
ax.bar(range(len(ch)),mm,color=['tab:orange' if 'BASELINE' not in x['tag'] else 'k' for x in ch])
ax.axhline(3.308620867791414,color='r',ls='--',label='S177 baseline 3.3086')
ax.set_xticks(range(len(ch)));ax.set_xticklabels(labels,rotation=15,fontsize=7);ax.set_ylabel('42-row minimax')
ax.legend(fontsize=8);ax.set_title('gate-passing poses after root Newton (all root-balanced):\nnone beats the baseline')
ax=axs[2]
ax.scatter([x['geometry']['bearing_legs']+np.random.RandomState(1).uniform(-.1,.1) for x in ch],[x['minimax'] for x in ch],s=60,c='tab:blue')
ax.axhline(3.308620867791414,color='r',ls='--');ax.set_xlabel('bearing legs');ax.set_ylabel('minimax')
ax.set_title('root-balanced, geometry-valid candidates\n(minimax floor 3.42 vs baseline 3.31)')
fig.suptitle('S183 geometry-targeted search: %d poses, %d gate-passing, %d forwarded to root Newton+LP' % (s['poses'],s['n_gate_passing_poses'],len(ch)))
fig.tight_layout();fig.savefig(r/'S183_GEOMETRY_TARGETED.png',dpi=150)
print('saved S183_GEOMETRY_TARGETED.png')
