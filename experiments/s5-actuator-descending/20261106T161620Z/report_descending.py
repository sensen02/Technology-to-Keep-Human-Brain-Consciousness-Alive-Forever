import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z')
fig,axs=plt.subplots(2,3,figsize=(13,7),sharex=True)
summary=[]
for i,dt in enumerate([.0005,.0000625]):
 folder=O/('joint_descending_'+str(dt));s=json.loads((folder/'SUMMARY.json').read_text())
 assert json.loads((folder/'VERIFICATION.json').read_text())['passed']
 for j,arm in enumerate(['torque','incremental','muscle']):
  for c in ['normal','zero_output','sensory_cut','matched','candidate_cut']:
   x=np.load(folder/f'{arm}_{c}_pos.npz')['rows'];axs[i,j].plot(x[:,0]*1000,x[:,1],label=c,alpha=.8)
  axs[i,j].axhline(.4789,color='gray',ls='--');axs[i,j].axhline(2.502,color='gray',ls='--');axs[i,j].set_title(f'{arm}, neural dt={dt*1000:g}ms');axs[i,j].set_ylabel('Joint angle (rad)');axs[i,j].set_xlabel('Time (ms)')
  normal=[r for r in s['runs'] if r['arm']==arm and r['condition']=='normal']
  summary.append(dict(dt_s=dt,arm=arm,terminal_error_range=[min(r['terminal_error'] for r in normal),max(r['terminal_error'] for r in normal)],in_range=all(r['within_range'] for r in normal),restoration=s['neural_contribution_screen'][arm]))
axs[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(O/'DESCENDING_JOINT.png',dpi=150);plt.close(fig)
(O/'DESCENDING_COMPARISON.json').write_text(json.dumps(dict(importance='核心必需',rows=summary),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary))
