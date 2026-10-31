"""S171 final plot from saved points and certified separators; no solves."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output
check=json.loads((p/'S171_SAVED_INDEPENDENT_CHECK.json').read_text());assert check['status']=='VERIFIED_SAVED_ONLY'
fig,axs=plt.subplots(3,2,figsize=(16,11))
labels=['real66, missing zero','real66 + independent missing16','independent90, half amplitude']
for k in [1,2,3]:
 with np.load(p/f'CASE_{k}.npz') as z, np.load(p/f'CERTIFICATE_{k}.npz') as c:
  axs[k-1,0].plot(z['residual']);axs[k-1,0].axhline(0,color='black',lw=.5);axs[k-1,0].set_title(f'{k}: CERTIFIED INFEASIBLE — {labels[k-1]}');axs[k-1,0].set_xlabel(f'48 DOFs; phase-I diagnostic only. separator gap={float(c["separator_gap"]):.8g}')
  axs[k-1,1].bar(np.arange(90),z['activation']);axs[k-1,1].set_ylim(0,.55);axs[k-1,1].set_title('Phase-I activation (NOT equilibrium or deployed ctrl)');axs[k-1,1].set_xlabel('90 actuator slots')
fig.suptitle('S171 same fixed pose / native soft f0: all three systems certified infeasible; NOT other poses');fig.tight_layout();fig.savefig(p/'THREE_CONDITIONS_CERTIFIED.png',dpi=150)
