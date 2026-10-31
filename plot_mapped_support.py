"""S166 saved-only display; never draw unavailable feasible mapped solutions."""
from pathlib import Path
import argparse,json,numpy as np,csv
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
S165=Path('/home/sensen/Desktop/engine修复验收/s165-static-support-corrected/20261024T231400000000Z')
def main(p):
 with np.load(p/'MAPPING.npz') as z:ap=z['actuator_pool'].copy()
 fig,axs=plt.subplots(3,3,figsize=(17,12));summary=json.loads((p/'SUMMARY.json').read_text());audit=json.loads((p/'INDEPENDENT_CHECK.json').read_text())
 for row,i in enumerate([0,15,83]):
  with np.load(S165/f'STATIC_AFFINE_{i:03d}.npz') as z:act=z['actual_activation'].copy()
  with np.load(S165/f'rigid_upper_{i:03d}_upper0.5.npz') as z:ind=z['x'][:90].copy()
  axs[row,0].plot(act,label='Actual saved activation');axs[row,0].plot(ind,label='S165 independent90 <=.5');axs[row,0].scatter(np.flatnonzero(ap<0),ind[ap<0],marker='x',color='red',label='Unmapped slots in independent witness');axs[row,0].set_title(f'State {i}: actual and independent witness');axs[row,0].legend(fontsize=7)
  axs[row,1].bar(range(90),np.where(ap<0,0,.5));axs[row,1].set_title('Interface max act: 16 slots fixed zero\nshared pool equality also imposed');axs[row,1].set_xlabel('Actuator index');axs[row,1].set_ylim(0,.55)
  axs[row,2].axis('off');axs[row,2].text(.05,.8,f'State {i}\nMapped LP: no feasible solution\nSee independent certificate audit\n\nNo invented pool/activation/leg force\nor solution residual is plotted.\n\n90 independent -> 66 mapped pools\nNative J unchanged: 72 edges / 48 DOF',va='top',fontsize=11)
 fig.suptitle('S166 interface-topology constrained rigid upper bound — NOT standing or neural attainability');fig.tight_layout();fig.savefig(p/'MAPPED_SUPPORT_COMPARISON.png',dpi=150)
if __name__=='__main__':
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);main(pa.parse_args().output)
