"""S165 corrected saved-only feasibility display; missing solutions explicit."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',required=True,type=Path);p=pa.parse_args().output
 schema=json.loads((p/'PLOT_DATA.json').read_text());fig,axs=plt.subplots(3,3,figsize=(17,13))
 for row,(index,state) in enumerate(schema['states'].items()):
  state['index']=index
  with np.load(p/f'STATIC_AFFINE_{int(index):03d}.npz') as z: actual_residual=z['actual_soft_static_residual'].copy()
  state['leg_Fz']={'Actual moving state':state['actual_legFz'],'Native-soft prescribed qacc=0':state['soft_legFz']}
  state['residuals']={'Actual act / prescribed soft qacc=0':actual_residual}
  state['solutions']={f"{s['kind']} <= {s['upper']}":s for s in state['solutions']}
  for label,s in state['solutions'].items():
   if s.get('activation') is not None:
    state['leg_Fz'][label]=s['legFz'];state['residuals'][label]=s['residuals']
  x=np.arange(90);axs[row,0].plot(x,state['actual_act'],label='Actual saved activation')
  for label,sol in state['solutions'].items():
   if sol.get('activation') is not None:axs[row,0].plot(x,sol['activation'],label=label)
  axs[row,0].set_title(f"State {state['index']} activation (solutions only if feasible)");axs[row,0].set_xlabel('Actuator index');axs[row,0].legend(fontsize=7)
  for label,force in state['leg_Fz'].items():axs[row,1].plot(np.arange(6),force,'o-',label=label)
  axs[row,1].set_xticks(np.arange(6),['LF','LM','LH','RF','RM','RH']);axs[row,1].set_title('Leg vertical support (native units)');axs[row,1].legend(fontsize=7)
  for label,res in state['residuals'].items():axs[row,2].plot(np.arange(48),res,label=label)
  axs[row,2].set_title('All 48 DOF residuals / failed cases explicit');axs[row,2].set_xlabel('DOF');axs[row,2].legend(fontsize=7)
 fig.suptitle('S165 corrected static diagnostic: native-soft vs rigid upper bound; NOT standing/controller');fig.tight_layout();fig.savefig(p/'STATIC_SUPPORT_COMPARISON.png',dpi=150)
 print(p/'STATIC_SUPPORT_COMPARISON.png')
if __name__=='__main__':main()
