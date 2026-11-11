"""Standalone saved-learning-curve figure, no physical/optimization work."""
from pathlib import Path
import sys,json
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
p=torch.load(ROOT/'continuation3000/TRAIN/checkpoint_003000.pt',map_location='cpu',weights_only=False)
curve=p['curve'];floors=json.loads((ROOT/'FAST_RANGE_FLOOR.json').read_text())['cases']
fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
for ax,name in zip(axes.flat,('left_full','right_full','straight_085','straight_090')):
 x=[r['step'] for r in curve];y=[r['postprime_MSE'][name] for r in curve]
 ax.semilogy(x,y,lw=1,label='Actual pre-update teacher-fit MSE')
 ax.axvline(1000,color='black',ls='--',lw=1,label='Exact resume; lr .001 -> .0001')
 ax.axhline(floors[name]['irreducible_bounded_action_MSE'],color='tab:red',ls=':',label='Pointwise gate/bound floor')
 ax.set_title(name);ax.set_ylabel('Bounded-action MSE');ax.grid(alpha=.25);ax.set_xlabel('Global optimization step')
axes.flat[0].legend(fontsize=8)
fig.suptitle('Saved fitting stability only — no claim of closed-loop motor success',fontsize=14)
fig.tight_layout();fig.savefig(HERE/'SAVED_FIT_STABILITY.png',dpi=150);plt.close(fig)
print(str(HERE/'SAVED_FIT_STABILITY.png'))
