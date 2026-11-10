"""Parent analysis of the adaptation screen: modality structure of leg rates; 0 simulation steps."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
OUT=Path('/home/sensen/Desktop/engine修复验收/banc-descending-learning/20261107T203049Z')
LEGS=['LF','LH','LM','RF','RH','RM']
res=json.loads((HERE/'RESULT.json').read_text())
modes={}
for name in res['schemes']:
    z=np.load(HERE/f'RATES_{name}.npz');rates=z['rates_hz'].astype(float);leg=z['leg']
    series=np.stack([rates[:,leg==L.lower()].mean(1) for L in LEGS],1)
    X=series-series.mean(0);U,S,Vt=np.linalg.svd(X,full_matrices=False);frac=S**2/np.sum(S**2)
    modes[name]={'variance_fraction_PC1_PC2_PC3':[float(x) for x in frac[:3]],
                 'PC1_loadings':{L:float(v) for L,v in zip(LEGS,Vt[0])},
                 'PC2_loadings':{L:float(v) for L,v in zip(LEGS,Vt[1])},
                 'left_right_PC1_asymmetry':float(abs(abs(Vt[0][0])-abs(Vt[0][3]))),
                 'front_hind_PC1_opposition':float(-np.sign(Vt[0][0]*Vt[0][1])),
                 'tripod_vs_front_hind_alignment':float(abs(np.dot(Vt[0],[1,1,0,-1,-1,0])/np.linalg.norm([1,1,0,-1,-1,0]))),
                 'tripod_alignment':float(abs(np.dot(Vt[0],[1,1,0,-1,-1,0])/np.linalg.norm(np.array([1,1,0,-1,-1,0]))))}
out={'status':'ADAPTATION_SCREEN_MODAL_ANALYSIS','schemes':{k:{kk:vv for kk,vv in v.items() if kk not in ('per_leg_mean_hz','per_leg_cv','pair_correlations','dominant_hz')} for k,v in res['schemes'].items()},
     'modality':modes,'equivalence_scale1_bit_exact':res['equivalence_scale1_bit_exact'],'alternation_positive':res['alternation_positive'],
     'limits':'Saved rates only, 0 steps. PCA of 6 leg-mean rates over 1.5s of open-loop neural replay; small-dimensional summary, not a gait analysis and not a claim about fly VNC.'}
(OUT/'ADAPTATION_MODALITY.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
fig,ax=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
names=list(res['schemes']);x=np.arange(len(names))
ax[0].bar(x,[res['schemes'][n]['tripod_alternation_score'] for n in names],color=['tab:gray']+['tab:blue']*(len(names)-1))
ax[0].axhline(.3,color='tab:green',ls='--',label='preregistered positive threshold');ax[0].axhline(0,color='k',lw=.8)
ax[0].set_xticks(x,names,rotation=30,ha='right');ax[0].set(ylabel='tripod alternation score',title='Alternation never becomes positive');ax[0].legend()
ax[1].bar(x,[res['schemes'][n]['ipsilateral_score'] for n in names],color=['tab:gray']+['tab:orange']*(len(names)-1))
ax[1].axhline(0,color='k',lw=.8);ax[1].set_xticks(x,names,rotation=30,ha='right');ax[1].set(ylabel='left-vs-right correlation',title='Scale only weakly breaks same-side synchrony')
base=modes['baseline']
ax[2].bar(np.arange(6),list(base['PC1_loadings'].values()),label='PC1 %.0f%%'%(100*base['variance_fraction_PC1_PC2_PC3'][0]))
ax[2].bar(np.arange(6),list(base['PC2_loadings'].values()),alpha=.55,label='PC2 %.0f%%'%(100*base['variance_fraction_PC1_PC2_PC3'][1]))
ax[2].set_xticks(np.arange(6),LEGS);ax[2].axhline(0,color='k',lw=.8);ax[2].set(ylabel='loading',title='Dominant leg-rate modes (baseline)');ax[2].legend()
fig.suptitle('Adaptation-asymmetry screen and leg-rate modality | neural open loop, not walking')
fig.savefig(OUT/'ADAPTATION_SCREEN.png',dpi=150);plt.close(fig)
print(json.dumps({n:{k:(round(v,3) if isinstance(v,float) else v) for k,v in out['modality'][n].items() if k not in ('PC1_loadings','PC2_loadings')} for n in out['modality']},indent=2))
print(json.dumps({'PC1':base['PC1_loadings'],'PC2':base['PC2_loadings']},indent=2))
