"""Round12 summary figures from saved screens; zero simulation steps."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SRC=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-block-reweight/20261107T224429Z')
OUT=Path('/home/sensen/Desktop/engine修复验收/banc-block-reweight/20261107T224429Z')
s1=json.loads((SRC/'RESULT.json').read_text());s2=json.loads((SRC/'halfcenter/RESULT.json').read_text());s3=json.loads((SRC/'halfcenter_adapt/RESULT.json').read_text())
def vals(d,key='tripod_alternation_score'):
    return {k:v[key] for k,v in d['results'].items() if k!='baseline'}
fig,ax=plt.subplots(1,3,figsize=(15.5,4.8),layout='constrained')
v1=vals(s1);ax[0].boxplot([list(v1.values())],widths=.5);ax[0].scatter(np.ones(len(v1))+np.random.default_rng(0).uniform(-.12,.12,len(v1)),list(v1.values()),s=14,alpha=.75)
ax[0].axhline(.30,color='tab:green',ls='--',label='positive threshold +0.30');ax[0].axhline(0,color='k',lw=.8)
ax[0].set(xticks=[1],xticklabels=['screen 1\n124 coarse-class combos'],ylabel='tripod alternation score',title='Screen 1: all negative')
ax[0].legend(loc='lower right')
v2=vals(s2);order=sorted(v2,key=lambda k:-v2[k])[:14]
ax[1].barh(range(len(order)),[v2[k] for k in order],color='tab:blue')
ax[1].axvline(.30,color='tab:green',ls='--');ax[1].axvline(0,color='k',lw=.8)
ax[1].set(yticks=list(range(len(order))),yticklabels=[o.replace('_',' ') for o in order],xlabel='tripod alternation score',title='Screen 2: best crosses 0 only by silencing a group')
ax[1].tick_params(axis='y',labelsize=7)
g=s3['results'][s3['best_six_by_tripod_score'][0][0]]['group_mean_hz']
ax[2].scatter([v['group_mean_hz']['A_LF_RM_LH'] for k,v in s3['results'].items() if k!='baseline'],
              [v['tripod_alternation_score'] for k,v in s3['results'].items() if k!='baseline'],s=45,label='screen 3 combos')
ax[2].axhline(.30,color='tab:green',ls='--');ax[2].axhline(0,color='k',lw=.8);ax[2].axvline(2.,color='tab:red',ls=':',label='both-groups >2 Hz requirement')
ax[2].set(xlabel='group A mean rate (Hz)',ylabel='tripod alternation score',title='Screen 3: higher adaptation kills activity');ax[2].legend(fontsize=8)
fig.suptitle('Round12 preregistered screens | existing edges only, per-row 0.020 uS, neural open loop, no physics')
fig.savefig(OUT/'SCREENS_SUMMARY.png',dpi=150);plt.close(fig)
print('screen1 combos',len(v1),'best',max(v1.values()),'screen2 best',max(v2.values()),'screen3 best',max(vals(s3).values()))
