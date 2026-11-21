import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
here=Path(__file__).resolve().parent
def read(f):return json.loads((here/f).read_text())
live=read('registered.json')
assert live==read('registered-replay.json')
arms=['silent','left','right','both']
bg=read('background.json');rx=read('readback.json')
def stats(rows):
    return {k:sum(r[k] for r in rows) for k in ('correct','total','blank_dn','blanks')}
summary={}
for a in arms:
    rows=[r for r in rx['rows'] if r['arm']==a]
    methods={'uncompensated':lambda r:r['artifact_residual']==(0 if a=='silent' else 1)}
    if a!='silent':
        methods.update(ideal_1pct=lambda r:r['artifact_residual']==.01,ideal_exact=lambda r:r['artifact_residual']==0,
            learned_locked=lambda r:r['artifact_residual']=='calibration64' and r['jitter_samples']==0,
            learned_jitter=lambda r:r['artifact_residual']=='calibration64' and r['jitter_samples']==1,
            estimated_lag_gain=lambda r:r['artifact_residual']=='align_calibration64')
    summary[a]={m:stats([r for r in rows if fn(r)]) for m,fn in methods.items()}
    assert all(q['total']==306 and q['blanks']==768 for q in summary[a].values())
base=summary['silent']['uncompensated']['correct']/306
def passing(method):
    return all(summary[a][method]['correct']/306>=base-.05 and summary[a][method]['blank_dn']/768<=.05 for a in arms[1:])
result=dict(named=all(live['gates'].values()),background_selective=all(
    bg['2'][a]['extra_local_crossings']==0 for a in arms[1:]),raw_read_during_write=passing('uncompensated'),
    learned_aligned_paired_readback=passing('estimated_lag_gain'),score=summary,training_started=False)
assert not result['background_selective']
(here/'RESULT.json').write_text(json.dumps(result,indent=2))
fig,ax=plt.subplots(1,2,figsize=(9.2,3.7),layout='constrained')
xx=np.arange(4)
for i,res in enumerate(['1','2','4']):
    ax[0].bar(xx+(i-1)*.23,[bg[res][a]['extra_local_crossings'] for a in arms],width=.22,
              label=res+' µm',color=['#659d9c','#21595d','#9bbfbb'][i])
ax[0].set(title='Non-target local fibres above threshold',ylabel='Number of distinct roots',xticks=xx,xticklabels=arms)
ax[0].legend(title='Centroid bin',fontsize=8)
x=np.arange(3)
for i,(method,colour,display) in enumerate([('uncompensated','#b96840','Raw'),
        ('learned_jitter','#b7a170','Trigger template'),('estimated_lag_gain','#307477','Fit lag and gain')]):
    ax[1].bar(x+(i-1)*.23,[100*summary[a][method]['correct']/306 for a in arms[1:]],width=.22,color=colour,label=display)
ax[1].axhline(100*base,ls='--',color='#41424d',label='No stimulation')
ax[1].set(title='Descending identity during stimulation',ylabel='Correct spontaneous DN (%)',xticks=x,xticklabels=arms[1:],ylim=(0,100))
ax[1].legend(fontsize=8)
for a in ax:a.spines[['top','right']].set_visible(False)
fig.savefig(here/'PATCH226-WRITE.png',dpi=155)
print(json.dumps({k:v for k,v in result.items() if k!='score'}))
print(summary)
