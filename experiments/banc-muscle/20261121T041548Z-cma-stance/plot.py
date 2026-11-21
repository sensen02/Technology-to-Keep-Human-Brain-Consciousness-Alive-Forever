"""Plot native stance-gate observables for baseline, optimized nominal and held-outs."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from bench import HERE, HANDOFF

result=json.loads((HERE/'ACCEPTANCE.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(10,6.5),sharex=True,layout='constrained')
for ax in axes.flat:
    ax.grid(alpha=.18)
for name,label,color,style in [('base','Original','#92979e','-'),(result['selected'],'CMA-ES, nominal','#1358a2','-'),
                              ('positive','CMA-ES, +0.03 mm/s','#12805e','--'),('negative','CMA-ES, −0.03 mm/s','#b35611',':')]:
    with np.load(HERE/'validation'/name/'warm_release.npz') as d:
        loads=d['loads'][HANDOFF:]; height=d['torso'][HANDOFF:,0]/d['torso'][0,0]
        t=np.arange(len(height))*.1
        for ax,v in zip(axes.flat,(height,loads[:,:6].sum(axis=1),loads[:,-1],(loads[:,:6]>.01).sum(axis=1))):
            ax.plot(t,v,style,color=color,label=label,lw=1.45)
axes[0,0].axhline(.9,color='k',lw=.8,ls=':')
axes[0,1].axhspan(.8,1.2,color='#378a59',alpha=.1)
axes[1,1].axhline(3,color='k',lw=.8,ls=':')
for ax,title in zip(axes.flat,['COM height / initial height','Total foot load / body weight','Nonfoot load / body weight','Number of loaded legs']):
    ax.set_title(title,loc='left',fontsize=11)
for ax in axes[1]:
    ax.set_xlabel('Time after release (ms)')
axes[0,0].legend(frameon=False,fontsize=8)
axes[0,0].set_xlim(0,250)
fig.savefig(HERE/'STANCE.png',dpi=155)
