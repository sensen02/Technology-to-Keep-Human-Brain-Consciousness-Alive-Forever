"""Plot independent intake and stance readouts for matched local positive controls."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
d=Path(__file__).resolve().parent
fig,axes=plt.subplots(2,2,figsize=(9.5,6),layout='constrained')
for col,stem in enumerate(('FOOD_','FOOD_release_')):
    for arm,color in [('off','#888888'),('on','#256d72')]:
        p=d/(stem+arm)
        if not (p/'CONFIG.json').exists():continue
        cfg=json.loads((p/'CONFIG.json').read_text())
        with np.load(p/'BODY.npz') as b:
            tr=b['trace'];m=b['meal']
            axes[0,col].plot(m[:,0],m[:,1],label='Tarsal reflex '+arm,color=color)
            axes[1,col].plot(tr[:,0],tr[:,6],color=color)
        axes[0,col].set_title(f"Apple ground-003 · facet {cfg['geometry']['source_face']}")
    axes[0,col].axhline(.18,color='#385994',linestyle=':',label='Meal threshold')
    axes[1,col].axhline(.94,color='#385994',linestyle=':')
    axes[0,col].legend(fontsize=8)
    axes[1,col].set_xlabel('Time (s)')
    axes[0,col].set_ylabel('Cumulative intake (model units)')
    axes[1,col].set_ylabel('Thorax upright cosine')
fig.savefig(d/'INGESTION.png',dpi=160)
