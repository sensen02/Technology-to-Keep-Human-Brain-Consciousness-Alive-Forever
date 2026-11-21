"""Plot matched physical trajectories and measured left DNa02 events."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

HERE=Path(__file__).resolve().parent
audit=json.loads((HERE/'JOINT_AUDIT.json').read_text())
cases=audit['matched_cases']
labels=['Intact','Left DNa02 partial','Left DNa02 block','Full shaft profile']
colors=['#426b46','#c0781d','#217aab','#9c4d70']
fig, axes=plt.subplots(1,2,figsize=(11,4.7),gridspec_kw={'width_ratios':[1.55,1]})
ax, events=axes
for i,((name,case),label,color) in enumerate(zip(cases.items(),labels,colors,strict=True)):
    directory=HERE/f'live_{name}_left'
    geom=json.loads((directory/'RESULT.json').read_text())['geometry']
    with np.load(directory/'BODY.npz') as body:
        xy=body['trace'][:,1:3]*10
    ax.plot(*xy.T,color=color,label=label,lw=1.8)
    ax.scatter(*xy[-1],color=color,s=22)
    row=case['roots'][audit['target_root']]
    for j,key in enumerate(['source','arrived']):
        y=i+(j-.5)*.31
        events.barh(y,row[key],height=.27,color=color,alpha=.42 if j==0 else 1.)
        events.text(row[key]+1,y,str(row[key]),va='center',fontsize=8)
ax.add_patch(Rectangle((10*geom['near'],10*geom['low']),10*(geom['far']-geom['near']),
                       10*(geom['high']-geom['low']),color='#7c8088',alpha=.7))
ax.add_patch(Circle((10*geom['goal_x'],10*geom['goal_y']),4.9,
                    facecolor='#db7865',edgecolor='#b45a48',alpha=.18))
ax.scatter(10*geom['goal_x'],10*geom['goal_y'],s=70,color='#af4432',marker='*',label='Food center')
ax.set_aspect('equal',adjustable='datalim')
ax.set(xlabel='x (mm)',ylabel='y (mm)',title='Thorax trajectory on the left detour')
ax.legend(frameon=False,fontsize=8)
events.set_yticks(range(4),labels)
events.invert_yaxis()
events.set(xlabel='Spike count during each trial',title='Left DNa02 source and arrival')
events.text(.02,-.22,'Pale bars: source     Solid bars: arrival',transform=events.transAxes,fontsize=9)
events.margins(x=.18)
fig.tight_layout()
fig.savefig(HERE/'JOINT_PATHS.png',dpi=155,bbox_inches='tight')
