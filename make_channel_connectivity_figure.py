from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from engine.embodied.closure_bundle import validate_bundle
P=Path(__file__).resolve().parent/'outputs/neural_muscle_closure'
b=validate_bundle(P);d=json.loads((Path(b['path'])/'closure_results.json').read_text())['measured']['sensory_motor_structural_coverage']['channels']
legs=list(d);channels=list(d[legs[0]])
a=np.array([[d[l][c]['reachable_motor_cells'] for c in channels] for l in legs])
fig,ax=plt.subplots(figsize=(11,5));im=ax.imshow(a,vmin=0,vmax=202,cmap='Blues',aspect='auto')
ax.set_xticks(range(len(channels)),channels,rotation=25,ha='right');ax.set_yticks(range(len(legs)),legs)
for i in range(len(legs)):
 for j in range(len(channels)):ax.text(j,i,str(a[i,j]),ha='center',va='center',color='white' if a[i,j]>100 else 'black')
ax.set_title('Reachable mapped motor cells per sensory channel (structural paths only)')
fig.colorbar(im,ax=ax,label='Mapped motor cells reachable / 202')
fig.text(.5,.02,'Unknown transmitter edges remain excluded. Club/noxious inputs are unmodeled: a path does not mean an active input.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,1));p=P/'channel_connectivity.png';fig.savefig(p,dpi=140,bbox_inches='tight');plt.close(fig);print(p)
