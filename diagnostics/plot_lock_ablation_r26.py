from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
d=json.loads((ROOT/'outputs/lock_ablation_r26.json').read_text())
rows=d['rows']; labels=['Locked\nmuscles on','Locked\nactuation off','Unlocked\nmuscles on','Unlocked\nactuation off']
fig,axs=plt.subplots(1,2,figsize=(11,4.7))
for ax,key,title in [(axs[0],'final_tilt_deg','Tilt after 0.5 s (degrees)'),(axs[1],'final_height_mm','Thorax height after 0.5 s (mm)')]:
 vals=[r[key] for r in rows];ax.bar(labels,vals,color=['#ad5555','#ad8888','#448888','#77aaaa'])
 for i,v in enumerate(vals):ax.text(i,v,f'{v:.3f}',ha='center',va='bottom')
 ax.set_title(title);ax.set_ylim(0,max(vals)*1.2)
fig.suptitle('R26: seven right-foreleg locks confounded the previous standing tests')
fig.text(.5,.02,'Simulator-only hand-built prototype. Actuation-off verified zero force at every step. Unlocked body still collapses.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,.92));fig.savefig(ROOT/'outputs/lock_ablation_r26.png',dpi=150)
