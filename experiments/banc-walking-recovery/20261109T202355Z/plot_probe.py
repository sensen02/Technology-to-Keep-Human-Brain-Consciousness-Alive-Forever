from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
d=json.loads((HERE/'SUPPORT_PROBE_V2.json').read_text())
fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
for r in d['runs']:
    s=r['summary']; trace=r['trace'];t=[x['k']*.001 for x in trace]
    label=f"height gain {s['height_gain']}, position gain {s['velocity_gain']}, rate fraction {s['rate_scale']}"
    for ax,key in zip(axes,['support','com','padz']):ax.plot(t,[x[key] for x in trace],label=label,lw=1)
for ax,v,lab in zip(axes,[.9,1.,.015],['Foot load / body weight','Thorax subtree COM (native)','RH pad clearance (native)']):
    ax.axhline(v,color='black',ls='--',lw=.8);ax.set_ylabel(lab);ax.grid(alpha=.3)
axes[0].legend(fontsize=8)
axes[-1].set_xlabel('Physical time (s), continuous simulation')
fig.suptitle('Whole-body teacher diagnosis, corrected adhesion wrench\nNOT autonomous walking; probes failed to complete a safe step')
fig.tight_layout();fig.savefig(HERE/'support_probe.png',dpi=150)
