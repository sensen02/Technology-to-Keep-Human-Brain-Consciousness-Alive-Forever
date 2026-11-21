"""Score frozen optical predictions and draw the time-resolved observations."""
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

d=Path(__file__).resolve().parent
def read(rel):
    with (d/rel).open() as f:
        return list(csv.DictReader(f))
def digest(rel):
    return hashlib.sha256((d/rel).read_bytes()).hexdigest()
rows=read('predictions.csv')
ref=json.loads((d/'geometry_reference.json').read_text())
assert digest('predictions.csv')==(d/'predictions.sha256').read_text().split()[0]==(d/'replay.sha256').read_text().split()[0]
assert [(int(r['frame']),r['leg']) for r in rows]==[(r['frame'],r['leg']) for r in ref]
g=read('gauge/predictions.csv')
known=json.loads((d/'gauge/gauge_reference.json').read_text())
def errors(key,ref_key):
    e=[1000*abs(float(r[key])-t[ref_key]) for r,t in zip(rows,ref) if r[key]]
    return dict(zip(('median','p90','max'),np.percentile(e,[50,90,100])))
legs=list(dict.fromkeys(r['leg'] for r in rows))
events=[]
for leg in legs:
    trace=[r for r in rows if r['leg']==leg]
    for a,b in zip(trace,trace[1:]):
        if a['state']!=b['state']:
            events.append([leg,a['frame'],b['frame'],a['time_ms'],b['time_ms'],a['state'],b['state']])
with (d/'transitions.csv').open('w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['leg','from_frame','to_frame','from_ms','to_ms','from_observation','to_observation'])
    w.writerows(events)
audit=dict(samples=len(rows),state_counts=Counter(r['state'] for r in rows),transition_brackets=len(events),
           ground_absolute_error_um=errors('ground_z_mm','ground_at_estimated_xy_mm'),
           marker_gap_absolute_error_um=errors('marker_gap_mm','marker_gap_mm'),
           reference_marker_gap_range_um=[fn(r['marker_gap_mm']*1000 for r in ref) for fn in (min,max)],
           minimum_reference_tarsus_vertex_height_um=min(r['tarsus_min_vertex_vertical_mm']*1000 for r in ref),
           tarsus_mesh_intersections=sum(r['tarsus_surface_intersection'] for r in ref),
           covered=sum(float(r['low_mm'])<=s['marker_gap_mm']<=float(r['high_mm'])
                       for r,s in zip(rows,ref) if r['low_mm']),
           gauge=[{'leg':r['leg'],'reference_um':known[r['leg']+'_tarsus_tip'],
                   'estimated_um':1000*float(r['marker_gap_mm']),'state':r['state']} for r in g],
           predictions_sha256=digest('predictions.csv'),same_bytes_replay=True)
(d/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
fig,axes=plt.subplots(3,2,figsize=(10,6.1),sharex=True,sharey=True,layout='constrained')
for ax,leg in zip(axes.T.flat,legs):
    rr=[r for r in rows if r['leg']==leg]
    x=np.array([float(r['time_ms']) for r in rr])
    y=np.array([1000*float(r['marker_gap_mm']) for r in rr])
    guard=np.array([1000*float(r['guard_mm']) for r in rr])
    truth=[1000*r['marker_gap_mm'] for r in ref if r['leg']==leg]
    ax.axhline(0,color='#595959',lw=.8)
    ax.errorbar(x,y,yerr=guard,fmt='none',ecolor='#899ca3',capsize=2)
    ax.scatter(x,truth,marker='_',s=170,color='#171b24',label='Geometry reference',zorder=3)
    for state,color,label in [('clear','#116b85','Clear'),('contact_compatible','#cb721d','Contact compatible')]:
        keep=[r['state']==state for r in rr]
        ax.scatter(x[keep],y[keep],s=22,color=color,label=label,zorder=4)
    ax.set_title(leg,loc='left',fontweight='bold',fontsize=10)
    ax.grid(alpha=.16)
    ax.set_xticks([0,28,56,84,112,140])
fig.suptitle('Foot-marker clearance above the observed sand surface',fontweight='bold')
fig.supxlabel('Time from frame 450 (ms)  |  Samples every 14 ms')
fig.supylabel('Surface-normal gap (µm)')
axes[0,0].legend(fontsize=7,loc='upper left',framealpha=.95)
fig.savefig(d/'TOUCH.png',dpi=160)
print(json.dumps(audit,indent=2))
