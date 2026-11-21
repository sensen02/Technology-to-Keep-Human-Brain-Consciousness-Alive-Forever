"""Rotate and translate a held-out dual punch to test sub-cell position dependence."""
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from floor import Floor,HERE
from acquire import image
from estimate import decode
rng=np.random.default_rng(431)
truth=[];depth=[]
for k in range(12):
    angle=k*np.pi/12
    a=rng.uniform([-1.3,-1.4],[4.6,1.4])
    xy=np.array([a,a+.50*np.array([np.cos(angle),np.sin(angle)])])
    f=Floor('probes',xy)
    v=[f.m.joint(f'weight_{i}').dofadr[0] for i in range(2)]
    f.d.qfrc_applied[v]=[-1.4,-4.8]
    for _ in range(220):f.step()
    depth.append(f.depth().copy());truth.append(xy)
folder=HERE/'spatial'
image(depth,folder,912)
records,_=decode(folder,json.loads((HERE/'cal/response.json').read_text()))
rows=[];errors=[];locations=[]
for i,xy in enumerate(truth):
    r=[p for p in records if p['frame']==i]
    pred=np.array([[p['x_mm'],p['y_mm'],p['fz_uN']] for p in r])
    split=len(r)==2
    if split:
        a,b=linear_sum_assignment(np.linalg.norm(xy[:,None,:]-pred[None,:,:2],axis=-1))
        errors.extend(np.abs(np.array([1.4,4.8])[a]-pred[b,2]))
        locations.extend(np.linalg.norm(xy[a]-pred[b,:2],axis=1))
    rows.append(dict(angle_deg=15*i,xy_mm=xy.tolist(),patches=pred.tolist(),two_resolved=split))
result=dict(configurations=len(rows),resolved=sum(r['two_resolved'] for r in rows),
            resolved_force_p90_uN=float(np.percentile(errors,90)) if errors else None,
            resolved_position_p90_mm=float(np.percentile(locations,90)) if locations else None,rows=rows)
(folder/'result.json').write_text(json.dumps(result,indent=2))
print({k:v for k,v in result.items() if k!='rows'})
