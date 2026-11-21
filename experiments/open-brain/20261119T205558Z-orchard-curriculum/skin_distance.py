"""Compute unbiased mouth-to-source-surface distances for terminal credit diagnosis."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def distances(points,tris):
    best=np.full(len(points),np.inf)
    for a,b,c in tris:
        edges=[(a,b),(b,c),(c,a)]
        norm=np.cross(b-a,c-a)
        size=np.linalg.norm(norm)
        if size<1e-10:
            continue
        norm/=size
        delta=(points-a)@norm
        projected=points-delta[:,None]*norm
        inside=np.ones(len(points),dtype=bool)
        for start,end in edges:
            edge=end-start
            inside&=(np.cross(edge,projected-start)@norm>=-1e-12)
            u=np.clip((points-start)@edge/(edge@edge),0,1)
            best=np.minimum(best,np.linalg.norm(points-start-u[:,None]*edge,axis=1))
        best[inside]=np.minimum(best[inside],np.abs(delta[inside]))
    return 10*best

if __name__=='__main__':
    rows={}
    for g in sorted(HERE.glob('*/TRAIN_MEASURES.npz')):
        r=json.loads((g.parent/'CONFIG.json').read_text())
        with np.load(HERE.parent/'20261119T194948Z-orchard-odor-transfer'/r['site']/'CROP.npz') as mesh,np.load(g) as z,np.load(g.parent/'BODY.npz') as b:
            oral=z['oral']; t=b['trace']
            dx=distances(oral[:,1:4],mesh['skin'])
            eligible=(oral[:,4]<=4.9)&(t[:,4]>=3)&(t[:,6]>=.88)
            rows[g.parent.name]=dict(source_surface_min_mm=float(dx[eligible].min()) if eligible.any() else None,
                under_055_samples=int(np.count_nonzero(eligible&(dx<=.55))),mouth_gate_samples_5ms=int(oral[:,5].sum()))
    (HERE/'SKIN_DISTANCE.json').write_text(json.dumps(rows,indent=2)+'\n')
