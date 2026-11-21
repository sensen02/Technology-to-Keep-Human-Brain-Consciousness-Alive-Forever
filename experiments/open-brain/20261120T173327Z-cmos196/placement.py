"""Select the 196-contact layout by geometry alone, before inspecting voltage."""
from pathlib import Path
import json
import numpy as np

HOME_DIR=Path(__file__).resolve().parent
DATA=Path('/run/media/sensen/Data1/omnisim-results/implant-site/banc-neck')
REF=DATA/'cmos-17um-wide'
SIDES=('720575941633589179','720575941626534090')
PITCH=136/14

def main():
    ref=json.loads((REF/'cmos-audit.json').read_text())
    center=np.array(ref['center_um'])[[0,2]]
    with np.load(REF/'native-face.npz') as z, np.load(REF/'cmos-local-probes.npz') as local:
        labels=z['labels']
        ids=z['roots'].astype(str)
        assert np.array_equal(ids,local['roots'])
        neural=local['native_present']
        axis=[(np.arange(labels.shape[i])+z['lower_xz'][i]+.5)*z['pitch_nm'][[0,2]][i]/1000 for i in range(2)]
    targets=[np.argwhere(labels==np.flatnonzero(ids==s)[0]+1) for s in SIDES]
    xyz=[np.column_stack((axis[0][t[:,0]],axis[1][t[:,1]])) for t in targets]
    allpix=np.nonzero(labels)
    lo=np.array([a[i.min()] for a,i in zip(axis,allpix)])
    hi=np.array([a[i.max()] for a,i in zip(axis,allpix)])
    def evaluate(c,n,p,w):
        half=n*p/2
        if np.any(lo<c-half) or np.any(hi>c+half):return None
        contacts=(np.arange(n)-(n-1)/2)*p
        mask=[np.abs(a[:,None]-v-contacts).min(axis=1)<w/2 for a,v in zip(axis,c)]
        unique=np.unique(labels[np.ix_(*mask)])
        hits=int(neural[unique[unique>0]-1].sum())
        cover=[int((mask[0][t[:,0]]&mask[1][t[:,1]]).sum()) for t in targets]
        clearance=min(float((w/2-np.abs((q-(c+contacts[0])+p/2)%p-p/2)).min()) for q in xyz)
        return dict(center_xz_um=c.tolist(),side=n,pitch_um=p,pad_um=w,root_hits=hits,
                    target_pixels=cover,minimum_target_edge_clearance_um=clearance)
    old=evaluate(center,8,17,12)
    extended=evaluate(center,14,17,12)
    dense_center=evaluate(center,14,PITCH,8)
    candidates=[]
    for x in range(-5,6):
        for z in range(-5,6):
            trial=evaluate(center+[x,z],14,PITCH,8)
            if trial and trial['target_pixels']==list(map(len,targets)):
                candidates.append(trial)
    best=max(candidates,key=lambda q:(q['root_hits'],q['minimum_target_edge_clearance_um']))
    result=dict(native_neural_roots=int(neural.sum()),target_sizes=list(map(len,targets)),old=old,
                extended=extended,dense_at_original_center=dense_center,dense=best,
                feasible_dense_count=len(candidates))
    assert old['root_hits']==2417 and extended['root_hits']==old['root_hits']
    (HOME_DIR/'placement.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
