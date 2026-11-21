"""Counterfactual DN readout at identical camera pose, pressure, and oral phase."""
import json
import numpy as np
import torch
from conditioned import ResponsiveSettling
from settling import SRC,OUT,OLD,load_floor,T
@torch.inference_mode()
def test(seed):
    with np.load(OUT/f'live-{seed}-on-responsive/replay.npz') as z:
        p=T(z['markers']);calls=T(z['dn_read']);gate=z['gate']
    h=torch.nn.functional.pad(calls,(0,0,3,0)).unfold(0,4,1).transpose(1,2)
    prev,now=p[:-2][gate],p[1:-1][gate];h=h[gate]
    if not len(h):return {'frames':0}
    f=load_floor(OLD/'floor.pt')(prev,now)
    m=ResponsiveSettling().eval();m.load_state_dict(torch.load(OUT/'responsive.pt',weights_only=True))
    d=(m(prev,now,f,h)-m(prev,now,f,torch.zeros_like(h))).norm(dim=-1)*1000
    # micrometers of change in one 5 ms output with observed DN vs the zero-history perturbation.
    tail={}
    if gate[-28:].all():
        start=len(gate)-28
        all_hist=torch.nn.functional.pad(calls,(0,0,3,0)).unfold(0,4,1).transpose(1,2)
        floor=load_floor(OLD/'floor.pt')
        for arm in ('observed','silenced'):
            a,b=p[start:start+1],p[start+1:start+2];speeds=[]
            for j in range(start,len(gate)):
                command=all_hist[j:j+1].contiguous()
                w=m(a,b,floor(a,b),command if arm=='observed' else torch.zeros_like(command))
                if arm=='observed':assert torch.equal(w[0],p[j+2]),(seed,j)
                speeds.append(float((w[0,0,:2]-b[0,0,:2]).norm()/.005))
                a,b=b,w
            tail[arm+'_peak_mm_s']=max(speeds)
            tail[arm+'_rms_mm_s']=float(np.mean(np.square(speeds))**.5)
    return dict(tail_matched=tail,frames=len(h),nonzero_history=int((h.flatten(1).sum(1)>0).sum()),
        changed_frames=int((d.amax(1)>.001).sum()),max_marker_um=float(d.max()),
        rms_marker_um=float((d.square().mean()).sqrt()))
if __name__=='__main__':
    rows={s:test(s) for s in (88207,88224,88225,91001)}
    (OUT/'command_audit.json').write_text(json.dumps(rows,indent=2))
    print(rows)
