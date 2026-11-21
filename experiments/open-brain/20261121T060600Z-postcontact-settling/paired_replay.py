"""Matched recorded-DN replay. No inference of ingestion from replay contact."""
import json
import numpy as np
import torch
from conditioned import ResponsiveSettling
from settling import OLD,SRC,OUT,ContactGate,Settling,local_axes,T,food,load_floor
from practice import ObservedBody

@torch.inference_mode()
def run(responsive=False):
    train=json.loads((SRC/'twenty/cohort-identity.json').read_text())['seeds']
    seeds=train+[88224,88225]
    episodes=[food(s) for s in seeds]
    lengths=np.array([len(ep[0][0]) for ep in episodes])
    dn=torch.nn.utils.rnn.pad_sequence([ep[0][2] for ep in episodes],batch_first=True)
    goals=T([[ep[-1]['geometry']['goal_x']*10,ep[-1]['geometry']['goal_y']*10] for ep in episodes])
    offset=T(json.loads((OLD/'mouth_offset.json').read_text()))
    motion=ObservedBody(OLD/'weights.pt').eval()
    floor=load_floor(OLD/'floor.pt').eval()
    correction=(ResponsiveSettling() if responsive else Settling()).eval()
    correction.load_state_dict(torch.load(OUT/('responsive.pt' if responsive else 'settling.pt'),weights_only=True))
    runs=[]
    for apply in (False,True):
        a,b=[torch.stack([ep[0][0][i] for ep in episodes]) for i in (0,1)]
        last=floor(a,a)
        gates=[ContactGate() for _ in seeds];poses=[];contacts=[];active=[]
        for t in range(1,max(lengths)-1):
            f=floor(a,b);axes=local_axes(b)
            mouth=b[:,0]+axes@offset
            contact=((mouth[:,:2]-goals).norm(dim=1)<=4.55)&(mouth[:,2]>=.02)&(mouth[:,2]<=1.44)&((f>=.5).sum(1)>=3)&(axes[:,2,2]>=.88)
            en=torch.tensor([g.step(bool(c)) for g,c in zip(gates,contact)]) & apply
            nxt,_=motion(a,b,torch.stack((last,f),1),dn[:,t])
            if en.any():
                adjusted=correction(a,b,f,dn[:,t]) if responsive else correction(a,b,f)
                nxt=torch.where(en[:,None,None],adjusted,nxt)
            poses.append(b.numpy());contacts.append(contact.numpy());active.append(en.numpy())
            a,b,last=b,nxt,f
        runs.append(tuple(map(np.stack,(poses,contacts,active))))
    rows=[]
    for i,s in enumerate(seeds):
        n=lengths[i]-2
        base,cur=[tuple(col[:n,i] for col in r) for r in runs]
        gated=np.flatnonzero(cur[2])
        pre=(gated[0]+1) if len(gated) else n
        assert np.array_equal(base[0][:pre],cur[0][:pre]) and np.array_equal(base[1][:pre],cur[1][:pre])
        rows.append(dict(seed=s,old_contact_frames=int(base[1].sum()),new_contact_frames=int(cur[1].sum()),equal_prefix_frames=int(pre),first_gate_frame=int(pre-1) if len(gated) else None))
    (OUT/('paired_responsive.json' if responsive else 'paired_replay.json')).write_text(json.dumps(rows,indent=2))
    print(sum(r['new_contact_frames']>0 for r in rows),'of',len(rows))
if __name__=='__main__':
    import sys
    run('--responsive' in sys.argv)
