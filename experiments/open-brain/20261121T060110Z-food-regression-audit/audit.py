"""Factorial replay with identical initial camera observations and recorded electrode DN."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
OUT=Path(__file__).resolve().parent
SRC=OUT.parent/'20261121T044921Z-twenty-food-teachers'
sys.path.insert(0,str(SRC))
from practice import food,read_json,ObservedBody,load_floor,local_axes,T,FOOD_PRACTICE
old=FOOD_PRACTICE/'long-rollout'
new=SRC/'twenty'
def fingerprint(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.inference_mode()
def run():
    train=read_json(new/'cohort-identity.json')['seeds']
    seeds=train+[88224,88225]
    episodes=[food(s) for s in seeds]
    lengths=[len(r[0][0]) for r in episodes]
    dn=torch.nn.utils.rnn.pad_sequence([r[0][2] for r in episodes],batch_first=True)
    recorded=torch.nn.utils.rnn.pad_sequence([r[0][1] for r in episodes],batch_first=True)
    initial=[torch.stack([r[0][0][i] for r in episodes]) for i in (0,1)]
    goals=T([[r[-1]['geometry']['goal_x']*10,r[-1]['geometry']['goal_y']*10] for r in episodes])
    results=[]
    for key in ['mouth_offset.json','skeleton.json']:
        assert fingerprint(old/key)==fingerprint(new/key)
    for body_label,body_dir in [('old',old),('new',new)]:
        body=ObservedBody(body_dir/'weights.pt').eval()
        offset=T(read_json(body_dir/'mouth_offset.json'))
        for floor_label,floor_dir in [('old',old),('new',new),('recorded',None)]:
            pressure=load_floor((floor_dir or old)/'floor.pt').eval()
            a,b=initial;last=pressure(a,a);frames=[]
            for t in range(1,max(lengths)-1):
                est=pressure(a,b)
                touch=recorded[:,t] if floor_dir is None else torch.stack((last,est),1)
                axis=local_axes(b);mouth=b[:,0]+axis@offset
                dist=(mouth[:,:2]-goals).norm(dim=1)
                height=(mouth[:,2]>=.02)&(mouth[:,2]<=1.44)
                support=(touch[:,1]>=.5).sum(1)>=3
                upright=axis[:,2,2]>=.88
                valid=(dist<=4.55)&height&support&upright
                frames.append(torch.stack([dist,valid.float(),height.float(),support.float(),upright.float(),b[:,0,0],b[:,0,1]],dim=1))
                nxt,_=body(a,b,touch,dn[:,t]);a,b=b,nxt;last=est
            rows=torch.stack(frames).numpy()
            for j,s in enumerate(seeds):
                m=rows[:lengths[j]-2,j];contact=m[:,1]>0
                results.append(dict(seed=s,split='original' if j<2 else ('added' if j<20 else 'held'),motion=body_label,pressure=floor_label,radial=bool(m[:,0].min()<=4.55),strict=bool(contact.any()),seconds=float(contact.sum()*.005),closest_mm=float(m[:,0].min()),net_mm=float(np.linalg.norm(m[-1,5:7]-m[0,5:7]))))
    (OUT/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    for b in ('old','new'):
        for f in ('old','new','recorded'):
            groups=[[r for r in results if (r['motion'],r['pressure'],r['split'])==(b,f,g)] for g in ('original','added','held')]
            print(b,f,'strict', [sum(r['strict'] for r in g) for g in groups], 'radial',[sum(r['radial'] for r in g) for g in groups],'distance',[round(np.mean([r['closest_mm'] for r in g]),2) for g in groups],'time',[round(np.mean([r['seconds'] for r in g]),3) for g in groups])
if __name__=='__main__':run()
