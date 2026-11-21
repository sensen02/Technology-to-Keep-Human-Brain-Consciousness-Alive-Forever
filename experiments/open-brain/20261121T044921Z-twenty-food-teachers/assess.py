"""Compare shared-DN replay before attempting the full live electrode loop."""
import sys
import numpy as np
import torch
from practice import (OUT,PRE,OLD,PRIOR,FOOD_PRACTICE,DEV,HELD,ObservedBody,read_json,save_json,food,flat,load_floor,local_axes,T)


def model_paths(variant):
    return dict(previous=(FOOD_PRACTICE/'long-rollout'/'weights.pt',FOOD_PRACTICE/'long-rollout'/'floor.pt'),
                current=(OUT/variant/'weights.pt',OUT/variant/'floor.pt'))


@torch.inference_mode()
def open_loop(net,floor,p,f,dn,valid,goal=None,offset=None):
    metric={}
    valid2=valid[:-1]&valid[1:]
    metric['floor_mae_uN']=float((floor(p[:-1][valid2],p[1:][valid2])-f[1:,1][valid2]).abs().mean())
    for h in (10,40):
        steps=torch.tensor([t for t in range(1,len(p)-h,h) if valid[t-1:t+h+1].all()])
        a,b=p[steps-1],p[steps]
        for k in range(h):
            c,_=net(a,b,f[steps+k],dn[steps+k]);a,b=b,c
        metric[f'error_{5*h}_mm']=float((b-p[steps+h]).norm(dim=-1).mean())
    for estimated in (False,True):
        a,b=p[:1],p[1:2];last=floor(a,a);sequence=[a,b]
        for k in range(1,len(p)-1):
            load=floor(a,b)
            touch=torch.stack((last,load),1) if estimated else f[k:k+1]
            c,_=net(a,b,touch,dn[k:k+1])
            last=load;a,b=b,c;sequence.append(b)
        trajectory=torch.cat(sequence)
        assert torch.isfinite(trajectory).all()
        name='estimated' if estimated else 'recorded'
        metric[name+'_endpoint_mm']=float((trajectory[-1,0]-p[-1,0]).norm())
        metric[name+'_net_mm']=float((trajectory[-1,0,:2]-trajectory[0,0,:2]).norm())
        metric[name+'_tail_rms_mm_s']=float((trajectory[-41:,0,:2].diff(dim=0).square().sum(-1).mean()).sqrt()/.005)
        if goal is not None:
            mouth=trajectory[:,0]+local_axes(trajectory)@offset
            metric[name+'_mouth_min_mm']=float((mouth[:,:2]-goal).norm(dim=1).min())
    return metric


@torch.inference_mode()
def run(variant,seeds):
    rows=[]
    for seed in seeds:
        (p,f,dn),valid,_,_,cfg=food(seed)
        result=dict(seed=seed,teacher_intake=cfg['intake'],split='heldout' if seed in HELD else 'development')
        goal=T([cfg['geometry']['goal_x']*10,cfg['geometry']['goal_y']*10])
        for arm,(weights,pressure) in model_paths(variant).items():
            offset=T(read_json(weights.parent/'mouth_offset.json'))
            result[arm]=open_loop(ObservedBody(weights).eval(),load_floor(pressure).eval(),p,f,dn,valid,goal,offset)
        rows.append(result);print(result,flush=True)
    save_json(OUT/variant/'replay-evaluation.json',rows)
    rows=[]
    for seed in (11701,11903):
        name,p,f,dn,_=next(r for r in flat(seed) if r[0].endswith(('intact','clean_on')))
        result=dict(trial=name)
        for arm,weights in [('parent',PRE/'weights.pt'),('previous',FOOD_PRACTICE/'long-rollout'/'weights.pt'),
                            ('current',OUT/variant/'weights.pt')]:
            net=ObservedBody(PRE/'weights.pt')
            net.raw.load_state_dict(torch.load(weights,weights_only=True,map_location='cpu'))
            pressure=OLD/'floor.pt'
            result[arm]=open_loop(net.eval(),load_floor(pressure).eval(),p,f,dn,np.ones(len(p),bool))
        rows.append(result);print(result,flush=True)
    save_json(OUT/variant/'flat-regression.json',rows)

if __name__=='__main__':run(sys.argv[1],[int(x) for x in sys.argv[2:]])
