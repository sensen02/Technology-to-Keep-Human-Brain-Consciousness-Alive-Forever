from skeleton import ObservedBody
from train import load,BodyModel,HERE,tensor
import numpy as np
import torch
import json

def dist(a,b):
    return torch.linalg.vector_norm(a-b,dim=-1)

def brier(x,y):
    return ((x-y)**2).mean((0,1))

@torch.inference_mode()
def main(seed=11701):
    prefix='' if seed==11701 else f'{seed}_'
    net=ObservedBody(HERE/'weights.pt').eval()
    dev=load(10901)+load(11311)
    sens=tensor(json.loads((HERE/'calibration.json').read_text())['sensitivity'])
    base=torch.cat([r[-1] for r in dev]).mean((0,1))/sens
    results=[]; paths={}
    for name,pos,f,dn,observed_an in load(seed):
        with np.load(HERE/'data'/name/'nerve.npz') as z:
            actual=tensor(z['check_an_emitted'].reshape(-1,5,10,2).any(2))
        curr=pos[1:-1];prev=pos[:-2];target=pos[2:]
        out,l=net(prev,curr,f[1:-1],dn[1:-1])
        nodn,_=net(prev,curr,f[1:-1],torch.zeros_like(dn[1:-1]))
        shuf,_=net(prev,curr,f[1:-1],dn.roll(67,0)[1:-1])
        noforce,_=net(prev,curr,torch.zeros_like(f[1:-1]),dn[1:-1])
        p=l.sigmoid()
        metrics={'name':name,'n_transitions':len(curr)}
        for key,est in dict(network=out,hold=curr,velocity=2*curr-prev,no_dn=nodn,shift_dn=shuf,no_touch=noforce).items():
            metrics[key+'_marker_mm']=float(dist(target,est).mean())
        metrics['feedback_brier']=brier(p,actual[2:]).tolist()
        metrics['base_brier']=brier(base,actual[2:]).tolist()
        metrics['mean_rate_hz']=(p.mean((0,1))*1000).tolist()
        metrics['actual_rate_hz']=(actual[2:].mean((0,1))*1000).tolist()
        for h in (10,40):
            start=torch.arange(1,len(pos)-h,h)
            a,b=pos[start-1],pos[start]
            for j in range(h):
                a,b=b,net(a,b,f[start+j],dn[start+j])[0]
            metrics[f'roll_{5*h}ms_marker_mm']=float(dist(b,pos[start+h]).mean())
            metrics[f'hold_{5*h}ms_marker_mm']=float(dist(pos[start],pos[start+h]).mean())
        generated=[pos[0:1],pos[1:2]]
        prob=[]
        for t in range(1,len(pos)-1):
            following,logit=net(*generated[-2:],f[t:t+1],dn[t:t+1])
            generated.append(following)
            prob.append(logit.sigmoid())
        trajectory=torch.cat(generated)
        assert torch.isfinite(trajectory).all()
        metrics['final_center_error_mm']=float(dist(trajectory[-1,0],pos[-1,0]))
        metrics['stop_last_100ms_net_speed_mm_s']=float(dist(trajectory[-1,0],trajectory[-21,0])/.1)
        metrics['teacher_stop_last_100ms_net_speed_mm_s']=float(dist(pos[-1,0],pos[-21,0])/.1)
        results.append(metrics)
        paths[name+'_expected']=pos.numpy()
        paths[name+'_predicted']=trajectory.numpy()
        paths[name+'_an_probability']=torch.cat(prob).numpy()
    np.savez_compressed(HERE/(prefix+'heldout.npz'),**paths)
    (HERE/(prefix+'evaluation.json')).write_text(json.dumps(results,indent=2,allow_nan=False))
    print(json.dumps(results[0],indent=2))
    print('mean',{k:np.asarray([r[k] for r in results]).mean(axis=0).tolist() for k in results[0] if k!='name'})

if __name__=='__main__':
    import argparse
    cli=argparse.ArgumentParser()
    cli.add_argument('--seed',type=int,default=11701)
    main(cli.parse_args().seed)
