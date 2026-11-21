from pathlib import Path
import sys,shutil,json,hashlib
import numpy as np
import torch
OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'20261120T155833Z-body-surrogate'
sys.path.insert(0,str(OLD))
from train import tensor,build
from floor import FlatFloor
from skeleton import ObservedBody
def write(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def load(seed):
    rows=[]
    for dest in sorted((OUT/'data').glob(str(seed)+'_*')):
        with np.load(OLD/'data'/dest.name/'external.npz') as z:
            pos=tensor(z['markers_mm']);force=tensor(z['force_uN'])
        with np.load(dest/'observed.npz') as z:
            counts=tensor(z['dn'].reshape(-1,50,17).sum(1))
            current_labels=tensor(z['an'].reshape(-1,5,10,2).any(2))
        # Independent intact-body electrode observation supplies teacher-side AN labels.
        with np.load(OLD/'data'/dest.name/'nerve.npz') as z:
            labels=tensor(z['an'].reshape(-1,5,10,2).any(2))
        touch=torch.stack((torch.cat([force[:1],force[:-1]]),force),1)
        hist=torch.stack([torch.cat([torch.zeros_like(counts[:i]),counts[:-i]]) if i else counts for i in (3,2,1,0)],1)
        rows.append((dest.name,pos,touch,hist,torch.stack((current_labels,labels),-2)))
    return rows
def loss(net,batch,sens,horizon):
    pos,touch,dn,labels=batch
    prev,now=pos[:,0],pos[:,1]+torch.randn_like(pos[:,1])*.0018
    total=pos.new_zeros(())
    for t in range(horizon):
        nxt,logits=net(prev,now,touch[:,t+1],dn[:,t+1])
        motion=torch.nn.functional.smooth_l1_loss((nxt-pos[:,t+2])/.035,torch.zeros_like(nxt))
        detected=(logits.sigmoid().unsqueeze(-2)*sens).clamp(1e-6,1-1e-6)
        total+=motion+.8*torch.nn.functional.binary_cross_entropy(detected,labels[:,t+2])
        prev,now=now,nxt
    return total/horizon

def fit():
    torch.manual_seed(22617)
    shutil.copyfile(OLD/'skeleton.json',OUT/'skeleton.json')
    net=ObservedBody(OLD/'weights.pt')
    sens=tensor([json.loads((d/'calibration.json').read_text())['sensitivity'] for d in (OUT,OLD)])
    episodes=load(10901)+load(11311)
    assert len(episodes)==12
    batch=build(episodes)
    opt=torch.optim.AdamW(net.raw.parameters(),lr=.00022,weight_decay=.0005)
    history=[];horizon=8
    for epoch in range(20):
        values=[]
        for _ in range(12):
            idx=torch.randint(len(episodes),(72,))
            ticks=torch.randint(3,400-horizon-1,(72,))[:,None]+torch.arange(horizon+2)
            value=loss(net,[a[idx[:,None],ticks] for a in batch],sens,horizon=horizon)
            opt.zero_grad();value.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
            opt.step();values.append(value.item())
        history.append(float(np.mean(values)))
    torch.save(net.raw.state_dict(),OUT/'weights.pt')
    write('training.json',dict(epochs=20,updates=240,seed=22617,episodes=[r[0] for r in episodes],loss=history,source_sha=sha(OLD/'weights.pt'),new_sha=sha(OUT/'weights.pt'),horizon=horizon))
    print('training',history[0],history[-1],flush=True)
@torch.inference_mode()
def evaluate():
    rows=[]
    floor=FlatFloor().eval()
    floor.load_state_dict(torch.load(OLD/'floor.pt',weights_only=True,map_location='cpu'))
    arms={k:ObservedBody(folder/'weights.pt').eval() for k,folder in [('old',OLD),('new',OUT)]}
    def error(a,b):return (a-b).norm(dim=-1).mean().item()
    for name,pos,f,dn,labels in load(11701)+load(11903):
        with np.load(OLD/'data'/name/'nerve.npz') as z:
            actual=tensor(z['check_an_emitted'].reshape(-1,5,10,2).any(2))
        row={'name':name,'hold_mm':error(pos[1:-1],pos[2:]),'velocity_mm':error(2*pos[1:-1]-pos[:-2],pos[2:])}
        for k,net in arms.items():
            m={}; row[k]=m
            prediction,logit=net(pos[:-2],pos[1:-1],f[1:-1],dn[1:-1])
            m['step_mm']=error(prediction,pos[2:])
            m['feedback_brier']=((logit.sigmoid()-actual[2:])**2).mean((0,1)).tolist()
            m['rate_hz']=(1000*logit.sigmoid().mean((0,1))).tolist()
            m['no_dn_mm']=error(net(pos[:-2],pos[1:-1],f[1:-1],torch.zeros_like(dn[1:-1]))[0],pos[2:])
            for h in (10,40):
                start=torch.arange(1,len(pos)-h,h)
                a,b=pos[start-1],pos[start]
                for j in range(h): a,b=b,net(a,b,f[start+j],dn[start+j])[0]
                m[str(h*5)+'ms_mm']=error(b,pos[start+h])
            trajectory=[pos[0:1],pos[1:2]]
            for t in range(1,len(pos)-1):
                step,_=net(*trajectory[-2:],f[t:t+1],dn[t:t+1]);trajectory.append(step)
            rollout=torch.cat(trajectory)
            assert torch.isfinite(rollout).all()
            m['endpoint_mm']=error(rollout[-1,0],pos[-1,0])
            m['travel_mm']=error(rollout[-1,0,:2],rollout[0,0,:2])
            m['last100ms_mm_s']=error(rollout[-1,0],rollout[-21,0])/.1
            generated=[pos[:1],pos[1:2]]
            previous_load=floor(generated[0],generated[0])
            for t in range(1,len(pos)-1):
                force_est=floor(*generated[-2:])
                pair=torch.stack((previous_load,force_est),1)
                next_pose,_=net(*generated[-2:],pair,dn[t:t+1])
                previous_load=force_est
                generated.append(next_pose)
            free=torch.cat(generated)
            assert torch.isfinite(free).all()
            m['self_floor_travel_mm']=error(free[-1,0,:2],free[0,0,:2])
            m['self_floor_last100ms_mm_s']=error(free[-1,0],free[-21,0])/.1
            m['self_floor_last200ms_xy_rms_speed_mm_s']=float((free[-41:,0,:2].diff(dim=0).square().sum(-1).mean()).sqrt()/.005)
            row['teacher_travel_mm']=error(pos[-1,0,:2],pos[0,0,:2])
            row['teacher_last100ms_mm_s']=error(pos[-1,0],pos[-21,0])/.1
            row['teacher_hz']=(actual[2:].mean((0,1))*1000).tolist()
        rows.append(row)
    write('evaluation.json',rows)
if __name__=='__main__':
    if '--score-only' not in sys.argv:fit()
    evaluate()
