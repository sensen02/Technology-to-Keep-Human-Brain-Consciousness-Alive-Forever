"""Motion-only adaptation on external labels, taxels and sorted voltage. Seed 88211 is held out."""
from pathlib import Path
import shutil,json,sys,hashlib
import numpy as np
import torch
OUT=Path(__file__).resolve().parent
BASE=OUT.parent
OLD=BASE/'20261120T155833Z-body-surrogate'
PRE=BASE/'20261121T025425Z-mild-terrain-100x/run'
sys.path.insert(0,str(OLD))
from skeleton import ObservedBody
from model import FeedbackSampler
from floor import FlatFloor
torch.set_num_threads(1)
T=lambda x:torch.as_tensor(x,dtype=torch.float32)
def data(seed):
    d=OUT/f'aligned-{seed}'
    with np.load(d/'external.npz') as f:
        pos=T(f['markers_mm']);loads=T(f['force_uN']);valid=f['valid'];mouth=T(f['mouth_mm'])
    with np.load(d/'electrode.npz') as f:counts=T(f['counts'])
    assert len(pos)==len(counts) and np.isfinite(loads).all()
    hist=torch.stack([torch.cat([torch.zeros_like(counts[:k]),counts[:-k]]) if k else counts for k in (3,2,1,0)],1)
    force=torch.stack((torch.cat([loads[:1],loads[:-1]]),loads),1)
    return pos,force,hist,mouth,valid
def local_axes(p):
    fw=torch.nn.functional.normalize(p[:,1]-p[:,0],dim=1)
    z=torch.nn.functional.normalize(torch.linalg.cross(fw,p[:,2]-p[:,0]),dim=1)
    return torch.stack((fw,torch.linalg.cross(z,fw),z),dim=2)
def fit_floor(train,d):
    floor=FlatFloor()
    floor.load_state_dict(torch.load(OLD/'floor.pt',weights_only=True))
    allpairs=[(r[0][:-1][r[4][1:]&r[4][:-1]],r[0][1:][r[4][1:]&r[4][:-1]],r[1][1:,1][r[4][1:]&r[4][:-1]]) for r in train]
    prev,now,target=[torch.cat([row[k] for row in allpairs]) for k in range(3)]
    opt=torch.optim.AdamW(floor.parameters(),lr=.0008)
    torch.manual_seed(922)
    for _ in range(500):
        i=torch.randint(len(now),(80,))
        estimate=floor(prev[i],now[i])
        err=torch.nn.functional.smooth_l1_loss(estimate,target[i])
        opt.zero_grad();err.backward();opt.step()
    torch.save(floor.state_dict(),d/'floor.pt')
    return floor.eval().requires_grad_(False)

def fit(adapt=False):
    train=[data(s) for s in (88207,88209)];d=OUT/('adapted' if adapt else 'student');d.mkdir(exist_ok=True)
    allp=torch.cat([r[0][r[4]] for r in train])
    # External, training-only segment measurements for the FlyBody morphology.
    limb=allp[:,3:].reshape(-1,6,3,3);bounds=[]
    for k in range(3):
        length=(limb[:,:,k]-(limb[:,:,k-1] if k else allp[:,:1])).norm(dim=-1)
        bounds.append(torch.stack([(length.amin(0)-.015).clamp_min(0),length.amax(0)+.015]).tolist())
    cfg=dict(bounds_mm=bounds,thorax_mm=(allp[:,1:3]-allp[:,:1]).norm(dim=-1).median(dim=0).values.tolist())
    (d/'skeleton.json').write_text(json.dumps(cfg))
    shutil.copyfile(PRE/'weights.pt',d/'initial.pt')
    net=ObservedBody(d/'initial.pt');net.raw.feedback.requires_grad_(False)
    coord=[]
    for p,_,_,mouth,valid in train:
        ok=torch.from_numpy(valid)&torch.isfinite(mouth).all(1)
        coord.append(((mouth[ok]-p[ok,0])[:,None,:]@local_axes(p[ok]))[:,0])
    offset=torch.cat(coord).median(0).values
    (d/'mouth_offset.json').write_text(json.dumps(offset.tolist()))
    predictor=fit_floor(train,d) if adapt else None
    torch.manual_seed(62441)
    opt=torch.optim.AdamW(net.raw.motion.parameters(),lr=.00014,weight_decay=.0005)
    horizon=10;windows=[]
    for j,r in enumerate(train):
        windows+=[(j,t) for t in range(1,len(r[0])-horizon-1) if r[4][t-1:t+horizon+1].all()]
    assert min(sum(j==s for j,t in windows) for s in (0,1))>40
    logs=[]
    for epoch in range(16):
        epoch_loss=[]
        for _ in range(12):
            batch=[windows[i] for i in torch.randint(len(windows),(48,))]
            pos,f,dn=[torch.stack([train[j][index][t-1:t+horizon+1] for j,t in batch]) for index in (0,1,2)]
            a,b=pos[:,0],pos[:,1];loss=0.
            if adapt:
                previous_load=predictor(a,a)
                choose=torch.arange(len(pos))%2==0
            for t in range(horizon):
                loads=f[:,t+1]
                if adapt:
                    predicted=predictor(a,b)
                    loads=torch.where(choose[:,None,None],torch.stack((previous_load,predicted),1),loads)
                    previous_load=predicted
                v,_=net(a,b,loads,dn[:,t+1])
                loss+=torch.nn.functional.smooth_l1_loss((v-pos[:,t+2])/.035,torch.zeros_like(v))
                a,b=b,v
            loss=loss/horizon
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.raw.motion.parameters(),2.);opt.step()
            epoch_loss.append(loss.item())
        logs.append(float(np.mean(epoch_loss)))
    torch.save(net.raw.state_dict(),d/'weights.pt')
    (d/'training.json').write_text(json.dumps(dict(updates=192,train_seeds=[88207,88209],heldout=88211,windows=len(windows),loss=logs,inputs=['external.npz','electrode.npz'],initial_sha=hashlib.sha256((PRE/'weights.pt').read_bytes()).hexdigest()),indent=2))
@torch.inference_mode()
def evaluate(adapt=False):
    p,f,dn,mouth,valid=data(88211);d=OUT/('adapted' if adapt else 'student')
    offset=T(json.loads((d/'mouth_offset.json').read_text()))
    floor=FlatFloor();floor.load_state_dict(torch.load(OLD/'floor.pt',weights_only=True))
    ok=torch.from_numpy(valid)&torch.isfinite(mouth).all(1)
    reconstructed=p[ok,0]+(local_axes(p[ok])@offset)
    metrics={'external_mouth_mean_mm':float((reconstructed-mouth[ok]).norm(dim=-1).mean()),'frames':len(p),'complete':int(valid.sum())}
    for arm,path in [('parent',PRE/'weights.pt'),('parent_refit_geometry',d/'initial.pt'),('student',d/'weights.pt')]:
        floor.load_state_dict(torch.load(d/'floor.pt' if adapt and arm=='student' else OLD/'floor.pt',weights_only=True))
        net=ObservedBody(path).eval();v={}
        valid_pairs=valid[1:]&valid[:-1]
        v['floor_mae_uN']=float((floor(p[:-1][valid_pairs],p[1:][valid_pairs])-f[1:,1][valid_pairs]).abs().mean())
        for h in [1,10,40]:
            times=torch.as_tensor([t for t in range(1,len(p)-h,h) if valid[t-1:t+h+1].all()])
            if not len(times):continue
            a,b=p[times-1],p[times]
            for t in range(h):a,b=b,net(a,b,f[times+t],dn[times+t])[0]
            v[str(5*h)+'ms_mm']=float((b-p[times+h]).norm(dim=-1).mean())
        times=torch.as_tensor([t for t in range(1,len(p)-10,10) if valid[t-1:t+11].all()])
        a,b=p[times-1],p[times]
        for t in range(10):a,b=b,net(a,b,torch.zeros_like(f[times+t]),dn[times+t])[0]
        v['no_pressure_50ms_mm']=float((b-p[times+10]).norm(dim=-1).mean())
        assert valid[:2].all() and valid[-1]
        for estimating in [False,True]:
            a,b=p[:1],p[1:2];last=floor(a,a);trajectory=[a,b];sampler=FeedbackSampler(1);requests=[]
            gen=torch.Generator().manual_seed(88211)
            for t in range(1,len(p)-1):
                force=floor(a,b)
                loads=torch.stack((last,force),1) if estimating else f[t:t+1]
                v2,logits=net(a,b,loads,dn[t:t+1]);prob=logits.sigmoid()
                requests.append(sampler.sample(prob,torch.rand(prob.shape,generator=gen)))
                trajectory.append(v2);a,b=b,v2;last=force
            seq=torch.cat(trajectory)
            assert torch.isfinite(seq).all()
            key='estimated' if estimating else 'recorded'
            v[key+'_endpoint_mm']=float((seq[-1,0]-p[-1,0]).norm())
            v[key+'_travel_mm']=float((seq[-1,0,:2]-seq[0,0,:2]).norm())
            v[key+'_tail_rms_mm_s']=float((seq[-41:,0,:2].diff(dim=0).square().sum(-1).mean()).sqrt()/.005)
            if arm=='student' and estimating:
                np.savez_compressed(d/'heldout_replay.npz',trajectory=seq.numpy(),mouth=(seq[:,0]+local_axes(seq)@offset).numpy(),requests=torch.cat(requests).reshape(-1,2).numpy(),request_first_ms=10)
        metrics[arm]=v
    (d/'evaluation.json').write_text(json.dumps(metrics,indent=2,allow_nan=False));print(metrics,flush=True)
if __name__=='__main__':fit();evaluate()
