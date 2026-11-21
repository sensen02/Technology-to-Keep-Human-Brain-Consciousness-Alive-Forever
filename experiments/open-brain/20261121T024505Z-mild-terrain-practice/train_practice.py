"""Continue only the motion head; isolate the effect of new physics data with a matched flat control."""
from pathlib import Path
import hashlib,json,sys,shutil
import torch
import numpy as np
OUT=Path(__file__).resolve().parent
PARENT=OUT.parent/'20261121T012712Z-body226-distal-cut'
sys.path.insert(0,str(PARENT))
from fit_body import load as flat,ObservedBody,FlatFloor,OLD,tensor
torch.set_num_threads(1)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,data):p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def floor_model():
    model=FlatFloor().eval()
    model.load_state_dict(torch.load(OLD/'floor.pt',weights_only=True,map_location='cpu'))
    model.requires_grad_(False)
    return model
def corpus(split):
    seeds=(10901,11311) if split=='train' else (11701,11903)
    original={r[0]:r[:4] for seed in seeds for r in flat(seed)}
    terrain=[];match=[]
    for dest in sorted((OUT/'episodes').glob('terrain*')):
        info=json.loads((dest/'receipt.json').read_text())
        if info['split']!=split:continue
        aligned=original[f"{info['seed']}_{info['mode']}"]
        with np.load(dest/'external.npz') as z:
            pos=tensor(z['markers_mm']);f=tensor(z['force_uN'])
        touch=torch.stack((torch.cat((f[:1],f[:-1])),f),dim=1)
        terrain.append((dest.name,pos,touch,aligned[3]))
        match.append(aligned)
    assert len(terrain)==(16 if split=='train' else 4)
    return list(original.values()),terrain,match
def fit(terrain_enabled):
    out=OUT/('expanded' if terrain_enabled else 'flat_control');out.mkdir(exist_ok=True)
    base,terrain,match=corpus('train')
    rows=base+(terrain if terrain_enabled else match)
    tensors=[torch.stack([row[i] for row in rows]) for i in (1,2,3)]
    net=ObservedBody(PARENT/'weights.pt')
    net.raw.feedback.requires_grad_(False)
    floor=floor_model()
    opt=torch.optim.AdamW(net.raw.motion.parameters(),lr=.00016,weight_decay=.0005)
    torch.manual_seed(51603)
    log=[];h=14
    for ep in range(24):
        errors=[]
        for batch in range(12):
            idx=torch.randint(len(rows),(64,))
            start=torch.randint(1,400-h-1,(64,))
            times=start[:,None]+torch.arange(h+2)
            p,touch,commands=[v[idx[:,None],times] for v in tensors]
            prev,now=p[:,0],p[:,1]
            last=floor(prev,prev)
            use_est=torch.rand((len(idx),1,1))<.50
            loss=0.
            for k in range(1,h+1):
                est=floor(prev,now)
                loads=torch.where(use_est,torch.stack((last,est),dim=1),touch[:,k])
                step,_=net(prev,now,loads,commands[:,k])
                loss+=torch.nn.functional.smooth_l1_loss((step-p[:,k+1])/.035,torch.zeros_like(step))
                last=est;prev,now=now,step
            loss=loss/h
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(net.raw.motion.parameters(),2.)
            opt.step();errors.append(loss.item())
        log.append(float(np.mean(errors)))
    torch.save(net.raw.state_dict(),out/'weights.pt')
    shutil.copyfile(PARENT/'skeleton.json',out/'skeleton.json')
    write(out/'training.json',dict(seed=51603,epochs=24,updates=288,rollout_steps=h,
          input_weights_sha256=digest(PARENT/'weights.pt'),weights_sha256=digest(out/'weights.pt'),
          rows=[r[0] for r in rows],motion_loss=log))
    print(out.name,log[0],log[-1],flush=True)
if __name__=='__main__':fit(sys.argv[1]=='expanded')
