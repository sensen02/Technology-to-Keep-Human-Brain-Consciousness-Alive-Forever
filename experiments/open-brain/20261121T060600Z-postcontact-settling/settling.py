"""Local camera-frame settling, enabled only by sustained student oral contact."""
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent
SRC = OUT.parent / '20261121T044921Z-twenty-food-teachers'
sys.path.insert(0, str(SRC))
from practice import food, demo, local_axes, T, load_floor, FOOD_PRACTICE
from skeleton import bounded
OLD = FOOD_PRACTICE / 'long-rollout'
DT = .005

class ContactGate:
    def __init__(self):
        self.ticks = self.misses = 0
        self.active = False

    def step(self, contact):
        if contact:
            self.ticks += 1
            self.misses = 0
        else:
            self.misses += 1
            if not self.active or self.misses >= 30:
                self.ticks = 0
        self.active = self.ticks >= 12
        return self.active

def frame(p):
    fw = F.normalize(p[:,1,:2] - p[:,0,:2],dim=-1)
    side = torch.stack((-fw[:,1],fw[:,0]),1)
    xy = torch.stack((fw,side),2)
    origin = p[:,:1,:2]
    return xy,origin

def local(p):
    xy,origin = frame(p)
    return torch.cat(((p[:,:,:2]-origin) @ xy,p[:,:,2:]),2)

class Settling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rest = torch.nn.Parameter(torch.zeros(21,3))
        self.response = torch.nn.Linear(3,2)
        torch.nn.init.zeros_(self.response.weight)
        torch.nn.init.zeros_(self.response.bias)
        cfg=json.loads((OLD/'skeleton.json').read_text())
        self.register_buffer('bounds',T(cfg['bounds_mm']))
        self.register_buffer('chest',T(cfg['thorax_mm']))

    def project(self,p):
        axes=local_axes(p)
        center=p[:,:1]
        chest=torch.cat((center, center+self.chest[0]*axes[:,:,0][:,None],
                        center+self.chest[1]*axes[:,:,1][:,None]),1)
        limbs=p[:,3:].reshape(-1,6,3,3)
        vectors=torch.cat((limbs[:,:,:1]-center[:,:,None],limbs[:,:,1:]-limbs[:,:,:-1]),2)
        lengths=self.bounds.permute(2,0,1)
        segments=bounded(vectors,lengths[:,:,0,None],lengths[:,:,1,None])
        return torch.cat((chest,(center[:,:,None]+segments.cumsum(2)).flatten(1,2)),1)

    def forward(self,prev,now,load):
        xy,origin=frame(now)
        velocity=now[:,:1,:2]-prev[:,:1,:2]
        features=torch.stack((velocity[:,0].norm(dim=-1)/.05,
                              (load.sum(1)/15).clamp(0,4),
                              (load>=.5).float().mean(1)),1)
        # Bounded first-order relaxation of velocity and posture at a 5 ms step.
        decay,step=(.035+.23*self.response(features).sigmoid()).unbind(1)
        rel=self.rest[None]-torch.cat((self.rest[:1,:2],self.rest[:1,2:]*0),1)
        targetxy=rel[:,:,:2] @ xy.transpose(1,2) + origin
        target=torch.cat((targetxy,rel[:,:,2:].expand(len(now),-1,-1)),2)
        shift=velocity*(1-decay[:,None,None])
        move=torch.cat((shift,torch.zeros_like(shift[:,:,:1])),2)
        return self.project(now + step[:,None,None]*(target-now) + move)

def train():
    import numpy as np
    torch.manual_seed(463)
    seeds=json.loads((SRC/'twenty/cohort-identity.json').read_text())['seeds']
    samples=[];final=[]
    for seed in seeds:
        (p,load,_),valid,_,_,cfg=food(seed)
        # Only teacher timing labels define which frames to supervise.
        first=cfg['first_intake_s']
        for t in range(max(1,int(np.ceil(first/DT))),len(p)-11,2):
            if valid[t-1:t+12].all():
                samples.append((p[t-1],p[t],load[t,1],p[t+10],load[t+10,1]))
        mask=(torch.arange(len(p))>=len(p)-28)&torch.as_tensor(valid)&((load[:,1]>=.5).sum(1)>=4)
        final.append(local(p[mask]).median(0).values)
    a,b,f,y,fy=map(torch.stack,zip(*samples))
    target=torch.stack(final).median(0).values
    m=Settling();m.rest.data.copy_(target)
    floor=load_floor(OLD/'floor.pt').requires_grad_(False)
    opt=torch.optim.Adam(m.parameters(),lr=.002)
    losses=[]
    for epoch in range(320):
        idx=torch.randint(len(b),(40,))
        u,v,pressure=a[idx],b[idx],f[idx]
        for step in range(10):
            w=m(u,v,pressure)
            u,v=v,w
            pressure=floor(u,v)
        pose=F.smooth_l1_loss(v,y[idx],beta=.08)
        loads=F.smooth_l1_loss(pressure.log1p(),fy[idx].log1p(),beta=.2)
        loss=pose+.025*loads+.25*F.smooth_l1_loss(m.rest,target,beta=.025)
        opt.zero_grad();loss.backward();opt.step()
        losses.append(float(loss.detach()))
    torch.save(m.state_dict(),OUT/'settling.pt')
    (OUT/'fit.json').write_text(json.dumps(dict(seeds=seeds,windows=len(b),updates=len(losses),
        first40_loss=np.mean(losses[:40]),last40_loss=np.mean(losses[-40:]),
        terminal_teacher_medians=len(final),loss=losses),indent=2))
    print('trained',len(b),losses[-1],flush=True)
if __name__=='__main__':train()
