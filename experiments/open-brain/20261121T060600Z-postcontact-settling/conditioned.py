"""Learn command-conditioned post-contact pose residuals on the same 20 teachers."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from settling import OUT, SRC, OLD, DT, Settling, frame, food, load_floor

class ResponsiveSettling(Settling):
    def __init__(self):
        super().__init__()
        self.command=torch.nn.Linear(68,63,bias=False)
        torch.nn.init.zeros_(self.command.weight)

    def forward(self,prev,now,load,dn):
        base=super().forward(prev,now,load)
        # Learn a finite displacement in body axes, retaining the teacher skeleton.
        delta=.05*self.command(dn.flatten(1)).tanh().reshape(-1,21,3)
        xy,_=frame(now)
        delta=torch.cat((delta[:,:,:2] @ xy.transpose(1,2),delta[:,:,2:]),2)
        return self.project(base+delta)

def train():
    torch.manual_seed(855)
    fit=json.loads((OUT/'fit.json').read_text())
    samples=[]
    for s in fit['seeds']:
        (p,f,dn),valid,_,_,cfg=food(s)
        for t in range(max(1,int(np.ceil(cfg['first_intake_s']/DT))),len(p)-11,2):
            if valid[t-1:t+12].all():
                samples.append((p[t-1],p[t],f[t,1],dn[t:t+10],p[t+10],f[t+10,1]))
    a,b,f,commands,y,fy=map(torch.stack,zip(*samples))
    floor=load_floor(OLD/'floor.pt').requires_grad_(False)
    m=ResponsiveSettling()
    result=m.load_state_dict(torch.load(OUT/'settling.pt',weights_only=True),strict=False)
    assert result.missing_keys==['command.weight'] and not result.unexpected_keys
    m.requires_grad_(False);m.command.requires_grad_(True)
    opt=torch.optim.AdamW(m.command.parameters(),lr=.0007)
    losses=[]
    for _ in range(320):
        i=torch.randint(len(b),(40,))
        u,v,pressure=a[i],b[i],f[i]
        for t in range(10):
            w=m(u,v,pressure,commands[i,t])
            u,v=v,w
            pressure=floor(u,v)
        loss=F.smooth_l1_loss(v,y[i],beta=.08)+.025*F.smooth_l1_loss(pressure.log1p(),fy[i].log1p(),beta=.2)
        opt.zero_grad();loss.backward();opt.step()
        losses.append(float(loss.detach()))
    torch.save(m.state_dict(),OUT/'responsive.pt')
    (OUT/'conditioned_fit.json').write_text(json.dumps(dict(seeds=fit['seeds'],samples=len(samples),losses=losses),indent=2))
    print('fit',len(samples),'loss',np.mean(losses[-40:]))
if __name__=='__main__':train()
