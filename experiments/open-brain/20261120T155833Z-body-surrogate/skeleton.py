"""Fit and apply marker-distance limits using external training measurements in mm."""
import json
from pathlib import Path
import torch
from torch import nn
from model import BodyModel

def extent(vec):
    return torch.linalg.vector_norm(vec,dim=-1,keepdim=True)

def bounded(vec,lower,upper):
    length=extent(vec).clamp_min(1e-8)
    return vec*(length.clamp(lower,upper)/length)

class ObservedBody(nn.Module):
    def __init__(self,weights):
        super().__init__()
        self.raw=BodyModel()
        self.raw.load_state_dict(torch.load(weights,weights_only=True,map_location='cpu'))
        cfg=json.loads(Path(weights).with_name('skeleton.json').read_text())
        self.register_buffer('limits',torch.tensor(cfg['bounds_mm']))
        self.register_buffer('thorax',torch.tensor(cfg['thorax_mm']))

    def forward(self,prev,now,touch,dn):
        pose,logits=self.raw(prev,now,touch,dn)
        center=pose[:,0:1]
        forward=nn.functional.normalize(pose[:,1:2]-center,dim=-1)
        across=pose[:,2:3]-center
        side=nn.functional.normalize(across-(across*forward).sum(-1,keepdim=True)*forward,dim=-1)
        # Keep the observed thorax triangle rigid. Bounds on distal tip length allow tarsal flexion.
        chest=torch.cat([center,center+self.thorax[0]*forward,center+self.thorax[1]*side],1)
        limbs=pose[:,3:].reshape(-1,6,3,3)
        anchored=center
        result=[]
        for j in range(3):
            delta=limbs[:,:,j]-(center if j==0 else limbs[:,:,j-1])
            delta=bounded(delta,self.limits[j,0,:,None],self.limits[j,1,:,None])
            anchored=anchored+delta
            result.append(anchored)
        return torch.cat((chest,torch.stack(result,2).flatten(1,2)),1),logits

if __name__=='__main__':
    from train import load,HERE
    pos=torch.cat([row[1] for seed in (10901,11311) for row in load(seed)])
    limbs=pos[:,3:].reshape(-1,6,3,3)
    bounds=[]
    for j in range(3):
        delta=limbs[:,:,j]-(pos[:,:1] if j==0 else limbs[:,:,j-1])
        length=extent(delta).squeeze(-1)
        # 15 µm measurement allowance at either edge of the observed training range.
        bounds.append(torch.stack(((length.amin(0)-.015).clamp_min(0),length.amax(0)+.015)).tolist())
    thorax=extent(pos[:,1:3]-pos[:,:1]).squeeze(-1).median(0).values.tolist()
    (HERE/'skeleton.json').write_text(json.dumps(dict(bounds_mm=bounds,thorax_mm=thorax),indent=2))
