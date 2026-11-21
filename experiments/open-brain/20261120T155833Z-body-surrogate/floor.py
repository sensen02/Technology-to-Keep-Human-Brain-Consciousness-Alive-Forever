"""External flat-floor load estimator used only as an environment in the closed-loop test."""
import json
import torch
from torch import nn
from paths import HERE
from train import load
class FlatFloor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(9,48),nn.SiLU(),nn.Linear(48,32),nn.SiLU(),nn.Linear(32,1))
    def forward(self,prev,now):
        tips=now[:,5::3]; old=prev[:,5::3]
        knee=now[:,3::3]; ankle=now[:,4::3]
        height=now[:,0,2:3].expand_as(tips[:,:,2])
        features=torch.stack((tips[:,:,2],ankle[:,:,2],knee[:,:,2],height,
            (tips[:,:,2]-old[:,:,2])/.05,(ankle[:,:,2]-prev[:,4::3,2])/.05,
            torch.linalg.vector_norm(tips[:,:,:2]-old[:,:,:2],dim=-1)/.05,
            torch.linalg.vector_norm(tips[:,:,:2]-now[:,:1,:2],dim=-1),
            (height-tips[:,:,2])),dim=-1)
        return self.net(features).squeeze(-1).clamp(-2,2).expm1().clamp_min(0)*4
def main():
    torch.manual_seed(43)
    floor=FlatFloor()
    opt=torch.optim.AdamW(floor.parameters(),lr=.001)
    def samples(rows):
        return tuple(torch.cat(v) for v in zip(*[(p[:-1],p[1:],f[1:,1]) for _,p,f,_,_ in rows]))
    a,b,y=samples(load(10901)+load(11311))
    for _ in range(1000):
        i=torch.randint(len(y),(96,))
        loss=nn.functional.smooth_l1_loss(floor(a[i],b[i]),y[i])
        opt.zero_grad();loss.backward();opt.step()
    torch.save(floor.state_dict(),HERE/'floor.pt')
    a,b,y=samples(load(11701))
    with torch.no_grad():err=(floor(a,b)-y).abs().mean().item()
    (HERE/'floor.json').write_text(json.dumps({'test_mae_uN':err,'zero_mae_uN':y.mean().item()}))
if __name__=='__main__':main()
