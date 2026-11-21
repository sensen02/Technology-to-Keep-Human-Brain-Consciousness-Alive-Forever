"""Stateless sensory neural drive plus fitted local motor/pad readouts."""
from shared import *
from torch import nn

HARMONICS=np.arange(1,129)


def basis(p):
    angle=p[...,None]*HARMONICS
    return np.concatenate((np.ones((*angle.shape[:-1],1)),np.cos(angle),np.sin(angle)),axis=-1)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.reflex=nn.Sequential(nn.Linear(272,192),nn.SiLU(),nn.Linear(192,192),nn.SiLU(),nn.Linear(192,6))
        self.register_buffer('center',torch.zeros(272));self.register_buffer('scale',torch.ones(272))
        self.register_buffer('fourier',torch.zeros(6,257,7,dtype=torch.float64))
        self.register_buffer('order',torch.zeros(42,dtype=torch.long))
        self.register_buffer('neutral',torch.zeros(42,dtype=torch.float64))
        self.register_buffer('synergy',torch.zeros(42,6,dtype=torch.float64))
        self.register_buffer('lambda_scale',torch.ones(6))
        self.register_buffer('pad_weights',torch.zeros(6,3,dtype=torch.float64))
        self.register_buffer('minimum',torch.zeros(42,dtype=torch.float64));self.register_buffer('maximum',torch.zeros(42,dtype=torch.float64))
        self.register_buffer('q_feature_columns',torch.zeros(42,dtype=torch.long))
        self.register_buffer('kinematic_reference',torch.zeros(66))
        nn.init.zeros_(self.reflex[-1].weight);nn.init.constant_(self.reflex[-1].bias,.03)

    def rhythmic(self,phase):
        value=np.einsum('nle,lej->nlj',basis(phase[:,:6]),self.fourier.cpu().numpy())
        return (value*phase[:,6:,None]).reshape(-1,42)[:,self.order.cpu().numpy()]+self.neutral.cpu().numpy()

    def inputs(self,x,base):
        error=x[...,self.q_feature_columns]-base.to(x.dtype)
        z=torch.cat((x,error),dim=-1)
        return ((z-self.center)/self.scale).clamp(-10,10)

    def forward(self,x,base):
        raw=self.reflex(self.inputs(x,base))
        gate=(x[...,-6:].abs().amax(-1,keepdim=True)/.05).clamp(0,1)
        neural_lambda=(raw*self.lambda_scale*gate).clamp(0,80)
        target=base+neural_lambda.to(base.dtype)@self.synergy.T
        return torch.maximum(torch.minimum(target,self.maximum),self.minimum),raw,gate

    @torch.inference_mode()
    def act(self,x,phase):
        base=self.rhythmic(phase[None])[0]
        target,raw,gate=self(torch.from_numpy(x[None]),torch.from_numpy(base[None]))
        p=phase[:6]
        pf=np.stack((np.sin(p),np.cos(p),np.ones(6)),axis=-1)
        logits=np.sum(pf*self.pad_weights.numpy(),axis=-1)
        return target.numpy()[0],logits>=0,raw.numpy()[0]


def read(path):
    value=torch.load(path,map_location='cpu',weights_only=True)
    model=Policy();model.load_state_dict(value['state']);model.eval()
    return model


def calibrate_pad(b):
    weights=[];calls=0
    for leg in b.legs:
        def label(t):
            nonlocal calls
            calls+=1;return b.motor._get_adhesion_onoff(leg,t%(2*np.pi))
        grid=np.linspace(0,2*np.pi,4097);labels=[label(t) for t in grid]
        edges=[]
        for i in range(4096):
            if labels[i]!=labels[i+1]:
                lo,hi=grid[i:i+2];old=labels[i]
                for _ in range(45):
                    mid=(lo+hi)/2
                    if label(mid)==old:lo=mid
                    else:hi=mid
                edges.append((lo+hi)/2)
        assert len(edges)==2,edges
        mid=sum(edges)/2;half=(edges[1]-edges[0])/2
        w=np.array([-np.sin(mid),-np.cos(mid),np.cos(half)])
        features=np.stack((np.sin(grid),np.cos(grid),np.ones(len(grid))),axis=-1)
        # Boundary floating-point ties are checked separately from uniform samples.
        assert np.array_equal(features@w>=0,np.array(labels))
        weights.append(w)
    return np.asarray(weights),calls
