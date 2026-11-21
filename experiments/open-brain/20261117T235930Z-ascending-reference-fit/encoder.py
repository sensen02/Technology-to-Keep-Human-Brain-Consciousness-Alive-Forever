"""Per-source causal engineering encoder, no action, clock or cross-source mixing."""
from shared import *


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mix_logits=torch.nn.Parameter(torch.zeros(2,3,dtype=torch.float64))
        self.log_gain=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64))

    def forward(self,x,cut=None):
        if x.shape[-2:]!=(2,3) or not torch.isfinite(x).all() or (x<0).any():
            raise ValueError('expected actual causal post-neck rates in bound source order')
        level=(x*self.mix_logits.softmax(-1)).sum(-1)
        gain=self.log_gain.clamp(-2,2).exp()
        y=CAP_HZ*(-torch.expm1(-gain*level/CAP_HZ))
        if cut is not None:
            if cut.dtype!=torch.bool or cut.shape!=(2,):raise ValueError('cut source order mismatch')
            y=y.masked_fill(cut,0.)
        return y


def load(path):
    c=torch.load(path,weights_only=True,map_location='cpu')
    m=Encoder();m.load_state_dict(c['state']);m.eval()
    return m,c
