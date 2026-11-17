"""Parent-only exact-source frozen-component ordering; no numerical child import.
Only the frozen oldmean reconstruction changes. ANN/integration/loss are unchanged.
"""
import math
from typing import List, Optional, Tuple
import torch
from torch import nn
ALPHA=-math.expm1(-.005/.1)
BOUND=6.

class TargetFeedbackScan(nn.Module):
    def __init__(self,m):
        super().__init__();self.encoder=m.encoder;self.gru=m.gru;self.head=m.head;self.alpha=ALPHA;self.bound=BOUND
    def forward(self,physical:torch.Tensor,oldafter:torch.Tensor,
                commands:torch.Tensor,basehead:torch.Tensor,right:torch.Tensor,left:torch.Tensor,
                gain:torch.Tensor,initial_target:torch.Tensor,q0:torch.Tensor,halfspan:torch.Tensor,
                bounds:torch.Tensor,max_rate:torch.Tensor,
                forced_pre_targets:Optional[torch.Tensor]=None
                )->Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        T=physical.size(0);B=physical.size(1)
        if T<1 or physical.dim()!=3 or physical.size(2)!=112:raise ValueError('Actualphysical[T,B,112]')
        if oldafter.size(0)!=T or oldafter.size(1)!=B or oldafter.size(2)!=256:raise ValueError('Actualoldafter[T,B,256]')
        if commands.size(0)!=T or commands.size(1)!=B or commands.size(2)!=2:raise ValueError('Actualtask[T,B,2]')
        for component in (basehead,right,left):
            if component.size(0)!=T or component.size(1)!=B or component.size(2)!=42:raise ValueError('ExactB1cached frozencomponent[T,B,42]')
        if gain.dim()!=1 or gain.size(0)!=42 or q0.dim()!=1 or q0.size(0)!=42:raise ValueError('Gain/q0 shape42')
        if halfspan.dim()!=1 or halfspan.size(0)!=42 or bool((halfspan<.1).any()):raise ValueError('Actualhalfspan42')
        if bounds.dim()!=2 or bounds.size(0)!=42 or bounds.size(1)!=2:raise ValueError('Actualbounds42x2')
        if max_rate.dim()!=1 or max_rate.size(0)!=42 or bool((max_rate<0.).any()):raise ValueError('Actualmaxrate42')
        if initial_target.dim()!=2 or initial_target.size(0)!=B or initial_target.size(1)!=42:raise ValueError('ActualinitialtargetBx42')
        if forced_pre_targets is not None:
            if forced_pre_targets.size(0)!=T or forced_pre_targets.size(1)!=B or forced_pre_targets.size(2)!=42:raise ValueError('Forcedpretargets[T,B,42]')
        h=physical.new_zeros((B,96));target=initial_target;lower=bounds[:,0];upper=bounds[:,1]
        means=torch.jit.annotate(List[torch.Tensor],[]);residuals=torch.jit.annotate(List[torch.Tensor],[]);states=torch.jit.annotate(List[torch.Tensor],[])
        pretargets=torch.jit.annotate(List[torch.Tensor],[]);posttargets=torch.jit.annotate(List[torch.Tensor],[]);actions=torch.jit.annotate(List[torch.Tensor],[])
        for k in range(T):
            if forced_pre_targets is not None:target=forced_pre_targets[k]
            pretargets.append(target)
            offset=torch.clamp((target-q0)/halfspan,-10.,10.).to(torch.float32).to(torch.float64)
            oldmean=basehead[k]-gain*offset+right[k]+left[k]
            feature=torch.cat((physical[k],oldafter[k],commands[k],oldmean),dim=-1)
            # Always preserve productionB1 linear/GRU/head kernels persequence;
            # loss aggregation remains fullB12/micro4, shared8gradients unchanged.
            nextstates=torch.jit.annotate(List[torch.Tensor],[])
            corrections=torch.jit.annotate(List[torch.Tensor],[])
            for b in range(B):
                encoded=torch.tanh(self.encoder(feature[b:b+1]));candidate=self.gru(encoded,h[b:b+1]);oneh=(1.-self.alpha)*h[b:b+1]+self.alpha*candidate
                corrections.append(self.bound*torch.tanh(self.head(torch.cat((oneh,encoded),dim=-1))));nextstates.append(oneh)
            h=torch.cat(nextstates,dim=0);residual=torch.cat(corrections,dim=0);mean=oldmean+residual;action=torch.tanh(mean)
            rate=torch.clamp(action*2.,-2.,2.);limited=torch.minimum(torch.maximum(rate,-max_rate),max_rate)
            for _ in range(5):target=torch.minimum(torch.maximum(target+limited*.001,lower),upper)
            means.append(mean);residuals.append(residual);states.append(h);posttargets.append(target);actions.append(action)
        return torch.stack(means),torch.stack(residuals),torch.stack(states),torch.stack(pretargets),torch.stack(posttargets),torch.stack(actions)

def make_scan(m,script=True):
    if tuple(m.encoder.weight.shape)!=(96,412) or tuple(m.head.weight.shape)!=(42,192):raise ValueError('Unchanged412/192architecture')
    scan=TargetFeedbackScan(m);expected={n:p for n,p in m.named_parameters() if not n.startswith('old.')};actual=dict(scan.named_parameters())
    if len(expected)!=8 or expected.keys()!=actual.keys() or any(expected[n] is not actual[n] for n in expected):raise ValueError('Shareexact8trainabletensors')
    return torch.jit.script(scan) if script else scan
