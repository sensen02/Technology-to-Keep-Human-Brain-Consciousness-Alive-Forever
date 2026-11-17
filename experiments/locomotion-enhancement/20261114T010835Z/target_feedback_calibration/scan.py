"""Parent-only conditional target-register scan; ONLYCODE must not import here.
Physical observations and old256 history are exogenous saved inputs. This is NOT
new physical dynamics or a deployed action generator outside the learned ANN.
"""
import math
from typing import List, Optional, Tuple
import torch
from torch import nn

ALPHA=-math.expm1(-.005/.1)
BOUND=6.


class TargetFeedbackScan(nn.Module):
    """Share exactly the existing encoder/GRU/head8 tensors, never oldmodel.

    Result order: mean, residual, h96, pretarget, posttarget, action.
    Target integration holds each learned action over FIVE sequential1ms updates.
    forced_pre_targets is a value/parity probe, not the training feedback path.
    """
    def __init__(self,m):
        super().__init__()
        self.encoder=m.encoder
        self.gru=m.gru
        self.head=m.head
        self.alpha=ALPHA
        self.bound=BOUND

    def forward(self,physical:torch.Tensor,oldafter:torch.Tensor,
                commands:torch.Tensor,nontarget:torch.Tensor,gain:torch.Tensor,
                initial_target:torch.Tensor,q0:torch.Tensor,halfspan:torch.Tensor,
                bounds:torch.Tensor,max_rate:torch.Tensor,
                forced_pre_targets:Optional[torch.Tensor]=None
                )->Tuple[torch.Tensor,torch.Tensor,torch.Tensor,
                         torch.Tensor,torch.Tensor,torch.Tensor]:
        T=physical.size(0)
        B=physical.size(1)
        if T<1 or physical.dim()!=3 or physical.size(2)!=112:
            raise ValueError('Nonempty[T,B,112] saved physical features required')
        if oldafter.size(0)!=T or oldafter.size(1)!=B or oldafter.size(2)!=256:
            raise ValueError('Actual saved/cached afterold[T,B,256] required')
        if commands.size(0)!=T or commands.size(1)!=B or commands.size(2)!=2:
            raise ValueError('Independent commands[T,B,2] required')
        if nontarget.size(0)!=T or nontarget.size(1)!=B or nontarget.size(2)!=42:
            raise ValueError('Frozen old non-target mean[T,B,42] required')
        if gain.dim()!=1 or gain.size(0)!=42 or q0.dim()!=1 or q0.size(0)!=42:
            raise ValueError('Source frozen gain42 and source q0_42 required')
        if halfspan.dim()!=1 or halfspan.size(0)!=42 or bool((halfspan<.1).any()):
            raise ValueError('Actual finite source halfspan42>=.1 required')
        if bounds.dim()!=2 or bounds.size(0)!=42 or bounds.size(1)!=2:
            raise ValueError('Actual actuator bounds[42,2] required')
        if max_rate.dim()!=1 or max_rate.size(0)!=42 or bool((max_rate<0.).any()):
            raise ValueError('Actual actuator max_rate42 required')
        if initial_target.dim()!=2 or initial_target.size(0)!=B or initial_target.size(1)!=42:
            raise ValueError('Actual handover target[B,42] required')
        if forced_pre_targets is not None:
            if forced_pre_targets.size(0)!=T or forced_pre_targets.size(1)!=B or forced_pre_targets.size(2)!=42:
                raise ValueError('Forced parity targets must be[T,B,42]')
        h=physical.new_zeros((B,96))
        target=initial_target
        lower=bounds[:,0]
        upper=bounds[:,1]
        means=torch.jit.annotate(List[torch.Tensor],[])
        residuals=torch.jit.annotate(List[torch.Tensor],[])
        states=torch.jit.annotate(List[torch.Tensor],[])
        pretargets=torch.jit.annotate(List[torch.Tensor],[])
        posttargets=torch.jit.annotate(List[torch.Tensor],[])
        actions=torch.jit.annotate(List[torch.Tensor],[])
        for k in range(T):
            if forced_pre_targets is not None:
                target=forced_pre_targets[k]
            pretargets.append(target)
            # Match actual proprioception: clip in native normalized values,
            # cast tofloat32, then deployment obs float64. Do not detach: frozen
            # oldgain does NOT imply cutting target->oldprior->newANN gradient.
            offset=torch.clamp((target-q0)/halfspan,-10.,10.).to(torch.float32).to(torch.float64)
            oldmean=nontarget[k]-gain*offset
            feature=torch.cat((physical[k],oldafter[k],commands[k],oldmean),dim=-1)
            encoded=torch.tanh(self.encoder(feature))
            candidate=self.gru(encoded,h)
            h=(1.-self.alpha)*h+self.alpha*candidate
            residual=self.bound*torch.tanh(self.head(torch.cat((h,encoded),dim=-1)))
            mean=oldmean+residual
            action=torch.tanh(mean)
            # Core.tick clips action*MAX_RATE first, then actuator.update clips
            # to actual per-joint rate. Repeated additions preserve source order.
            rate=torch.clamp(action*2.,-2.,2.)
            limited=torch.minimum(torch.maximum(rate,-max_rate),max_rate)
            for _ in range(5):
                target=torch.minimum(torch.maximum(target+limited*.001,lower),upper)
            means.append(mean)
            residuals.append(residual)
            states.append(h)
            posttargets.append(target)
            actions.append(action)
        return (torch.stack(means),torch.stack(residuals),torch.stack(states),
                torch.stack(pretargets),torch.stack(posttargets),torch.stack(actions))


def make_scan(m,script=True):
    """Build a shared8-parameter scan; script=False is the independent Python path."""
    if tuple(m.encoder.weight.shape)!=(96,412) or tuple(m.head.weight.shape)!=(42,192):
        raise ValueError('Exact parent412encoder/head192 architecture required')
    scan=TargetFeedbackScan(m)
    expected={name:p for name,p in m.named_parameters() if not name.startswith('old.')}
    actual=dict(scan.named_parameters())
    if len(expected)!=8 or expected.keys()!=actual.keys() or any(expected[k] is not actual[k] for k in expected):
        raise ValueError('Conditional scan must share exactly all8 parent ANN tensors')
    return torch.jit.script(scan) if script else scan
