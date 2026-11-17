"""Diagnostic copy of original scan core, with exact old three-component ordering.
All integration/ANN expressions below are copied from target_feedback_calibration/scan.py.
Frozen sources: models.py:142 and combined_turning/model.py:44.
This file is a probe only; no deployment or archived source is edited.
"""
from typing import List, Optional, Tuple
import torch
from torch import nn
from scan import TargetFeedbackScan

class ExactComponentsScan(TargetFeedbackScan):
    def forward(self,physical:torch.Tensor,oldafter:torch.Tensor,
                commands:torch.Tensor,basehead:torch.Tensor,right:torch.Tensor,left:torch.Tensor,
                gain:torch.Tensor,initial_target:torch.Tensor,q0:torch.Tensor,halfspan:torch.Tensor,
                bounds:torch.Tensor,max_rate:torch.Tensor,
                forced_pre_targets:Optional[torch.Tensor]=None
                )->Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        T=physical.size(0)
        B=physical.size(1)
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
            offset=torch.clamp((target-q0)/halfspan,-10.,10.).to(torch.float32).to(torch.float64)
            oldmean=basehead[k]-gain*offset+right[k]+left[k]
            feature=torch.cat((physical[k],oldafter[k],commands[k],oldmean),dim=-1)
            encoded=torch.tanh(self.encoder(feature))
            candidate=self.gru(encoded,h)
            h=(1.-self.alpha)*h+self.alpha*candidate
            residual=self.bound*torch.tanh(self.head(torch.cat((h,encoded),dim=-1)))
            mean=oldmean+residual
            action=torch.tanh(mean)
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
