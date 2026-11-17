"""Unchanged conditionalB1scan plus actualsourcefeatureB1scan; same shared8 tensors."""
from pathlib import Path
import importlib.util,sys
from typing import List,Tuple
# Keep the conditionalclass originalmodule/__file__ for TorchScript inspection.
_PARENT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_order_repair_v4/scan.py')
_spec=importlib.util.spec_from_file_location('_statealigned_exact_conditional_scan',_PARENT)
_conditional=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_conditional;_spec.loader.exec_module(_conditional)
torch=_conditional.torch;nn=_conditional.nn;ALPHA=_conditional.ALPHA;BOUND=_conditional.BOUND
TargetFeedbackScan=_conditional.TargetFeedbackScan
make_scan=_conditional.make_scan


class ActualSourceScan(nn.Module):
    """Actualsource features412 -> sameANN ops with sequential B1 valuepath."""
    def __init__(self,m):
        super().__init__();self.encoder=m.encoder;self.gru=m.gru;self.head=m.head;self.alpha=ALPHA;self.bound=BOUND
    def forward(self,features:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
        if features.dim()!=3 or features.size(2)!=412 or features.size(0)<1 or features.size(1)<1:raise ValueError('Actualsourcefeatures[T,B,412]')
        T=features.size(0);B=features.size(1);h=features.new_zeros((B,96))
        residuals=torch.jit.annotate(List[torch.Tensor],[]);states=torch.jit.annotate(List[torch.Tensor],[])
        for k in range(T):
            nextstates=torch.jit.annotate(List[torch.Tensor],[]);corrections=torch.jit.annotate(List[torch.Tensor],[])
            for b in range(B):
                encoded=torch.tanh(self.encoder(features[k,b:b+1]));candidate=self.gru(encoded,h[b:b+1]);oneh=(1.-self.alpha)*h[b:b+1]+self.alpha*candidate
                corrections.append(self.bound*torch.tanh(self.head(torch.cat((oneh,encoded),dim=-1))));nextstates.append(oneh)
            h=torch.cat(nextstates,dim=0);residual=torch.cat(corrections,dim=0);residuals.append(residual);states.append(h)
        return torch.stack(residuals),torch.stack(states)


def make_actual_scan(m,script=True):
    if tuple(m.encoder.weight.shape)!=(96,412) or tuple(m.head.weight.shape)!=(42,192):raise ValueError('Unchanged412/192architecture')
    scan=ActualSourceScan(m);expected={n:p for n,p in m.named_parameters() if not n.startswith('old.')};actual=dict(scan.named_parameters())
    if len(expected)!=8 or expected.keys()!=actual.keys() or any(expected[n] is not actual[n] for n in expected):raise ValueError('Actualscan must sharecanonical8trainabletensors')
    return torch.jit.script(scan) if script else scan
