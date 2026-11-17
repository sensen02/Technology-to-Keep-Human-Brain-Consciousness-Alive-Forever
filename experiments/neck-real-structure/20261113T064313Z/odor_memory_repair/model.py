"""Parent numerical backend. Source66f architecture/ALPHA unchanged; offline loss only."""
from common import module,SOURCE
from typing import List,Tuple
old=module('_odor_memory_original_class_source',SOURCE);SteeringBehavior=old.SteeringBehavior;torch=old.torch;nn=old.nn
class Scan(nn.Module):
 def __init__(self,m):
  super().__init__();self.encoder=m.encoder;self.gru=m.gru;self.head=m.head;self.alpha=float(m.ALPHA)
 def forward(self,z:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
  h=z.new_zeros((z.shape[1],32));turns=torch.jit.annotate(List[torch.Tensor],[]);latents=torch.jit.annotate(List[torch.Tensor],[]);states=torch.jit.annotate(List[torch.Tensor],[])
  for k in range(1920):
   candidate=self.gru(torch.tanh(self.encoder(z[k])),h);h=(1.-self.alpha)*h+self.alpha*candidate;latent=self.head(h);turns.append(latent.tanh());latents.append(latent);states.append(h)
  return torch.stack(turns),torch.stack(latents),torch.stack(states)
def make_scan(m):
 scan=torch.jit.script(Scan(m));a=dict(m.named_parameters());b=dict(scan.named_parameters())
 if a.keys()!=b.keys() or any(a[k] is not b[k] for k in a):raise ValueError('Shared8parameters required')
 return scan
def production(m,z):
 h=m.initial_state(z.shape[1]);rows=[[],[],[]]
 for k in range(1920):
  latent,h=m.mean_step(z[k],h)
  for row,v in zip(rows,(latent.tanh(),latent,h)):row.append(v)
 return tuple(torch.stack(v) for v in rows)
def objective(out,target,weights,m,reference):
 pred,latent,h=out;action=((pred-target).square()*weights).sum(0).mean();anchor=sum((p-q).square().sum() for p,q in zip(m.parameters(),reference))/sum(q.numel() for q in reference);return action+.001*anchor,dict(weighted_turn_MSE=action,parent_anchor_MSE=anchor)
def accumulated_backward(scan,m,z,y,w,reference):
 if z.shape!=(1920,16,2):raise ValueError('Exacteffective16fullhistory')
 total=0.;terms={}
 for start in (0,8):
  out=scan(z[:,start:start+8]);loss,stats=objective(out,y[:,start:start+8],w[:,start:start+8],m,reference);(loss/2.).backward();total+=float(loss.detach())/2.
  for k,v in stats.items():terms[k]=terms.get(k,0.)+float(v.detach())/2.
  del out,loss,stats
 return total,terms
