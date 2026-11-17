"""Parent-only Torch backend. Brain3 input architecture, no runtime sensory fusion formula."""
from pathlib import Path
import importlib.util,sys
from typing import List,Tuple
P=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/neural_calibration/model.py');s=importlib.util.spec_from_file_location('_fusion_original_bridge',P);old=importlib.util.module_from_spec(s);sys.modules[s.name]=old;s.loader.exec_module(old)
torch=old.torch;nn=old.nn
class NeuralBridge(old.NeuralBridge):
 def __init__(self,mean,std):
  super().__init__(mean,std);self.brain=nn.GRUCell(3,16,dtype=torch.float64)
 def brain_step(self,delivered_an,modalities,h):
  if modalities.ndim!=2 or modalities.shape[1]!=2:raise ValueError('Actualneural visual1/odor1 only')
  after=self.brain(torch.cat((delivered_an,modalities),-1),h);return self.brain_out(after).tanh(),after
 def identity(self):
  d=super().identity();d.update(format='SOURCE_NECK_BRAIN_SIDE_VISUAL_OLFACTORY_BRIDGE_V1',brain_input=['deliveredAN1','frozen_actual_RGB_visual_rawturn1','frozen_receptor_odor_rawturn1'],GRU_input3_column_mapping='parentAN/vision columns copied, odor column initializedzero',visual_hidden=32,odor_hidden=32,total_recurrent_state_with_motor=336,visual_feature_cache16_separate=True,odor_filter2_pending2_separate=True,sensory_networks_frozen=True,vision_odor_warm_hidden_held_zero=True,visual_sha256='996980806ac107137d6d5f4743de207fb3350e84a8a145dc5f52b8807398bb87',odor_sha256='66f423df4f00d801b2f0a194f539baa63ce84973765694e3d3111fbeb62aa034',runtime_fusion_rule=False,food_arrival_qualified=False);return d
class Scan(nn.Module):
 def __init__(self,m):
  super().__init__();self.feedback=m.feedback;self.brain=m.brain;self.brain_out=m.brain_out;self.decoder=m.decoder;self.reconstruction=m.reconstruction;self.register_buffer('observation_mean',m.observation_mean);self.register_buffer('observation_std',m.observation_std)
 def forward(self,body111:torch.Tensor,modalities:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
  an=self.feedback((body111-self.observation_mean)/self.observation_std);recon=self.reconstruction(an);h=body111.new_zeros((body111.shape[1],16));held_an=body111.new_zeros((body111.shape[1],1));held_dn=body111.new_zeros((body111.shape[1],2));turns=torch.jit.annotate(List[torch.Tensor],[]);dns=torch.jit.annotate(List[torch.Tensor],[]);hs=torch.jit.annotate(List[torch.Tensor],[])
  for k in range(1920):
   h=self.brain(torch.cat((held_an,modalities[k]),-1),h);dn=self.brain_out(h).tanh();turns.append(self.decoder(held_dn));dns.append(dn);hs.append(h);held_an=an[k];held_dn=dn
  return torch.stack(turns),torch.stack(dns),an,recon,torch.stack(hs)
def make_scan(m):
 scan=torch.jit.script(Scan(m));a=dict(m.named_parameters());b=dict(scan.named_parameters())
 if a.keys()!=b.keys() or any(a[k] is not b[k] for k in a):raise ValueError('Shared exact modelparameters')
 return scan
def python_rollout(m,x,u):
 h=x.new_zeros((x.shape[1],16));ad=x.new_zeros((x.shape[1],1));dd=x.new_zeros((x.shape[1],2));rows=[[],[],[],[],[]]
 for k in range(1920):
  an=m.feedback(m.normalize(x[k]));dn,h=m.brain_step(ad,u[k],h);turn=m.decoder(dd);rec=m.reconstruction(an)
  for row,v in zip(rows,(turn,dn,an,rec,h)):row.append(v)
  ad=an;dd=dn
 return tuple(torch.stack(v) for v in rows)
def objective(out,target,weights,m,ref):
 turn,dn,an,rec,h=out;action=((turn-target).square()*weights).sum(0).mean();silent=m.decoder(dn.new_zeros((dn.shape[1],2))).square().mean();activity=dn.square().mean();params=dict(m.named_parameters());anchor=sum((params[k]-v).square().sum() for k,v in ref.items())/sum(v.numel() for v in ref.values())
 return action+.1*silent+.001*activity+.001*anchor,dict(turn_weighted_MSE=action,silent_DN_decoder_MSE=silent,DN_activity_MSE=activity,parent_trainable_anchor_MSE=anchor)
def accumulated_backward(scan,m,x,u,y,w,ref):
 if x.shape[1]!=27:raise ValueError('Exactlyeffective27')
 total=0.;stats={}
 for start in (0,9,18):
  output=scan(x[:,start:start+9],u[:,start:start+9]);loss,terms=objective(output,y[:,start:start+9],w[:,start:start+9],m,ref);(loss/3.).backward();total+=float(loss.detach())/3.
  for k,v in terms.items():stats[k]=stats.get(k,0.)+float(v.detach())/3.
  del output,loss,terms
 return total,stats
