"""RIGHT_ONLY anatomical LH-only parameter split; load mask never enters inference."""
from pathlib import Path
import importlib.util,sys,ast,xml.etree.ElementTree as ET
import torch,numpy as np
from torch import nn
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
PARENT_DIR=HERE.parent/'head_refit';PARENT=PARENT_DIR/'TRAIN/checkpoint_004000.pt'
PARENT_SHA='6407d6dbb85e1a7f40bf915e959433aa8c0896752924f06fab302e407b8f59b5'
s=importlib.util.spec_from_file_location('_LH_refit_parent_head_io',PARENT_DIR/'model_io.py');old=importlib.util.module_from_spec(s);sys.modules[s.name]=old;s.loader.exec_module(old)
sha256=old.sha256;exact=old.exact;verify_hashes=old.verify_hashes;BASE=old.BASE;REFERENCE=old.REFERENCE;DATA=old.DATA
TRACE=HERE.parent.parent/'attempt02/right_yaw_m060/TRACE_1MS.npz'
XML=BASE.parent.parent/'banc-triad-pad/20261110T010000Z/MODEL_TRIAD_x200.xml'
MAPPING_SOURCE=BASE/'sources/repair_core.py';ROWS=list(range(7,14));W=24.446354755878005
class LHHead(nn.Module):
 def __init__(self,weight,bias):
  super().__init__();self.register_buffer('frozen_weight',weight.detach().clone());self.register_buffer('frozen_bias',bias.detach().clone());self.lh_weight=nn.Parameter(weight[7:14].detach().clone());self.lh_bias=nn.Parameter(bias[7:14].detach().clone())
 @property
 def weight(self):
  w=self.frozen_weight.clone();w[7:14]=self.lh_weight;return w
 @property
 def bias(self):
  b=self.frozen_bias.clone();b[7:14]=self.lh_bias;return b
 def forward(self,x):return F.linear(x,self.weight,self.bias)
class LHRefit(old.HeadRefit):
 VERSION='RIGHT_ONLY_yaw006_bound3_anatomical_LH_stance_refit_v1'
 def __init__(self,baseline):
  super().__init__(baseline);self.head=LHHead(self.head.weight,self.head.bias)
 def adapter_parameters(self):return [self.head.lh_weight,self.head.lh_bias]
 def identity(self):
  i=super().identity();i.update(version=self.VERSION,kind='RIGHT_ONLY_LH_stance_refit',parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,trainable='LH rows7:14 only455',output_rows=ROWS,other_head_rows_frozen=True,head_parameterization='frozen42matrix clone with LH7 learnable slice, one unchanged F.linear42',optimizer='freshAdam_LH_only_2states');return i

def initialize():
 parent,p=old.load_checked(PARENT,PARENT_SHA);m=LHRefit(parent.base);m.encoder.load_state_dict(parent.encoder.state_dict(),strict=True);m.gru.load_state_dict(parent.gru.state_dict(),strict=True);m.head=LHHead(parent.head.weight,parent.head.bias)
 if sum(v.numel() for v in m.parameters() if v.requires_grad)!=455 or not torch.equal(m.head.weight,parent.head.weight) or not torch.equal(m.head.bias,parent.head.bias):raise ValueError('exact455 LH initial model required')
 return m

def mapping():
 tree=ast.parse(MAPPING_SOURCE.read_text());legs=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LEG_ORDER' for t in n.targets));weight=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='WEIGHT' for t in n.targets))
 names=[n.attrib['joint'] for n in ET.parse(XML).getroot().find('actuator')];rows=[i for i,n in enumerate(names) if n.startswith('joint_LH')]
 if legs!=['LF','LH','LM','RF','RH','RM'] or legs.index('LH')!=1 or rows!=ROWS or len(names)!=42 or weight!=W:raise ValueError('frozen physical leg/output mapping mismatch')
 return dict(legs=legs,LH_contact_index=legs.index('LH'),LH_output_rows=rows,actuator_joint_names=names,body_weight_native=weight,source_sha256={str(p):sha256(p) for p in (XML,MAPPING_SOURCE)})
def prepare(m):
 x,u,y=old.old.dataset();pc=old.old.cached(m,x,u)
 with torch.no_grad():means,hist=m.sequence(x,u)
 with np.load(DATA,allow_pickle=False) as z:times=z['actual_time'].copy()
 with np.load(TRACE,allow_pickle=False) as z:clock=z['time'].copy();load=z['loads'].copy()
 mp=mapping();indices=999+5*np.arange(2880);error=float(np.max(np.abs(clock[indices]-times)))
 if indices[960]!=5799 or error>=1e-8 or np.max(np.abs(times-(1+np.arange(2880)*.005)))>=1e-8:raise ValueError('actual pre-action load timestamp mismatch')
 stance=load[indices,mp['LH_contact_index']]>=.1*mp['body_weight_native'];weights=torch.from_numpy(np.where(stance,4.,1.))[:,None,None]
 return dict(x=x,u=u,y=y,hidden=hist[1].detach(),base=pc['means'],gate=pc['gate'],means=means,base_states=hist[0],weights=weights,stance=stance,preaction_indices=indices,time_error=error,mapping=mp)
def cached_forward(m,c):return c['base']+c['gate']*m.RESIDUAL_BOUND*torch.tanh(m.head(c['hidden']))
def loss(pred,c):
 e=(pred[960:,:,7:14].tanh()-c['y'][960:,:,7:14]).square();w=c['weights'][960:];return (e*w).sum()/(7*w.sum())
neutral=old.neutral
def frozen(m):return {k:v.detach().clone() for k,v in m.state_dict().items() if k not in ('head.lh_weight','head.lh_bias')}
def load_checked(path,digest,require_final=True):
 if sha256(path)!=digest:raise ValueError('LH checkpoint digest')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p.get('format_version')!='RIGHT_LH_STANCE_REFIT_V1' or p['config']['physical_eval_only_step']!=4000:raise ValueError('strict separate RIGHT LH format')
 verify_hashes(p['config']['source_hashes']);verify_hashes(p['config']['data_hashes']);m=LHRefit(old.old.legacy.load_baseline_actor())
 if p['identity']!=m.identity():raise ValueError('LH identity mismatch')
 m.load_state_dict(p['state_dict'],strict=True);reference=initialize()
 if not exact(frozen(m),frozen(reference)):raise ValueError('frozen base/encoder/GRU/nonLH rows changed')
 if not all(torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('nonfinite LH checkpoint')
 step=p['curve'][-1]['step']
 if step not in (2000,4000) or (require_final and step!=4000):raise ValueError('only fixed final4000 for physics')
 opt=torch.optim.Adam(m.adapter_parameters(),lr=.001);opt.load_state_dict(p['optimizer'])
 if len(opt.state)!=2 or any(float(v['step'])!=step for v in opt.state.values()) or sum(v.numel() for v in m.adapter_parameters())!=455:raise ValueError('LH455 Adam2 counters required')
 for param in m.adapter_parameters():
  if opt.state[param]['exp_avg'].shape!=param.shape or opt.state[param]['exp_avg_sq'].shape!=param.shape:raise ValueError('LH-only Adam shape')
 return m.eval(),p
