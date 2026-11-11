"""Separate strict head-only format; frozen body encoder GRU, original production step."""
from pathlib import Path
import importlib.util,sys
import torch,numpy as np
HERE=Path(__file__).resolve().parent
PARENT_DIR=HERE.parent
PARENT=PARENT_DIR/'TRAIN/checkpoint_002000.pt'
PARENT_SHA='f25b920b1d9bc74a2ff3092dd926908d728b1b0ee49d211d4295e59f630aee15'
s=importlib.util.spec_from_file_location('_head_refit_original_right_model',PARENT_DIR/'model.py');old=importlib.util.module_from_spec(s);sys.modules[s.name]=old;s.loader.exec_module(old)
sha256=old.sha256;exact=old.exact;verify_hashes=old.verify_hashes;BASE=old.BASE;REFERENCE=old.REFERENCE;DATA=old.DATA
class HeadRefit(old.RightMotor):
 VERSION='RIGHT_ONLY_yaw006_bound3_uniform_head_refit_v1'
 def __init__(self,baseline):
  super().__init__(baseline);self.encoder.requires_grad_(False);self.gru.requires_grad_(False)
 def adapter_parameters(self):return list(self.head.parameters())
 def identity(self):
  i=super().identity();i.update(version=self.VERSION,kind='RIGHT_ONLY_uniform_head_refit',parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,encoder_frozen=True,gru_frozen=True,trainable='head42x64+42=2730',optimizer='freshAdam_head_only_2states');return i

def initialize():
 parent,p=old.load_checked(PARENT,PARENT_SHA);m=HeadRefit(old.legacy.load_baseline_actor());m.load_state_dict(parent.state_dict(),strict=True)
 if not exact(m.state_dict(),parent.state_dict()) or sum(v.numel() for v in m.parameters() if v.requires_grad)!=2730:raise ValueError('parent/frozen count mismatch')
 return m

def prepare(m):
 x,u,y=old.dataset();pc=old.cached(m,x,u)
 with torch.no_grad():means,hist=m.sequence(x,u)
 hidden=hist[1].detach();base=pc['means'];gate=pc['gate']
 return dict(x=x,u=u,y=y,hidden=hidden,base=base,gate=gate,means=means,base_states=hist[0])
def cached_forward(m,c):return c['base']+c['gate']*m.RESIDUAL_BOUND*torch.tanh(m.head(c['hidden']))
def neutral(m):
 with np.load(REFERENCE,allow_pickle=False) as z:x=torch.from_numpy(z['X238'].copy())[:,None];ref=torch.from_numpy(z['mean'].copy())[:,None];hb=torch.from_numpy(z['hidden_after'].copy())[:,None]
 with torch.no_grad():means,h=m.sequence(x,torch.zeros((2880,1,2),dtype=torch.float64))
 return dict(means_exact=torch.equal(means,ref),base_hidden_exact=torch.equal(h[0],hb))
def frozen(m):return {k:v.detach().clone() for k,v in m.state_dict().items() if not k.startswith('head.')}
def load_checked(path,digest,require_final=True):
 if sha256(path)!=digest:raise ValueError('head-only digest mismatch')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p.get('format_version')!='RIGHT_HEAD_REFIT_V1' or p['config']['physical_eval_only_step']!=4000:raise ValueError('strict separate head-refit format required')
 verify_hashes(p['config']['source_hashes']);verify_hashes(p['config']['data_hashes'])
 m=HeadRefit(old.legacy.load_baseline_actor())
 if p['identity']!=m.identity():raise ValueError('head identity mismatch')
 m.load_state_dict(p['state_dict'],strict=True)
 if not all(torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('nonfinite model')
 parent,_=old.load_checked(PARENT,PARENT_SHA)
 if not exact(frozen(m),frozen(parent)):raise ValueError('base/encoder/GRU differ from exact parent')
 step=p['curve'][-1]['step']
 if step not in (2000,4000) or (require_final and step!=4000):raise ValueError('only fixed final4000 for physics')
 opt=torch.optim.Adam(m.adapter_parameters(),lr=.001);opt.load_state_dict(p['optimizer'])
 if len(opt.state)!=2 or any(float(v['step'])!=step for v in opt.state.values()):raise ValueError('exactly two head-only Adam states/counters')
 return m.eval(),p
