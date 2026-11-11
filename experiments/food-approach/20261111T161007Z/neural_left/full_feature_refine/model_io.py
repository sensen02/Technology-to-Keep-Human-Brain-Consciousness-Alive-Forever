"""New strict full LEFT feature refinement, only original base remains frozen."""
from pathlib import Path
import importlib.util,sys,json
import torch,numpy as np
HERE=Path(__file__).resolve().parent
PARENT_DIR=HERE.parent
s=importlib.util.spec_from_file_location('_left_feature_parent_io',PARENT_DIR/'model_io.py');old=importlib.util.module_from_spec(s);sys.modules[s.name]=old;s.loader.exec_module(old)
PARENT=PARENT_DIR/'TRAIN/checkpoint_004000.pt'
PARENT_SHA='9c58e2a8772c7ef034f2fc0e18d01d5e4e59ace8d7cfb89b31e6f8569b6ae799'
FORMAT='LEFT_FEATURE_REFINE_V1'
LR=.0003
ROOT=old.ROOT;BASE=old.BASE;DATA=old.DATA;TEACHER=old.TEACHER;AUDIT=old.AUDIT;REFERENCE=old.REFERENCE;legacy=old.legacy
sha256=old.sha256;exact=old.exact;verify_hashes=old.verify_hashes;module=old.module;dataset=old.dataset;neutral=old.neutral
ENGINE=ROOT/'20261110T235830Z/train_two_only_control/engine.py'

class FeatureRefine(old.LeftHead):
 VERSION='LEFT_FEATURE_REFINE_V1_parenthead4000_fulladapter43242_basefrozen'
 def __init__(self,baseline):
  super().__init__(baseline);self.encoder.requires_grad_(True);self.gru.requires_grad_(True);self.head.requires_grad_(True);self.base.requires_grad_(False)
 def adapter_parameters(self):return [p for n,p in self.named_parameters() if not n.startswith('base.')]
 def identity(self):
  i=super().identity();i.update(version=self.VERSION,kind=FORMAT,parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,encoder_frozen=False,gru_frozen=False,trainable_parameters=43242,trainable_tensors=8,optimizer='fresh Adam ALL8, parent head optimizer discarded',initialization='all parent4000 weights exact',head_initialization='parent4000 learned head');return i

def frozen_base(m):return {k:v.detach().clone() for k,v in m.base.state_dict().items()}

def initialize():
 parent,p=old.load_checked(PARENT,PARENT_SHA)
 m=FeatureRefine(legacy.load_baseline_actor());m.load_state_dict(parent.state_dict(),strict=True)
 if not exact(m.state_dict(),parent.state_dict()):raise ValueError('initial all parent weights mismatch')
 if sum(v.numel() for v in m.parameters() if v.requires_grad)!=43242 or len(m.adapter_parameters())!=8:raise ValueError('exact original adapter43242 in8 tensors required')
 return m

def cached(m,x,u):
 bh=m.base.initial_state(x.shape[1]);means=[];states=[]
 with torch.no_grad():
  for row in x:
   mean,bh=m.base.step(row,bh);means.append(mean);states.append(bh)
 means=torch.stack(means);states=torch.stack(states)
 return dict(features=torch.cat((x[:,:,m.base.physical_mask.bool()],states,u),-1),gate=torch.relu(u[:,:,1:2]),means=means,valid=torch.ones((*x.shape[:2],1),dtype=torch.bool),base_states=states)

def make(m):return module('_left_feature_shared_scan',ENGINE).make(m)

def source_paths():return [HERE/n for n in ('PREREG.json','model_io.py','train.py','verify_saved.py')]+old.source_paths()+[ENGINE]
def data_paths():return list(dict.fromkeys(old.data_paths()+[PARENT,PARENT_DIR/'TRAIN/VERIFY_004000.json']))

def load_checked(path,digest,require_final=True):
 if sha256(path)!=digest:raise ValueError('refinement SHA mismatch')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p.get('format_version')!=FORMAT or p['config']['kind']!=FORMAT or p['config']['physical_eval_only_step']!=2000:raise ValueError('strict LEFT_FEATURE_REFINE_V1 required')
 verify_hashes(p['config']['source_hashes']);verify_hashes(p['config']['data_hashes'])
 m=initialize();expected=frozen_base(m)
 if p['identity']!=m.identity():raise ValueError('refinement architecture/domain identity mismatch')
 m.load_state_dict(p['state_dict'],strict=True)
 # Encoder/GRU/head are intentionally trainable. Only original baseline must remain exact.
 if not exact(expected,frozen_base(m)) or not all(torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('base drift or nonfinite model')
 step=p['curve'][-1]['step']
 if step not in (1000,2000) or len(p['curve'])!=step or (require_final and step!=2000):raise ValueError('only refinementfinal2000 physically eligible')
 if p['config']['lr']!=LR or p['config']['seed']!=20261111 or p['config']['parent_sha256']!=PARENT_SHA:raise ValueError('fixedrefinementconfig mismatch')
 opt=torch.optim.Adam(m.adapter_parameters(),lr=LR);opt.load_state_dict(p['optimizer'])
 if len(opt.state)!=8 or any(float(v['step'])!=step for v in opt.state.values()) or any(g['lr']!=LR for g in opt.param_groups):raise ValueError('fresh8Adam counters/lr mismatch')
 return m.eval(),p
