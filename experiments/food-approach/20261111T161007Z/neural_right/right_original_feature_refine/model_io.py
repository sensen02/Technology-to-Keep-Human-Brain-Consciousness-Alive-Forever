"""RIGHT qualified-only fulladapter43242 from ef768 densehead, base remains exact."""
from pathlib import Path
import importlib.util,sys,json
import torch,numpy as np
HERE=Path(__file__).resolve().parent;PARENT_DIR=HERE.parent/'left_hind_stance_refit';PARENT=PARENT_DIR/'TRAIN/checkpoint_004000.pt';PARENT_SHA='ef768c788c94aeecbb49f0bd1ccb5c424333af36ec514941b72e49ab0d4917ff'
s=importlib.util.spec_from_file_location('_right_original_refine_parent_LH_io',PARENT_DIR/'model_io.py');old=importlib.util.module_from_spec(s);sys.modules[s.name]=old;s.loader.exec_module(old)
right=old.old.old;FORMAT='RIGHT_ORIGINAL_FEATURE_REFINE_V1';LR=.0003
BASE=old.BASE;DATA=old.DATA;REFERENCE=old.REFERENCE;QUALIFIED=DATA.parent;ACTUAL_STREAM=PARENT_DIR/'EVAL_right/neural_decisions.npz';ENGINE=right.ENGINE
sha256=old.sha256;exact=old.exact;verify_hashes=old.verify_hashes
class FeatureRefine(right.RightMotor):
 VERSION='RIGHT_ORIGINAL_FEATURE_REFINE_V1_denseparentef_all8_basefrozen'
 def __init__(self,baseline):
  super().__init__(baseline);self.encoder.requires_grad_(True);self.gru.requires_grad_(True);self.head.requires_grad_(True);self.base.requires_grad_(False)
 def identity(self):
  i=super().identity();i.update(version=self.VERSION,kind=FORMAT,parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,encoder_frozen=False,gru_frozen=False,head_initialization='exact parentef768 assembleddense42matrix',trainable_parameters=43242,trainable_tensors=8,optimizer='fresh Adam ALL8 lr.0003; parent4000 optimizer not resumed',training_episode='ONLY originalqualifiedright');return i

def frozen_base(m):return {k:v.detach().clone() for k,v in m.base.state_dict().items()}
def initialize():
 parent,p=old.load_checked(PARENT,PARENT_SHA);m=FeatureRefine(parent.base);m.encoder.load_state_dict(parent.encoder.state_dict(),strict=True);m.gru.load_state_dict(parent.gru.state_dict(),strict=True)
 with torch.no_grad():m.head.weight.copy_(parent.head.weight);m.head.bias.copy_(parent.head.bias)
 if not exact(frozen_base(m),frozen_base(parent)) or not torch.equal(m.head.weight,parent.head.weight) or not torch.equal(m.head.bias,parent.head.bias) or len(m.adapter_parameters())!=8 or sum(q.numel() for q in m.adapter_parameters())!=43242:raise ValueError('exact denseef assembly/43242 ALL8 required')
 return m

def dataset():
 r=json.loads((QUALIFIED/'RESULT.json').read_text())
 if not r['original_all5_gates'] or not r['cycles']['all_legs_completed'] or r['native']['post_nonfoot_peak_native']!=0 or -r['actual_signed_yaw_deg']<30:raise ValueError('original qualifiedRIGHT teacher required')
 if sha256(DATA)!=r['samples']['sha256']:raise ValueError('original qualified sampleSHA changed')
 return right.dataset()
def cached(m,x,u):return right.cached(m,x,u)
def make(m):return right.module('_right_original_sharedparameter_scan',ENGINE).make(m)
neutral=old.old.neutral

def load_checked(path,digest,require_final=True):
 if sha256(path)!=digest:raise ValueError('RIGHToriginalfeature checkpointSHA')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p.get('format_version')!=FORMAT or p['config']['kind']!=FORMAT or p['config']['physical_eval_only_step']!=2000 or p['config']['training_episode_count']!=1 or p['config']['training_dataset']!=str(DATA):raise ValueError('strict qualified-only RIGHT originalfeature format')
 verify_hashes(p['config']['source_hashes']);verify_hashes(p['config']['data_hashes']);verify_hashes(p['config']['preflight_only_hashes']);m=FeatureRefine(right.legacy.load_baseline_actor())
 if p['identity']!=m.identity():raise ValueError('RIGHToriginalfeature identity/domain mismatch')
 m.load_state_dict(p['state_dict'],strict=True);parent,_=old.load_checked(PARENT,PARENT_SHA)
 if not exact(frozen_base(m),frozen_base(parent)) or not all(torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('base drift/nonfinite refinement')
 step=p['curve'][-1]['step']
 if step not in (1000,2000) or len(p['curve'])!=step or (require_final and step!=2000):raise ValueError('only fixedfinal2000 for physics')
 opt=torch.optim.Adam(m.adapter_parameters(),lr=LR);opt.load_state_dict(p['optimizer'])
 if len(opt.state)!=8 or any(float(v['step'])!=step for v in opt.state.values()) or any(g['lr']!=LR for g in opt.param_groups):raise ValueError('freshALL8 optimizer counters/lr mismatch')
 return m.eval(),p
