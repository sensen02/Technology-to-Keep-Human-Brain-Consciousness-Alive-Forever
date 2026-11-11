"""Separate immutable proposal-delta model format, no training/Adam impersonation."""
from pathlib import Path
import importlib.util,sys,json,copy
import numpy as np,torch
ROOT=HERE=Path(__file__).resolve().parent
PARENT_DIR=ROOT.parent/'left_hind_stance_refit';PARENT=PARENT_DIR/'TRAIN/checkpoint_004000.pt'
PARENT_SHA='ef768c788c94aeecbb49f0bd1ccb5c424333af36ec514941b72e49ab0d4917ff'
GLOBAL_ROWS=[8,11,13];LOCAL_ROWS=[1,4,6];JOINT_NAMES=['joint_LHCoxa_pitch','joint_LHTrochanter_pitch','joint_LHTibia_pitch']
s=importlib.util.spec_from_file_location('_RIGHT3bias_parent_LH_io',PARENT_DIR/'model_io.py');legacy=importlib.util.module_from_spec(s);sys.modules[s.name]=legacy;s.loader.exec_module(legacy)
sha256=legacy.sha256;exact=legacy.exact;verify_hashes=legacy.verify_hashes;BASE=legacy.BASE
FORMAT='RIGHT_THREE_BIAS_ACTUAL_V1'
def proposal():
 rng=np.random.Generator(np.random.PCG64(20261111));before=copy.deepcopy(rng.bit_generator.state);z=rng.standard_normal((2,3));d=np.clip(np.stack((.005*z[0],-.005*z[0],.005*z[1],-.005*z[1])),-.02,.02)
 return dict(kind=FORMAT,generation=1,mean=[0.,0.,0.],sigma=[.005]*3,bounds=[-.02,.02],rng_before=before,rng_after=copy.deepcopy(rng.bit_generator.state),normal_directions=z.tolist(),deltas=d.tolist(),order=['+d0','-d0','+d1','-d1'],seed=20261111)
def verify_mapping():
 m=legacy.mapping();names=m['actuator_joint_names']
 if [names[i] for i in GLOBAL_ROWS]!=JOINT_NAMES or [r-7 for r in GLOBAL_ROWS]!=LOCAL_ROWS or m['LH_contact_index']!=1:raise ValueError('actual LH3 joint mapping mismatch')
 return dict(global_rows=GLOBAL_ROWS,local_rows=LOCAL_ROWS,joint_names=JOINT_NAMES,actual_mapping=m)
def expected_state(parent,delta):
 state={k:v.detach().clone() for k,v in parent.items()}
 for i,d in zip(LOCAL_ROWS,delta):state['head.lh_bias'][i]+=float(d)
 return state
def load_checked(path,digest,require_final=True,require_frozen=True):
 path=Path(path).resolve()
 if not path.is_relative_to(ROOT) or path.name!='candidate.pt' or sha256(path)!=digest:raise ValueError('scoped candidate digest mismatch')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p.get('format_version')!=FORMAT or p.get('parent_sha256')!=PARENT_SHA or p.get('training_updates')!=0 or p.get('optimizer') is not None:raise ValueError('new bias-calibration format only, no new training/Adam')
 verify_hashes(p['calibration_source_hashes']);proposal_path=ROOT/'PROPOSAL.json'
 if p['proposal_path']!=str(proposal_path) or p['proposal_sha256']!=sha256(proposal_path):raise ValueError('immutable proposal identity mismatch')
 actual=json.loads(proposal_path.read_text());expected=proposal()
 for k,v in expected.items():
  if actual[k]!=v:raise ValueError('fixed PCG64 proposal mismatch '+k)
 idx=p['candidate_index']
 if not isinstance(idx,int) or not 0<=idx<4 or path.parent.name!=f'c{idx:02d}' or p['delta']!=actual['deltas'][idx]:raise ValueError('fixed candidate/index/delta mismatch')
 model,parent=legacy.load_checked(PARENT,PARENT_SHA)
 if p['identity']!=model.identity() or p['mapping']!=verify_mapping():raise ValueError('unchanged ANN architecture/mapping mismatch')
 if not exact(expected_state(parent['state_dict'],p['delta']),p['state_dict']):raise ValueError('candidate changed tensor outside exact3 LH biases')
 model.load_state_dict(p['state_dict'],strict=True)
 if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('nonfinite candidate')
 if require_frozen:
  frozen=json.loads((ROOT/'FROZEN.json').read_text());c=frozen['candidates'][idx]
  if len(frozen['candidates'])!=4 or frozen['proposal_sha256']!=sha256(proposal_path) or c['checkpoint']!=str(path) or c['sha256']!=digest:raise ValueError('all4 pre-actual freeze mismatch')
  for item in frozen['candidates']:
   if sha256(item['checkpoint'])!=item['sha256']:raise ValueError('another frozen candidate drift')
 return model.eval(),p

def selection(results):
 from pipeline import choose
 return choose(results)
