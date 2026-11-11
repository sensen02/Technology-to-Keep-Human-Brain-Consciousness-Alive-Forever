"""Immutable ANN candidate loader for fixed three-neuron CEM proposals."""
from pathlib import Path
import importlib.util,sys,hashlib,json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'20261111T010218Z'
PARENT=OLD/'train/TRAIN/checkpoint_002000.pt'
PARENT_SHA='eeced49c5bdd50a7e5bce2cfd63d56d9929b3bf2d2cb00c1985019d91321e6df'
INDICES=[29,32,34]
spec=importlib.util.spec_from_file_location('_cem_parent_io',OLD/'train/model_io.py');legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
def sha256(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load_checked(path,digest,require_final=True):
 if sha256(path)!=digest:raise ValueError('candidate SHA mismatch')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p['kind']!='CEM3bias' or p['parent_sha256']!=PARENT_SHA:raise ValueError('wrong candidate identity')
 legacy.verify_hashes(p['calibration_source_hashes'])
 proposal=json.loads(Path(p['proposal_path']).read_text())
 if sha256(p['proposal_path'])!=p['proposal_sha256']:raise ValueError('proposal changed')
 if not 0<=p['candidate_index']<4 or not 1<=p['generation']<=4:raise ValueError('outside fixed generation budget')
 if proposal['generation']!=p['generation'] or proposal['deltas'][p['candidate_index']]!=p['delta']:raise ValueError('candidate not exact immutable proposal')
 rng=np.random.default_rng();rng.bit_generator.state=proposal['rng_before']
 z=rng.standard_normal((2,3));m=np.asarray(proposal['mean']);s=np.asarray(proposal['sigma'])
 expected_d=np.clip(np.stack((m+s*z[0],m-s*z[0],m+s*z[1],m-s*z[1])),-.25,.25)
 if not np.array_equal(expected_d,np.asarray(proposal['deltas'])):raise ValueError('proposal RNG/order mismatch')
 model,parent=legacy.load_checked(PARENT,PARENT_SHA)
 if p['identity']!=model.identity():raise ValueError('ANN architecture changed')
 expected={k:v.clone() for k,v in parent['state_dict'].items()}
 for index,delta in zip(INDICES,p['delta']):expected['readout.bias'][index]+=delta
 if not legacy.exact(expected,p['state_dict']):raise ValueError('unregistered ANN tensor change')
 model.load_state_dict(p['state_dict'],strict=True)
 if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('nonfinite candidate')
 return model.eval(),p
