"""Four fixed neuron-bias copies, exact intervention checked on every load."""
from pathlib import Path
import importlib.util,sys,hashlib,json
import torch
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'20261111T010218Z'
PARENT=OLD/'train/TRAIN/checkpoint_002000.pt'
PARENT_SHA='eeced49c5bdd50a7e5bce2cfd63d56d9929b3bf2d2cb00c1985019d91321e6df'
spec=importlib.util.spec_from_file_location('_bias_parent_io',OLD/'train/model_io.py');legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
def sha256(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load_checked(path,sha,require_final=True):
 if sha256(path)!=sha:raise ValueError('candidate digest mismatch')
 p=torch.load(path,map_location='cpu',weights_only=False)
 allowed={c['id']:c for c in json.loads((ROOT/'PREREG.json').read_text())['candidates']}
 if p['candidate_id'] not in allowed:raise ValueError('unregistered candidate')
 c=allowed[p['candidate_id']]
 if p['bias_index']!=c['index'] or p['bias_delta']!=c['bias_delta']:raise ValueError('unregistered scalar intervention')
 if p['parent_sha256']!=PARENT_SHA or sha256(PARENT)!=PARENT_SHA:raise ValueError('parent mismatch')
 legacy.verify_hashes(p['calibration_source_hashes'])
 m,parent=legacy.load_checked(PARENT,PARENT_SHA)
 expected={k:v.clone() for k,v in parent['state_dict'].items()};expected['readout.bias'][c['index']]+=c['bias_delta']
 if not legacy.exact(expected,p['state_dict']):raise ValueError('changes beyond exact one bias scalar')
 if p['identity']!=m.identity():raise ValueError('architecture changed')
 m.load_state_dict(p['state_dict'],strict=True)
 if not all(torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('nonfinite model')
 return m.eval(),p
