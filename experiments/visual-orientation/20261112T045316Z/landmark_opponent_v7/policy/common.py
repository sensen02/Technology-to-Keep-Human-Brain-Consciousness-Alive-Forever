"""Strictbe622lineage,newneuralencoder/fixed1000,actual728RGBproof."""
from pathlib import Path
import importlib.util,sys
HERE=Path(__file__).resolve().parent;SCOPE=HERE.parent;ROOT=SCOPE.parent;OLD=ROOT/'landmark_neutral_v6/policy';V1=ROOT/'visual_policy_v1';LIB=SCOPE/'library'
spec=importlib.util.spec_from_file_location('_opponent_previous_neutral_common',OLD/'common.py');old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
np=old.np;torch=old.torch;json=old.json;hashlib=old.hashlib;os=old.os
sha=old.sha;read=old.read;save=old.save;jsonable=old.jsonable;cpu1=old.cpu1;guard=old.guard;exact=old.exact;verify=old.verify;MOTOR=old.MOTOR;MOTOR_SHA=old.MOTOR_SHA
FORMAT='VISION_OPPONENT_POLICY_V7';PARENT=OLD/'TRAIN/checkpoint_001000.pt';PARENT_SHA='be62264966789c59a82e27dc696d710d0751f4c9ac8c1b575d26b0a1920a7651'
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def initialize():
 guard();sentinel=object();prior=sys.modules.get('model',sentinel)
 try:
  original=module('_opponent_original_parent_class',V1/'model.py');sys.modules['model']=original;parent,p=old.load_checked(PARENT,PARENT_SHA,require_final=True)
 finally:
  if prior is sentinel:sys.modules.pop('model',None)
  else:sys.modules['model']=prior
 # Newmodel import must be explicit and avoid leaking originalclass via basenamealias.
 architecture=module('_opponent_new_architecture',HERE/'model.py')
 with torch.random.fork_rng(devices=[]):torch.manual_seed(20261112);m=architecture.VisualPolicy()
 state=m.state_dict();previous=parent.state_dict()
 for key in previous:
  if not key.startswith('conv1.'):state[key]=previous[key].clone()
 m.load_state_dict(state,strict=True);m.requires_grad_(True)
 if not all(torch.equal(m.state_dict()[k],v) for k,v in previous.items() if not k.startswith('conv1.')):raise ValueError('exactother12parenttensors')
 return m,p
def sources():
 names=('PREREG.json','common.py','model.py','data.py','generate_data.py','preflight.py','extra_checks.py','adapt_source.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py')
 return [HERE/n for n in names]+[SCOPE/'PREREG.json']+[V1/n for n in ('model.py','data.py','preflight.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py')]+[OLD/n for n in ('common.py','data.py','adapt_source.py','generate_data.py','DATASET.json','train.npz','TRAIN/VERIFY_001000.json','VALIDATION/RESULT.json','VALIDATION/VERIFY.json')]+[PARENT,ROOT/'camera.py',LIB/'LIBRARY.json',LIB/'RGB.npz',LIB/'VERIFY.json',LIB/'PHYSICAL_IDENTITY.json',LIB/'TRAIN_OPPONENT_DIAGNOSTIC.json',ROOT/'landmark_neutral_v6/library/LIBRARY.json',ROOT/'landmark_neutral_v6/library/RGB.npz',MOTOR]
def frozen():
 f=read(HERE/'FROZEN.json');verify(f['source_hashes']);return f
def library():
 m=read(LIB/'LIBRARY.json');a=read(LIB/'VERIFY.json')
 if m['kind']!='VISION_OPPONENT_HEAD_LIBRARY_V7' or Path(m['rgb_path']).resolve()!=LIB/'RGB.npz' or sha(LIB/'RGB.npz')!=m['rgb_sha256']:raise ValueError('actual728identity')
 if not a['integrity_all_pass'] or not all(v is True for v in a['checks'].values()) or a['library_sha256']!=sha(LIB/'LIBRARY.json') or a['rgb_sha256']!=sha(LIB/'RGB.npz') or not a['invisible_label_training_admitted']:raise ValueError('independentactual728proof')
 verify(a['source_hashes']);verify(m['source_hashes'])
 if sha(a['training_opponent_diagnostic'])!=a['training_opponent_diagnostic_sha256'] or read(a['training_opponent_diagnostic'])['model_forward_calls']!=0:raise ValueError('trainpixelopponentdiagnostic')
 if m['physics_steps']!=0 or not m['compiled_unchanged'] or m['eye_order']!=['LEFT','RIGHT'] or sha(m['physical_identity'])!=m['physical_identity_sha256']:raise ValueError('actualphysicalidentity')
 with np.load(LIB/'RGB.npz',allow_pickle=False) as z:
  if z.files!=['rgb']:raise ValueError('actualRGBonly')
  rgb=z['rgb'].copy()
 with np.load(ROOT/'landmark_neutral_v6/library/RGB.npz',allow_pickle=False) as z:
  if not np.array_equal(rgb[:672],z['rgb']):raise ValueError('old672exactprefix')
 if rgb.shape!=(728,2,32,32,3) or rgb.dtype!=np.uint8 or len(m['rows'])!=728:raise ValueError('actual728shape')
 for i,(r,img) in enumerate(zip(m['rows'],rgb)):
  expected='train' if r['config_id'].startswith('T') else 'development' if r['config_id'] in ('V8','V9','V10','V11') else 'validation'
  if r['frame_id']!=i or r['split']!=expected or hashlib.sha256(img.tobytes()).hexdigest()!=r['rgb_sha256'] or not r['eye_outside_boxes'] or r['eye_box_min_clearance']<=.005:raise ValueError('rowhash/split/eyeoutside')
 return rgb,m
def load_checked(path,digest,require_final=True):
 guard()
 if sha(path)!=digest:raise ValueError('opponentcheckpointSHA')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p['format_version']!=FORMAT or p['deployment_qualified'] is not False or p['parent_sha256']!=PARENT_SHA:raise ValueError('opponentformat/lineage')
 verify(p['source_hashes']);verify(p['data_hashes']);m,parent=initialize()
 if p['identity']!=m.identity():raise ValueError('newfixedopponentarchitectureidentity')
 m.load_state_dict(p['state_dict'],strict=True);step=p['step']
 if step not in (0,500,1000) or (require_final and step!=1000) or len(p['curve'])!=step+1 or len(p['pair_schedule'])!=step:raise ValueError('fixed1000lineage')
 if not torch.equal(m.opponent_weight,torch.tensor([1.,-.5,-.5],dtype=torch.float64).reshape(1,3,1,1)) or m.opponent_bias.item()!=0:raise ValueError('fixedneuralreceptorchanged')
 if not all(torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('nonfinitevisual')
 opt=torch.optim.Adam(m.parameters(),lr=1e-4);opt.load_state_dict(p['optimizer'])
 if step and (len(opt.state)!=14 or any(float(v['step'])!=step for v in opt.state.values())):raise ValueError('fresh14trainableAdamcounter')
 return m.eval(),p
