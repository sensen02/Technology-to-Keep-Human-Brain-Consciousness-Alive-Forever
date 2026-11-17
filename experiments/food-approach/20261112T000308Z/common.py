"""New strict segmented-recovery identity, frozen inherited architecture/utilities."""
from pathlib import Path
import importlib.util,sys
HERE=Path(__file__).resolve().parent;EXP=HERE.parent;OLD=EXP/'20261111T234150Z'
spec=importlib.util.spec_from_file_location('_segmented_frozen_longhistory_common',OLD/'common.py');old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
np=old.np;torch=old.torch;json=old.json;hashlib=old.hashlib;os=old.os
module=old.module;sha=old.sha;read=old.read;save=old.save;jsonable=old.jsonable;exact=old.exact;cpu1=old.cpu1;guard=old.guard;verify_hashes=old.verify_hashes
GEOMETRY=old.GEOMETRY;MOTOR=old.MOTOR;MOTOR_SHA=old.MOTOR_SHA
PARENT=OLD/'TRAIN/checkpoint_001000.pt';PARENT_SHA='4be0cd6a9d1f77e3be17323713f3fc8783adedd4b9e8a010df69922c7d1a09e7';FORMAT='ODOR_SEGMENT_RECOVERY_V1'
def initialize():
 guard();m,p=old.load_checked(PARENT,PARENT_SHA,require_final=True);m.requires_grad_(True)
 if not exact(m.state_dict(),p['state_dict']):raise ValueError('exact4be0initialization required')
 return m,p
def source_paths():
 names=('PREREG.json','common.py','data.py','engine.py','generate_data.py','preflight.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py','adapt_source.py')
 return [HERE/n for n in names]+[OLD/n for n in ('common.py','data.py','engine.py','preflight.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py','DATASET.json','train.npz')]+[PARENT,OLD/'TRAIN/VERIFY_001000.json',OLD/'VALIDATION/RESULT.json',OLD/'VALIDATION/VERIFY.json',GEOMETRY,MOTOR]
def freeze_proof():
 f=read(HERE/'FROZEN.json');verify_hashes(f['source_hashes']);return f
def load_checked(path,digest,require_final=True):
 guard()
 if sha(path)!=digest:raise ValueError('segmentedcheckpointSHA mismatch')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p.get('format_version')!=FORMAT or p.get('kind')!='food_steering_behavior_long_history' or p.get('deployment_qualified') is not False:raise ValueError('strict segmented recoveryformat')
 verify_hashes(p['config']['source_hashes']);verify_hashes(p['config']['data_hashes']);m,parent=initialize()
 if p['identity']!=m.identity() or p['parent_sha256']!=PARENT_SHA:raise ValueError('sameoriginalbehaviorarchitecture/newparent identity required')
 m.load_state_dict(p['state_dict'],strict=True);step=p['curve'][-1]['step']
 if step not in (0,250,500) or (require_final and step!=500):raise ValueError('onlyfinal500 validation')
 if len(p['curve'])!=step+1:raise ValueError('actualcurveincludesstep0')
 if not all(torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('nonfinite model')
 opt=torch.optim.Adam(m.parameters(),lr=.0001);opt.load_state_dict(p['optimizer'])
 if step and (len(opt.state)!=8 or any(float(v['step'])!=step for v in opt.state.values())):raise ValueError('freshALL8Adam exactstep required')
 return m.eval(),p
