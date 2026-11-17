"""New visual wall-domain lineage, unchanged v1 architecture/physical baselines."""
from pathlib import Path
import importlib.util,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OLD=ROOT/'visual_policy_v1';LIB=ROOT/'frame_library_v2'
spec=importlib.util.spec_from_file_location('_wallvision_frozen_v1_common',OLD/'common.py');old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
np=old.np;torch=old.torch;json=old.json;hashlib=old.hashlib;os=old.os
sha=old.sha;read=old.read;save=old.save;jsonable=old.jsonable;cpu1=old.cpu1;guard=old.guard;exact=old.exact;verify=old.verify;MOTOR=old.MOTOR;MOTOR_SHA=old.MOTOR_SHA
FORMAT='VISION_STATIC_HEAD_OCCLUSION_RECOVERY_V2';PARENT=OLD/'TRAIN/checkpoint_000500.pt';PARENT_SHA='f1a339dd86cc2a7e5c7b7d4ba81ef4b39fe8a99b9a74c482ab19980668f137a2'
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def initialize():
 guard();m,p=old.load_checked(PARENT,PARENT_SHA,require_final=True);m.requires_grad_(True)
 if not exact(m.state_dict(),p['state_dict']):raise ValueError('exactf1ainitialweights')
 return m,p
def sources():
 names=('PREREG.json','common.py','model.py','data.py','generate_data.py','preflight.py','extra_checks.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py','adapt_source.py')
 return [HERE/n for n in names]+[OLD/n for n in ('common.py','model.py','data.py','generate_data.py','preflight.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py','DATASET.json','train.npz')]+[PARENT,OLD/'TRAIN/VERIFY_000500.json',OLD/'VALIDATION/RESULT.json',OLD/'VALIDATION/VERIFY.json',ROOT/'camera.py',LIB/'LIBRARY.json',LIB/'RGB.npz',LIB/'VERIFY.json',LIB/'PHYSICAL_IDENTITY.json',ROOT/'frame_library_v1/RGB.npz',ROOT/'frame_library_v1/LIBRARY.json',ROOT/'frame_library_v1/VERIFY.json',MOTOR]
def frozen():
 f=read(HERE/'FROZEN.json');verify(f['source_hashes']);return f
def library():
 m=read(LIB/'LIBRARY.json');a=read(LIB/'VERIFY.json')
 if m['kind']!='VISION_STATIC_HEAD_LIBRARY_V2' or Path(m['rgb_path']).resolve()!=LIB/'RGB.npz' or sha(LIB/'RGB.npz')!=m['rgb_sha256']:raise ValueError('strict actual126library2')
 if not a['integrity_all_pass'] or not all(v is True for v in a['checks'].values()) or a['library_sha256']!=sha(LIB/'LIBRARY.json') or a['rgb_sha256']!=sha(LIB/'RGB.npz') or a['invisible_label_training_admitted'] is not True:raise ValueError('library2independentactualauditrequired')
 verify(a['source_hashes']);verify(m['source_hashes'])
 if m['eye_order']!=['LEFT','RIGHT'] or m['physics_steps']!=0 or m['compiled_unchanged'] is not True:raise ValueError('sameactualoptics/physicalscope')
 if Path(m['physical_identity']).resolve()!=LIB/'PHYSICAL_IDENTITY.json' or sha(m['physical_identity'])!=m['physical_identity_sha256']:raise ValueError('actualphysicalidentity')
 with np.load(LIB/'RGB.npz',allow_pickle=False) as z:
  if z.files!=['rgb']:raise ValueError('RGBonly')
  rgb=z['rgb'].copy()
 if rgb.dtype!=np.uint8 or rgb.shape!=(126,2,32,32,3) or len(m['rows'])!=126 or [r['frame_id'] for r in m['rows']]!=list(range(126)):raise ValueError('126actualorderedRGBpairs')
 with np.load(ROOT/'frame_library_v1/RGB.npz',allow_pickle=False) as z:
  if not np.array_equal(rgb[:112],z['rgb']):raise ValueError('old112actualRGBprefixchanged')
 seen=set()
 for r,img in zip(m['rows'],rgb):
  key=(r['config_id'],r['pose_id'],r['stimulus'])
  if key in seen or hashlib.sha256(img.tobytes()).hexdigest()!=r['rgb_sha256']:raise ValueError('libraryrow/hash')
  seen.add(key);expected='train' if r['config_id'].startswith('T') else 'development' if r['config_id'] in ('V0','V1') else 'validation'
  if r['split']!=expected:raise ValueError('strictnewvalidationnotoldV0V1')
  if np.asarray(r['actual_head_position']).shape!=(3,) or np.asarray(r['actual_head_rotation']).shape!=(3,3):raise ValueError('actualheadmetadata')
  if r['stimulus'] in ('left','right') and (not r['target_visible'] or (r['bearing_rad']>0)!=(r['stimulus']=='left')):raise ValueError('actualvisiblebearingsign')
 for cfg,poses in [(f'T{i}',('pose0','pose1','pose2')) for i in range(4)]+[('V0',('pose3','pose4')),('V1',('pose3','pose4')),('V2',('pose5',)),('V3',('pose6',))]:
  for pose in poses:
   def image(s):return rgb[next(r['frame_id'] for r in m['rows'] if (r['config_id'],r['pose_id'],r['stimulus'])==(cfg,pose,s))]
   for s in ('off','center','left','right','occluded_left','occluded_right','occluder_only'):image(s)
   if not np.array_equal(image('occluded_left'),image('occluder_only')) or not np.array_equal(image('occluded_right'),image('occluder_only')):raise ValueError('actualwallnotcompleteocclusion')
 return rgb,m
def load_checked(path,digest,require_final=True):
 guard()
 if sha(path)!=digest:raise ValueError('visual2checkpointSHA')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p['format_version']!=FORMAT or p['deployment_qualified'] is not False or p['parent_sha256']!=PARENT_SHA:raise ValueError('visual2strictlineage')
 verify(p['source_hashes']);verify(p['data_hashes']);m,parent=initialize()
 if m.identity()!=p['identity']:raise ValueError('samevisualarchitecture')
 m.load_state_dict(p['state_dict'],strict=True);step=p['step']
 if step not in (0,250,500) or (require_final and step!=500) or len(p['curve'])!=step+1 or len(p['pair_schedule'])!=step:raise ValueError('fixed500only')
 if not all(torch.isfinite(x).all() for x in m.state_dict().values()):raise ValueError('nonfinitevisual2')
 opt=torch.optim.Adam(m.parameters(),lr=1e-4);opt.load_state_dict(p['optimizer'])
 if step and (len(opt.state)!=len(list(m.parameters())) or any(float(v['step'])!=step for v in opt.state.values())):raise ValueError('freshALLparamAdamcounter')
 return m.eval(),p
