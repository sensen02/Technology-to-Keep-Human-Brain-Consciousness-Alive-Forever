"""Strict far-domain lineage; inherited exact architecture/utilities, no physical backend."""
from pathlib import Path
import importlib.util,sys
HERE=Path(__file__).resolve().parent;SCOPE=HERE.parent;ROOT=SCOPE.parent;OLD=ROOT/'visual_policy_v2';V1=ROOT/'visual_policy_v1';LIB=SCOPE/'library'
spec=importlib.util.spec_from_file_location('_far_previous_visual_common',OLD/'common.py');old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
np=old.np;torch=old.torch;json=old.json;hashlib=old.hashlib;os=old.os
sha=old.sha;read=old.read;save=old.save;jsonable=old.jsonable;cpu1=old.cpu1;guard=old.guard;exact=old.exact;verify=old.verify;MOTOR=old.MOTOR;MOTOR_SHA=old.MOTOR_SHA
FORMAT='VISION_FAR_LANDMARK_POLICY_V3';PARENT=OLD/'TRAIN/checkpoint_000500.pt';PARENT_SHA='79dbcdcd38c1c3de94588bff9923b3758f82b1abbc63f017e499ece464340fee'
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def initialize():
 guard();m,p=old.load_checked(PARENT,PARENT_SHA,require_final=True);m.requires_grad_(True)
 if not exact(m.state_dict(),p['state_dict']):raise ValueError('strict79dbparent initialization')
 return m,p
def sources():
 names=('PREREG.json','common.py','model.py','data.py','generate_data.py','preflight.py','extra_checks.py','adapt_source.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py')
 return [HERE/n for n in names]+[SCOPE/'PREREG.json']+[V1/n for n in ('model.py','data.py','preflight.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py')]+[OLD/'common.py',OLD/'TRAIN/VERIFY_000500.json',OLD/'VALIDATION/RESULT.json',OLD/'VALIDATION/VERIFY.json',PARENT,ROOT/'camera.py',LIB/'LIBRARY.json',LIB/'RGB.npz',LIB/'VERIFY.json',LIB/'PHYSICAL_IDENTITY.json',MOTOR]
def frozen():
 f=read(HERE/'FROZEN.json');verify(f['source_hashes']);return f
def library():
 m=read(LIB/'LIBRARY.json');a=read(LIB/'VERIFY.json')
 if m['kind']!='VISION_FAR_HEAD_LIBRARY_V3' or Path(m['rgb_path']).resolve()!=LIB/'RGB.npz' or sha(LIB/'RGB.npz')!=m['rgb_sha256']:raise ValueError('far42actual RGB identity')
 if not a['integrity_all_pass'] or not all(v is True for v in a['checks'].values()) or a['library_sha256']!=sha(LIB/'LIBRARY.json') or a['rgb_sha256']!=sha(LIB/'RGB.npz') or not a['invisible_label_training_admitted']:raise ValueError('independent far optical proof required')
 verify(a['source_hashes']);verify(m['source_hashes'])
 if m['physics_steps']!=0 or m['compiled_unchanged'] is not True or m['eye_order']!=['LEFT','RIGHT'] or sha(m['physical_identity'])!=m['physical_identity_sha256']:raise ValueError('real complete-body optical identity')
 with np.load(LIB/'RGB.npz',allow_pickle=False) as z:
  if z.files!=['rgb']:raise ValueError('onlyactualRGB')
  rgb=z['rgb'].copy()
 if rgb.shape!=(42,2,32,32,3) or rgb.dtype!=np.uint8 or len(m['rows'])!=42 or [r['frame_id'] for r in m['rows']]!=list(range(42)):raise ValueError('exact42library')
 expected=[('T0','pose0'),('T1','pose1'),('T2','pose2'),('T3','pose0'),('V4','pose7'),('V5','pose8')]
 for cfg,pose in expected:
  rows=[r for r in m['rows'] if (r['config_id'],r['pose_id'])==(cfg,pose)]
  if len(rows)!=7:raise ValueError('exact far config/pose coverage')
  ims={r['stimulus']:rgb[r['frame_id']] for r in rows}
  for r in rows:
   if r['split']!=('train' if cfg.startswith('T') else 'validation') or hashlib.sha256(rgb[r['frame_id']].tobytes()).hexdigest()!=r['rgb_sha256'] or not r['eye_outside_boxes'] or r['eye_box_min_clearance']<=.005:raise ValueError('row identity/eye clearance')
   if r['stimulus'] in ('left','right') and (not r['target_visible'] or (r['bearing_rad']>0)!=(r['stimulus']=='left')):raise ValueError('actual visible sign')
  if not np.array_equal(ims['occluded_left'],ims['occluder_only']) or not np.array_equal(ims['occluded_right'],ims['occluder_only']):raise ValueError('closedboxes actual occlusion failed')
 return rgb,m
def load_checked(path,digest,require_final=True):
 guard()
 if sha(path)!=digest:raise ValueError('far policy SHA')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p['format_version']!=FORMAT or p['deployment_qualified'] is not False or p['parent_sha256']!=PARENT_SHA:raise ValueError('far format/lineage')
 verify(p['source_hashes']);verify(p['data_hashes']);m,parent=initialize()
 if m.identity()!=p['identity']:raise ValueError('same original visual architecture')
 m.load_state_dict(p['state_dict'],strict=True);step=p['step']
 if step not in (0,250,500) or (require_final and step!=500) or len(p['curve'])!=step+1 or len(p['pair_schedule'])!=step:raise ValueError('fixed500 actuallineage')
 if not all(torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('nonfinite visual model')
 opt=torch.optim.Adam(m.parameters(),lr=1e-4);opt.load_state_dict(p['optimizer'])
 if step and (len(opt.state)!=len(list(m.parameters())) or any(float(v['step'])!=step for v in opt.state.values())):raise ValueError('fresh14Adam counters')
 return m.eval(),p
