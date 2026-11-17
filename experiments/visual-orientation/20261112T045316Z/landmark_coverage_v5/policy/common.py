"""Strict79db initialization/actual616library; many staticposes notindependenttrajectories."""
from pathlib import Path
import importlib.util,sys
HERE=Path(__file__).resolve().parent;SCOPE=HERE.parent;ROOT=SCOPE.parent;OLD=ROOT/'visual_policy_v2';V1=ROOT/'visual_policy_v1';LIB=SCOPE/'library'
spec=importlib.util.spec_from_file_location('_coverage_original_visual_common',OLD/'common.py');old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
np=old.np;torch=old.torch;json=old.json;hashlib=old.hashlib;os=old.os
sha=old.sha;read=old.read;save=old.save;jsonable=old.jsonable;cpu1=old.cpu1;guard=old.guard;exact=old.exact;verify=old.verify;MOTOR=old.MOTOR;MOTOR_SHA=old.MOTOR_SHA
FORMAT='VISION_COVERAGE_POLICY_V5';PARENT=OLD/'TRAIN/checkpoint_000500.pt';PARENT_SHA='79dbcdcd38c1c3de94588bff9923b3758f82b1abbc63f017e499ece464340fee'
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def initialize():
 guard();m,p=old.load_checked(PARENT,PARENT_SHA,require_final=True);m.requires_grad_(True)
 if not exact(m.state_dict(),p['state_dict']):raise ValueError('strict79dbinitialweights')
 return m,p
def sources():
 names=('PREREG.json','common.py','model.py','data.py','generate_data.py','preflight.py','extra_checks.py','adapt_source.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py')
 return [HERE/n for n in names]+[SCOPE/'PREREG.json']+[V1/n for n in ('model.py','data.py','preflight.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py')]+[OLD/n for n in ('common.py','TRAIN/VERIFY_000500.json','VALIDATION/RESULT.json','VALIDATION/VERIFY.json')]+[PARENT,ROOT/'camera.py',LIB/'LIBRARY.json',LIB/'RGB.npz',LIB/'VERIFY.json',LIB/'PHYSICAL_IDENTITY.json',LIB/'TRAIN_COVERAGE_DIAGNOSTIC.json',LIB/'layouts.py',MOTOR]
def frozen():
 f=read(HERE/'FROZEN.json');verify(f['source_hashes']);return f
def library():
 m=read(LIB/'LIBRARY.json');a=read(LIB/'VERIFY.json')
 if m['kind']!='VISION_COVERAGE_HEAD_LIBRARY_V5' or Path(m['rgb_path']).resolve()!=LIB/'RGB.npz' or sha(LIB/'RGB.npz')!=m['rgb_sha256']:raise ValueError('actual616libraryidentity')
 if not a['integrity_all_pass'] or not all(v is True for v in a['checks'].values()) or a['library_sha256']!=sha(LIB/'LIBRARY.json') or a['rgb_sha256']!=sha(LIB/'RGB.npz') or not a['invisible_label_training_admitted']:raise ValueError('independentactual616proof')
 verify(a['source_hashes']);verify(m['source_hashes'])
 if sha(a['training_coverage_diagnostic'])!=a['training_coverage_diagnostic_sha256']:raise ValueError('trainRGBdiagnosticSHA')
 diagnostic=read(a['training_coverage_diagnostic'])
 if diagnostic['model_forward_calls']!=0 or diagnostic['training_actual_frames']!=560 or diagnostic['training_saved_pose_count']!=20:raise ValueError('trainonlyfielddiagnostic')
 if m['physics_steps']!=0 or not m['compiled_unchanged'] or m['eye_order']!=['LEFT','RIGHT'] or sha(m['physical_identity'])!=m['physical_identity_sha256']:raise ValueError('physicalopticalidentity')
 with np.load(LIB/'RGB.npz',allow_pickle=False) as z:
  if z.files!=['rgb']:raise ValueError('rawactualRGBonly')
  rgb=z['rgb'].copy()
 if rgb.shape!=(616,2,32,32,3) or rgb.dtype!=np.uint8 or len(m['rows'])!=616 or [r['frame_id'] for r in m['rows']]!=list(range(616)):raise ValueError('actual616shape')
 seen=set()
 for r,image in zip(m['rows'],rgb):
  key=(r['config_id'],r['pose_id'],r['stimulus']);expected='train' if r['config_id'].startswith('T') else 'validation'
  if key in seen or r['split']!=expected or hashlib.sha256(image.tobytes()).hexdigest()!=r['rgb_sha256'] or not r['eye_outside_boxes'] or r['eye_box_min_clearance']<=.005:raise ValueError('row/hash/split/clearance')
  seen.add(key)
  if r['stimulus'] in ('left','right') and (not r['target_visible'] or (r['bearing_rad']>0)!=(r['stimulus']=='left')):raise ValueError('actualvisiblesign')
 for cfg in ('T0','T1','T2','T3','V8','V9'):
  poses=range(20) if cfg.startswith('T') else range(4);prefix='trainpose' if cfg.startswith('T') else 'valpose'
  for i in poses:
   rows=[r for r in m['rows'] if r['config_id']==cfg and r['pose_id']==prefix+str(i)]
   if len(rows)!=7:raise ValueError('completeactualcoverage')
   ims={r['stimulus']:rgb[r['frame_id']] for r in rows}
   if not np.array_equal(ims['occluded_left'],ims['occluder_only']) or not np.array_equal(ims['occluded_right'],ims['occluder_only']):raise ValueError('actualclosedboxocclusion')
 return rgb,m
def load_checked(path,digest,require_final=True):
 guard()
 if sha(path)!=digest:raise ValueError('coveragecheckpointSHA')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p['format_version']!=FORMAT or p['deployment_qualified'] is not False or p['parent_sha256']!=PARENT_SHA:raise ValueError('coverageformat/parentlineage')
 verify(p['source_hashes']);verify(p['data_hashes']);m,parent=initialize()
 if p['identity']!=m.identity():raise ValueError('sameoriginalvisualarchitecture')
 m.load_state_dict(p['state_dict'],strict=True);step=p['step']
 if step not in (0,500,1000) or (require_final and step!=1000) or len(p['curve'])!=step+1 or len(p['pair_schedule'])!=step:raise ValueError('fixed1000lineage')
 if not all(torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('nonfinitemodel')
 opt=torch.optim.Adam(m.parameters(),lr=1e-4);opt.load_state_dict(p['optimizer'])
 if step and (len(opt.state)!=14 or any(float(v['step'])!=step for v in opt.state.values())):raise ValueError('freshALL14Adamcounts')
 return m.eval(),p
