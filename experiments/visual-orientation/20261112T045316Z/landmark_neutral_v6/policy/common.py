"""Strict812fparent/new672actuallibrary, explicitneutral-objective lineage."""
from pathlib import Path
import importlib.util,sys
HERE=Path(__file__).resolve().parent;SCOPE=HERE.parent;ROOT=SCOPE.parent;OLD=ROOT/'landmark_coverage_v5/policy';V1=ROOT/'visual_policy_v1';LIB=SCOPE/'library'
spec=importlib.util.spec_from_file_location('_neutral_previous_coverage_common',OLD/'common.py');old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
np=old.np;torch=old.torch;json=old.json;hashlib=old.hashlib;os=old.os
sha=old.sha;read=old.read;save=old.save;jsonable=old.jsonable;cpu1=old.cpu1;guard=old.guard;exact=old.exact;verify=old.verify;MOTOR=old.MOTOR;MOTOR_SHA=old.MOTOR_SHA
FORMAT='VISION_NEUTRAL_REBALANCE_POLICY_V6';PARENT=OLD/'TRAIN/checkpoint_001000.pt';PARENT_SHA='812f0bf409f39550eed6e47d2dbafb0287ce894e725a632c0ffc48afc009f02b'
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def initialize():
 guard();m,p=old.load_checked(PARENT,PARENT_SHA,require_final=True);m.requires_grad_(True)
 if not exact(m.state_dict(),p['state_dict']):raise ValueError('strict812finitialweights')
 return m,p
def sources():
 names=('PREREG.json','common.py','model.py','data.py','generate_data.py','preflight.py','extra_checks.py','adapt_source.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py')
 return [HERE/n for n in names]+[SCOPE/'PREREG.json']+[V1/n for n in ('model.py','data.py','preflight.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py')]+[OLD/n for n in ('common.py','data.py','DATASET.json','train.npz','TRAIN/VERIFY_001000.json','VALIDATION/RESULT.json','VALIDATION/VERIFY.json')]+[PARENT,ROOT/'camera.py',LIB/'LIBRARY.json',LIB/'RGB.npz',LIB/'VERIFY.json',LIB/'PHYSICAL_IDENTITY.json',ROOT/'landmark_coverage_v5/library/LIBRARY.json',ROOT/'landmark_coverage_v5/library/RGB.npz',MOTOR]
def frozen():
 f=read(HERE/'FROZEN.json');verify(f['source_hashes']);return f
def library():
 m=read(LIB/'LIBRARY.json');a=read(LIB/'VERIFY.json')
 if m['kind']!='VISION_NEUTRAL_HEAD_LIBRARY_V6' or Path(m['rgb_path']).resolve()!=LIB/'RGB.npz' or sha(LIB/'RGB.npz')!=m['rgb_sha256']:raise ValueError('actual672identity')
 if not a['integrity_all_pass'] or not all(v is True for v in a['checks'].values()) or a['library_sha256']!=sha(LIB/'LIBRARY.json') or a['rgb_sha256']!=sha(LIB/'RGB.npz') or not a['invisible_label_training_admitted']:raise ValueError('independentactual672proof')
 verify(a['source_hashes']);verify(m['source_hashes'])
 if m['physics_steps']!=0 or not m['compiled_unchanged'] or m['eye_order']!=['LEFT','RIGHT'] or sha(m['physical_identity'])!=m['physical_identity_sha256']:raise ValueError('physicalopticalidentity')
 with np.load(LIB/'RGB.npz',allow_pickle=False) as z:
  if z.files!=['rgb']:raise ValueError('onlyactualRGB')
  rgb=z['rgb'].copy()
 with np.load(ROOT/'landmark_coverage_v5/library/RGB.npz',allow_pickle=False) as z:
  if not np.array_equal(rgb[:616],z['rgb']):raise ValueError('old616pixelprefix')
 if rgb.shape!=(672,2,32,32,3) or rgb.dtype!=np.uint8 or len(m['rows'])!=672 or [r['frame_id'] for r in m['rows']]!=list(range(672)):raise ValueError('actual672shape')
 seen=set()
 for r,img in zip(m['rows'],rgb):
  expected='train' if r['config_id'].startswith('T') else 'development' if r['config_id'] in ('V8','V9') else 'validation';key=(r['config_id'],r['pose_id'],r['stimulus'])
  if key in seen or r['split']!=expected or hashlib.sha256(img.tobytes()).hexdigest()!=r['rgb_sha256'] or not r['eye_outside_boxes'] or r['eye_box_min_clearance']<=.005:raise ValueError('actualrow/split/clearance')
  seen.add(key)
  if r['stimulus'] in ('left','right') and (not r['target_visible'] or (r['bearing_rad']>0)!=(r['stimulus']=='left')):raise ValueError('actualvisiblelabels')
 for cfg in ('T0','T1','T2','T3','V8','V9','V10','V11'):
  poseids=range(20) if cfg.startswith('T') else range(4);prefix='trainpose' if cfg.startswith('T') else 'valpose' if cfg in ('V8','V9') else 'neutralvalpose'
  for j in poseids:
   rows=[r for r in m['rows'] if r['config_id']==cfg and r['pose_id']==prefix+str(j)]
   if len(rows)!=7:raise ValueError('configposecoverage')
   ims={r['stimulus']:rgb[r['frame_id']] for r in rows}
   if not np.array_equal(ims['occluded_left'],ims['occluder_only']) or not np.array_equal(ims['occluded_right'],ims['occluder_only']):raise ValueError('actualclosedboxproof')
 return rgb,m
def load_checked(path,digest,require_final=True):
 guard()
 if sha(path)!=digest:raise ValueError('neutralcheckpointSHA')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p['format_version']!=FORMAT or p['deployment_qualified'] is not False or p['parent_sha256']!=PARENT_SHA:raise ValueError('neutralformat/lineage')
 verify(p['source_hashes']);verify(p['data_hashes']);m,parent=initialize()
 if p['identity']!=m.identity():raise ValueError('same14architecture')
 m.load_state_dict(p['state_dict'],strict=True);step=p['step']
 if step not in (0,500,1000) or (require_final and step!=1000) or len(p['curve'])!=step+1 or len(p['pair_schedule'])!=step:raise ValueError('fixed1000lineage')
 if not all(torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('nonfinitevisual')
 opt=torch.optim.Adam(m.parameters(),lr=1e-4);opt.load_state_dict(p['optimizer'])
 if step and (len(opt.state)!=14 or any(float(v['step'])!=step for v in opt.state.values())):raise ValueError('freshALL14counter')
 return m.eval(),p
