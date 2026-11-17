"""Actual visual-library identities and parent-owned CPU1 execution; no physics."""
from pathlib import Path
import hashlib,json,os,sys
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;LIB=ROOT/'frame_library_v1'
FORMAT='VISION_STATIC_HEAD_POLICY_V1'
MOTOR=ROOT.parent.parent/'food-approach/20261111T161007Z/combined_turning/COMBINED.pt'
MOTOR_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def jsonable(x):
 if torch.is_tensor(x):return x.detach().cpu().tolist()
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,Path):return str(x)
 if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [jsonable(v) for v in x]
 return x
def save(path,x):
 with Path(path).open('x',encoding='utf8') as f:json.dump(jsonable(x),f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def guard():
 if any(k in sys.modules for k in ('mujoco','physics','student_wave','flygym')):raise RuntimeError('visual policy offline forbids physical backend')
def cpu1():
 guard();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});torch.set_num_threads(1)
 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
  if os.environ.get(k)!='1':raise ValueError(k+'=1 required')
def exact(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return type(a)==type(b) and a==b
def verify(hashes):
 for p,h in hashes.items():
  if not Path(p).is_absolute() or sha(p)!=h:raise ValueError('source/input drift '+p)
def sources():
 names=('PREREG.json','common.py','model.py','data.py','generate_data.py','preflight.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py')
 return [HERE/n for n in names]+[ROOT/'camera.py',ROOT/'probe.py',LIB/'LIBRARY.json',LIB/'RGB.npz',LIB/'VERIFY.json',LIB/'PHYSICAL_IDENTITY.json',MOTOR]
def frozen():
 x=read(HERE/'FROZEN.json');verify(x['source_hashes']);return x
def library():
 m=read(LIB/'LIBRARY.json');audit=read(LIB/'VERIFY.json')
 if audit['integrity_all_pass'] is not True or not all(v is True for v in audit['checks'].values()) or audit['library_sha256']!=sha(LIB/'LIBRARY.json') or audit['rgb_sha256']!=sha(LIB/'RGB.npz') or audit['invisible_label_training_admitted'] is not True:raise ValueError('independent actualRGB library audit required')
 verify(audit['source_hashes'])
 if m['kind']!='VISION_STATIC_HEAD_LIBRARY_V1' or Path(m['rgb_path']).resolve()!=LIB/'RGB.npz' or sha(LIB/'RGB.npz')!=m['rgb_sha256']:raise ValueError('actual image library identity')
 if m['eye_order']!=['LEFT','RIGHT'] or m['physics_steps']!=0 or m['compiled_unchanged'] is not True:raise ValueError('actual camera library scope')
 verify(m['source_hashes'])
 if Path(m['physical_identity']).resolve()!=LIB/'PHYSICAL_IDENTITY.json' or sha(m['physical_identity'])!=m['physical_identity_sha256']:raise ValueError('actual rendered physical identity')
 with np.load(LIB/'RGB.npz',allow_pickle=False) as z:
  if z.files!=['rgb']:raise ValueError('RGB-only library array')
  rgb=z['rgb'].copy()
 if rgb.dtype!=np.uint8 or rgb.shape!=(112,2,32,32,3) or len(m['rows'])!=112:raise ValueError('exact112 binocular actualframes')
 if [r['frame_id'] for r in m['rows']]!=list(range(112)):raise ValueError('frame IDs contiguous')
 seen=set()
 for r,img in zip(m['rows'],rgb):
  key=(r['config_id'],r['pose_id'],r['stimulus'])
  if key in seen or hashlib.sha256(img.tobytes()).hexdigest()!=r['rgb_sha256']:raise ValueError('duplicate row/hash')
  seen.add(key)
  if r['split']!=('train' if r['config_id'].startswith('T') else 'validation'):raise ValueError('config split')
  if np.asarray(r['actual_head_position']).shape!=(3,) or np.asarray(r['actual_head_rotation']).shape!=(3,3):raise ValueError('actual head pose required')
  if r['stimulus'] in ('left','right'):
   if not r['target_visible'] or (r['bearing_rad']>0)!=(r['stimulus']=='left'):raise ValueError('visiblebearing sign required')
 for cfg,poses in [(f'T{i}',('pose0','pose1','pose2')) for i in range(4)]+[(f'V{i}',('pose3','pose4')) for i in range(2)]:
  for pose in poses:
   def image(s):return rgb[next(r['frame_id'] for r in m['rows'] if (r['config_id'],r['pose_id'],r['stimulus'])==(cfg,pose,s))]
   for s in ('off','center','left','right','occluded_left','occluded_right','occluder_only'):image(s)
   if not np.array_equal(image('occluded_left'),image('occluder_only')) or not np.array_equal(image('occluded_right'),image('occluder_only')):raise ValueError('actual complete occlusion counterfactual failed')
 return rgb,m
def load_checked(path,digest,require_final=True):
 guard();path=Path(path)
 if sha(path)!=digest:raise ValueError('visual checkpoint SHA')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p['format_version']!=FORMAT or p['deployment_qualified'] is not False:raise ValueError('offline vision format')
 verify(p['source_hashes']);verify(p['data_hashes'])
 from model import VisualPolicy
 rng=torch.get_rng_state();m=VisualPolicy();torch.set_rng_state(rng)
 if m.identity()!=p['identity']:raise ValueError('vision architecture identity')
 m.load_state_dict(p['state_dict'],strict=True);step=p['step']
 if step not in (0,250,500) or (require_final and step!=500) or len(p['curve'])!=step+1 or len(p['pair_schedule'])!=step:raise ValueError('fixed500 lineage')
 if not all(torch.isfinite(x).all() for x in m.state_dict().values()):raise ValueError('nonfinite model')
 opt=torch.optim.Adam(m.parameters(),lr=1e-4);opt.load_state_dict(p['optimizer'])
 if step and (len(opt.state)!=len(list(m.parameters())) or any(float(v['step'])!=step for v in opt.state.values())):raise ValueError('freshAdam actual parameter counters')
 return m.eval(),p
