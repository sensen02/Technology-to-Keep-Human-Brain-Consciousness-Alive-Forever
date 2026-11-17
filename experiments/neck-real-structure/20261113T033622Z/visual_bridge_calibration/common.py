"""Stdlib until parent explicitflag. Strict4153 bridge/99698vision/9e88source lineage."""
from pathlib import Path
import json,hashlib,sys,importlib.util,os
HERE=Path(__file__).resolve().parent;SCOPE=HERE.parent;PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype')
PREV=PROJECT/'experiments/neck-real-structure/20261113T000915Z';PARENT_CAL=PREV/'feedback_aggregation_v2';BODY=PROJECT/'experiments/neck-real-structure/20261112T225331Z/body_pilot';NEURO=BODY.parent/'neural_calibration';PORTS=BODY.parent/'dnp09_interface'
VIS=PROJECT/'experiments/visual-orientation/20261112T045316Z';VIS_POLICY=VIS/'landmark_opponent_v7/policy';VIS_BODY=VIS/'landmark_opponent_v7/body'
PARENT=PARENT_CAL/'TRAIN/checkpoint_000400.pt';PARENT_SHA='415304b7027dda17f359350052934670c1aa2bd44a0025b1dd0089d549997b27';VISION=VIS_POLICY/'TRAIN/checkpoint_001000.pt';VISION_SHA='996980806ac107137d6d5f4743de207fb3350e84a8a145dc5f52b8807398bb87';NECK_SHA='9e88a04dd91aa6f9c1c1c68e0562325205e0f36f5cdd9ae41d1dd46c8171dc23'
FORMAT='SOURCE_NECK_VISUAL_SCALAR_CALIBRATION_V1';NORMAL=('V_empty','V_center','V_left_pulse','V_right_pulse','V_left_to_right','V_right_to_left');AUX=('P_empty','P_left_pulse','P_right_pulse');TRAIN=NORMAL+AUX;DEV=('V_left_occluded','V_flip_fixedframe');MASK=list(range(42))+list(range(84,126))+list(range(168,195));PREFIXES=('brain.','brain_out.','decoder.')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def verify(hashes):
 for p,h in hashes.items():
  if sha(p)!=h:raise ValueError('Source drift '+str(p))
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def backend():
 for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
  if os.environ.get(key)!='1':raise ValueError(key+'=1 required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});import torch,numpy as np;torch.set_num_threads(1)
 if any(k in sys.modules for k in ('physics','mujoco','student_wave','flygym')):raise ValueError('No physical backend allowed')
 return torch,np
def trainable(m):return [(k,p) for k,p in m.named_parameters() if p.requires_grad]
def frozen_tensors(m):return {k:v.detach().clone() for k,v in m.state_dict().items() if not k.startswith(PREFIXES)}
def check_frozen(m,ref):
 torch,np=backend()
 if any(not torch.equal(m.state_dict()[k],v) for k,v in ref.items()):raise ValueError('Frozen feedback/reconstruction/mask/normalizer drift')
def initialize():
 torch,np=backend();sentinel=object();prior=sys.modules.get('model',sentinel)
 try:
  arch=module('_visual_neck_parent_original_arch',NEURO/'model.py');sys.modules['model']=arch;io=module('_visual_neck_parent4153_io',PARENT_CAL/'common.py');parent,p=io.load_checkpoint(PARENT,PARENT_SHA,final=True)
 finally:
  if prior is sentinel:sys.modules.pop('model',None)
  else:sys.modules['model']=prior
 current=module('_visual_neck_current_arch',HERE/'model.py');m=current.NeuralBridge(p['state_dict']['observation_mean'],p['state_dict']['observation_std']);m.load_state_dict(p['state_dict'],strict=True);m.requires_grad_(False)
 for k,param in m.named_parameters():param.requires_grad_(k.startswith(PREFIXES))
 if len(trainable(m))!=10 or any(not torch.equal(v,p['state_dict'][k]) for k,v in m.state_dict().items()):raise ValueError('Exact4153allstate and ten trainable tensors required')
 return m,p
def anchors(parent):return {k:v.detach().clone() for k,v in parent['state_dict'].items() if k.startswith(PREFIXES)}
def load_visual():
 backend();sentinel=object();prior=sys.modules.get('model',sentinel)
 try:
  arch=module('_visual_neck_frozen_v7_arch',VIS_POLICY/'model.py');sys.modules['model']=arch;io=module('_visual_neck_frozen_v7_io',VIS_POLICY/'common.py');m,p=io.load_checked(VISION,VISION_SHA,require_final=True)
 finally:
  if prior is sentinel:sys.modules.pop('model',None)
  else:sys.modules['model']=prior
 if p['step']!=1000 or m.identity()['format']!='VISION_FIXED_NEURAL_OPPONENT_ARCH_V7':raise ValueError('Actualfinal99698vision required')
 m.requires_grad_(False);return m,p
def binding():
 p=PORTS/'SOURCE_BINDING.json';b=read(p)
 if sha(p)!=NECK_SHA or [(x['direction'],x['root_id']) for x in b['ports']]!=[('DN','720575941433155799'),('DN','720575941566493282'),('AN','720575941351919408')]:raise ValueError('Exactsource2DN1AN ordering')
 for x in b['source_files']:verify({x['path']:x['sha256']})
 return p,b,NECK_SHA
def real_queue_rollout(m,x,u,bp,digest):
 io=module('_visual_neck_original_queue_io',NEURO/'common.py');return io.real_queue_rollout(m,x,u,bp,digest)
def sources():
 paths=[HERE/n for n in ('CALIBRATION_PREREG.json','common.py','model.py','data.py','preflight.py','train.py','verify_saved.py','evaluate.py','run.py','codecheck.py')]+[SCOPE/'visual_source_audit/NEURAL_VISUAL_SOURCE_REVIEW.json',PARENT,VISION,NEURO/'common.py',NEURO/'model.py',PARENT_CAL/'common.py',PARENT_CAL/'FROZEN.json',PARENT_CAL/'TRAIN/VERIFY_000400.json',VIS_POLICY/'model.py',VIS_POLICY/'common.py',VIS/'camera.py',PORTS/'source_transport.py',PORTS/'api.py']
 bp,b,d=binding();paths.append(bp);paths.extend(Path(x['path']) for x in b['source_files']);return sorted(set(paths),key=str)
def frozen():
 f=read(HERE/'FROZEN.json');verify(f['source_hashes']);verify(f['data_hashes']);binding();return f
def load_checkpoint(path,digest,final=True):
 torch,np=backend();f=frozen()
 if sha(path)!=digest:raise ValueError('ActualsavedcheckpointSHA')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p['format_version']!=FORMAT or p['parent_sha256']!=PARENT_SHA or p['visual_sha256']!=VISION_SHA or p['source_binding_sha256']!=NECK_SHA or p['step'] not in (0,150,300) or (final and p['step']!=300) or p['batch_size']!=9 or len(p['curve'])!=p['step']+1:raise ValueError('Strictfinal300 visual-source/parent/step/batch lineage')
 verify(p['source_hashes']);verify(p['data_hashes']);m,parent=initialize();ref=frozen_tensors(m);m.load_state_dict(p['state_dict'],strict=True);check_frozen(m,ref)
 if m.identity()!=p['identity'] or p['trainable_names']!=[k for k,v in trainable(m)] or any(not torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('Architecture/trainableorder/nonfinite tensor')
 opt=torch.optim.Adam([v for k,v in trainable(m)],lr=3e-4);opt.load_state_dict(p['optimizer'])
 if p['step'] and (len(opt.state)!=10 or any(float(v['step'])!=p['step'] for v in opt.state.values())):raise ValueError('TenAdam300 counters required')
 return m.eval(),p
