"""Strictdb25/newodor250 binding; stdlib until parentflag backend."""
from pathlib import Path
import importlib.util,sys,hashlib,json,os
HERE=Path(__file__).resolve().parent;SCOPE=HERE.parent;PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype');OLDROOT=PROJECT/'experiments/neck-real-structure/20261113T045302Z';OLD=OLDROOT/'sensory_fusion_recovery_v1';BODY=OLDROOT/'sensory_body';VISROOT=PROJECT/'experiments/neck-real-structure/20261113T033622Z';VISBODY=VISROOT/'visual_body';ODORCAL=SCOPE/'odor_memory_repair';PORTS=PROJECT/'experiments/neck-real-structure/20261112T225331Z/dnp09_interface';VPOLICY=PROJECT/'experiments/visual-orientation/20261112T045316Z/landmark_opponent_v7/policy'
PARENT=OLD/'TRAIN/checkpoint_000400.pt';PARENT_SHA='db2510d46900644c7f8c7b556b4b892dad2a88e3e37ce13792811916a928d4c6';VISION=VPOLICY/'TRAIN/checkpoint_001000.pt';VISION_SHA='996980806ac107137d6d5f4743de207fb3350e84a8a145dc5f52b8807398bb87';NECK_SHA='9e88a04dd91aa6f9c1c1c68e0562325205e0f36f5cdd9ae41d1dd46c8171dc23';FORMAT='SOURCE_NECK_MULTISENSORY_AGGREGATION_REPAIR_V1';PREFIXES=('brain.','brain_out.','decoder.');MASK=list(range(42))+list(range(84,126))+list(range(168,195));NORMAL=('V_empty','V_center','V_left_pulse','V_right_pulse','V_left_to_right','V_right_to_left');LATEST=NORMAL+('O_left','O_right','O_noodor','F_left_agree','F_right_agree','F_left_DN_cut','O_left_odorcut','O_left_swap');AUX=('F_left_agree','F_right_agree','O_left','O_noodor');TRAIN=tuple('LATEST_'+n for n in LATEST)+tuple('PRIOR_'+n for n in NORMAL)+tuple('ZERO_'+n for n in AUX);DEV=('V_left_occluded','V_flip_fixedframe')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def verify(h):
 for p,d in h.items():
  if sha(p)!=d:raise ValueError('Sourcedrift '+p)
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def backend():
 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
  if os.environ.get(k)!='1':raise ValueError(k+'=1 required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});import torch,numpy as np;torch.set_num_threads(1)
 if any(k in sys.modules for k in ('physics','mujoco','student_wave','flygym')):raise ValueError('Offline forbidsphysicalbackend')
 return torch,np
def trainable(m):return [(k,v) for k,v in m.named_parameters() if v.requires_grad]
def frozen_tensors(m):return {k:v.detach().clone() for k,v in m.state_dict().items() if not k.startswith(PREFIXES)}
def check_frozen(m,ref):
 torch,np=backend()
 if any(not torch.equal(m.state_dict()[k],v) for k,v in ref.items()):raise ValueError('Frozenfeedback/norm/mask/reconstructionchanged')
def anchors(m):return {k:v.detach().clone() for k,v in trainable(m)}
def odor_binding():
 p=HERE/'PARENT_ODOR_BINDING.json';b=read(p)
 if b['format']!='ODOR_NEURAL_OFFSET_REPAIR_V1' or b['step']!=250 or Path(b['checkpoint']).resolve()!=ODORCAL/'TRAIN/checkpoint_000250.pt' or b['unknown_SHA_not_admitted'] is not False:raise ValueError('Actualnewodor250bindingneeded')
 if sha(b['checkpoint'])!=b['sha256']:raise ValueError('Boundodorbyteschanged')
 verify(b['source_hashes']);return b
def initialize():
 torch,np=backend();sentinel=object();previous={k:sys.modules.get(k,sentinel) for k in ('model','common')};paths=list(sys.path)
 try:
  io=module('_fusionagg_db25_io',OLD/'common.py');sys.modules['common']=io;arch=module('_fusionagg_db25_arch',OLD/'model.py');sys.modules['model']=arch;parent,p=io.load_checkpoint(PARENT,PARENT_SHA,final=True)
 finally:
  sys.path[:]=paths
  for k,v in previous.items():
   if v is sentinel:sys.modules.pop(k,None)
   else:sys.modules[k]=v
 arch=module('_fusionagg_current_arch',HERE/'model.py');m=arch.NeuralBridge(p['state_dict']['observation_mean'],p['state_dict']['observation_std']);m.load_state_dict(p['state_dict'],strict=True);m.new_odor_sha256=odor_binding()['sha256'];m.requires_grad_(False)
 for k,v in m.named_parameters():v.requires_grad_(k.startswith(PREFIXES))
 if len(trainable(m))!=10 or any(not torch.equal(v,p['state_dict'][k]) for k,v in m.state_dict().items()):raise ValueError('Exactdb25allstate/tenparametersrequired')
 return m,p
def load_sensory():
 backend();b=odor_binding();sentinel=object();previous={k:sys.modules.get(k,sentinel) for k in ('model','common','steering_behavior')};paths=list(sys.path)
 try:
  vm=module('_fusionagg_frozenvisual_arch',VPOLICY/'model.py');sys.modules['model']=vm;vi=module('_fusionagg_frozenvisual_io',VPOLICY/'common.py');vision,vp=vi.load_checked(VISION,VISION_SHA,require_final=True)
  oi=module('_fusionagg_newodor_io',ODORCAL/'common.py');sys.modules['common']=oi;odor,op=oi.load_checkpoint(Path(b['checkpoint']),b['sha256'],final=True)
 finally:
  sys.path[:]=paths
  for k,v in previous.items():
   if v is sentinel:sys.modules.pop(k,None)
   else:sys.modules[k]=v
 if vp['step']!=1000 or op['format_version']!='ODOR_NEURAL_OFFSET_REPAIR_V1' or op['step']!=250 or odor.identity()['recurrent_state']!=32:raise ValueError('Qualifiedsensoryidentities')
 vision.requires_grad_(False);odor.requires_grad_(False);return vision,odor,vp,op
def binding():
 p=PORTS/'SOURCE_BINDING.json';b=read(p)
 if sha(p)!=NECK_SHA:raise ValueError('Necksourceidentity')
 for row in b['source_files']:verify({row['path']:row['sha256']})
 return p,b,NECK_SHA
def real_queue_rollout(m,x,u,p,d):return module('_fusionagg_actualsourcequeue',OLD/'common.py').real_queue_rollout(m,x,u,p,d)
def sources():
 p,b,d=binding();ob=odor_binding();paths=[HERE/n for n in ('PREREG.json','common.py','model.py','prepare_odor_binding.py','data.py','preflight.py','train.py','verify_saved.py','evaluate.py','run.py','codecheck.py','PARENT_ODOR_BINDING.json')]+[PARENT,VISION,Path(ob['checkpoint']),OLD/'common.py',OLD/'model.py',OLD/'FROZEN.json',OLD/'TRAIN/VERIFY_000400.json',BODY/'SENSORY_BODY_SUMMARY.json',BODY/'PARENT_READY_REVIEW.json',BODY/'JOBS.json',BODY/'runtime.py',BODY/'audit.py',VISBODY/'NEURAL_VISUAL_BODY_SUMMARY.json',VISBODY/'PARENT_READY_REVIEW.json',VPOLICY/'model.py',VPOLICY/'common.py',ODORCAL/'common.py',ODORCAL/'model.py',p,PORTS/'source_transport.py',PORTS/'api.py']
 paths.extend(Path(v['path']) for v in b['source_files']);paths.extend(Path(v) for v in ob['source_hashes']);return sorted(set(paths),key=str)
def frozen():
 f=read(HERE/'FROZEN.json');verify(f['source_hashes']);verify(f['data_hashes']);binding();ob=odor_binding()
 if f['odor_sha256']!=ob['sha256']:raise ValueError('Newodorbindingdrift')
 return f
def load_checkpoint(path,digest,final=True):
 torch,np=backend();f=frozen();ob=odor_binding()
 if sha(path)!=digest:raise ValueError('FusioncheckpointSHA')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p['format_version']!=FORMAT or p['parent_sha256']!=PARENT_SHA or p['visual_sha256']!=VISION_SHA or p['odor_sha256']!=ob['sha256'] or p['source_binding_sha256']!=NECK_SHA or p['step'] not in (0,200,400) or (final and p['step']!=400) or p['effective_batch']!=24 or len(p['curve'])!=p['step']+1:raise ValueError('Strictrepair400lineage')
 verify(p['source_hashes']);verify(p['data_hashes']);m,parent=initialize();fixed=frozen_tensors(m);m.load_state_dict(p['state_dict'],strict=True);check_frozen(m,fixed)
 if m.identity()!=p['identity'] or any(not torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('Architecture/nonfinitetensor')
 opt=torch.optim.Adam([v for k,v in trainable(m)],lr=3e-4);opt.load_state_dict(p['optimizer'])
 if p['step'] and (len(opt.state)!=10 or any(float(v['step'])!=p['step'] for v in opt.state.values())):raise ValueError('TenAdam400counters')
 return m.eval(),p
