"""Stdlib-only lineage andpublicloader; numerical imports require explicit parent invocation."""
from pathlib import Path
import hashlib,json,importlib.util,sys,os
HERE=Path(__file__).resolve().parent;PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype');OLD=PROJECT/'experiments/food-approach/20261112T001916Z';PARENT=OLD/'TRAIN/checkpoint_000500.pt';PARENT_SHA='66f423df4f00d801b2f0a194f539baa63ce84973765694e3d3111fbeb62aa034';ORIGINAL=PROJECT/'experiments/food-approach/20261110T191449Z';SOURCE=ORIGINAL/'steering_behavior.py';PRIOR=PROJECT/'experiments/neck-real-structure/20261113T045302Z';BODY=PRIOR/'sensory_body';DIAGNOSTIC=PRIOR/'failure_diagnostic/RESULT.json'
FORMAT='ODOR_NEURAL_OFFSET_REPAIR_V1';B_NAMES=('B_noodor','B_equal','B_left_pulse','B_right_pulse','B_left_to_right','B_right_to_left');CURRENT=('O_left','O_right','O_noodor','F_left_agree','F_right_agree','O_left_odorcut','O_left_swap');AUG=('AUG_cut_0p9','AUG_cut_2p6','AUG_cut_4p8');CUTS=(.9,2.6,4.8);TRAIN=B_NAMES+CURRENT+AUG;DEV=TRAIN

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def verify(h):
 for p,d in h.items():
  if sha(p)!=d:raise ValueError('Source drift '+p)
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def backend():
 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
  if os.environ.get(k)!='1':raise ValueError(k+'=1required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});import torch,numpy as np;torch.set_num_threads(1)
 if any(k in sys.modules for k in ('mujoco','physics','student_wave','flygym')):raise ValueError('No physical backend permitted')
 return torch,np

def initialize():
 torch,np=backend();sentinel=object();prior={k:sys.modules.get(k,sentinel) for k in ('common','model','steering_behavior')};paths=list(sys.path)
 try:
  io=module('_odor_memory_exact_parent66f_io',OLD/'common.py');m,p=io.load_checked(PARENT,PARENT_SHA,require_final=True)
 finally:
  sys.path[:]=paths
  for k,v in prior.items():
   if v is sentinel:sys.modules.pop(k,None)
   else:sys.modules[k]=v
 if sha(PARENT)!=PARENT_SHA or p['curve'][-1]['step']!=500 or p['format_version']!='ODOR_PAIRED_BATCH_RECOVERY_V1' or p['deployment_qualified'] is not False:raise ValueError('Strict original66f500 engineeringlineage')
 if any(not torch.equal(v,p['state_dict'][k]) for k,v in m.state_dict().items()) or len(list(m.parameters()))!=8:raise ValueError('Allparent66fweights exact/eightparameters')
 m.requires_grad_(True);return m,p

def parameters(m):return list(m.parameters())
def anchors(m):return [p.detach().clone() for p in m.parameters()]
def hashes_of(m):return {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in m.state_dict().items()}
def sources():
 paths=[HERE/n for n in ('PREREG.json','common.py','model.py','data.py','preflight.py','train.py','verify_saved.py','evaluate.py','run.py','codecheck.py','load_api.py')]+[SOURCE,PARENT,OLD/'common.py',OLD/'FROZEN.json',OLD/'TRAIN/VERIFY_000500.json',OLD/'VALIDATION/RESULT.json',OLD/'VALIDATION/VERIFY.json',DIAGNOSTIC,BODY/'runtime.py',BODY/'runner.py',BODY/'FROZEN.json',BODY/'JOBS.json']
 return sorted(set(paths),key=str)
def frozen():
 f=read(HERE/'FROZEN.json');verify(f['source_hashes']);verify(f['data_hashes']);return f

def load_checkpoint(path,digest,final=True):
 torch,np=backend();f=frozen()
 if sha(path)!=digest:raise ValueError('ActualodorrepaircheckpointSHA')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p.get('format_version')!=FORMAT or p.get('parent_sha256')!=PARENT_SHA or p.get('step') not in (0,125,250) or (final and p['step']!=250) or len(p.get('curve',[]))!=p.get('step',-1)+1 or p.get('effective_batch')!=16 or p.get('microbatch')!=8:raise ValueError('Strictodorrepairfixed250lineage')
 verify(p['source_hashes']);verify(p['data_hashes']);m,parent=initialize();m.load_state_dict(p['state_dict'],strict=True)
 if p['identity']!=m.identity() or m.identity()!=parent['identity'] or any(not torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('Exactsourceclassidentity/finiteweights')
 opt=torch.optim.Adam(m.parameters(),lr=1e-4);opt.load_state_dict(p['optimizer'])
 if p['step'] and (len(opt.state)!=8 or any(float(v['step'])!=p['step'] for v in opt.state.values())):raise ValueError('EightAdam250counters')
 return m.eval(),p
