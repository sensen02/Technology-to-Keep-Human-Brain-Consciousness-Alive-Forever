"""Stdlib-only lineage helpers. Parent explicitflag must precede numerical backend imports."""
from pathlib import Path
import json,hashlib,sys,importlib.util,os
HERE=Path(__file__).resolve().parent;SCOPE=HERE.parent;PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype')
VISROOT=PROJECT/'experiments/neck-real-structure/20261113T033622Z';OLD=VISROOT/'visual_bridge_calibration';VISBODY=VISROOT/'visual_body';FOOD=PROJECT/'experiments/food-approach';ODORCAL=FOOD/'20261112T001916Z';B=ODORCAL/'physical_B';C=ODORCAL/'spatial_C';AUXBODY=PROJECT/'experiments/neck-real-structure/20261112T225331Z/body_pilot';PORTS=AUXBODY.parent/'dnp09_interface'
VIS=PROJECT/'experiments/visual-orientation/20261112T045316Z';VPOLICY=VIS/'landmark_opponent_v7/policy'
PARENT=OLD/'TRAIN/checkpoint_000300.pt';PARENT_SHA='2123eeadaffbdcf722e361ad1da26b1816833279918e9231773839ed1cfb924c';VISION=VPOLICY/'TRAIN/checkpoint_001000.pt';VISION_SHA='996980806ac107137d6d5f4743de207fb3350e84a8a145dc5f52b8807398bb87';ODOR=ODORCAL/'TRAIN/checkpoint_000500.pt';ODOR_SHA='66f423df4f00d801b2f0a194f539baa63ce84973765694e3d3111fbeb62aa034';NECK_SHA='9e88a04dd91aa6f9c1c1c68e0562325205e0f36f5cdd9ae41d1dd46c8171dc23'
FORMAT='SOURCE_NECK_VISUAL_OLFACTORY_FUSION_CALIBRATION_V1';MASK=list(range(42))+list(range(84,126))+list(range(168,195));PREFIXES=('brain.','brain_out.','decoder.')
NORMAL=('V_empty','V_center','V_left_pulse','V_right_pulse','V_left_to_right','V_right_to_left');BNAMES=('B_noodor','B_equal','B_left_pulse','B_right_pulse','B_left_to_right','B_right_to_left');CNAMES=('C_noodor','C_front','C_left_near','C_right_near','C_left_far','C_right_far');AUX=('P_empty','P_left_pulse','P_right_pulse');TRAIN=tuple('VIS_'+n for n in NORMAL)+BNAMES+CNAMES+tuple('PAIR_'+n for n in NORMAL)+AUX;DEV=('V_left_occluded','V_flip_fixedframe')+tuple('DEV_'+n for n in CNAMES)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def verify(h):
 for p,d in h.items():
  if sha(p)!=d:raise ValueError('Source drift '+p)
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def backend():
 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
  if os.environ.get(k)!='1':raise ValueError(k+'=1 required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});import torch,numpy as np;torch.set_num_threads(1)
 if any(k in sys.modules for k in ('physics','mujoco','student_wave','flygym')):raise ValueError('No physical backend allowed')
 return torch,np
def trainable(m):return [(k,p) for k,p in m.named_parameters() if p.requires_grad]
def frozen_tensors(m):return {k:v.detach().clone() for k,v in m.state_dict().items() if not k.startswith(PREFIXES)}
def tensor_hashes(m):return {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in m.state_dict().items()}
def check_frozen(m,ref):
 torch,np=backend()
 if any(not torch.equal(m.state_dict()[k],v) for k,v in ref.items()):raise ValueError('Frozen feedback/reconstruction/normalizer/mask drift')
def initialize():
 torch,np=backend();sentinel=object();previous={k:sys.modules.get(k,sentinel) for k in ('common','model')};paths=list(sys.path)
 try:
  io=module('_fusion_parent_visual_io',OLD/'common.py');sys.modules['common']=io;arch=module('_fusion_parent_visual_arch',OLD/'model.py');sys.modules['model']=arch;parent,p=io.load_checkpoint(PARENT,PARENT_SHA,final=True)
 finally:
  sys.path[:]=paths
  for k,v in previous.items():
   if v is sentinel:sys.modules.pop(k,None)
   else:sys.modules[k]=v
 arch=module('_fusion_new_brain_arch',HERE/'model.py');m=arch.NeuralBridge(p['state_dict']['observation_mean'],p['state_dict']['observation_std']);state={k:v.detach().clone() for k,v in p['state_dict'].items()};expanded=state['brain.weight_ih'].new_zeros((48,3));expanded[:,:2]=state['brain.weight_ih'];state['brain.weight_ih']=expanded;m.load_state_dict(state,strict=True);m.requires_grad_(False)
 for k,v in m.named_parameters():v.requires_grad_(k.startswith(PREFIXES))
 if len(trainable(m))!=10 or any(not torch.equal(v,state[k]) for k,v in m.state_dict().items()):raise ValueError('Exact parent column expansion andten tensors required')
 return m,p,parent
def anchors(m):return {k:v.detach().clone() for k,v in trainable(m)}
def load_sensory():
 backend();sentinel=object();previous={k:sys.modules.get(k,sentinel) for k in ('common','model')};paths=list(sys.path)
 try:
  vm=module('_fusion_frozen_visual_arch',VPOLICY/'model.py');sys.modules['model']=vm;vio=module('_fusion_frozen_visual_io',VPOLICY/'common.py');vision,vp=vio.load_checked(VISION,VISION_SHA,require_final=True)
  oi=module('_fusion_frozen_odor_io',ODORCAL/'common.py');odor,op=oi.load_checked(ODOR,ODOR_SHA,require_final=True)
 finally:
  sys.path[:]=paths
  for k,v in previous.items():
   if v is sentinel:sys.modules.pop(k,None)
   else:sys.modules[k]=v
 if vp['step']!=1000 or op['curve'][-1]['step']!=500 or op['format_version']!='ODOR_PAIRED_BATCH_RECOVERY_V1' or op['deployment_qualified'] is not False:raise ValueError('Exact frozen sensory identities')
 vision.requires_grad_(False);odor.requires_grad_(False);return vision,odor,vp,op

def binding():
 p=PORTS/'SOURCE_BINDING.json';b=read(p)
 if sha(p)!=NECK_SHA:raise ValueError('Neck binding identity')
 if [(v['direction'],v['root_id']) for v in b['ports']]!=[('DN','720575941433155799'),('DN','720575941566493282'),('AN','720575941351919408')]:raise ValueError('Original sourceportorder')
 for v in b['source_files']:verify({v['path']:v['sha256']})
 return p,b,NECK_SHA

def real_queue_rollout(m,x,u,bp,digest):
 import torch
 if x.shape[1]!=1 or u.shape[1:]!=(1,2):raise ValueError('Source queue batch1/two modalityvalues required')
 sentinel=object();prior=sys.modules.get('api',sentinel);paths=list(sys.path)
 try:
  sys.path.insert(0,str(PORTS));api=module('_fusion_source_api',PORTS/'api.py');sys.modules['api']=api;tm=module('_fusion_source_transport',PORTS/'source_transport.py');queue=tm.SourceTransport(bp,digest)
 finally:
  sys.path[:]=paths
  if prior is sentinel:sys.modules.pop('api',None)
  else:sys.modules['api']=prior
 h=x.new_zeros((1,16));held={'AN':x.new_zeros((1,1)),'DN':x.new_zeros((1,2))};packets={};rows=[[],[],[],[],[]];events=[]
 with torch.no_grad():
  for k in range(1920):
   if queue.step!=5*k:raise ValueError('Sourceclock drift')
   for d,p in packets.items():
    use=dict(p,consume_step=queue.step);api.validate_delivery(use,queue.ports[d],d,digest,queue.step);events.append(dict(direction=d,capture_step=p['capture_step'],queue_release_step=p['release_step'],neural_consume_step=queue.step))
   an=m.feedback(m.normalize(x[k]));dn,h=m.brain_step(held['AN'],u[k],h);turn=m.decoder(held['DN']);rec=m.reconstruction(an)
   for row,value in zip(rows,(turn,dn,an,rec,h)):row.append(value.clone())
   for d,value in (('DN',dn),('AN',an)):queue.send({'from':'brain' if d=='DN' else 'other','to':'neck','direction':d,'root_ids':queue.ports[d],'values':value[0].tolist(),'capture_step':queue.step,'source_bindings_sha256':digest})
   for _ in range(5):
    for p in queue.advance():held[p['direction']]=x.new_tensor([p['values']]);packets[p['direction']]=dict(p)
 proof=queue.verify_final();proof.update(neural_consumption_events=events,initial_delivery_zero=True,actual_queue_journal=queue.journal(),state336_modalities_not_transported=True)
 if proof['actual_steps']!=9600:raise ValueError('Full sourceclock required')
 return tuple(torch.stack(v) for v in rows),proof

def sources():
 p,b,d=binding();paths=[HERE/n for n in ('PREREG.json','common.py','model.py','data.py','preflight.py','train.py','verify_saved.py','evaluate.py','run.py','codecheck.py')]+[p,PARENT,VISION,ODOR,OLD/'common.py',OLD/'model.py',OLD/'FROZEN.json',OLD/'TRAIN/VERIFY_000300.json',VISBODY/'NEURAL_VISUAL_BODY_SUMMARY.json',VISBODY/'PARENT_READY_REVIEW.json',VISBODY/'runtime.py',VISBODY/'runner.py',VISBODY/'audit.py',VPOLICY/'model.py',VPOLICY/'common.py',ODORCAL/'common.py',ODORCAL/'FROZEN.json',ODORCAL/'TRAIN/VERIFY_000500.json',ODORCAL/'VALIDATION/RESULT.json',ODORCAL/'VALIDATION/VERIFY.json',PORTS/'source_transport.py',PORTS/'api.py',VISROOT/'olfactory_source_audit/OLFACTORY_SOURCE_REVIEW.json',SCOPE/'olfactory_design/OLFACTORY_INTEGRATION_DESIGN_REVIEW.json']
 paths.extend(Path(v['path']) for v in b['source_files']);return sorted(set(paths),key=str)
def frozen():
 f=read(HERE/'FROZEN.json');verify(f['source_hashes']);verify(f['data_hashes']);binding();return f

def load_checkpoint(path,digest,final=True):
 torch,np=backend();f=frozen()
 if sha(path)!=digest:raise ValueError('CheckpointSHA')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p['format_version']!=FORMAT or p['parent_sha256']!=PARENT_SHA or p['visual_sha256']!=VISION_SHA or p['odor_sha256']!=ODOR_SHA or p['source_binding_sha256']!=NECK_SHA or p['step'] not in (0,200,400) or (final and p['step']!=400) or len(p['curve'])!=p['step']+1 or p['effective_batch']!=27:raise ValueError('Strict fusionfinal400/lineage')
 verify(p['source_hashes']);verify(p['data_hashes']);m,parent,old=initialize();fixed=frozen_tensors(m);m.load_state_dict(p['state_dict'],strict=True);check_frozen(m,fixed)
 if p['identity']!=m.identity() or any(not torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('Nonfinite/architecture')
 opt=torch.optim.Adam([v for k,v in trainable(m)],lr=3e-4);opt.load_state_dict(p['optimizer'])
 if p['step'] and (len(opt.state)!=10 or any(float(v['step'])!=p['step'] for v in opt.state.values())):raise ValueError('Ten exact optimizer updatecounters')
 return m.eval(),p
