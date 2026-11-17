"""Actual brain/neck/body identities and source queue orchestration; no heavy import by default."""
from pathlib import Path
import json,hashlib,sys,importlib.util,os
ROOT=Path(__file__).resolve().parent;SCOPE=ROOT.parent;PROJECT=SCOPE.parents[2]
CAL=SCOPE/'neural_calibration';PORTS=SCOPE/'dnp09_interface'
BASE=PROJECT/'experiments/banc-walking-recovery/20261110T152714Z'
MOTOR_DIR=PROJECT/'experiments/food-approach/20261111T161007Z/combined_turning';MOTOR=MOTOR_DIR/'COMBINED.pt';MOTOR_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
BRIDGE=CAL/'TRAIN/checkpoint_000200.pt';BRIDGE_SHA='eed1ad3cc61a1b59086ff2324175f6bf3b5da5aa197165e6c4b009c17027c95e'
BINDING=PORTS/'SOURCE_BINDING.json';BINDING_SHA='9e88a04dd91aa6f9c1c1c68e0562325205e0f36f5cdd9ae41d1dd46c8171dc23'
PEAK_SOURCE=PROJECT/'experiments/food-approach/20261111T224234Z/runtime.py';NEUTRAL=MOTOR_DIR/'EVAL_neutral'
IDS=('P_empty','P_left_pulse','P_right_pulse','P_left_DN_cut','P_left_AN_cut','P_left_cuezero')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def jsonable(v):
 if isinstance(v,dict):return {k:jsonable(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [jsonable(x) for x in v]
 if type(v).__module__.startswith('numpy'):return v.tolist() if hasattr(v,'tolist') else v.item()
 if isinstance(v,Path):return str(v)
 return v
def save(p,v):
 with Path(p).open('x') as f:json.dump(jsonable(v),f,indent=2,allow_nan=False);f.write('\n')
def verify():
 f=read(ROOT/'FROZEN.json')
 for p,h in f['source_hashes'].items():
  if sha(p)!=h:raise ValueError('Frozen input drift '+p)
 return f
def cpu1():
 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
  if os.environ.get(k)!='1':raise ValueError(k+'=1 required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});import torch;torch.set_num_threads(1)
def load_policies():
 if any(k in sys.modules for k in ('physics','mujoco','student_wave','flygym')):raise ValueError('Strict neural load before physical backend required')
 sentinel=object();old=sys.modules.get('model',sentinel);prior=list(sys.path)
 try:
  cm=module('_bodypilot_bridge_architecture',CAL/'model.py');sys.modules['model']=cm;cio=module('_bodypilot_bridge_io',CAL/'common.py');bridge,bp=cio.load_checkpoint(BRIDGE,BRIDGE_SHA,final=True)
  mm=module('_bodypilot_motor_architecture',MOTOR_DIR/'model.py');sys.modules['model']=mm;sys.path.insert(0,str(MOTOR_DIR));mio=module('_bodypilot_motor_io',MOTOR_DIR/'model_io.py');motor,mp=mio.load_checked(MOTOR,MOTOR_SHA,require_final=True)
 finally:
  sys.path[:]=prior
  if old is sentinel:sys.modules.pop('model',None)
  else:sys.modules['model']=old
 if sys.modules.get('model',sentinel) is not old:raise ValueError('model alias not restored')
 if bridge.identity()['brain_hidden']!=16 or bridge.identity()['DN']!=2 or bridge.identity()['AN']!=1 or motor.identity()['state_dims']!=[128,64,64]:raise ValueError('brain16/motor256/source2DN1AN ABI')
 bridge.requires_grad_(False);motor.requires_grad_(False);return bridge,motor,bp,mp
def transport_module():
 sentinel=object();old=sys.modules.get('api',sentinel);prior=list(sys.path)
 try:
  sys.path.insert(0,str(PORTS));api=module('_bodypilot_source_api',PORTS/'api.py');sys.modules['api']=api;transport=module('_bodypilot_source_transport',PORTS/'source_transport.py')
 finally:
  sys.path[:]=prior
  if old is sentinel:sys.modules.pop('api',None)
  else:sys.modules['api']=old
 return api,transport

def pin():
 sys.path.insert(0,str(BASE));import student_wave as sw,identity
 if Path(sw.__file__).resolve()!=BASE/'student_wave.py':raise ValueError('Frozen walking import collision')
 return sw,identity
def flat(motorstate,brainstate):
 import torch
 return torch.cat((*motorstate,brainstate),-1)
def project(turn):
 import torch
 t=turn.reshape(-1,1);return torch.cat((torch.relu(t)/3.,t),-1)
def cue(condition,ms):
 e=ms-5800
 if condition in ('P_empty','P_left_cuezero') or e<600 or e>=3600:return 0.
 return -1. if condition=='P_right_pulse' else 1.
def stable(auto,cycles,native):return bool(all(auto['gates'][k] for k in ('support','nonfoot','com','forward')) and cycles['all_legs_completed'] and native['post_nonfoot_peak_native']==0 and native['post_native_steps']==768000 and native['extra_force_max']==0)
def arrays(p):
 import numpy as np
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

class BrainNeckLoop:
 """One current observation at 5ms, identity queues at1ms, no physics or motor access."""
 def __init__(self,model,condition):
  import torch
  if condition not in IDS:raise ValueError('condition')
  api,tm=transport_module();self.api=api;self.transport=tm.SourceTransport(BINDING,BINDING_SHA);self.model=model;self.condition=condition;self.brain=torch.zeros((1,16),dtype=torch.float64);self.held={'DN':torch.zeros((1,2),dtype=torch.float64),'AN':torch.zeros((1,1),dtype=torch.float64)};self.packets={};self.cut={'DN':False,'AN':False};self.cut_applied=False;self.consumption=[]
 def snapshot(self):return dict(step=self.transport.step,brain=self.brain.tolist(),held={k:v.tolist() for k,v in self.held.items()},packets=self.packets,cut=self.cut,queue=self.transport._queue,journal_length=len(self.transport.journal()))
 def decision(self,x,ms):
  import torch
  import numpy as np
  if self.transport.step!=ms or ms%5:raise ValueError('Neck/neural integer clocks drift')
  if not self.cut_applied and ms>=7400 and self.condition in ('P_left_DN_cut','P_left_AN_cut'):
   d='DN' if self.condition=='P_left_DN_cut' else 'AN';self.transport.set_cut(d,True);self.held[d]=torch.zeros_like(self.held[d]);self.packets.pop(d,None);self.cut[d]=True;self.cut_applied=True;self.consumption.append(dict(event='engineering_port_silence_on_cut',direction=d,consume_step=ms,no_brain_reset=True,no_motor_reset=True))
  used={}
  for d in ('DN','AN'):
   p=self.packets.get(d)
   if p is not None:
    actual=dict(p,consume_step=ms);self.api.validate_delivery(actual,self.transport.ports[d],d,self.transport.digest,ms);self.consumption.append(dict(event='consume',**actual));used[d]=(p['capture_step'],p['release_step'])
   else:used[d]=(-1,-1)
  body=torch.tensor(np.asarray(x)[self.model.allowed_raw_indices.numpy()][None],dtype=torch.float64);before=self.brain[0].numpy().copy();used_dn=self.held['DN'].clone();used_an=self.held['AN'].clone();instruction=cue(self.condition,ms)
  with torch.no_grad():
   an=self.model.feedback(self.model.normalize(body))
   if ms<5800:dn=torch.zeros((1,2),dtype=torch.float64)
   else:dn,self.brain=self.model.brain_step(used_an,torch.tensor([[instruction]],dtype=torch.float64),self.brain)
   decoded=self.model.decoder(used_dn);turn=decoded[:,0] if ms>=5800 else torch.zeros(1,dtype=torch.float64);command=project(turn)
  for d,activity in (('DN',dn),('AN',an)):
   if not self.cut[d]:self.transport.send(dict(**{'from':'brain' if d=='DN' else 'other','to':'neck'},direction=d,root_ids=self.transport.ports[d],values=activity[0].tolist(),capture_step=ms,source_bindings_sha256=self.transport.digest))
  return dict(cue=instruction,brain_hidden_before=before,brain_hidden_after=self.brain[0].numpy().copy(),AN_encoded=an[0].numpy().copy(),DN_encoded=dn[0].numpy().copy(),AN_held=used_an[0].numpy().copy(),DN_held=used_dn[0].numpy().copy(),AN_capture_step=used['AN'][0],AN_release_step=used['AN'][1],DN_capture_step=used['DN'][0],DN_release_step=used['DN'][1],AN_cut=self.cut['AN'],DN_cut=self.cut['DN'],shadow_decoder_turn=float(decoded[0,0]),raw_turn=float(turn[0]),command=command[0].numpy().copy())
 def advance(self):
  import torch
  for p in self.transport.advance():self.held[p['direction']]=torch.tensor([p['values']],dtype=torch.float64);self.packets[p['direction']]=dict(p)
 def finalize(self):
  final=self.transport.verify_final()
  if final['actual_steps']!=15400:raise ValueError('Full source queue duration required')
  return dict(queue_final=final,queue_journal=self.transport.journal(),consumption_events=self.consumption)
