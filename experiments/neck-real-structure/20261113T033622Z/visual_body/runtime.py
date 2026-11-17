"""Actual released RGB brain-side source-neck-body runtime; no heavy imports at module load."""
from pathlib import Path
import json,hashlib,sys,importlib.util,os,types
ROOT=Path(__file__).resolve().parent;SCOPE=ROOT.parent;PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype')
VIS=PROJECT/'experiments/visual-orientation/20261112T045316Z';V7=VIS/'landmark_opponent_v7';POLICY=V7/'policy';CAL=SCOPE/'visual_bridge_calibration'
BASE=PROJECT/'experiments/banc-walking-recovery/20261110T152714Z';MOTOR_DIR=PROJECT/'experiments/food-approach/20261111T161007Z/combined_turning';MOTOR=MOTOR_DIR/'COMBINED.pt';MOTOR_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
VISION=POLICY/'TRAIN/checkpoint_001000.pt';VISION_SHA='996980806ac107137d6d5f4743de207fb3350e84a8a145dc5f52b8807398bb87';BRIDGE=CAL/'TRAIN/checkpoint_000300.pt';FORMAT='SOURCE_NECK_VISUAL_SCALAR_CALIBRATION_V1'
PORTS=PROJECT/'experiments/neck-real-structure/20261112T225331Z/dnp09_interface';BINDING=PORTS/'SOURCE_BINDING.json';BINDING_SHA='9e88a04dd91aa6f9c1c1c68e0562325205e0f36f5cdd9ae41d1dd46c8171dc23'
OLD_BODY=PROJECT/'experiments/neck-real-structure/20261113T000915Z/body_pilot_v2';NEUTRAL=MOTOR_DIR/'EVAL_neutral';PEAK_SOURCE=PROJECT/'experiments/food-approach/20261111T224234Z/runtime.py'
NORMAL_IDS=('V_empty','V_center','V_left_pulse','V_right_pulse','V_left_to_right','V_right_to_left');CONTROL_IDS=('V_left_occluded','V_flip_fixedframe');CUT_IDS=('V_left_DN_cut','V_left_AN_cut');IDS=NORMAL_IDS+CONTROL_IDS+CUT_IDS
KIND='ACTUAL_RGB_SOURCE_NECK_VISUAL_BODY_V1';SUMMARY_KIND='ACTUAL_RGB_SOURCE_NECK_VISUAL_BODY_SUMMARY_V1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def jsonable(v):
 if isinstance(v,dict):return {str(k):jsonable(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [jsonable(x) for x in v]
 if type(v).__module__.startswith('numpy'):return v.tolist() if hasattr(v,'tolist') else v.item()
 if isinstance(v,Path):return str(v)
 return v
def save(p,v):
 with Path(p).open('x') as f:json.dump(jsonable(v),f,indent=2,allow_nan=False);f.write('\n')
def verify():
 f=read(ROOT/'FROZEN.json')
 for p,h in f['source_hashes'].items():
  if sha(p)!=h:raise ValueError('Frozen source drift '+p)
 return f
def cpu1():
 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):
  if os.environ.get(k)!='1':raise ValueError(k+'=1 required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});import torch;torch.set_num_threads(1)
def load_policies():
 if any(k in sys.modules for k in ('physics','mujoco','student_wave','flygym')):raise ValueError('Load strict three neural models before physics')
 b=read(ROOT/'INPUT_BINDINGS.json');sentinel=object();previous={n:sys.modules.get(n,sentinel) for n in ('model','common')};paths=list(sys.path)
 try:
  vm=module('_neckvision_frozen_visual_arch',POLICY/'model.py');sys.modules['model']=vm;vio=module('_neckvision_visual_strict_io',POLICY/'common.py');visual,vp=vio.load_checked(VISION,VISION_SHA,require_final=True)
  cio=module('_neckvision_final300_common',CAL/'common.py');sys.modules['common']=cio;cm=module('_neckvision_bridge_arch',CAL/'model.py');sys.modules['model']=cm
  bridge,bp=cio.load_checkpoint(BRIDGE,b['brain_sha256'],final=True)
  if bp['format_version']!=FORMAT or bp['step']!=300:raise ValueError('Actual final300 calibration required')
  mm=module('_neckvision_frozen_motor_arch',MOTOR_DIR/'model.py');sys.modules['model']=mm;sys.path.insert(0,str(MOTOR_DIR));mio=module('_neckvision_motor_strict_io',MOTOR_DIR/'model_io.py');motor,mp=mio.load_checked(MOTOR,MOTOR_SHA,require_final=True)
 finally:
  sys.path[:]=paths
  for n,v in previous.items():
   if v is sentinel:sys.modules.pop(n,None)
   else:sys.modules[n]=v
 if any(sys.modules.get(n,sentinel) is not v for n,v in previous.items()):raise ValueError('Restore model/common aliases')
 if visual.identity()['hidden']!=32 or visual.identity()['ordered_feature']!=16 or motor.identity()['state_dims']!=[128,64,64] or bridge.identity()['brain_hidden']!=16:raise ValueError('304 state ABI required')
 for m in (visual,bridge,motor):m.requires_grad_(False)
 return visual,bridge,motor,vp,bp,mp
def pin():
 sys.path.insert(0,str(BASE));import student_wave as sw,identity
 if Path(sw.__file__).resolve()!=BASE/'student_wave.py':raise ValueError('Frozen walking module collision')
 return sw,identity
def project(turn):
 import torch
 t=turn.reshape(-1,1);return torch.cat((torch.relu(t)/3.,t),-1)
def flat(motor,vision,brain):
 import torch
 return torch.cat((*motor,vision,brain),-1)
def arrays(p):
 import numpy as np
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def stable(auto,cycles,native):return bool(all(auto['gates'][k] for k in ('support','nonfoot','com','forward')) and cycles['all_legs_completed'] and native['post_nonfoot_peak_native']==0 and native['post_native_steps']==768000 and native['extra_force_max']==0)
def camera_condition(condition):
 if condition not in IDS:raise ValueError('Condition')
 return 'V_left_pulse' if condition in CUT_IDS else condition
def camera_runtime():
 old=module('_neckvision_frozen_far_camera_runtime',V7/'body/runtime.py');g=dict(old.__dict__);g.update(ROOT=ROOT)
 for n,v in list(g.items()):
  if isinstance(v,types.FunctionType):g[n]=types.FunctionType(v.__code__,g,v.__name__,v.__defaults__,v.__closure__)
 for name in ('_OpticalProxy','CameraStream'):
  cls=getattr(old,name);attrs={k:(types.FunctionType(v.__code__,g,v.__name__,v.__defaults__,v.__closure__) if isinstance(v,types.FunctionType) else v) for k,v in cls.__dict__.items() if k not in ('__dict__','__weakref__')};attrs['__module__']=__name__;g[name]=type(name,cls.__bases__,attrs)
 g['_g']=g;return g

def actor_observation(stream,time_s):
 # Original actor_query assumes >=1s; preserve contiguous camera queries fromt0.
 if time_s<.020-1e-10:
  raw=stream.queue.query(time_s)
  if raw['valid']:raise ValueError('First20ms unexpectedly alreadyreleased')
  return dict(raw,raw_capture_id=-1,raw_capture_time_s=-1.,raw_release_time_s=-1.,actor_frame_frozen=False)
 return stream.actor_query(time_s)

def transport_module():
 previous=sys.modules.get('api');present='api' in sys.modules;paths=list(sys.path)
 try:
  sys.path.insert(0,str(PORTS));api=module('_neckvision_source_api',PORTS/'api.py');sys.modules['api']=api;tm=module('_neckvision_source_transport',PORTS/'source_transport.py')
 finally:
  sys.path[:]=paths
  if present:sys.modules['api']=previous
  else:sys.modules.pop('api',None)
 return api,tm

class BrainNeckLoop:
 """Only source-ordered released AN and current neural visual scalar enter brain16."""
 def __init__(self,bridge,condition):
  import torch
  if condition not in IDS:raise ValueError('Condition')
  self.bridge=bridge;self.condition=condition;self.brain=torch.zeros((1,16),dtype=torch.float64);self.held={'DN':torch.zeros((1,2),dtype=torch.float64),'AN':torch.zeros((1,1),dtype=torch.float64)};self.api,tm=transport_module();self.transport=tm.SourceTransport(BINDING,BINDING_SHA);self.packets={};self.cut={'DN':False,'AN':False};self.cut_applied=False;self.consumption=[]
 def snapshot(self):return dict(step=self.transport.step,brain=self.brain.tolist(),held={k:v.tolist() for k,v in self.held.items()},packets=self.packets,cut=self.cut,queue=self.transport._queue,journal_length=len(self.transport.journal()))
 def decision(self,x,visual_raw,ms):
  import torch,numpy as np
  if self.transport.step!=ms or ms%5:raise ValueError('Source/neural clock drift')
  if not self.cut_applied and ms>=7400 and self.condition in CUT_IDS:
   d='DN' if self.condition=='V_left_DN_cut' else 'AN';self.transport.set_cut(d,True);self.held[d]=torch.zeros_like(self.held[d]);self.packets.pop(d,None);self.cut[d]=True;self.cut_applied=True;self.consumption.append(dict(event='engineering_port_silence_on_cut',direction=d,consume_step=ms,no_brain_reset=True,no_visual_reset=True,no_motor_reset=True))
  used={}
  for d in ('DN','AN'):
   p=self.packets.get(d)
   if p is not None:
    actual=dict(p,consume_step=ms);self.api.validate_delivery(actual,self.transport.ports[d],d,self.transport.digest,ms);self.consumption.append(dict(event='consume',**actual));used[d]=(p['capture_step'],p['release_step'])
   else:used[d]=(-1,-1)
  body=torch.tensor(np.asarray(x)[self.bridge.allowed_raw_indices.numpy()][None],dtype=torch.float64);before=self.brain[0].numpy().copy();ad=self.held['AN'].clone();dd=self.held['DN'].clone()
  with torch.no_grad():
   an=self.bridge.feedback(self.bridge.normalize(body))
   if ms<5800:dn=torch.zeros((1,2),dtype=torch.float64)
   else:dn,self.brain=self.bridge.brain_step(ad,visual_raw.reshape(1,1),self.brain)
   decoded=self.bridge.decoder(dd);turn=decoded[:,0] if ms>=5800 else torch.zeros(1,dtype=torch.float64);command=project(turn)
  for d,a in (('DN',dn),('AN',an)):
   if not self.cut[d]:self.transport.send({'from':'brain' if d=='DN' else 'other','to':'neck','direction':d,'root_ids':self.transport.ports[d],'values':a[0].tolist(),'capture_step':ms,'source_bindings_sha256':self.transport.digest})
  return dict(brain_hidden_before=before,brain_hidden_after=self.brain[0].numpy().copy(),visual_brain_input=float(visual_raw[0]),AN_encoded=an[0].numpy().copy(),DN_encoded=dn[0].numpy().copy(),AN_held=ad[0].numpy().copy(),DN_held=dd[0].numpy().copy(),AN_capture_step=used['AN'][0],AN_release_step=used['AN'][1],DN_capture_step=used['DN'][0],DN_release_step=used['DN'][1],AN_cut=self.cut['AN'],DN_cut=self.cut['DN'],shadow_decoder_turn=float(decoded[0,0]),raw_turn=float(turn[0]),command=command[0].numpy().copy())
 def advance(self):
  import torch
  for p in self.transport.advance():self.held[p['direction']]=torch.tensor([p['values']],dtype=torch.float64);self.packets[p['direction']]=dict(p)
 def finalize(self):
  final=self.transport.verify_final()
  if final['actual_steps']!=15400:raise ValueError('Full source queue duration')
  return dict(queue_final=final,queue_journal=self.transport.journal(),consumption_events=self.consumption)
