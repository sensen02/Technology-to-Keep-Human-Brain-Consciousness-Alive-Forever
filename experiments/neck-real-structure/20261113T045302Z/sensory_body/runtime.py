"""ActualRGB + physical head odor through source-neck fusion; stdlib until parentflag."""
from pathlib import Path
import importlib.util,sys,types,json,hashlib,os
ROOT=Path(__file__).resolve().parent;SCOPE=ROOT.parent;PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype');PREV=PROJECT/'experiments/neck-real-structure/20261113T033622Z/visual_body';CAL=SCOPE/'sensory_fusion_recovery_v1';ODOR_DIR=PROJECT/'experiments/food-approach/20261112T001916Z';ODOR=ODOR_DIR/'TRAIN/checkpoint_000500.pt';ODOR_SHA='66f423df4f00d801b2f0a194f539baa63ce84973765694e3d3111fbeb62aa034'
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
old=module('_fusion_body_previous_visual_runtime',PREV/'runtime.py');g=dict(old.__dict__)
for name,value in list(g.items()):
 if isinstance(value,types.FunctionType):g[name]=types.FunctionType(value.__code__,g,value.__name__,value.__defaults__,value.__closure__)
g.update(ROOT=ROOT);g['_g']=g
for n in ('read','sha','save','jsonable','verify','cpu1','pin','project','arrays','stable','camera_runtime','actor_observation','transport_module'):globals()[n]=g[n]
VIS=old.VIS;V7=old.V7;POLICY=old.POLICY;VISION=old.VISION;VISION_SHA=old.VISION_SHA;BASE=old.BASE;MOTOR_DIR=old.MOTOR_DIR;MOTOR=old.MOTOR;MOTOR_SHA=old.MOTOR_SHA;PORTS=old.PORTS;BINDING=old.BINDING;BINDING_SHA=old.BINDING_SHA;NEUTRAL=old.NEUTRAL;PEAK_SOURCE=old.PEAK_SOURCE
BRIDGE=CAL/'TRAIN/checkpoint_000400.pt';FORMAT='SOURCE_NECK_VISUAL_OLFACTORY_FUSION_RECOVERY_V1';KIND='SOURCE_NECK_MULTISENSORY_BODY_V1';SUMMARY_KIND='SOURCE_NECK_MULTISENSORY_BODY_SUMMARY_V1'
V_IDS=('V_empty','V_center','V_left_pulse','V_right_pulse','V_left_to_right','V_right_to_left');O_IDS=('O_left','O_right','O_noodor');F_IDS=('F_left_agree','F_right_agree','F_left_DN_cut');SENSOR_IDS=('O_left_odorcut','O_left_swap');IDS=V_IDS+O_IDS+F_IDS+SENSOR_IDS
HEAD_AUDIT=PROJECT/'experiments/food-approach/20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json'
def camera_condition(name):
 if name in V_IDS:return name
 if name in ('F_left_agree','F_left_DN_cut'):return 'V_left_pulse'
 if name=='F_right_agree':return 'V_right_pulse'
 if name in O_IDS+SENSOR_IDS:return 'V_empty'
 raise ValueError('condition')
def load_policies():
 if any(k in sys.modules for k in ('physics','mujoco','student_wave','flygym')):raise ValueError('AllstrictNNloadbeforephysicalbackend')
 binding=read(ROOT/'INPUT_BINDINGS.json');sentinel=object();previous={k:sys.modules.get(k,sentinel) for k in ('model','common','steering_behavior')};paths=list(sys.path)
 try:
  ci=module('_fusion_body_calibration_io',CAL/'common.py');sys.modules['common']=ci;visual,odor,vp,op=ci.load_sensory();cm=module('_fusion_body_calibration_arch',CAL/'model.py');sys.modules['model']=cm;bridge,bp=ci.load_checkpoint(BRIDGE,binding['brain_sha256'],final=True)
  mm=module('_fusion_body_motor_arch',MOTOR_DIR/'model.py');sys.modules['model']=mm;sys.path.insert(0,str(MOTOR_DIR));mi=module('_fusion_body_motor_io',MOTOR_DIR/'model_io.py');motor,mp=mi.load_checked(MOTOR,MOTOR_SHA,require_final=True)
 finally:
  sys.path[:]=paths
  for k,v in previous.items():
   if v is sentinel:sys.modules.pop(k,None)
   else:sys.modules[k]=v
 if any(sys.modules.get(k,sentinel) is not v for k,v in previous.items()):raise ValueError('model/common/steeringaliasesrestore')
 if bp['format_version']!=FORMAT or bp['step']!=400 or visual.identity()['hidden']!=32 or odor.identity()['recurrent_state']!=32 or bridge.brain.input_size!=3 or bridge.brain.hidden_size!=16 or motor.identity()['state_dims']!=[128,64,64]:raise ValueError('Strict400/336memoryABI')
 for m in (visual,odor,bridge,motor):m.requires_grad_(False)
 return visual,odor,bridge,motor,vp,op,bp,mp
def flat(motor,visual,odor,brain):
 import torch
 return torch.cat((*motor,visual,odor,brain),-1)
def sensors(headpos,headrot):
 import numpy as np
 local=np.asarray(read(HEAD_AUDIT)['sensor_offsets']['local_xyz']);expected=np.asarray([[.571317656,.459157176,0],[.571317656,-.459157176,0]])
 if local.shape!=(2,3) or not np.allclose(local,expected,atol=1e-8,rtol=0):raise ValueError('Sourcequalifiedactualheadproxyoffsets')
 return np.asarray(headpos)+local@np.asarray(headrot).T
class OdorField:
 def __init__(self,condition):
  import numpy as np
  if condition not in IDS:raise ValueError('condition')
  self.condition=condition;self.center=None;self.filter=module('_fusion_exact_original_clampfilter',PEAK_SOURCE).ClampFilter();self.last_step=None
 def snapshot(self):return dict(center=None if self.center is None else self.center.tolist(),state=self.filter.state.tolist(),pending=self.filter.pending.tolist(),last_step=self.last_step)
 def anchor(self,headpos,headrot):
  import numpy as np
  if self.center is not None:raise ValueError('Sourceanchoronce')
  sites=sensors(headpos,headrot);ex=np.asarray(headrot)[:2,0].copy();ex/=np.linalg.norm(ex);ey=np.array([-ex[1],ex[0]]);sign=-1 if self.condition in ('O_right','F_right_agree') else 1;self.center=sites.mean(0)[:2]+.6*ex+.6*np.tan(sign*.3)*ey
 def concentration(self,sites,ms):
  import numpy as np
  if self.center is None or self.condition in V_IDS or self.condition=='O_noodor':return np.zeros(2)
  u=np.clip((ms*.001-5.8-.6)/.2,0.,1.);on=u*u*(3-2*u);delta=(np.asarray(sites)[:,:2]-self.center)/.5;return 1.25*on*np.exp(-.5*np.sum(delta*delta,axis=1))
 def sample(self,headpos,headrot,ms):
  import numpy as np
  if self.last_step is not None and ms!=self.last_step+5:raise ValueError('Contiguous5msodorclock')
  self.last_step=ms;sites=sensors(headpos,headrot);C=self.concentration(sites,ms);sensory=self.filter.advance(C);used=sensory['receptors'].copy();actor=used.copy();cut=self.condition=='O_left_odorcut' and ms>=7400;swap=self.condition=='O_left_swap' and ms>=5800
  if cut:actor[:]=0
  if swap:actor=actor[::-1].copy()
  return dict(sensor_xyz_now=sites,source_anchor_valid=self.center is not None,odor_source_xy=np.zeros(2) if self.center is None else self.center.copy(),concentration=C,filter_before=sensory['filter_before'],filter_after=sensory['filter_after'],pending_before=sensory['pending_before'],pending_after=sensory['pending_after'],receptors_used=used,actor_receptors=actor,odor_actor_cut=cut,odor_actor_swap=swap,odor_capture_step=ms-5,odor_consume_step=ms)
class BrainNeckLoop:
 def __init__(self,bridge,condition):
  import torch
  self.bridge=bridge;self.condition=condition;self.brain=torch.zeros((1,16),dtype=torch.float64);self.held={'DN':torch.zeros((1,2),dtype=torch.float64),'AN':torch.zeros((1,1),dtype=torch.float64)};self.api,tm=transport_module();self.transport=tm.SourceTransport(BINDING,BINDING_SHA);self.packets={};self.cut=False;self.consumption=[]
 def snapshot(self):return dict(step=self.transport.step,brain=self.brain.tolist(),held={k:v.tolist() for k,v in self.held.items()},packets=self.packets,cut=self.cut,queue=self.transport._queue,journal_length=len(self.transport.journal()))
 def decision(self,x,visual_raw,odor_raw,ms):
  import torch,numpy as np
  if ms!=self.transport.step or ms%5:raise ValueError('Sourcebrainclock')
  if self.condition=='F_left_DN_cut' and ms>=7400 and not self.cut:self.transport.set_cut('DN',True);self.held['DN'].zero_();self.packets.pop('DN',None);self.cut=True;self.consumption.append(dict(event='engineering_DN_receiver_silence',step=ms,no_state_reset=True))
  used={}
  for direction in ('AN','DN'):
   p=self.packets.get(direction)
   if p is None:used[direction]=(-1,-1)
   else:self.api.validate_delivery(dict(p,consume_step=ms),self.transport.ports[direction],direction,self.transport.digest,ms);self.consumption.append(dict(event='consume',**dict(p,consume_step=ms)));used[direction]=(p['capture_step'],p['release_step'])
  before=self.brain[0].numpy().copy();anheld=self.held['AN'].clone();dnheld=self.held['DN'].clone();body=torch.tensor(np.asarray(x)[self.bridge.allowed_raw_indices.numpy()][None],dtype=torch.float64)
  with torch.no_grad():
   an=self.bridge.feedback(self.bridge.normalize(body))
   if ms<5800:dn=torch.zeros((1,2),dtype=torch.float64)
   else:
    dn,self.brain=self.bridge.brain_step(anheld,torch.cat((visual_raw.reshape(1,1),odor_raw.reshape(1,1)),-1),self.brain)
   shadow=self.bridge.decoder(dnheld);turn=shadow[:,0] if ms>=5800 else torch.zeros(1,dtype=torch.float64);command=project(turn)
  for direction,value in (('DN',dn),('AN',an)):
   if not (self.cut and direction=='DN'):self.transport.send({'from':'brain' if direction=='DN' else 'other','to':'neck','direction':direction,'root_ids':self.transport.ports[direction],'values':value[0].tolist(),'capture_step':ms,'source_bindings_sha256':self.transport.digest})
  return dict(brain_hidden_before=before,brain_hidden_after=self.brain[0].numpy().copy(),visual_brain_input=float(visual_raw[0]),odor_brain_input=float(odor_raw[0]),AN_encoded=an[0].numpy().copy(),DN_encoded=dn[0].numpy().copy(),AN_held=anheld[0].numpy().copy(),DN_held=dnheld[0].numpy().copy(),AN_capture_step=used['AN'][0],AN_release_step=used['AN'][1],DN_capture_step=used['DN'][0],DN_release_step=used['DN'][1],DN_cut=self.cut,shadow_decoder_turn=float(shadow[0,0]),raw_turn=float(turn[0]),command=command[0].numpy().copy())
 def advance(self):
  import torch
  for p in self.transport.advance():self.held[p['direction']]=torch.tensor([p['values']],dtype=torch.float64);self.packets[p['direction']]=dict(p)
 def finalize(self):
  proof=self.transport.verify_final()
  if proof['actual_steps']!=15400:raise ValueError('Sourcequeuefull15400')
  return dict(queue_final=proof,queue_journal=self.transport.journal(),consumption_events=self.consumption)
