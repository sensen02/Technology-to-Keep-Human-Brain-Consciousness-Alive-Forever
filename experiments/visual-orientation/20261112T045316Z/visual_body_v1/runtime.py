"""Strict visual/motor identities, explicit module isolation, render-only fixedworld landmarks."""
from pathlib import Path
import importlib.util,sys,json,hashlib,os
ROOT=Path(__file__).resolve().parent;VIS=ROOT.parent;POLICY=VIS/'visual_policy_v1'
PROJECT=VIS.parents[2];EXP=PROJECT/'experiments'
BASE=EXP/'banc-walking-recovery/20261110T152714Z';MOTOR_DIR=EXP/'food-approach/20261111T161007Z/combined_turning';MOTOR=MOTOR_DIR/'COMBINED.pt';MOTOR_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
GEOMETRY=EXP/'food-approach/20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json';NEUTRAL=MOTOR_DIR/'EVAL_neutral';PEAK_SOURCE=EXP/'food-approach/20261111T224234Z/runtime.py'
IDS=('V_empty','V_center','V_left_pulse','V_right_pulse','V_left_to_right','V_right_to_left')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def jsonable(v):
 if isinstance(v,dict):return {str(k):jsonable(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [jsonable(x) for x in v]
 if type(v).__module__.startswith('numpy'):return v.tolist() if hasattr(v,'tolist') else v.item()
 if isinstance(v,Path):return str(v)
 return v
def save(p,v):
 with Path(p).open('x') as f:json.dump(jsonable(v),f,indent=2,allow_nan=False);f.write('\n')
def verify():
 f=read(ROOT/'FROZEN.json')
 for p,h in f['source_hashes'].items():
  if sha(p)!=h:raise ValueError('frozen visualbody inputdrift '+p)
 return f
def cpu1():
 import torch
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});torch.set_num_threads(1)
 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):
  if os.environ.get(k)!='1':raise ValueError(k+'=1 required')
def load_policies():
 if any(k in sys.modules for k in ('physics','mujoco','student_wave','flygym')):raise ValueError('strictvisual loader MUST precede physics')
 bindings=read(ROOT/'INPUT_BINDINGS.json');sentinel=object();prior=sys.modules.get('model',sentinel);priorpath=list(sys.path)
 try:
  vm=module('_online_visual_model',POLICY/'model.py');sys.modules['model']=vm
  io=module('_online_visual_strict_common',POLICY/'common.py');visual,vp=io.load_checked(bindings['visual_checkpoint'],bindings['visual_sha256'],require_final=True)
  mm=module('_online_combined_motor_model',MOTOR_DIR/'model.py');sys.modules['model']=mm;sys.path.insert(0,str(MOTOR_DIR))
  mio=module('_online_combined_strict_io',MOTOR_DIR/'model_io.py');motor,mp=mio.load_checked(MOTOR,MOTOR_SHA,require_final=True)
 finally:
  sys.path[:]=priorpath
  if prior is sentinel:sys.modules.pop('model',None)
  else:sys.modules['model']=prior
 if (prior is sentinel and 'model' in sys.modules) or (prior is not sentinel and sys.modules.get('model') is not prior):raise ValueError('basename model not restored')
 if visual.identity()['hidden']!=32 or motor.identity()['state_dims']!=[128,64,64]:raise ValueError('visual32+motor256 ABI')
 visual.requires_grad_(False);motor.requires_grad_(False);return visual,motor,vp,mp

def pin():
 sys.path.insert(0,str(BASE));import student_wave as sw,identity
 if Path(sw.__file__).resolve()!=BASE/'student_wave.py':raise ValueError('frozenwalking module collision')
 return sw,identity

def flatten(motorstate,visionstate):
 import torch
 return torch.cat((*motorstate,visionstate),dim=-1)
def project(turn):
 import torch
 t=turn.reshape(-1,1);return torch.cat((torch.relu(t)/3.,t),-1)
def frame_target(condition,elapsed):
 if elapsed<.6-1e-10 or condition=='V_empty':return 'off'
 if condition in ('V_center','V_left_pulse','V_right_pulse') and elapsed>=3.6-1e-10:return 'off'
 if condition=='V_center':return 'center'
 if condition=='V_left_pulse':return 'left'
 if condition=='V_right_pulse':return 'right'
 if condition=='V_left_to_right':return 'left' if elapsed<4.8-1e-10 else 'right'
 if condition=='V_right_to_left':return 'right' if elapsed<4.8-1e-10 else 'left'
 raise ValueError(condition)

def anchor_landmarks(headpos,headrot,camera,spec,audit):
 import numpy as np
 eyes=camera.eye_frames(headpos,headrot,audit,spec);mid=(eyes[0]['pos']+eyes[1]['pos'])/2;ex=np.asarray(headrot)[:,0].copy();ex[2]=0.;ex/=np.linalg.norm(ex);ey=np.array([-ex[1],ex[0],0.]);centers={}
 for name,s in (('left',1),('right',-1),('center',0)):
  p=mid+.8*ex+s*.4*ey;p[2]=1.2;centers[name]=p
 return centers
class CameraStream:
 def __init__(self,c,condition):
  import mujoco as mj
  self.c=c;self.condition=condition;self.camera=module('_online_camera_frozen',VIS/'camera.py');self.spec=read(VIS/'PREREG_CAMERA_PROBE.json')['eyes'];self.audit=read(GEOMETRY);self.head=c.plant.body('Head').id
  self.renderer=mj.Renderer(c.plant,height=32,width=32);self.queue=self.camera.PendingVision();self.centers=None;self.rows=[];self.images=[]
 def anchor(self):
  self.centers=anchor_landmarks(self.c.d.xpos[self.head],self.c.d.xmat[self.head].reshape(3,3),self.camera,self.spec,self.audit)
 def capture(self,ms):
  import numpy as np
  import mujoco as mj
  if ms%20 or not 0<=ms<15400:raise ValueError('captureexact20msindices')
  nominal=ms*.001;actual=float(self.c.d.time)
  if abs(nominal-actual)>1e-8:raise ValueError('capturephysicsclock')
  hp=self.c.d.xpos[self.head].copy();hr=self.c.d.xmat[self.head].reshape(3,3).copy();eyes=self.camera.eye_frames(hp,hr,self.audit,self.spec)
  stimulus=frame_target(self.condition,nominal-5.8) if self.centers is not None else 'off'
  def decor(scene):
   if stimulus!='off':self.camera.add_decor(scene,mj,mj.mjtGeom.mjGEOM_CYLINDER,[.12,1.2,0],self.centers[stimulus],np.eye(3),[1.,.08,.03,1.])
  statekeys=('qpos','qvel','ctrl','qacc','qacc_warmstart','qfrc_actuator','qfrc_applied','xfrc_applied','efc_force')
  before={key:np.asarray(getattr(self.c.d,key)).copy() for key in statekeys};contact_before=[(int(x.geom1),int(x.geom2),float(x.dist),x.pos.copy(),x.frame.copy()) for x in self.c.d.contact];before_time=float(self.c.d.time)
  rgb,metadata=self.camera.render_eye_pair(self.renderer,self.c.plant,self.c.d,eyes,self.spec,decor)
  unchanged=all(np.array_equal(value,np.asarray(getattr(self.c.d,key))) for key,value in before.items()) and before_time==float(self.c.d.time) and len(contact_before)==self.c.d.ncon
  if unchanged:
   for old,x in zip(contact_before,self.c.d.contact):
    unchanged=unchanged and old[0]==x.geom1 and old[1]==x.geom2 and old[2]==x.dist and np.array_equal(old[3],x.pos) and np.array_equal(old[4],x.frame)
  if not unchanged:raise RuntimeError('RGB rendering modified physical data/contactpipeline')
  self.queue.capture(nominal,rgb)
  self.images.append(rgb.copy());self.rows.append(dict(render_does_not_modify_physical_data=True,physical_before_sha256={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in before.items()},physical_after_sha256={key:hashlib.sha256(np.asarray(getattr(self.c.d,key)).tobytes()).hexdigest() for key in statekeys},contact_pipeline_unchanged=True,capture_id=len(self.rows),capture_time_s=nominal,actual_physical_time_s=actual,release_time_s=nominal+.020,actual_head_position=hp,actual_head_rotation=hr,camera_metadata=metadata,stimulus=stimulus,target_center_xyz=np.zeros(3) if stimulus=='off' else self.centers[stimulus].copy(),source_anchor_valid=self.centers is not None,rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
 def close(self):self.renderer.close()
def stable(auto,cycles,native):return bool(all(auto['gates'][k] for k in ('support','nonfoot','com','forward')) and cycles['all_legs_completed'] and native['post_nonfoot_peak_native']==0 and native['post_native_steps']==768000 and native['extra_force_max']==0)
def arrays(p):
 import numpy as np
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
