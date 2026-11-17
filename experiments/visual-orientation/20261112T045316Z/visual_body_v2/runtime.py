"""Explicit new globals and two sensory controls; frozenbody1 algorithms remain untouched."""
from pathlib import Path
import importlib.util,sys,types
ROOT=Path(__file__).resolve().parent;BODY1=ROOT.parent/'visual_body_v1';POLICY=ROOT.parent/'visual_policy_v2'
_s=importlib.util.spec_from_file_location('_body2_frozen_body1_runtime',BODY1/'runtime.py');_base=importlib.util.module_from_spec(_s);sys.modules[_s.name]=_base;_s.loader.exec_module(_base)
_g={k:v for k,v in _base.__dict__.items() if not k.startswith('__')}
for _k,_v in list(_g.items()):
 if isinstance(_v,types.FunctionType) and _v.__module__==_base.__name__:_g[_k]=types.FunctionType(_v.__code__,_g,_v.__name__,_v.__defaults__,_v.__closure__)
_g.update(ROOT=ROOT,POLICY=POLICY)
globals().update(_g)
NORMAL_IDS=('V_empty','V_center','V_left_pulse','V_right_pulse','V_left_to_right','V_right_to_left');CONTROL_IDS=('V_left_occluded','V_flip_fixedframe');IDS=NORMAL_IDS+CONTROL_IDS

def load_policies():
 if any(k in sys.modules for k in ('physics','mujoco','student_wave','flygym')):raise ValueError('newvisual strictloader precedes physics')
 b=read(ROOT/'INPUT_BINDINGS.json');sentinel=object();previous=sys.modules.get('model',sentinel);priorpath=list(sys.path)
 try:
  # v2 preserves originalclass but its model.py imports basenamecommon; bind originalclass directly.
  vm=module('_body2_original_visual_class',ROOT.parent/'visual_policy_v1/model.py');sys.modules['model']=vm;io=module('_body2_visual_v2_strict_io',POLICY/'common.py');visual,vp=io.load_checked(b['visual_checkpoint'],b['visual_sha256'],require_final=True)
  if vp['format_version']!='VISION_STATIC_HEAD_OCCLUSION_RECOVERY_V2' or vp['step']!=500:raise ValueError('onlyactualvisual2final500')
  mm=module('_body2_original_combined_class',MOTOR_DIR/'model.py');sys.modules['model']=mm;sys.path.insert(0,str(MOTOR_DIR));mio=module('_body2_motor_strict_io',MOTOR_DIR/'model_io.py');motor,mp=mio.load_checked(MOTOR,MOTOR_SHA,require_final=True)
 finally:
  sys.path[:]=priorpath
  if previous is sentinel:sys.modules.pop('model',None)
  else:sys.modules['model']=previous
 if (previous is sentinel and 'model' in sys.modules) or (previous is not sentinel and sys.modules.get('model') is not previous):raise ValueError('modelaliasrestorationfailed')
 visual.requires_grad_(False);motor.requires_grad_(False);return visual,motor,vp,mp

def frame_target(condition,elapsed):
 if condition=='V_left_occluded':return 'off' if elapsed<.6-1e-10 else 'left'
 if condition=='V_flip_fixedframe':return 'off' if elapsed<.6-1e-10 else 'left' if elapsed<4.8-1e-10 else 'right'
 return _base.frame_target(condition,elapsed)
_g.update(IDS=IDS,frame_target=frame_target,load_policies=load_policies)
_init=types.FunctionType(_base.CameraStream.__init__.__code__,_g,'_init')
_anchor=types.FunctionType(_base.CameraStream.anchor.__code__,_g,'_anchor')
_capture=types.FunctionType(_base.CameraStream.capture.__code__,_g,'_capture')
class _OpticalProxy:
 def __init__(self,owner):self.owner=owner
 def __getattr__(self,key):return getattr(self.owner.original_camera,key)
 def render_eye_pair(self,renderer,model,data,frames,spec,decorator=None):
  import numpy as np
  import mujoco as mj
  o=self.owner;active=o.condition=='V_left_occluded' and o.centers is not None and float(data.time)-5.8>=3.6-1e-8
  if not active:return o.original_camera.render_eye_pair(renderer,model,data,frames,spec,decorator)
  def wall(scene):o.original_camera.add_decor(scene,mj,mj.mjtGeom.mjGEOM_BOX,[.04,1.4,1.4],o.wall_center,o.wall_basis,[.12,.12,.12,1.])
  def both(scene):
   if decorator is not None:decorator(scene)
   wall(scene)
  actual,meta=o.original_camera.render_eye_pair(renderer,model,data,frames,spec,both);counter,_=o.original_camera.render_eye_pair(renderer,model,data,frames,spec,wall)
  o.pending_counter=counter.copy();o.pending_occlusion=dict(counterfactual_available=True,occlusion_exact=bool(np.array_equal(actual,counter)),occlusion_pixel_MAE=float(np.abs(actual.astype(float)-counter).mean()),counterfactual_rgb_sha256=hashlib.sha256(counter.tobytes()).hexdigest());return actual,meta
class CameraStream:
 def __init__(self,c,condition):
  _init(self,c,condition);self.original_camera=self.camera;self.camera=_OpticalProxy(self);self.frozen_actor=None;self.pending_counter=None;self.pending_occlusion=None;self.counter_images=[];self.counter_ids=[];self.additional_artifacts={}
 def anchor(self):
  import numpy as np
  _anchor(self);hr=self.c.d.xmat[self.head].reshape(3,3);eyes=self.original_camera.eye_frames(self.c.d.xpos[self.head],hr,self.audit,self.spec);mid=(eyes[0]['pos']+eyes[1]['pos'])/2;ex=hr[:,0].copy();ex[2]=0;ex/=np.linalg.norm(ex);ey=np.array([-ex[1],ex[0],0.]);self.wall_center=mid+.4*ex;self.wall_center[2]=1.4;self.wall_basis=np.column_stack((ex,ey,np.array([0.,0.,1.])))
 def capture(self,ms):
  self.pending_counter=None;self.pending_occlusion=None;_capture(self,ms);row=self.rows[-1];row.update(counterfactual_available=False,occlusion_exact=False,occlusion_pixel_MAE=0.,counterfactual_rgb_sha256='')
  if self.pending_counter is not None:self.counter_images.append(self.pending_counter);self.counter_ids.append(row['capture_id']);row.update(self.pending_occlusion)
 def actor_query(self,time_s):
  raw=self.queue.query(time_s);elapsed=time_s-5.8;freeze=self.condition=='V_flip_fixedframe' and elapsed>=4.8-1e-10
  if freeze and self.frozen_actor is None:
   if not raw['valid']:raise ValueError('fixedframe requires genuine released image')
   self.frozen_actor=dict(raw,image=raw['image'].copy())
  result=dict(self.frozen_actor if freeze else raw);result.update(raw_capture_id=int(round(raw['capture_time_s']/.020)),raw_capture_time_s=raw['capture_time_s'],raw_release_time_s=raw['release_time_s'],actor_frame_frozen=freeze);return result
 def close(self):
  import numpy as np
  self.renderer.close()
  if self.condition=='V_left_occluded':
   path=ROOT/'trials'/self.condition/'occlusion_counterfactual_rgb.npz';np.savez_compressed(path,rgb=np.stack(self.counter_images) if self.counter_images else np.empty((0,2,32,32,3),dtype=np.uint8),capture_ids=np.asarray(self.counter_ids,dtype=np.int64));self.additional_artifacts['occlusion_counterfactual']=dict(path=str(path),sha256=sha(path),rows=len(self.counter_images))

def expected_actor_ids(condition,rawids):
 import numpy as np
 return np.minimum(rawids,529) if condition=='V_flip_fixedframe' else rawids

def audit_controls(check,out,result,decisions,rgb,captures):
 import numpy as np
 rawids=np.floor((np.arange(2880)*5+1000-20)/20).astype(int);actor=expected_actor_ids(result['condition'],rawids);check('rawreleaseIDs_continuing',np.array_equal(decisions['raw_released_image_capture_id'],rawids));check('rawreleaseclock',np.max(abs(decisions['raw_released_capture_time_s']-rawids*.020))<=1e-10 and np.max(abs(decisions['raw_released_release_time_s']-(rawids*.020+.020)))<=1e-10)
 frozen=np.arange(2880)>=1920 if result['condition']=='V_flip_fixedframe' else np.zeros(2880,bool);check('actorfreezeonlypredeclaredcondition',np.array_equal(decisions['actor_frame_frozen'],frozen))
 if result['condition']=='V_flip_fixedframe':
  check('frozenleftframecaptureANN4_78',actor[1920]==529 and captures[529]['stimulus']=='left' and np.all(decisions['image_capture_id'][1920:]==529));check('frozen16features_exact',np.array_equal(decisions['held_features'][1920:],np.broadcast_to(decisions['held_features'][1920],(960,16))));check('actualrightscene_notfrozen',any(c['stimulus']=='right' for c in captures[530:]) and any(not np.array_equal(rgb[i],rgb[529]) for i in range(530,770)))
 if result['condition']=='V_left_occluded':
  artifact=result['occlusion_counterfactual'];check('counterfactualRGBartifactSHA',sha(artifact['path'])==artifact['sha256'])
  with np.load(artifact['path'],allow_pickle=False) as z:counter=z['rgb'];ids=z['capture_ids']
  expected=np.asarray([i for i,c in enumerate(captures) if c['counterfactual_available']],dtype=np.int64);check('wallcounterfactualsamecaptureIDs',np.array_equal(ids,expected) and counter.shape==(len(ids),2,32,32,3))
  for image,i in zip(counter,ids):
   cap=captures[int(i)];check('actualwallcounterfactual'+str(i),hashlib.sha256(image.tobytes()).hexdigest()==cap['counterfactual_rgb_sha256'] and np.array_equal(rgb[i],image)==cap['occlusion_exact'] and float(np.abs(rgb[i].astype(float)-image).mean())==cap['occlusion_pixel_MAE'])
