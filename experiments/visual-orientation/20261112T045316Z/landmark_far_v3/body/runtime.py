"""Far-only explicit namespace: immutable physical loop, genuine head RGB and closed covers."""
from pathlib import Path
import importlib.util,sys,types
PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype')
VIS=PROJECT/'experiments/visual-orientation/20261112T045316Z'
ROOT=VIS/'landmark_far_v3/body';POLICY=VIS/'landmark_far_v3/policy'
BODY1=VIS/'visual_body_v1';BODY2=VIS/'visual_body_v2'
def _module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
_base=_module('_far_body_frozen_body1_runtime',BODY1/'runtime.py')
_g={k:v for k,v in _base.__dict__.items() if not k.startswith('__')}
for _key,_value in list(_g.items()):
 if isinstance(_value,types.FunctionType) and _value.__module__==_base.__name__:_g[_key]=types.FunctionType(_value.__code__,_g,_value.__name__,_value.__defaults__,_value.__closure__)
_g.update(ROOT=ROOT,VIS=VIS,POLICY=POLICY,PROJECT=PROJECT)
globals().update(_g)
NORMAL_IDS=('V_empty','V_center','V_left_pulse','V_right_pulse','V_left_to_right','V_right_to_left')
CONTROL_IDS=('V_left_occluded','V_flip_fixedframe');IDS=NORMAL_IDS+CONTROL_IDS

def load_policies():
 if any(k in sys.modules for k in ('physics','mujoco','student_wave','flygym')):raise ValueError('far visual strict loader before physics required')
 b=read(ROOT/'INPUT_BINDINGS.json');sentinel=object();previous=sys.modules.get('model',sentinel);priorpath=list(sys.path)
 try:
  vm=module('_far_original_visual_class',VIS/'visual_policy_v1/model.py');sys.modules['model']=vm
  io=module('_far_visual_strict_io',POLICY/'common.py');visual,vp=io.load_checked(b['visual_checkpoint'],b['visual_sha256'],require_final=True)
  if vp['format_version']!=b['visual_format'] or vp['step']!=500:raise ValueError('bound far final500 format required')
  mm=module('_far_combined_motor_class',MOTOR_DIR/'model.py');sys.modules['model']=mm;sys.path.insert(0,str(MOTOR_DIR))
  mio=module('_far_combined_motor_strict_io',MOTOR_DIR/'model_io.py');motor,mp=mio.load_checked(MOTOR,MOTOR_SHA,require_final=True)
 finally:
  sys.path[:]=priorpath
  if previous is sentinel:sys.modules.pop('model',None)
  else:sys.modules['model']=previous
 if (previous is sentinel and 'model' in sys.modules) or (previous is not sentinel and sys.modules.get('model') is not previous):raise ValueError('model alias was not restored')
 if visual.identity()['hidden']!=32 or motor.identity()['state_dims']!=[128,64,64]:raise ValueError('same visual32+motor256 required')
 visual.requires_grad_(False);motor.requires_grad_(False);return visual,motor,vp,mp

def frame_target(condition,elapsed):
 if condition=='V_left_occluded':return 'off' if elapsed<.6-1e-10 else 'left'
 if condition=='V_flip_fixedframe':return 'off' if elapsed<.6-1e-10 else 'left' if elapsed<4.8-1e-10 else 'right'
 return _base.frame_target(condition,elapsed)

def anchor_landmarks(headpos,headrot,camera,spec,audit):
 import numpy as np
 eyes=camera.eye_frames(headpos,headrot,audit,spec);mid=(eyes[0]['pos']+eyes[1]['pos'])/2
 ex=np.asarray(headrot)[:,0].copy();ex[2]=0.;ex/=np.linalg.norm(ex);ey=np.array([-ex[1],ex[0],0.]);centers={}
 for name,angle in (('left',.4),('right',-.4),('center',0.)):
  p=mid+2.*ex+2.*np.tan(angle)*ey;p[2]=1.2;centers[name]=p
 return centers

def cover_centers(centers):
 import numpy as np
 result=[]
 for side in ('left','right'):
  p=np.asarray(centers[side]).copy();p[2]=1.4;result.append(p)
 return result

def cover_basis_from_head(headrot):
 import numpy as np
 ex=np.asarray(headrot)[:,0].copy();ex[2]=0.;ex/=np.linalg.norm(ex);ey=np.array([-ex[1],ex[0],0.]);return np.column_stack((ex,ey,np.array([0.,0.,1.])))

def eye_cover_diagnostics(frames,centers,basis):
 import numpy as np
 half=np.array([.18,.18,1.4]);distance=[];outside=[]
 for eye in frames:
  for center in cover_centers(centers):
   local=np.asarray(basis).T@(np.asarray(eye['pos'])-center);delta=np.abs(local)-half;distance.append(float(np.linalg.norm(np.maximum(delta,0.))));outside.append(bool(np.any(delta>0.)))
 return dict(eye_box_distances=distance,eyes_outside_all_boxes=all(outside),eye_box_min_clearance=min(distance),eye_box_near_clear=all(outside) and min(distance)>.005)

_g.update(IDS=IDS,load_policies=load_policies,frame_target=frame_target,anchor_landmarks=anchor_landmarks)
_init=types.FunctionType(_base.CameraStream.__init__.__code__,_g,'_far_camera_init')
_anchor=types.FunctionType(_base.CameraStream.anchor.__code__,_g,'_far_camera_anchor')
_capture=types.FunctionType(_base.CameraStream.capture.__code__,_g,'_far_camera_capture')

class _OpticalProxy:
 def __init__(self,owner):self.owner=owner
 def __getattr__(self,key):return getattr(self.owner.original_camera,key)
 def render_eye_pair(self,renderer,model,data,frames,spec,decorator=None):
  import numpy as np
  import mujoco as mj
  owner=self.owner;active=owner.condition=='V_left_occluded' and owner.centers is not None and float(data.time)-5.8>=3.6-1e-8
  if not active:return owner.original_camera.render_eye_pair(renderer,model,data,frames,spec,decorator)
  def boxes(scene):
   for center in cover_centers(owner.centers):owner.original_camera.add_decor(scene,mj,mj.mjtGeom.mjGEOM_BOX,[.18,.18,1.4],center,owner.cover_basis,[.12,.12,.12,1.])
  def both(scene):
   if decorator is not None:decorator(scene)
   boxes(scene)
  actual,meta=owner.original_camera.render_eye_pair(renderer,model,data,frames,spec,both)
  counter,_=owner.original_camera.render_eye_pair(renderer,model,data,frames,spec,boxes)
  owner.pending_counter=counter.copy();owner.pending_occlusion=dict(counterfactual_available=True,occlusion_exact=bool(np.array_equal(actual,counter)),occlusion_pixel_MAE=float(np.abs(actual.astype(float)-counter).mean()),counterfactual_rgb_sha256=hashlib.sha256(counter.tobytes()).hexdigest(),cover_rotation=owner.cover_basis.tolist(),**eye_cover_diagnostics(frames,owner.centers,owner.cover_basis))
  return actual,meta

class CameraStream:
 def __init__(self,c,condition):
  _init(self,c,condition);self.original_camera=self.camera;self.camera=_OpticalProxy(self);self.frozen_actor=None;self.pending_counter=None;self.pending_occlusion=None;self.counter_images=[];self.counter_ids=[];self.additional_artifacts={}
 def anchor(self):
  _anchor(self);self.cover_basis=cover_basis_from_head(self.c.d.xmat[self.head].reshape(3,3))
 def capture(self,ms):
  self.pending_counter=None;self.pending_occlusion=None;_capture(self,ms)
  row=self.rows[-1];row.update(counterfactual_available=False,occlusion_exact=False,occlusion_pixel_MAE=0.,counterfactual_rgb_sha256='',eye_box_distances=[],eyes_outside_all_boxes=False,eye_box_min_clearance=0.,eye_box_near_clear=False)
  if self.pending_counter is not None:self.counter_images.append(self.pending_counter);self.counter_ids.append(row['capture_id']);row.update(self.pending_occlusion)
 def actor_query(self,time_s):
  raw=self.queue.query(time_s);freeze=self.condition=='V_flip_fixedframe' and time_s-5.8>=4.8-1e-10
  if freeze and self.frozen_actor is None:
   if not raw['valid']:raise ValueError('freeze requires genuine released RGB')
   self.frozen_actor=dict(raw,image=raw['image'].copy())
  result=dict(self.frozen_actor if freeze else raw);result.update(raw_capture_id=int(round(raw['capture_time_s']/.020)),raw_capture_time_s=raw['capture_time_s'],raw_release_time_s=raw['release_time_s'],actor_frame_frozen=freeze);return result
 def close(self):
  import numpy as np
  self.renderer.close()
  if self.condition=='V_left_occluded':
   p=ROOT/'trials'/self.condition/'occlusion_counterfactual_rgb.npz';np.savez_compressed(p,rgb=np.stack(self.counter_images) if self.counter_images else np.empty((0,2,32,32,3),dtype=np.uint8),capture_ids=np.asarray(self.counter_ids,dtype=np.int64));self.additional_artifacts['occlusion_counterfactual']=dict(path=str(p),sha256=sha(p),rows=len(self.counter_images))

def expected_actor_ids(condition,rawids):
 import numpy as np
 return np.minimum(rawids,529) if condition=='V_flip_fixedframe' else rawids

def audit_controls(check,out,result,decisions,rgb,captures):
 import numpy as np
 rawids=np.floor((np.arange(2880)*5+1000-20)/20).astype(int);actor=expected_actor_ids(result['condition'],rawids)
 check('rawreleaseIDs_continuing',np.array_equal(decisions['raw_released_image_capture_id'],rawids))
 check('rawreleaseclock',np.max(abs(decisions['raw_released_capture_time_s']-rawids*.020))<=1e-10 and np.max(abs(decisions['raw_released_release_time_s']-(rawids*.020+.020)))<=1e-10)
 frozen=np.arange(2880)>=1920 if result['condition']=='V_flip_fixedframe' else np.zeros(2880,bool)
 check('actorfreezeonlypredeclaredcondition',np.array_equal(decisions['actor_frame_frozen'],frozen))
 if result['condition']=='V_flip_fixedframe':
  check('frozenleftframecaptureANN4_78',actor[1920]==529 and captures[529]['stimulus']=='left' and np.all(decisions['image_capture_id'][1920:]==529))
  check('frozen16features_exact',np.array_equal(decisions['held_features'][1920:],np.broadcast_to(decisions['held_features'][1920],(960,16))))
  check('actualrightscene_notfrozen',any(c['stimulus']=='right' for c in captures[530:]) and any(not np.array_equal(rgb[i],rgb[529]) for i in range(530,770)))
 if result['condition']=='V_left_occluded':
  artifact=result['occlusion_counterfactual'];check('counterfactualRGBartifactSHA',sha(artifact['path'])==artifact['sha256'])
  with np.load(artifact['path'],allow_pickle=False) as z:counter=z['rgb'];ids=z['capture_ids']
  expected=np.asarray([i for i,c in enumerate(captures) if c['counterfactual_available']],dtype=np.int64)
  check('closedcovers_captureIDs',np.array_equal(ids,expected) and counter.shape==(len(ids),2,32,32,3))
  centers=anchor_landmarks(np.asarray(captures[290]['actual_head_position']),np.asarray(captures[290]['actual_head_rotation']),module('_far_control_camera',VIS/'camera.py'),read(VIS/'PREREG_CAMERA_PROBE.json')['eyes'],read(GEOMETRY));basis=cover_basis_from_head(np.asarray(captures[290]['actual_head_rotation']))
  for image,i in zip(counter,ids):
   cap=captures[int(i)];diagnostics=eye_cover_diagnostics(cap['camera_metadata'],centers,basis);check('cover_world_rotation_fixed'+str(i),np.array_equal(cap['cover_rotation'],basis))
   check('actualclosedcovercounterfactual'+str(i),hashlib.sha256(image.tobytes()).hexdigest()==cap['counterfactual_rgb_sha256'] and bool(np.array_equal(rgb[i],image))==cap['occlusion_exact'] and float(np.abs(rgb[i].astype(float)-image).mean())==cap['occlusion_pixel_MAE'])
   check('eyeoutside_clearance'+str(i),all(cap[k]==v for k,v in diagnostics.items()))
