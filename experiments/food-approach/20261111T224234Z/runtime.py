"""Frozen stimulus ABI, strict component loading, causal environment transduction.
No Core construction, integration or training on import.
"""
from pathlib import Path
import importlib.util,sys,json,hashlib,os
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
EXP=ROOT.parent
MOTOR_DIR=EXP/'20261111T161007Z/combined_turning'
MOTOR=MOTOR_DIR/'COMBINED.pt'
MOTOR_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
SENSOR_DIR=EXP/'20261111T035953Z'
SENSOR=SENSOR_DIR/'TRAIN/checkpoint_003000.pt'
SENSOR_SHA='bfd65a3896d259496114695d7f7f3550272cf4c369895a4ddb34f3ddc3352c06'
BASE=EXP.parent/'banc-walking-recovery/20261110T152714Z'
GEOMETRY=EXP/'20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json'
ACCEPT=MOTOR_DIR/'USER_VISUAL_ACCEPTANCE.json'
A_IDS=('A_left_half','A_right_half','A_switch')
B_IDS=('B_noodor','B_equal','B_left_pulse','B_right_pulse','B_left_to_right','B_right_to_left')
IDS=A_IDS+B_IDS
ALPHA=-np.expm1(-.005/.05)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def read(p):return json.loads(Path(p).read_text())
def jsonable(v):
 if isinstance(v,dict):return {str(k):jsonable(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [jsonable(x) for x in v]
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,Path):return str(v)
 return v
def save_new(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(jsonable(v),f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def cpu1():
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});torch.set_num_threads(1)
 for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
  if os.environ.get(name)!='1':raise ValueError(name+' must be 1')
def load_policies():
 # Historical sensor-only loader correctly runs BEFORE any physical backend imports.
 if 'physics' in sys.modules or 'mujoco' in sys.modules:raise ValueError('strict sensor loader must precede physical backend')
 sys.path.insert(0,str(EXP/'20261111T034309Z'))
 import common as original_sensor_common
 if Path(original_sensor_common.__file__).resolve()!=EXP/'20261111T034309Z/common.py':raise ValueError('sensor common import collision')
 sensorio=module('_AB_balanced_sensor_loader',SENSOR_DIR/'common.py');behavior,bp=sensorio.load_behavior(SENSOR,SENSOR_SHA)
 sys.path.insert(0,str(MOTOR_DIR));motorio=module('_AB_strict_motor_loader',MOTOR_DIR/'model_io.py');motor,mp=motorio.load_checked(MOTOR,MOTOR_SHA,require_final=True)
 accept=read(ACCEPT)
 if accept['checkpoint_sha256']!=MOTOR_SHA or accept['visual_acceptance'] is not True or accept['engineering_qualified'] is not False:raise ValueError('bound experimental visual admission missing')
 if motor.identity()['state_dims']!=[128,64,64] or behavior.identity()['recurrent_state']!=32:raise ValueError('new256+32 ABI required')
 motor.requires_grad_(False);behavior.requires_grad_(False)
 return motor,behavior,mp,bp

def pin():
 sys.path.insert(0,str(BASE));import student_wave as sw,identity
 if Path(sw.__file__).resolve()!=BASE/'student_wave.py':raise ValueError('physical lineage collision')
 return sw,identity

def source_paths(mp,bp):
 names=('PREREG.json','runtime.py','runner.py','audit.py','preflight.py','orchestrate.py','summarize_B.py')
 paths=[ROOT/n for n in names]+[ACCEPT,GEOMETRY,MOTOR,SENSOR,SENSOR_DIR/'common.py',ROOT/'CLAMP_LEVELS.json']
 paths += [Path(p) for key in ('source_hashes','component_source_hashes','stream_hashes') for p in mp[key]]
 paths += [Path(p) for key in ('source_hashes','data_hashes') for p in bp['config'][key]]
 paths += [EXP/'20261111T034309Z/common.py',EXP/'20261111T034309Z/learning.py']
 paths += list(BASE.glob('*.py'))+list((BASE/'sources').glob('*.py'))
 paths += [MOTOR_DIR/'EVAL_neutral'/n for n in ('IDENTITY.json','RESULT.json','trace_1ms.npz','neural_decisions.npz','geometry_1ms.npz')]
 git=ROOT/'GIT_PREIMPLEMENTATION.json'
 if git.exists():paths.append(git)
 for p in paths:
  if not p.is_absolute() or not p.is_file():raise FileNotFoundError(p)
 return list(dict.fromkeys(paths))

def verify_frozen():
 frozen=read(ROOT/'FROZEN.json')
 for p,h in frozen['source_hashes'].items():
  if sha(p)!=h:raise ValueError('frozen source/input drift '+p)
 return frozen

def flat(state):return torch.cat(tuple(state),-1)
def fullflat(motor_state,behavior_state):return torch.cat((flat(motor_state),behavior_state),-1)
def project(raw):
 turn=raw[:,1:2]
 return torch.cat((torch.relu(turn)/3.,turn),dim=-1)
def smooth(x):
 x=np.clip(float(x),0.,1.);return float(x*x*(3.-2.*x))
def probe_turn(trial,elapsed):
 if elapsed<0:return 0.
 if trial=='A_left_half':return .5*smooth(elapsed-.6)
 if trial=='A_right_half':return -.5*smooth(elapsed-.6)
 if trial=='A_switch':
  if elapsed<4:return .5*smooth(elapsed-.6)
  if elapsed<5:return .5*(1.-smooth(elapsed-4.))
  return -.5*smooth(elapsed-5.8)
 raise ValueError('not an A probe')
def pulse(x):return smooth((x-.6)/.2)*(1.-smooth((x-3.6)/.2))
def concentration(trial,x):
 if trial in A_IDS:return np.zeros(2)
 levels=read(ROOT/'CLAMP_LEVELS.json');left=np.asarray(levels['left_pair']);right=left[::-1].copy()
 if trial=='B_noodor':return np.zeros(2)
 if trial=='B_equal':return np.full(2,float(left.mean()))*pulse(x)
 if trial=='B_left_pulse':return left*pulse(x)
 if trial=='B_right_pulse':return right*pulse(x)
 b=smooth((x-4.8)/.2);on=smooth((x-.6)/.2)
 if trial=='B_left_to_right':return ((1.-b)*left+b*right)*on
 if trial=='B_right_to_left':return ((1.-b)*right+b*left)*on
 raise ValueError('unknown trial')
class ClampFilter:
 def __init__(self):self.state=np.zeros(2);self.pending=np.zeros(2)
 def advance(self,C):
  used=self.pending.copy();before=self.state.copy();pending_before=self.pending.copy()
  target=np.asarray(C)/(1.+np.asarray(C));self.state=self.state+ALPHA*(target-self.state);self.pending=self.state.copy()
  return dict(receptors=used,concentration=np.asarray(C).copy(),filter_before=before,filter_after=self.state.copy(),pending_before=pending_before,pending_after=self.pending.copy())

class PeakRecorder:
 def __init__(self):self.count=0;self.post_count=0;self.peak=0.;self.post_peak=0.;self.com_min=float('inf');self.extra_max=0.
 def before(self,c):
  self.extra_max=max(self.extra_max,float(np.abs(c.d.xfrc_applied).max()),float(np.abs(c.d.qfrc_applied).max()))
  if c.adhesion_on or self.extra_max!=0:raise RuntimeError('extra force or adhesion')
  return float(c.d.time)
 def solved(self,c,before):
  import mujoco as mj
  nonfoot=0.
  for ci in range(c.d.ncon):
   ct=c.d.contact[ci];ga,gb=int(ct.geom1),int(ct.geom2)
   if not (c.planes[ga] or c.planes[gb]):continue
   geom=gb if c.planes[ga] else ga
   if int(c.geom_leg[geom])>=0:continue
   f=np.zeros(6);mj.mj_contactForce(c.plant,c.d,ci,f);world=ct.frame.reshape(3,3).T@f[:3];nonfoot+=max(0.,float(world[2])*(1 if c.planes[ga] else -1))
  return nonfoot
 def after(self,c,before,solved):
  self.count+=1;self.peak=max(self.peak,solved)
  if before>=5.8-1e-8:self.post_count+=1;self.post_peak=max(self.post_peak,solved);self.com_min=min(self.com_min,float(c.d.subtree_com[c.th][2]))
 def close(self):return dict(actual_native_steps=self.count,post_native_steps=self.post_count,nonfoot_peak_native=self.peak,post_nonfoot_peak_native=self.post_peak,post_COM_min=self.com_min,extra_force_max=self.extra_max,full_arrays_retained=False,scope='Every native solved NF peak, not full D1-D8')

def basic_functional(auto,cycles,native):
 return bool(all(auto['gates'][k] for k in ('support','nonfoot','com','forward')) and cycles['all_legs_completed'] and native['post_nonfoot_peak_native']==0 and native['post_native_steps']==768000 and native['extra_force_max']==0)
def strict_flags(auto,cycles,native,yaw):
 f=dict(original_five=bool(auto['all_gates']),all_six=bool(cycles['all_legs_completed']),strict_nonfoot_zero=bool(auto['nonfoot_force_max_native']==0),strict_native_nonfoot_zero=bool(native['post_nonfoot_peak_native']==0),legacy_large_turn_angle30=bool(abs(np.degrees(yaw))>=30))
 f['eligible']=bool(all(f.values()));return f

def arrays(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def delta_at(times,yaw,a,b):
 return float(yaw[int(np.argmin(abs(times-(5.8+b))))]-yaw[int(np.argmin(abs(times-(5.8+a))))])
def evaluate_A(trial,auto,cycles,native,g):
 baseline=arrays(MOTOR_DIR/'EVAL_neutral/geometry_1ms.npz');times=g['time'];y=np.unwrap(g['thorax_yaw']);by=np.unwrap(baseline['thorax_yaw'])
 relative=delta_at(times,y,0,9.6)-delta_at(times,by,0,9.6)
 stable=basic_functional(auto,cycles,native)
 if trial!='A_switch':
  direction=1 if trial=='A_left_half' else -1;effect=bool(direction*np.degrees(relative)>=1.)
  details=dict(fullANN_baseline_corrected_yaw_deg=float(np.degrees(relative)))
 else:
  left=delta_at(times,y,1.6,4.)-delta_at(times,by,1.6,4.);right=delta_at(times,y,6.8,9.6)-delta_at(times,by,6.8,9.6)
  effect=bool(np.degrees(left)>=.25 and np.degrees(right)<=-.25);details=dict(left_interval_corrected_yaw_deg=float(np.degrees(left)),right_interval_corrected_yaw_deg=float(np.degrees(right)))
 return dict(stable_body=stable,direction_effect=effect,functional_pass=bool(stable and effect),**details)
