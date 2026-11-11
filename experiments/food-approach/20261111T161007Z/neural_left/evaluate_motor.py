"""LEFT final4000 evaluator using explicitly adapted immutable complete physical rollout."""
from pathlib import Path
import argparse,types,sys,json
import numpy as np
import model_io as io
HERE=io.HERE;BASE=io.BASE;sha256=io.sha256
PRIMARY=io.ROOT/'20261110T235830Z/evaluation/evaluate_motor.py'
HEAD_AUDIT=io.ROOT/'20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json'
BASE_CHECKPOINT=io.legacy.CHECKPOINT
ORIGINAL=BASE.parent/'20261110T152754Z/NATIVE_ROLE_2500'
CHECKPOINT=HERE/'TRAIN/checkpoint_004000.pt'
COMMANDS={'neutral':[0.,0.],'left':[1./3.,1.]}

def jsonable(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,Path):return str(v)
 if isinstance(v,dict):return {str(k):jsonable(w) for k,w in v.items()}
 if isinstance(v,(list,tuple)):return [jsonable(w) for w in v]
 return v

def command_at(condition,decision):
 if condition not in COMMANDS or not isinstance(decision,(int,np.integer)) or not 0<=decision<2880:raise ValueError('LEFT/neutral decision domain')
 v=min(max((int(decision)-960)*.005,0.),1.);b=v*v*(3.-2.*v)
 return np.array([b/3.,b]) if condition=='left' else np.zeros(2)

def flatten(state):
 import torch
 return torch.cat(tuple(state),dim=-1)

def score(condition,auto,cycles,yaw,native):
 if condition not in COMMANDS:raise ValueError('LEFT/neutral only')
 flags=dict(original_five=bool(auto['all_gates']),all_six=bool(cycles['all_legs_completed']),strict_nonfoot_zero=bool(auto['nonfoot_force_max_native']==0.),turn_sign_magnitude=bool(condition=='neutral' or np.degrees(yaw)>=30.),strict_native_nonfoot_zero=bool(native['post_nonfoot_peak_native']==0. and native['post_native_steps']==768000 and native['actual_native_steps']==1232000 and native['extra_force_max']==0.))
 flags['eligible']=bool(all(flags.values()));return flags

class PeakRecorder:
 """Observe every solved native step, no state writes or large native arrays."""
 def __init__(self):self.count=0;self.post_count=0;self.peak=0.;self.post_peak=0.;self.com_min=float('inf');self.extra_max=0.
 def before(self,c):
  self.extra_max=max(self.extra_max,float(np.abs(c.d.xfrc_applied).max()),float(np.abs(c.d.qfrc_applied).max()))
  if c.adhesion_on or self.extra_max!=0:raise ValueError('no adhesion or extra forces permitted')
  return float(c.d.time)
 def solved(self,c,before):
  import mujoco as mj
  nonfoot=0.
  for ci in range(c.d.ncon):
   ct=c.d.contact[ci];ga,gb=int(ct.geom1),int(ct.geom2)
   if not (c.planes[ga] or c.planes[gb]):continue
   geom=gb if c.planes[ga] else ga
   if int(c.geom_leg[geom])>=0:continue
   force=np.zeros(6);mj.mj_contactForce(c.plant,c.d,ci,force);world=ct.frame.reshape(3,3).T@force[:3]
   nonfoot+=max(0.,float(world[2])*(1 if c.planes[ga] else -1))
  return nonfoot
 def after(self,c,before,solved):
  self.count+=1;self.peak=max(self.peak,solved)
  self.extra_max=max(self.extra_max,float(np.abs(c.d.xfrc_applied).max()),float(np.abs(c.d.qfrc_applied).max()))
  if before>=5.8-1e-8:
   self.post_count+=1;self.post_peak=max(self.post_peak,solved);self.com_min=min(self.com_min,float(c.d.subtree_com[c.th][2]))
 def close(self):return dict(actual_native_steps=self.count,post_native_steps=self.post_count,nonfoot_peak_native=self.peak,post_nonfoot_peak_native=self.post_peak,post_COM_min=self.com_min,extra_force_max=self.extra_max,full_arrays_retained=False,scope='every solved native NF peak only, not full D1-D8')

def pin():
 for name in ('physics','static_teacher','student_wave','models','identity','cycle_reward','wave_reference'):
  if name in sys.modules and Path(sys.modules[name].__file__).resolve()!=BASE/(name+'.py'):raise ValueError('conflicting immutable baseline '+name)
 sys.path.insert(0,str(BASE));import student_wave as sw;import identity
 return sw,identity,io

def checked_load(module,path,digest):
 if Path(path).resolve()!=CHECKPOINT:raise ValueError('physical LEFT_HEAD final4000 only')
 return io.load_checked(path,digest,require_final=True)

def source_paths(include_readiness=True):
 names=['model_io.py','train.py','verify_saved.py','evaluate_motor.py','verify_actual.py','prepare_eval.py','PREREG.json','EVAL_PREREG.json','CODEONLY.json']
 if include_readiness:names+=['READINESS.json']
 paths=[HERE/n for n in names]+[PRIMARY,io.ROOT/'20261110T235830Z/evaluation/verify_actual.py',HEAD_AUDIT,HERE/'TRAIN/VERIFY_002000.json',HERE/'TRAIN/VERIFY_004000.json']
 paths+=list(BASE.glob('*.py'))
 sw,_,_=pin();paths += [Path(p) for p in sw.teacher_preflight()['files']]+[BASE/'STUDENT_CHECKS.json']
 import torch
 payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
 paths += [Path(p) for p in payload['config']['source_hashes']]+[Path(p) for p in payload['config']['data_hashes']]
 paths=list(dict.fromkeys(paths))
 for path in paths:
  if not path.is_absolute() or not path.is_file():raise FileNotFoundError(path)
 return paths

def replace_once(source,old,new):
 if source.count(old)!=1:raise ValueError('immutable rollout adaptation drift '+old)
 return source.replace(old,new)

def adapted_source():
 source=PRIMARY.read_text()
 for old,new in [
  ("prereg['checkpoint_step']!=1500","prereg['checkpoint_step']!=4000"),
  ("recorder=sw.NativeRecorder(out/'native') if args.native else None","recorder=PeakRecorder()"),
  ("criteria=score(args.condition,auto,cycles,delta)","criteria=score(args.condition,auto,cycles,delta,recorder.close())"),
  ("kind='round2 fixed1500 slow residual actual physical evaluation'","kind='strict_LEFT_HEAD_V1 final4000 coupled positive turn bound6 actual physical evaluation'"),
  ("native_audit='NOT_PERFORMED'","native_audit='LIGHTWEIGHT_EVERY_NATIVE_STEP_NONFOOT_PEAK_ONLY'")]:source=replace_once(source,old,new)
 return source

def build():
 namespace={'__file__':str(PRIMARY),'__name__':'_left_head_complete_frozen_rollout'}
 exec(compile(adapted_source(),str(PRIMARY)+'::STRICT_LEFT_HEAD4000','exec'),namespace)
 fn=namespace['evaluate'];g=dict(fn.__globals__)
 g.update(HERE=HERE,ROOT=HERE,TRAIN=HERE,CHECKPOINT=CHECKPOINT,BASE=BASE,BASE_CHECKPOINT=BASE_CHECKPOINT,HEAD_AUDIT=HEAD_AUDIT,ORIGINAL=ORIGINAL,COMMANDS=COMMANDS,pin=pin,checked_load=checked_load,source_paths=source_paths,command_at=command_at,score=score,PeakRecorder=PeakRecorder,flatten=flatten)
 return types.FunctionType(fn.__code__,g,fn.__name__,fn.__defaults__,fn.__closure__)

def require_left_pass():
 result=json.loads((HERE/'EVAL_left/RESULT.json').read_text());audit=json.loads((HERE/'VERIFY_left.json').read_text())
 if not result['engineering_pass'] or not audit['integrity_all_pass'] or not audit['engineering_pass'] or not all(audit['checks'].values()):raise ValueError('neutral only after independently audited actual LEFTPASS')
 if result['checkpoint_sha256']!=sha256(CHECKPOINT) or audit['checkpoint_sha256']!=sha256(CHECKPOINT):raise ValueError('LEFT pass checkpoint identity')
 if audit.get('result_sha256')!=sha256(HERE/'EVAL_left/RESULT.json'):raise ValueError('LEFT audit/result binding')

def main():
 p=argparse.ArgumentParser();p.add_argument('--condition',choices=list(COMMANDS),required=True);p.add_argument('--parent-resource-released',action='store_true');args=p.parse_args()
 if not args.parent_resource_released:raise ValueError('parent resource release required')
 import os,torch
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});torch.set_num_threads(1)
 if args.condition=='neutral':require_left_pass()
 run=argparse.Namespace(condition=args.condition,checkpoint=str(CHECKPOINT),sha256=sha256(CHECKPOINT),out=str(HERE/('EVAL_'+args.condition)),training_complete=True,native=False)
 build()(run)
if __name__=='__main__':main()
