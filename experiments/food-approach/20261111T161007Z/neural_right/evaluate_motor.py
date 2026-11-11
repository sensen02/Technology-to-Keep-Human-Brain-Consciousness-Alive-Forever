"""Independent .06 RIGHT_ONLY loader; frozen original rollout schema, native peak recorder.
No legacy parent-right-preservation hook. Every textual adaptation is exact-count checked.
"""
from pathlib import Path
import argparse,types,sys,json,hashlib
import numpy as np
import model as io
from model import HERE,BASE,sha256,module
PRIMARY=HERE.parent.parent/'20261110T235830Z/evaluation/evaluate_motor.py'
PEAK_SOURCE=HERE.parent/'attempt02/diagnose.py'
HEAD_AUDIT=HERE.parent.parent/'20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json'
BASE_CHECKPOINT=io.legacy.CHECKPOINT
ORIGINAL=BASE.parent/'20261110T152754Z/NATIVE_ROLE_2500'
CHECKPOINT=HERE/'TRAIN/checkpoint_002000.pt'
COMMANDS={'neutral':[0.,0.],'right':[0.,-1.]}
def jsonable(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,Path):return str(v)
 if isinstance(v,dict):return {str(k):jsonable(w) for k,w in v.items()}
 if isinstance(v,(list,tuple)):return [jsonable(w) for w in v]
 return v
def command_at(condition,decision):
 if condition not in COMMANDS or not 0<=decision<2880:raise ValueError('RIGHT_ONLY fixed command')
 v=min(max((int(decision)-960)*.005,0.),1.);return np.asarray(COMMANDS[condition])*v*v*(3-2*v)
def flatten(state):
 import torch
 return torch.cat(tuple(state),dim=-1)
def score(condition,auto,cycles,yaw,native=None):
 flags=dict(original_five=bool(auto['all_gates']),all_six=bool(cycles['all_legs_completed']),strict_nonfoot_zero=bool(auto['nonfoot_force_max_native']==0.),turn_sign_magnitude=bool(condition=='neutral' or -np.degrees(yaw)>=30.))
 if native is not None:flags['strict_native_nonfoot_zero']=bool(native['post_nonfoot_peak_native']==0. and native['post_native_steps']==768000 and native['extra_force_max']==0.)
 flags['eligible']=bool(all(flags.values()));return flags
def pin():
 for name in ('physics','static_teacher','student_wave','models','identity','cycle_reward','wave_reference'):
  if name in sys.modules and Path(sys.modules[name].__file__).resolve()!=BASE/(name+'.py'):raise ValueError('conflicting baseline module '+name)
 sys.path.insert(0,str(BASE));import student_wave as sw;import identity
 return sw,identity,io
def checked_load(module,path,digest):
 if Path(path).resolve()!=CHECKPOINT:raise ValueError('physical only final2000')
 return io.load_checked(path,digest,require_final=True)
def source_paths():
 paths=[HERE/n for n in ('evaluate_motor.py','verify_actual.py','EVAL_PREREG.json','READINESS.json','model.py','PREREG.json')]
 paths+=[PRIMARY,PEAK_SOURCE,PEAK_SOURCE.parent/'turning_teacher.py',HEAD_AUDIT,io.OLD,io.ENGINE,HERE/'TRAIN/VERIFY_001000.json',HERE/'TRAIN/VERIFY_002000.json']
 # Include actual baseline local dependencies before Core; identity.capture adds physics/assets/compiled arrays.
 paths += list(BASE.glob('*.py'))
 sw,_,_=pin();paths += [Path(p) for p in sw.teacher_preflight()['files']]+[BASE/'STUDENT_CHECKS.json']
 import torch
 checkpoint_payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
 paths += [Path(p) for p in checkpoint_payload['config']['source_hashes']]+[Path(p) for p in checkpoint_payload['config']['data_hashes']]
 for p in paths:
  if not p.is_file():raise FileNotFoundError(p)
 return list(dict.fromkeys(paths))
def replace_once(source,old,new):
 if source.count(old)!=1:raise ValueError('adaptation drift: '+old)
 return source.replace(old,new)
def build():
 src=PRIMARY.read_text();src=replace_once(src,"prereg['checkpoint_step']!=1500","prereg['checkpoint_step']!=2000")
 src=replace_once(src,"recorder=sw.NativeRecorder(out/'native') if args.native else None","recorder=PeakRecorder()")
 src=replace_once(src,"criteria=score(args.condition,auto,cycles,delta)","criteria=score(args.condition,auto,cycles,delta,recorder.close())")
 src=replace_once(src,"kind='round2 fixed1500 slow residual actual physical evaluation'","kind='RIGHT_ONLY fixed2000 yaw006 bound3 actual physical evaluation'")
 src=replace_once(src,"native_audit='NOT_PERFORMED'","native_audit='LIGHTWEIGHT_EVERY_NATIVE_STEP_NONFOOT_PEAK_ONLY'")
 namespace={'__file__':str(PRIMARY),'__name__':'_right_only_frozen_rollout'};exec(compile(src,str(PRIMARY)+'::RIGHT_ONLY_explicit_adaptations','exec'),namespace)
 sw,identity,_=pin()
 # Import original PeakRecorder with all its actual dependencies source-bound, no physics construction on import.
 sys.path.insert(0,str(PEAK_SOURCE.parent));peak=module('_right_only_native_peak',PEAK_SOURCE)
 fn=namespace['evaluate'];g=dict(fn.__globals__);g.update(HERE=HERE,ROOT=HERE,TRAIN=HERE,CHECKPOINT=CHECKPOINT,COMMANDS=COMMANDS,pin=pin,checked_load=checked_load,source_paths=source_paths,command_at=command_at,score=score,PeakRecorder=peak.PeakRecorder,flatten=flatten)
 return types.FunctionType(fn.__code__,g,fn.__name__,fn.__defaults__,fn.__closure__)
def parser():
 p=argparse.ArgumentParser();p.add_argument('--condition',choices=list(COMMANDS),required=True);return p
def main():
 import os,torch
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});torch.set_num_threads(1)
 a=parser().parse_args();args=argparse.Namespace(condition=a.condition,checkpoint=str(CHECKPOINT),sha256=sha256(CHECKPOINT),out=str(HERE/('EVAL_'+a.condition)),training_complete=True,native=False)
 build()(args)
if __name__=='__main__':main()
