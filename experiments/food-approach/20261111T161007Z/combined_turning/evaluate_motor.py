"""Future single-COMBINED.pt neutral/LEFT/RIGHT rollout with explicit256-state semantics.
Research demonstration and original strict qualification are reported separately.
"""
from pathlib import Path
import argparse,ast,types,sys,json
import numpy as np
import model_io as io
HERE=io.HERE;BASE=io.BASE
PRIMARY=HERE.parent.parent/'20261110T235830Z/evaluation/evaluate_motor.py'
HEAD_AUDIT=HERE.parent.parent/'20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json'
BASE_CHECKPOINT=BASE/'BC_GRU_ROLE_3000/checkpoint_002500.pt';ORIGINAL=BASE.parent/'20261110T152754Z/NATIVE_ROLE_2500';CHECKPOINT=HERE/'COMBINED.pt'
PEAK_SOURCE=HERE.parent/'attempt02/diagnose.py'
COMMANDS={'neutral':[0.,0.],'left':[1./3.,1.],'right':[0.,-1.]}
sha256=io.sha256;jsonable=io.jsonable

def command_at(condition,decision):
 if condition not in COMMANDS or not isinstance(decision,(int,np.integer)) or not 0<=decision<2880:raise ValueError('combined fixed-condition decision0..2879')
 v=min(max((int(decision)-960)*.005,0.),1.);b=v*v*(3.-2.*v);turn=b if condition=='left' else (-b if condition=='right' else 0.)
 return np.asarray([max(turn,0.)/3.,turn],dtype=np.float64)
def flatten(state):
 import torch
 if not isinstance(state,(tuple,list)) or len(state)!=3 or [x.shape[-1] for x in state]!=[128,64,64]:raise ValueError('three explicit shared128/RIGHT64/LEFT64 states required')
 return torch.cat(tuple(state),dim=-1)
def score(condition,auto,cycles,yaw,native):
 signed=yaw if condition=='left' else -yaw
 flags=dict(original_five=bool(auto['all_gates']),all_six=bool(cycles['all_legs_completed']),strict_nonfoot_zero=bool(auto['nonfoot_force_max_native']==0.),turn_sign_magnitude=bool(condition=='neutral' or np.degrees(signed)>=30.),strict_native_nonfoot_zero=bool(native['post_nonfoot_peak_native']==0. and native['post_native_steps']==768000 and native['actual_native_steps']==1232000 and native['extra_force_max']==0.))
 flags['eligible']=bool(all(flags.values()));return flags
def demonstration(condition,auto,cycles,yaw,native):
 signed=yaw if condition=='left' else -yaw
 return bool(all(auto['gates'][k] for k in ('support','nonfoot','com','forward')) and cycles['all_legs_completed'] and native['post_nonfoot_peak_native']==0. and native['post_native_steps']==768000 and native['actual_native_steps']==1232000 and native['extra_force_max']==0. and (condition=='neutral' or signed>0.))
def pin():
 for name in ('physics','static_teacher','student_wave','models','identity','cycle_reward','wave_reference'):
  if name in sys.modules and Path(sys.modules[name].__file__).resolve()!=BASE/(name+'.py'):raise ValueError('conflicting immutable baseline '+name)
 sys.path.insert(0,str(BASE));import student_wave as sw;import identity
 return sw,identity,io
def checked_load(module,path,digest):return io.load_checked(path,digest,require_final=True)
def source_paths(include_readiness=True):
 names=['model.py','model_io.py','prepare.py','verify_saved.py','evaluate_motor.py','verify_actual.py','prepare_eval.py','check_codeonly.py','PREREG.json','AMENDMENT_01_LEFT_PENDING_FEATURE_REFINE.json','AMENDMENT_02_CONDITIONAL_RIGHT_FEATURE_REFINE.json','AMENDMENT_03_HONEST_DEMONSTRATION_COMPONENTS.json','AMENDMENT_04_POSTHOC_VISUAL_RIGHT_COMPONENT.json','COMPONENT_PLAN.json','COMPONENT_BINDINGS.json','PREPARATION.json','VERIFY_SAVED.json','EVAL_PREREG.json']
 if include_readiness:names+=['READINESS.json']
 paths=[HERE/n for n in names]+[PRIMARY,PEAK_SOURCE,PEAK_SOURCE.parent/'turning_teacher.py',HEAD_AUDIT,CHECKPOINT];paths+=list(BASE.glob('*.py'));sw,_,_=pin();paths += [Path(p) for p in sw.teacher_preflight()['files']]+[BASE/'STUDENT_CHECKS.json']
 import torch
 p=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);paths += [Path(s) for s in p['source_hashes']]+[Path(s) for s in p['component_source_hashes']]+[Path(s) for s in p['stream_hashes']]
 paths=list(dict.fromkeys(paths))
 for path in paths:
  if not path.is_absolute() or not path.is_file():raise FileNotFoundError(path)
 return paths

def replace_once(src,old,new):
 if src.count(old)!=1:raise ValueError('immutable rollout adaptation drift '+old)
 return src.replace(old,new)
def replace_function(src,name,new):
 node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name==name);return replace_once(src,ast.get_source_segment(src,node),new)
def adapted_source():
 src=PRIMARY.read_text();src=replace_once(src,"prereg['checkpoint_step']!=1500","prereg['composition_format']!='COMBINED_TURN_V1'");src=replace_once(src,"if not args.training_complete:","if not args.composition_ready:")
 if src.count('state_layout=[128,64]')!=2 or src.count('state_dims=[128,64]')!=1:raise ValueError('expected exactly two layouts and one handover dims in immutable source')
 src=src.replace("state_layout=[128,64]","state_layout=[128,64,64]").replace("state_dims=[128,64]","state_dims=[128,64,64]")
 src=replace_function(src,'flatten',"def flatten(state):\n    import torch\n    if len(state)!=3 or [s.shape[-1] for s in state]!=[128,64,64]:raise ValueError('requires explicit shared128/RIGHT64/LEFT64 state')\n    return torch.cat(tuple(state),dim=-1)")
 src=replace_function(src,'checked_load',"def checked_load(io,path,digest):\n    return io.load_checked(path,digest,require_final=True)")
 for old,new in [("recorder=sw.NativeRecorder(out/'native') if args.native else None","recorder=PeakRecorder()"),("criteria=score(args.condition,auto,cycles,delta)","criteria=score(args.condition,auto,cycles,delta,recorder.close())"),("kind='round2 fixed1500 slow residual actual physical evaluation'","kind='COMBINED_TURN_V1 frozencomposition actual256state researchdemonstration and originalstrictqualification'") ,("native_audit='NOT_PERFORMED'","native_audit='LIGHTWEIGHT_EVERY_NATIVE_STEP_NONFOOT_PEAK_ONLY'")]:src=replace_once(src,old,new)
 src=replace_once(src,"if args.condition=='neutral':","result['demonstration_stable_motion']=demonstration(args.condition,auto,cycles,delta,recorder.close())\n        result['engineering_qualified']=bool(result['engineering_pass'])\n        result['state_total']=256\n        result['training_performed']=False\n        if args.condition=='neutral':")
 return src

def build():
 src=adapted_source();namespace={'__file__':str(PRIMARY),'__name__':'_combined_frozen_rollout256'};exec(compile(src,str(PRIMARY)+'::COMBINED_TURN_V1_explicit256','exec'),namespace);pin();sys.path.insert(0,str(PEAK_SOURCE.parent));peak=io.module('_combined_native_peak',PEAK_SOURCE);fn=namespace['evaluate'];g=dict(fn.__globals__);g.update(HERE=HERE,ROOT=HERE,TRAIN=HERE,CHECKPOINT=CHECKPOINT,BASE=BASE,BASE_CHECKPOINT=BASE_CHECKPOINT,ORIGINAL=ORIGINAL,HEAD_AUDIT=HEAD_AUDIT,COMMANDS=COMMANDS,pin=pin,checked_load=checked_load,source_paths=source_paths,command_at=command_at,score=score,demonstration=demonstration,flatten=flatten,PeakRecorder=peak.PeakRecorder);return types.FunctionType(fn.__code__,g,fn.__name__,fn.__defaults__,fn.__closure__)
def main():
 a=argparse.ArgumentParser();a.add_argument('--condition',choices=list(COMMANDS),required=True);a.add_argument('--parent-resource-released',action='store_true');args=a.parse_args()
 if not args.parent_resource_released:raise ValueError('CODE_ONLY until parent compositionexecution release')
 import os,torch
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});torch.set_num_threads(1);run=argparse.Namespace(condition=args.condition,checkpoint=str(CHECKPOINT),sha256=sha256(CHECKPOINT),out=str(HERE/('EVAL_'+args.condition)),composition_ready=True,native=False);build()(run)
if __name__=='__main__':main()
