"""Actual RGB far-body controls with explicit new neural-opponent policy identity."""
from pathlib import Path
import importlib.util,sys,types
PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype');VIS=PROJECT/'experiments/visual-orientation/20261112T045316Z'
ROOT=VIS/'landmark_opponent_v7/body';POLICY=VIS/'landmark_opponent_v7/policy';COVERAGE_BODY=VIS/'landmark_coverage_v5/body'
_s=importlib.util.spec_from_file_location('_opponent_body_frozen_coverage_runtime',COVERAGE_BODY/'runtime.py');_coverage=importlib.util.module_from_spec(_s);sys.modules[_s.name]=_coverage;_s.loader.exec_module(_coverage)
_g=dict(_coverage.__dict__);_g.update(ROOT=ROOT,POLICY=POLICY,VIS=VIS,PROJECT=PROJECT,COVERAGE_BODY=COVERAGE_BODY)
for _key,_value in list(_g.items()):
 if isinstance(_value,types.FunctionType):_g[_key]=types.FunctionType(_value.__code__,_g,_value.__name__,_value.__defaults__,_value.__closure__)
for _name in ('_OpticalProxy','CameraStream'):
 _cls=getattr(_coverage,_name);_attrs={key:(types.FunctionType(value.__code__,_g,value.__name__,value.__defaults__,value.__closure__) if isinstance(value,types.FunctionType) else value) for key,value in _cls.__dict__.items() if key not in ('__dict__','__weakref__')};_attrs['__module__']=__name__;_g[_name]=type(_name,_cls.__bases__,_attrs)
_g['_g']=_g;globals().update({key:value for key,value in _g.items() if not key.startswith('__')})
POLICY_FORMAT='VISION_OPPONENT_POLICY_V7';FINAL_STEP=1000

def load_policies():
 if any(k in sys.modules for k in ('physics','mujoco','student_wave','flygym')):raise ValueError('new opponent model strictly before physical backend')
 binding=read(ROOT/'INPUT_BINDINGS.json')
 if binding['visual_format']!=POLICY_FORMAT:raise ValueError('actual opponent policy format required')
 sentinel=object();previous=sys.modules.get('model',sentinel);priorpath=list(sys.path)
 try:
  architecture=module('_body7_new_opponent_architecture',POLICY/'model.py');sys.modules['model']=architecture
  io=module('_body7_strict_opponent_common',POLICY/'common.py');visual,vp=io.load_checked(binding['visual_checkpoint'],binding['visual_sha256'],require_final=True)
  if vp['format_version']!=POLICY_FORMAT or vp['step']!=1000 or visual.identity()['format']!='VISION_FIXED_NEURAL_OPPONENT_ARCH_V7':raise ValueError('new neural opponent final1000 required')
  import torch
  if not torch.equal(visual.opponent_weight,torch.tensor([1.,-.5,-.5],dtype=torch.float64).reshape(1,3,1,1)) or visual.opponent_bias.item()!=0 or visual.conv1.in_channels!=1:raise ValueError('fixed declared neural receptor identity changed')
  mm=module('_body7_original_combined_architecture',MOTOR_DIR/'model.py');sys.modules['model']=mm;sys.path.insert(0,str(MOTOR_DIR));mio=module('_body7_strict_motor_common',MOTOR_DIR/'model_io.py');motor,mp=mio.load_checked(MOTOR,MOTOR_SHA,require_final=True)
 finally:
  sys.path[:]=priorpath
  if previous is sentinel:sys.modules.pop('model',None)
  else:sys.modules['model']=previous
 if (previous is sentinel and 'model' in sys.modules) or (previous is not sentinel and sys.modules.get('model') is not previous):raise ValueError('temporary model aliases not restored')
 if visual.identity()['hidden']!=32 or visual.identity()['ordered_feature']!=16 or motor.identity()['state_dims']!=[128,64,64]:raise ValueError('same288 neuralmemory and16cache required')
 visual.requires_grad_(False);motor.requires_grad_(False);return visual,motor,vp,mp
_g.update(load_policies=load_policies,POLICY_FORMAT=POLICY_FORMAT,FINAL_STEP=FINAL_STEP)
