"""Immutable old body/queue code with explicit newtimestamp/newfinal400 binding namespace."""
from pathlib import Path
import importlib.util,sys,types
PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype');ROOT=PROJECT/'experiments/neck-real-structure/20261113T000915Z/body_pilot_v2';CAL=ROOT.parent/'feedback_aggregation_v2';OLD_BODY=PROJECT/'experiments/neck-real-structure/20261112T225331Z/body_pilot'
_spec=importlib.util.spec_from_file_location('_aggregation_body_frozen_previous_runtime',OLD_BODY/'runtime.py');_old=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_old;_spec.loader.exec_module(_old)
_g=dict(_old.__dict__);_new_bridge=CAL/'TRAIN/checkpoint_000400.pt';_binding=ROOT/'INPUT_BINDINGS.json'
_digest=_old.read(_binding)['brain_sha256'] if _binding.is_file() else None
_g.update(ROOT=ROOT,CAL=CAL,BRIDGE=_new_bridge,BRIDGE_SHA=_digest,OLD_BODY=OLD_BODY,PROJECT=PROJECT)
for _name,_value in list(_g.items()):
 if isinstance(_value,types.FunctionType):_g[_name]=types.FunctionType(_value.__code__,_g,_value.__name__,_value.__defaults__,_value.__closure__)
_cls=_old.BrainNeckLoop;_attrs={key:(types.FunctionType(value.__code__,_g,value.__name__,value.__defaults__,value.__closure__) if isinstance(value,types.FunctionType) else value) for key,value in _cls.__dict__.items() if key not in ('__dict__','__weakref__')};_attrs['__module__']=__name__;_g['BrainNeckLoop']=type('BrainNeckLoop',_cls.__bases__,_attrs)
_g['_g']=_g;globals().update({key:value for key,value in _g.items() if not key.startswith('__')})
POLICY_FORMAT='SOURCE_NECK_FEEDBACK_AGGREGATION_V2';FINAL_STEP=400
def load_policies():
 if any(k in sys.modules for k in ('physics','mujoco','student_wave','flygym')):raise ValueError('Strict neural load before physical backend required')
 binding=read(ROOT/'INPUT_BINDINGS.json')
 if binding['brain_sha256']!=BRIDGE_SHA or binding['brain_final_step']!=400 or binding['brain_format']!=POLICY_FORMAT:raise ValueError('Actualnewfixed400sourcebinding required')
 sentinel=object();previous={name:sys.modules.get(name,sentinel) for name in ('model','common')};priorpath=list(sys.path)
 try:
  cio=module('_bodypilot_v2_calibration_common',CAL/'common.py');sys.modules['common']=cio
  cm=module('_bodypilot_v2_neural_architecture',CAL/'model.py');sys.modules['model']=cm;bridge,bp=cio.load_checkpoint(BRIDGE,BRIDGE_SHA,final=True)
  mm=module('_bodypilot_v2_motor_architecture',MOTOR_DIR/'model.py');sys.modules['model']=mm;sys.path.insert(0,str(MOTOR_DIR));mio=module('_bodypilot_v2_motor_io',MOTOR_DIR/'model_io.py');motor,mp=mio.load_checked(MOTOR,MOTOR_SHA,require_final=True)
 finally:
  sys.path[:]=priorpath
  for name,value in previous.items():
   if value is sentinel:sys.modules.pop(name,None)
   else:sys.modules[name]=value
 if any(sys.modules.get(name,sentinel) is not value for name,value in previous.items()):raise ValueError('Temporarymodel/common aliases notrestored')
 if bp['step']!=400 or bp['format_version']!=POLICY_FORMAT:raise ValueError('OnlyactualnewFORMATfinal400')
 if bridge.identity()['brain_hidden']!=16 or bridge.identity()['DN']!=2 or bridge.identity()['AN']!=1 or motor.identity()['state_dims']!=[128,64,64]:raise ValueError('Same272sourceABIrequired')
 bridge.requires_grad_(False);motor.requires_grad_(False);return bridge,motor,bp,mp
_g.update(load_policies=load_policies,POLICY_FORMAT=POLICY_FORMAT,FINAL_STEP=FINAL_STEP)
