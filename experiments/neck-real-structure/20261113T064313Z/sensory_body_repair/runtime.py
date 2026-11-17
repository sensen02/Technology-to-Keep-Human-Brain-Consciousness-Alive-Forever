"""Single complete namespace rebind over validated multisensory runtime; no heavy imports."""
from pathlib import Path
import importlib.util,sys,types
PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype');ROOT=Path(__file__).resolve().parent;CAL=ROOT.parent/'fusion_aggregation_repair';ODOR_DIR=ROOT.parent/'odor_memory_repair';OLD_BODY=PROJECT/'experiments/neck-real-structure/20261113T045302Z/sensory_body'
spec=importlib.util.spec_from_file_location('_sensoryrepair_previous_runtime',OLD_BODY/'runtime.py');old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
_PARENT_RUNTIME=old
g=dict(old.__dict__);bindingpath=ROOT/'INPUT_BINDINGS.json';bound=old.read(bindingpath) if bindingpath.is_file() else {};g.update(ROOT=ROOT,CAL=CAL,ODOR_DIR=ODOR_DIR,ODOR=ODOR_DIR/'TRAIN/checkpoint_000250.pt',ODOR_SHA=bound.get('odor_sha256'),BRIDGE=CAL/'TRAIN/checkpoint_000400.pt',FORMAT='SOURCE_NECK_MULTISENSORY_AGGREGATION_REPAIR_V1',KIND='SOURCE_NECK_MULTISENSORY_BODY_REPAIR_V1',SUMMARY_KIND='SOURCE_NECK_MULTISENSORY_BODY_REPAIR_SUMMARY_V1',OLD_BODY=OLD_BODY)
for name,value in list(g.items()):
 if isinstance(value,types.FunctionType):g[name]=types.FunctionType(value.__code__,g,value.__name__,value.__defaults__,value.__closure__)
for name in ('OdorField','BrainNeckLoop'):
 cls=getattr(old,name);attrs={k:(types.FunctionType(v.__code__,g,v.__name__,v.__defaults__,v.__closure__) if isinstance(v,types.FunctionType) else v) for k,v in cls.__dict__.items() if k not in ('__dict__','__weakref__')};attrs['__module__']=__name__;g[name]=type(name,cls.__bases__,attrs)
g['_g']=g;globals().update({k:v for k,v in g.items() if not k.startswith('__')})
ODOR_FORMAT='ODOR_NEURAL_OFFSET_REPAIR_V1'
def load_policies():
 if any(k in sys.modules for k in ('physics','mujoco','student_wave','flygym')):raise ValueError('AllfourstrictNNloadbeforephysicalbackend')
 binding=read(ROOT/'INPUT_BINDINGS.json')
 if binding['kind']!=KIND or binding['summary_kind']!=SUMMARY_KIND or binding['brain_format']!=FORMAT or binding['brain_final_step']!=400 or binding['odor_format']!=ODOR_FORMAT or binding['odor_final_step']!=250 or binding['odor_sha256']!=ODOR_SHA:raise ValueError('ActualtwoSHArepairbinding required')
 sentinel=object();prior={k:sys.modules.get(k,sentinel) for k in ('model','common','steering_behavior')};paths=list(sys.path)
 try:
  ci=module('_sensoryrepair_newfusion_common',CAL/'common.py');sys.modules['common']=ci;visual,odor,vp,op=ci.load_sensory();cm=module('_sensoryrepair_newfusion_model',CAL/'model.py');sys.modules['model']=cm;bridge,bp=ci.load_checkpoint(BRIDGE,binding['brain_sha256'],final=True)
  mm=module('_sensoryrepair_frozen_motor_arch',MOTOR_DIR/'model.py');sys.modules['model']=mm;sys.path.insert(0,str(MOTOR_DIR));mi=module('_sensoryrepair_frozen_motor_io',MOTOR_DIR/'model_io.py');motor,mp=mi.load_checked(MOTOR,MOTOR_SHA,require_final=True)
 finally:
  sys.path[:]=paths
  for k,v in prior.items():
   if v is sentinel:sys.modules.pop(k,None)
   else:sys.modules[k]=v
 if any(sys.modules.get(k,sentinel) is not v for k,v in prior.items()):raise ValueError('Restoremodel/common/steering aliases')
 if bp['format_version']!=FORMAT or bp['step']!=400 or visual.identity()['hidden']!=32 or odor.identity()['recurrent_state']!=32 or bridge.brain.input_size!=3 or bridge.brain.hidden_size!=16 or motor.identity()['state_dims']!=[128,64,64]:raise ValueError('Strictnew400/same336ABI')
 if op['format_version']!=ODOR_FORMAT or op['step']!=250:raise ValueError('Strictactualnewodor250FORMAT')
 if sha(ODOR)!=binding['odor_sha256'] or sha(BRIDGE)!=binding['brain_sha256']:raise ValueError('Actualnewodor250/brain400bytes')
 for m in (visual,odor,bridge,motor):m.requires_grad_(False)
 return visual,odor,bridge,motor,vp,op,bp,mp
g.update(load_policies=load_policies,ODOR_FORMAT=ODOR_FORMAT)
