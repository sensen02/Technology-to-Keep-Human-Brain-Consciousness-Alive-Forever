"""Explicit rebound frozen AB functions. Imports heavy modules only via released entries."""
from pathlib import Path
import importlib.util,sys,types
ROOT=Path(__file__).resolve().parent;CANDIDATE=ROOT.parent;AB=CANDIDATE.parent/'20261111T224234Z'
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def namespace():
 old=module('_new_behavior_frozen_AB_runtime',AB/'runtime.py')
 g=dict(old.__dict__)
 # Every inherited function receives NEW globals, including concentration/read helpers.
 for key,value in list(g.items()):
  if isinstance(value,types.FunctionType) and value.__module__==old.__name__:g[key]=types.FunctionType(value.__code__,g,value.__name__,value.__defaults__,value.__closure__)
 bindings=old.read(ROOT/'INPUT_BINDINGS.json')
 g.update(ROOT=ROOT,SENSOR_DIR=CANDIDATE,SENSOR=Path(bindings['behavior_checkpoint']),SENSOR_SHA=bindings['behavior_sha256'],INPUT_BINDINGS=bindings)
 def verify_frozen():
  frozen=old.read(ROOT/'FROZEN.json')
  for p,h in frozen['source_hashes'].items():
   if old.sha(p)!=h:raise ValueError('new physical_B source/input drift '+p)
  if old.sha(ROOT/'INPUT_BINDINGS.json')!=frozen['source_hashes'][str(ROOT/'INPUT_BINDINGS.json')]:raise ValueError('immutable admission changed')
  return frozen
 def load_policies():
  verify_frozen()
  if 'physics' in sys.modules or 'mujoco' in sys.modules:raise ValueError('new behavior strictload MUST precede physical backend')
  io=module('_candidate2_physical_strict_behavior',CANDIDATE/'common.py')
  behavior,bp=io.load_checked(g['SENSOR'],g['SENSOR_SHA'],require_final=True)
  if bp['format_version']!='ODOR_SEGMENT_RECOVERY_V1' or bp['curve'][-1]['step']!=500:raise ValueError('new candidate2 final500 only')
  sys.path.insert(0,str(old.MOTOR_DIR));motorio=module('_candidate2_physical_strict_motor',old.MOTOR_DIR/'model_io.py');motor,mp=motorio.load_checked(old.MOTOR,old.MOTOR_SHA,require_final=True)
  accept=old.read(old.ACCEPT)
  if accept['checkpoint_sha256']!=old.MOTOR_SHA or accept['visual_acceptance'] is not True or accept['engineering_qualified'] is not False:raise ValueError('accepted experimental motion baseline required')
  if motor.identity()['state_dims']!=[128,64,64] or behavior.identity()['recurrent_state']!=32:raise ValueError('same256+32 ABI required')
  motor.requires_grad_(False);behavior.requires_grad_(False);return motor,behavior,mp,bp
 def source_paths(mp,bp):
  paths=[Path(p) for p in verify_frozen()['source_hashes']]
  paths += [Path(p) for key in ('source_hashes','component_source_hashes','stream_hashes') for p in mp[key]]
  paths += [Path(p) for key in ('source_hashes','data_hashes') for p in bp['config'][key]]
  for p in paths:
   if not p.is_absolute() or not p.is_file():raise FileNotFoundError(p)
  return list(dict.fromkeys(paths))
 g.update(verify_frozen=verify_frozen,load_policies=load_policies,source_paths=source_paths)
 # Inherited concentration() reads the exact original CLAMP_LEVELS bytes via new path.
 g['IDS']=g['B_IDS']
 return g

def entry(script,name):
 g=namespace();specname='_candidate2_frozen_AB_'+script.removesuffix('.py');old=module(specname,AB/script)
 merged=dict(old.__dict__);merged.update({k:v for k,v in g.items() if not k.startswith('__')});merged['ROOT']=ROOT
 fn=getattr(old,name)
 return types.FunctionType(fn.__code__,merged,fn.__name__,fn.__defaults__,fn.__closure__)
