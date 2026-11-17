"""Candidate3 B-only namespace; no heavy import until parent-released namespace()."""
import sys,types
from pathlib import Path
from bindings import ROOT,CANDIDATE,AB,FORMAT,KIND,read,sha,require,module,clone,inherited_v2_proof

def namespace():
 old=module('_paired_behavior_frozen_AB_runtime',AB/'runtime.py');g=dict(old.__dict__)
 for key,fn in list(g.items()):
  if isinstance(fn,types.FunctionType) and fn.__module__==old.__name__:g[key]=clone(fn,g)
 binding=read(ROOT/'INPUT_BINDINGS.json')
 require(binding['kind']==KIND and binding['behavior_format']==FORMAT,'candidate3 bound format required')
 g.update(ROOT=ROOT,SENSOR_DIR=CANDIDATE,SENSOR=Path(binding['behavior_checkpoint']),SENSOR_SHA=binding['behavior_sha256'],INPUT_BINDINGS=binding)
 def verify_all():
  frozen=read(ROOT/'FROZEN.json')
  for p,h in frozen['source_hashes'].items():require(sha(p)==h,'physicalB frozen input drift '+p)
  for p,h in inherited_v2_proof()['source_hashes'].items():require(frozen['source_hashes'].get(p)==h,'missing inherited ABv2/baseline proof '+p)
  require(frozen['source_hashes'][str(ROOT/'INPUT_BINDINGS.json')]==sha(ROOT/'INPUT_BINDINGS.json'),'admission drift')
  return frozen
 def load_policies():
  verify_all()
  require(not any(n in sys.modules for n in ('physics','mujoco','student_wave','flygym')),'new behavior strictloader must precede physical imports')
  io=module('_paired_B_strict_common',CANDIDATE/'common.py');behavior,bp=io.load_checked(g['SENSOR'],g['SENSOR_SHA'],require_final=True)
  require(bp['format_version']==FORMAT and bp['curve'][-1]['step']==500,'paired format and exactfinal500 required')
  sys.path.insert(0,str(old.MOTOR_DIR));mi=module('_paired_B_strict_motor',old.MOTOR_DIR/'model_io.py');motor,mp=mi.load_checked(old.MOTOR,old.MOTOR_SHA,require_final=True)
  accept=read(old.ACCEPT);require(accept['checkpoint_sha256']==old.MOTOR_SHA and accept['visual_acceptance'] is True and accept['engineering_qualified'] is False,'original experimental visual admission')
  require(motor.identity()['state_dims']==[128,64,64] and behavior.identity()['recurrent_state']==32,'same motor256+behavior32 ABI')
  motor.requires_grad_(False);behavior.requires_grad_(False);return motor,behavior,mp,bp
 def source_paths(mp,bp):
  paths=[Path(p) for p in verify_all()['source_hashes']]
  paths += [Path(p) for key in ('source_hashes','component_source_hashes','stream_hashes') for p in mp[key]]
  paths += [Path(p) for key in ('source_hashes','data_hashes') for p in bp['config'][key]]
  for p in paths:require(p.is_absolute() and p.is_file(),'actual dependency missing '+str(p))
  return list(dict.fromkeys(paths))
 g.update(verify_frozen=verify_all,verify_all=verify_all,load_policies=load_policies,source_paths=source_paths,IDS=g['B_IDS'])
 return g

def entry(script,name):
 g=namespace()
 if script=='summarize_B.py':
  # Use the frozen v2 adaptation itself, not its obsolete v1 main function.
  sys.path.insert(0,str(AB));adapter=module('_paired_B_summary_v2',AB/'summarize_B_v2.py')
  ns={'__name__':'_paired_B_summary_v2_generated','__file__':str(AB/'summarize_B_v2.py')}
  exec(compile(adapter.build_source(),str(AB/'summarize_B_v2.py')+'::paired_B_new_globals','exec'),ns)
  fn=ns[name]
 else:
  old=module('_paired_B_frozen_'+script.removesuffix('.py'),AB/script);ns=dict(old.__dict__);fn=getattr(old,name)
 ns.update({k:v for k,v in g.items() if not k.startswith('__')});ns['ROOT']=ROOT
 return clone(fn,ns)
