"""Strict300 resume, explicit scientific-global rebinding only; stdlibhelpers preserved."""
from pathlib import Path
import importlib.util,sys,types
ROOT=Path(__file__).resolve().parent
SCOPE=ROOT.parent
V2=SCOPE/'target_feedback_order_repair_v2'
PARENT=SCOPE/'target_feedback_calibration_env_recovery_v1'
PARENT_CHECKPOINT=PARENT/'TRAIN/checkpoint_000300.pt'
PARENT_SHA='b562fcc3e39c2aa56381a87b17f1e80124f203fa1a73e3cdf8888778de119f14'
FORMAT='INDEPENDENT_MOTOR_TARGET_FEEDBACK_V1'
SAVE_STEPS=(300,450,600)
def _module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
_original=_module('_orderrepairv3_original_storage',SCOPE/'target_feedback_calibration/common.py')
g=dict(_original.__dict__)
# ONLY these scientific operations require relocated globals. Imported decorators,
# contextmanager wrappers, hashes and RNG utilities retain their defining globals.
REBIND_SCIENCE=('_validate_payload','load_checkpoint','save_checkpoint')
for name in REBIND_SCIENCE:
 value=g[name]
 if not isinstance(value,types.FunctionType) or value.__code__.co_filename.endswith('contextlib.py') or hasattr(value,'__wrapped__'):raise ValueError('Scientificwhitelist cannot include imported/decorated helpers')
 g[name]=types.FunctionType(value.__code__,g,value.__name__,value.__defaults__,value.__closure__)
sha=g['sha'];verify_hashes=g['verify_hashes'];_read=g['_read'];_preserve_rng=g['_preserve_rng'];restore_rng=g['restore_rng']
# The immutable v2 qualifier contains no decorated/global relocation; use it as a
# sourcequalified pure stdlib proof, then extend with v2 entry failure evidence.
_v2=_module('_orderrepairv3_failed_v2_metadata',V2/'common.py')
def resume_qualification_hashes():
 proof=_v2.resume_qualification_hashes()
 for path in list(V2.glob('*.py'))+[V2/'PREREG.json',V2/'INPUT_BINDINGS.json']+list(V2.glob('*.log')):
  if path.is_file():proof[str(path)]=sha(path)
 if (V2/'FAILURE.json').is_file():proof[str(V2/'FAILURE.json')]=sha(V2/'FAILURE.json')
 return proof
def load_resume_parent():
 resume_qualification_hashes();sentinel=object();aliases={k:sys.modules.get(k,sentinel) for k in ('model','common','sources','adapter')};paths=list(sys.path)
 try:
  sys.path.insert(0,str(PARENT));sys.modules['adapter']=_module('_v3_actual_recovery_adapter',PARENT/'adapter.py');loader=_module('_v3_strict_original300_io',PARENT/'common.py');m,p=loader.load_checkpoint(PARENT_CHECKPOINT,PARENT_SHA,final=False)
 finally:
  sys.path[:]=paths
  for k,v in aliases.items():
   if v is sentinel:sys.modules.pop(k,None)
   else:sys.modules[k]=v
 if p['step']!=300 or p['format_version']!=FORMAT or 'initial_anchor' not in p['config']:raise ValueError('Actual300originalanchorrequired')
 return m,p
def initialize():
 with _preserve_rng():
  parent,payload=load_resume_parent();backend=_module('_orderrepairv3_resume_model',ROOT/'model.py');m=backend.make_model(parent)
 return m,payload
def _check_config(config):
 if not isinstance(config,dict):raise ValueError('Configdictrequired')
 for key in ('source_hashes','data_hashes'):verify_hashes(config.get(key))
 for path in (ROOT/'model.py',ROOT/'common.py',PARENT/'model.py',PARENT/'common.py',g['OLD_DIR']/'model.py',g['OLD_DIR']/'model_io.py'):
  if config['source_hashes'].get(str(path))!=sha(path):raise ValueError('Newandresume sourcebindingsrequired')
 if config['data_hashes'].get(str(PARENT_CHECKPOINT))!=PARENT_SHA or config['data_hashes'].get(str(g['OLD_CHECKPOINT']))!=g['OLD_SHA']:raise ValueError('Actual300and789finputbinding')
 for path,h in resume_qualification_hashes().items():
  if config['data_hashes'].get(path)!=h:raise ValueError('Resumeproofdatahashrequired')
 _,p=load_resume_parent()
 import torch
 a=config.get('initial_anchor');b=p['config']['initial_anchor']
 if not isinstance(a,dict) or a.keys()!=b.keys() or any(not torch.equal(a[k],b[k]) for k in b):raise ValueError('Originalinitialanchorchanged')
 return True
g.update(ROOT=ROOT,SAVE_STEPS=SAVE_STEPS,FORMAT=FORMAT,initialize=initialize,_check_config=_check_config)
load_checkpoint=g['load_checkpoint'];save_checkpoint=g['save_checkpoint']
