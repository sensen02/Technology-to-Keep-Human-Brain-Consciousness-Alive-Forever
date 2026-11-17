"""Strict resume300 storage, stdlib import only; preserve optimizer/anchor/RNG.
Originalfailed300VERIFY is required FAILURE evidence, never a PASS admission.
"""
from pathlib import Path
import importlib.util
import sys
import types

ROOT=Path(__file__).resolve().parent
SCOPE=ROOT.parent
PARENT=SCOPE/'target_feedback_calibration_env_recovery_v1'
PARENT_CHECKPOINT=PARENT/'TRAIN/checkpoint_000300.pt'
PARENT_SHA='b562fcc3e39c2aa56381a87b17f1e80124f203fa1a73e3cdf8888778de119f14'
FORMAT='INDEPENDENT_MOTOR_TARGET_FEEDBACK_V1'
SAVE_STEPS=(300,450,600)


def _module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

# Load only the originalstdlib storage definitions, no neural backend.
_original=_module('_orderrepair_immutable_storage_definitions',SCOPE/'target_feedback_calibration/common.py')
g=dict(_original.__dict__)
for _name,_value in list(g.items()):
 if isinstance(_value,types.FunctionType):g[_name]=types.FunctionType(_value.__code__,g,_value.__name__,_value.__defaults__,_value.__closure__)
sha=g['sha'];verify_hashes=g['verify_hashes'];_read=g['_read'];_preserve_rng=g['_preserve_rng'];restore_rng=g['restore_rng']


def resume_qualification_hashes():
 if sha(PARENT_CHECKPOINT)!=PARENT_SHA:raise ValueError('Actual immutable300 checkpoint SHArequired')
 result_path=SCOPE/'conditional_parity_diagnostic/RESULT.json';d=_read(result_path)
 if d['status']!='COMPLETED' or d['training_updates']!=0 or d['optimizer_operations']!=0 or d['Core_constructions']!=0 or d['physical_steps']!=0 or not d['all_direct_components_pass'] or not d['weights_unchanged'] or not d['frozen_old_verified'] or not d['rng_unchanged_entire_diagnostic'] or not d['source_hashes_unchanged'] or not d['no_physical_modules'] or d['checkpoint_sha256_after']!=PARENT_SHA:raise ValueError('Actualzero-updateorderdiagnosticqualificationrequired')
 if len(d['cases'])!=2 or {c['id'] for c in d['cases']}!={'stop','old_walk_control'}:raise ValueError('Actual exacttwo diagnosticcasesrequired')
 for case in d['cases']:
  if case['rows']!=1920 or not case['old256exact'] or not case['direct_components_pass'] or not case['direct_components_actual_oldmean_exact']:raise ValueError('Exactdirectcomponentactualcasequalification')
  for key in ('direct_components_JIT_vs_full','direct_components_Python_vs_full','direct_components_Python_vs_JIT','direct_components_forced_vs_full'):
   if any(v!=0. for v in case[key].values()):raise ValueError('Directcomponentdiagnosticmustbeexact')
 zero=_read(PARENT/'TRAIN/VERIFY_000000.json');failed=_read(PARENT/'TRAIN/VERIFY_000300.json')
 if zero['step']!=0 or not zero['integrity_all_pass'] or not all(zero['checks'].values()):raise ValueError('Original0 savedqualificationrequired')
 if failed['step']!=300 or failed['checkpoint_sha256']!=PARENT_SHA or failed['integrity_all_pass'] is not False or not any(v is False for v in failed['checks'].values()):raise ValueError('Preserveactualfailed300VERIFY as FAILUREevidence')
 paths=[PARENT_CHECKPOINT,result_path,PARENT/'TRAIN/VERIFY_000000.json',PARENT/'TRAIN/VERIFY_000300.json',PARENT/'INPUT_BINDINGS.json',PARENT/'PREFLIGHT.json',PARENT/'JOBS.json',Path(d['arrays']['path'])]
 if sha(d['arrays']['path'])!=d['arrays']['sha256']:raise ValueError('Actual diagnosticartifactSHA')
 for proof in (d['source_hashes_after'],d['diagnostic_source_hashes']):verify_hashes(proof)
 paths.extend(Path(p) for p in d['diagnostic_source_hashes'])
 for parentdir in (SCOPE/'conditional_parity_diagnostic',PARENT):
  paths.extend(parentdir.glob('*.log'))
  if (parentdir/'FAILURE.json').is_file():paths.append(parentdir/'FAILURE.json')
 for name in ('common.py','model.py','adapter.py','RECOVERY_PREREG.json'):
  path=PARENT/name
  if path.is_file():paths.append(path)
 return {str(p):sha(p) for p in dict.fromkeys(paths)}


def load_resume_parent():
 """No RNG restore here: returnpayload RNG for explicit post-Adamload restore."""
 resume_qualification_hashes();sentinel=object();aliases={k:sys.modules.get(k,sentinel) for k in ('model','common','sources','adapter')};paths=list(sys.path)
 try:
  sys.path.insert(0,str(PARENT))
  sys.modules['adapter']=_module('_orderrepair_original_recovery_adapter',PARENT/'adapter.py')
  loader=_module('_orderrepair_strict_original300_io',PARENT/'common.py')
  m,payload=loader.load_checkpoint(PARENT_CHECKPOINT,PARENT_SHA,final=False)
 finally:
  sys.path[:]=paths
  for k,v in aliases.items():
   if v is sentinel:sys.modules.pop(k,None)
   else:sys.modules[k]=v
 if payload['step']!=300 or payload['format_version']!=FORMAT:raise ValueError('Resumeactual300 format only')
 if 'initial_anchor' not in payload['config']:raise ValueError('Originalinitialanchorrequired,nofreshanchor')
 # Originalstrictloader validatesall8Adamcounters/moments=300 andfullRNG.
 return m,payload


def initialize():
 with _preserve_rng():
  parent,payload=load_resume_parent();backend=_module('_orderrepair_resume_model',ROOT/'model.py');m=backend.make_model(parent)
 return m,payload


def _check_config(config):
 if not isinstance(config,dict):raise ValueError('Configdictrequired')
 for key in ('source_hashes','data_hashes'):verify_hashes(config.get(key))
 for path in (ROOT/'model.py',ROOT/'common.py',PARENT/'model.py',PARENT/'common.py',g['OLD_DIR']/'model.py',g['OLD_DIR']/'model_io.py'):
  if config['source_hashes'].get(str(path))!=sha(path):raise ValueError('Newscope andresumeoriginal sourcebindingsrequired')
 if config['data_hashes'].get(str(PARENT_CHECKPOINT))!=PARENT_SHA or config['data_hashes'].get(str(g['OLD_CHECKPOINT']))!=g['OLD_SHA']:raise ValueError('Actual300 andoriginal789f resumeinputs mustbebound')
 for path,digest in resume_qualification_hashes().items():
  if config['data_hashes'].get(path)!=digest:raise ValueError('Resumequalificationproof missing fromdata_hashes')
 if 'initial_anchor' not in config:raise ValueError('Originalanchorrequired')
 # Valueequality tocheckpoint300 originalanchor is a numerical storagecheck,
 # not a newanchor constructedfromcurrentweights.
 _,parent_payload=load_resume_parent()
 import torch
 a=config['initial_anchor'];b=parent_payload['config']['initial_anchor']
 if not isinstance(a,dict) or a.keys()!=b.keys() or any(not torch.equal(a[k],b[k]) for k in b):raise ValueError('Original300 initial_anchor changed')
 return True

# Each cloned validator/load/save resolves only this newglobals dictionary.
g.update(ROOT=ROOT,SAVE_STEPS=SAVE_STEPS,FORMAT=FORMAT,initialize=initialize,_check_config=_check_config)
load_checkpoint=g['load_checkpoint'];save_checkpoint=g['save_checkpoint']
