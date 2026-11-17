"""Stdlibactual causal-target helper test and recovery namespace contract checks, no model import."""
from pathlib import Path
import ast,json,hashlib,sys,importlib.util
HERE=Path(__file__).resolve().parent;ORIGINAL=HERE.parent/'sensory_fusion_calibration'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def main():
 if (HERE/'CODEONLY_CHECK.json').exists():raise FileExistsError('Inspectbeforecheckoverwrite')
 checks={};names=('common.py','model.py','data.py','preflight.py','train.py','verify_saved.py','evaluate.py','run.py','codecheck.py');trees={}
 for name in names:trees[name]=ast.parse((HERE/name).read_text());compile(trees[name],str(HERE/name),'exec');checks['syntax_'+name]=True
 c=module('_fusion_recovery_stdlib_common',HERE/'common.py');checks['newHERE_FORMAT']=c.HERE==HERE and c.FORMAT=='SOURCE_NECK_VISUAL_OLFACTORY_FUSION_RECOVERY_V1';checks['original_rootcorrect']=c.ORIGINAL==ORIGINAL;checks['source_names27']=len(c.TRAIN)==27 and len(c.DEV)==8
 sentinel=object();old=sys.modules.get('common',sentinel)
 try:
  sys.modules['common']=c
  for name in ('preflight.py','train.py','verify_saved.py','evaluate.py'):
   m=module('_fusion_recovery_contract_'+name.replace('.','_'),HERE/name);checks['namespace_'+name]=callable(m.main) and m.main.__globals__['HERE']==HERE and m.main.__globals__['FORMAT']==c.FORMAT
  entry=module('_fusion_recovery_contract_run',HERE/'run.py');checks['orchestrator_newroot']=entry.main.__globals__['HERE']==HERE;checks['orchestrator_same_codeobject']=entry.main.__code__.co_code==module('_fusion_recovery_comparison_original_run',ORIGINAL/'run.py').main.__code__.co_code
  data=module('_fusion_recovery_contract_data',HERE/'data.py');checks['actual_arrays_globals_newHERE']=data.arrays.__globals__['HERE']==HERE and data.arrays.__module__=='_fusion_recovery_contract_data';checks['data_main_newHERE']=data.main.__globals__['HERE']==HERE;checks['common_frozen_newHERE']=c.frozen.__globals__['HERE']==HERE and c.load_checkpoint.__globals__['HERE']==HERE and c.initialize.__globals__['HERE']==HERE
 finally:
  if old is sentinel:sys.modules.pop('common',None)
  else:sys.modules['common']=old
 # Execute precisely the new causal_target AST with stdlib list stand-ins, not a copied test algorithm.
 fn=next(n for n in trees['data.py'].body if isinstance(n,ast.FunctionDef) and n.name=='causal_target');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(HERE/'data.py')+'::stdlib_list_shape_test','exec'),ns)
 class ListStandin:
  @staticmethod
  def zeros_like(v):return [0. for _ in v]
  @staticmethod
  def concatenate(parts,axis=0):
   if axis!=0:raise ValueError('Onlyaxis0')
   return [item for part in parts for item in part]
 desired=[float(i+1) for i in range(1920)];actual=ns['causal_target'](desired,ListStandin);checks['actual_helper_length1920']=len(actual)==1920;checks['actual_helper_first0']=actual[0]==0.;checks['actual_helper_shift1919']=actual[1:]==desired[:-1];checks['targetAST_firstslice1']=any(isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='desired' and isinstance(n.slice,ast.Slice) and isinstance(n.slice.upper,ast.Constant) and n.slice.upper.value==1 for n in ast.walk(fn))
 text=(HERE/'data.py').read_text();checks['source_inputs_unchanged_proof']=all(s in text for s in ("'body111','modalities','current_desired','mean','std'", "np.array_equal(value[key],z[key])", "target[1:],desired[:-1]", "original_shape==[3839,expected,1]"));checks['no_sensory_reextraction']=not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('load_sensory','visual_replay','odor_replay') for n in ast.walk(trees['data.py']))
 j=json.loads((ORIGINAL/'JOBS.json').read_text());checks['actual122_only_preflight_exit1']=len(j)==1 and j[0]['stage']=='PREFLIGHT' and j[0]['exit_code']==1;checks['actual122_failurematches']='Fusionpreflight actual27full1920' in (ORIGINAL/'PREFLIGHT_EXECUTION.log').read_text();checks['original_no_training_frozen']=not (ORIGINAL/'TRAIN').exists() and not (ORIGINAL/'FROZEN.json').exists();proof=json.loads((ORIGINAL/'SENSORY_REPLAY.json').read_text());checks['actual_successful_source_extraction_kept']=proof['all_bitexact'] and proof['actual_visual_cases']==8 and proof['actual_odor_cases']==12
 for name in ('model.py','preflight.py','train.py','verify_saved.py','evaluate.py'):checks['oldcore_exists_'+name]=(ORIGINAL/name).is_file()
 checks['no_heavy_modules']=not any(k in sys.modules for k in ('numpy','torch','mujoco','physics','student_wave','flygym'));checks['new_not_executed']=not any((HERE/n).exists() for n in ('DATASET.json','FROZEN.json','TRAIN','JOBS.json'))
 if not all(checks.values()):raise ValueError('Recoverycontract '+str({k:v for k,v in checks.items() if not v}))
 paths=[HERE/n for n in names]+[HERE/'PREREG.json']+[ORIGINAL/n for n in names]+[ORIGINAL/n for n in ('PREREG.json','DATASET.json','SENSORY_REPLAY.json','PREFLIGHT_EXECUTION.log','PIPELINE.log','JOBS.json','train.npz','development.npz')];before={str(p):sha(p) for p in paths}
 with (HERE/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes=before,numerical_imports=False,array_payload_reads=0,model_loads=0,forward_calls=0,training_updates=0,physics_steps=0,original_failures_preserved=True),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),model_loads=0,numerical_imports=False,actual_stdlib_helper_length=len(actual))))
if __name__=='__main__':main()
