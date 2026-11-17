"""Stdlib newnamespace/final400 and identicalphysicalalgorithm checks; no numerical imports."""
from pathlib import Path
import ast,json,hashlib,sys
import runtime
from adapter import tree,namespace,OLD_BODY
ROOT=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def nums(t):return [n.value for n in ast.walk(t) if isinstance(n,ast.Constant) and type(n.value) in (int,float)]
def strings(t):return [n.value for n in ast.walk(t) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
def main():
 checks={};names=('runtime.py','adapter.py','runner.py','audit.py','prepare.py','preflight.py','summary.py','orchestrate.py','check_codeonly.py')
 for n in names:compile((ROOT/n).read_text(),str(ROOT/n),'exec')
 checks['all9_sources_compile']=True
 checks['default_no_heavy_imports']=not any(k in sys.modules for k in ('numpy','torch','mujoco','physics','model','common'))
 checks['new_ROOT_CAL_checkpoint400']=runtime.ROOT==ROOT and runtime.CAL==ROOT.parent/'feedback_aggregation_v2' and runtime.BRIDGE==runtime.CAL/'TRAIN/checkpoint_000400.pt'
 checks['new_format_final400']=runtime.POLICY_FORMAT=='SOURCE_NECK_FEEDBACK_AGGREGATION_V2' and runtime.FINAL_STEP==400
 checks['unknown_SHA_not_guessed']=runtime.BRIDGE_SHA is None if not (ROOT/'INPUT_BINDINGS.json').exists() else runtime.BRIDGE_SHA==runtime.read(ROOT/'INPUT_BINDINGS.json')['brain_sha256']
 old=runtime._old
 checks['original_motor_neck_exact']=runtime.MOTOR_SHA==old.MOTOR_SHA and runtime.BINDING_SHA==old.BINDING_SHA and runtime.PORTS==old.PORTS and runtime.BASE==old.BASE
 for name in ('decision','advance','snapshot','finalize','__init__'):
  new=getattr(runtime.BrainNeckLoop,name);previous=getattr(old.BrainNeckLoop,name);checks['same_neck_loop_code_'+name]=new.__code__ is previous.__code__
  checks['new_class_globals_'+name]=new.__globals__['ROOT']==ROOT and new.__globals__['CAL']==runtime.CAL and new.__globals__['BINDING']==old.BINDING
 for name in ('cue','flat','project','pin','stable','transport_module'):
  new=getattr(runtime,name);previous=getattr(old,name);checks['same_helper_code_'+name]=new.__code__ is previous.__code__ and new.__globals__['ROOT']==ROOT
 for name in ('runner.py','audit.py','summary.py','orchestrate.py'):
  original=ast.parse((OLD_BODY/name).read_text());t=tree(name);checks['all_physical_numericconstants_unchanged_'+name]=nums(t)==nums(original)
  ns=namespace(name);checks['complete_namespace_'+name]=ns['ROOT']==ROOT and ns['CAL']==runtime.CAL and ns['BrainNeckLoop'] is runtime.BrainNeckLoop and ns['load_policies'] is runtime.load_policies
 prepare=tree('prepare.py');preflight=tree('preflight.py');texts=strings(prepare)
 checks['prepare_saved1000notaccidentallyused']='TRAIN/VERIFY_000400.json' in texts and 'TRAIN/VERIFY_000200.json' not in texts
 boundcalls=[n for n in ast.walk(prepare) if isinstance(n,ast.Call) and any(k.arg=='brain_final_step' for k in n.keywords)]
 checks['prepare_final400_binding_and_newformat']=len(boundcalls)==1 and any(k.arg=='brain_final_step' and k.value.value==400 for k in boundcalls[0].keywords) and any(k.arg=='brain_format' and k.value.value==runtime.POLICY_FORMAT for k in boundcalls[0].keywords)
 stepcomp=[n for n in ast.walk(preflight) if isinstance(n,ast.Compare) and isinstance(n.left,ast.Subscript) and isinstance(n.left.value,ast.Name) and n.left.value.id=='bp' and isinstance(n.left.slice,ast.Constant) and n.left.slice.value=='step']
 checks['only_actualstep_comparison400']=len(stepcomp)==1 and stepcomp[0].comparators[0].value==400
 checks['settle200_and_neural3080_preserved']=200 in nums(tree('audit.py')) and 3080 in nums(tree('runner.py')) and 1160 in nums(tree('audit.py'))
 loader=ast.unparse(next(n for n in ast.parse((ROOT/'runtime.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='load_policies'))
 checks['loader_both_common_model_temporarilybound']="('model', 'common')" in loader and "sys.modules['common'] = cio" in loader and "sys.modules['model'] = cm" in loader and 'finally:' in loader
 checks['publicpreparemain_unknownarggate']=any(isinstance(n,ast.FunctionDef) and n.name=='main' for n in ast.parse((ROOT/'prepare.py').read_text()).body)
 checks['no_new_source_admission_or_physics']=not any((ROOT/n).exists() for n in ('INPUT_BINDINGS.json','FROZEN.json','PREFLIGHT.json','JOBS.json'))
 pr=json.loads((ROOT/'PREREG.json').read_text());checks['same6conditions_budget']=pr['conditions']==list(runtime.IDS) and pr['budget']['physical_trials']==6
 if not all(checks.values()):raise ValueError('BODY_V2_CODEONLY '+str({k:v for k,v in checks.items() if not v}))
 with (ROOT/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes={str(ROOT/n):sha(ROOT/n) for n in (*names,'PREREG.json')},heavy_imports=False,arrays_loaded=0,model_forward_calls=0,Core_constructions=0,physics_steps=0,training_updates=0,interpretation='No unknownfixed400bodyadmission. Parentmustbindactualcompletedproofs then0Corepreflight and6newfullphysicaltrials.'),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=False)))
if __name__=='__main__':main()
