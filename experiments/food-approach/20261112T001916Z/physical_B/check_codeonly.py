"""Stdlib-only syntax and two protocol nonregression checks; no admission/data/model."""
from pathlib import Path
import ast,json,sys,hashlib,types
from bindings import ROOT,AB,OLD,FORMAT
FILES=('bindings.py','prepare.py','runtime.py','runner.py','audit.py','summarize_B.py','preflight.py','orchestrate.py','check_codeonly.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 checks={}
 for name in FILES:
  path=ROOT/name;tree=ast.parse(path.read_text());compile(tree,str(path),'exec')
  imports=[]
  for node in tree.body:
   if isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
   if isinstance(node,ast.ImportFrom):imports.append(node.module)
  checks[name+'_syntax_and_light_top_imports']=not any(x and x.split('.')[0] in ('torch','numpy','mujoco','physics','common','matplotlib') for x in imports)
 # Execute ONLY frozen adapter's pure-stdlib build_source function, not its script/main.
 adapter=AB/'summarize_B_v2.py';tree=ast.parse(adapter.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_source')
 ns={'ROOT':AB};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(adapter)+'::stdlib_build_source_only','exec'),ns);source=ns['build_source']();compile(source,str(adapter)+'::not_executed','exec')
 checks['exact_frozen_v2_build_source_compiles']='effective_receptor_onset_elapsed_s' in source and 'first_response_latency_from_effective_input_s' in source and 'unique_physical_trials=6' in source
 checks['source_onset_latency_v1_not_used']='mask=window(.6,2.6)' not in source
 runtime=(ROOT/'runtime.py').read_text();proof=(ROOT/'bindings.py').read_text();prepare=(ROOT/'prepare.py').read_text()
 checks['new_summary_uses_build_source_new_globals']='adapter.build_source()' in runtime and "ns.update({k:v for k,v in g.items()" in runtime
 checks['full_ABv2_and_baseline_verified']="AB/'FROZEN_SUPPLEMENT.json'" in proof and "BASELINE/'VERIFY_neutral.json'" in proof and "audit['integrity_all_pass'] is True" in proof and "result['checkpoint_sha256']==audit['checkpoint_sha256']==MOTOR_SHA" in proof
 checks['prepare_and_runtime_both_inherit_v2proof']='inherited_v2_proof()' in prepare and 'inherited_v2_proof()' in runtime
 checks['paired_final500_loader_before_physics']="io.load_checked(g['SENSOR'],g['SENSOR_SHA'],require_final=True)" in runtime and "bp['format_version']==FORMAT and bp['curve'][-1]['step']==500" in runtime and FORMAT=='ODOR_PAIRED_BATCH_RECOVERY_V1'
 old={'ROOT':'old'};exec('def f(): return ROOT',old);new=dict(old);new['ROOT']='new';new['f']=types.FunctionType(old['f'].__code__,new)
 checks['explicit_function_globals_rebound']=new['f']()=='new' and old['f']()=='old'
 inherited=(OLD/'prepare.py').read_text();orchestrate=(OLD/'orchestrate.py').read_text()
 checks['retained_offline_ALLPASS_independent_gate']="r['all_targets_pass'] is True" in inherited and "v['all_targets_pass'] is True" in inherited and "v['integrity_all_pass'] is True" in inherited and "len(r['rows'])==12" in inherited
 checks['B_only_NEW_noodor_no_A_execution']="B=('B_noodor'" in orchestrate and 'for trial in B:' in orchestrate and 'for name in A:' not in orchestrate
 checks['no_admission_or_physics_outputs']=all(not (ROOT/name).exists() for name in ('INPUT_BINDINGS.json','FROZEN.json','PREFLIGHT.json','trials'))
 checks['heavy_imports_absent']=all(name not in sys.modules for name in ('torch','numpy','mujoco','physics','common','matplotlib','runtime'))
 result=dict(status='CANDIDATE3_PHYSICAL_B_CODE_ONLY_READY_NO_ADMISSION',all_passed=all(checks.values()),checks=checks,source_hashes={str(ROOT/n):sha(ROOT/n) for n in FILES},frozen_v2_adapter=str(adapter),frozen_v2_adapter_sha256=sha(adapter),model_loads=0,array_loads=0,forward_calls=0,Core_constructions=0,physics_steps=0,admission_generated=False)
 with (ROOT/'CODEONLY_CHECK.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result,indent=2));assert all(checks.values())
if __name__=='__main__':main()
