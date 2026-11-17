"""Small stdlib-only wrapper checks. No actual admission or numerical imports."""
from pathlib import Path
import ast,json,hashlib,sys,types
ROOT=Path(__file__).resolve().parent
FILES=('prepare.py','runtime.py','runner.py','audit.py','summarize_B.py','preflight.py','orchestrate.py','check_codeonly.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 checks={}
 for name in FILES:
  path=ROOT/name;tree=ast.parse(path.read_text());compile(tree,str(path),'exec');checks[name+'_syntax']=True
  tops=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
  imported=[a.name for n in tops if isinstance(n,ast.Import) for a in n.names]+[n.module for n in tops if isinstance(n,ast.ImportFrom)]
  checks[name+'_light_top_imports']=not any(x and x.split('.')[0] in ('torch','numpy','mujoco','physics','common') for x in imported)
 # Demonstrate why replacing only a module attribute is insufficient and our clone rebinds it.
 old={'ROOT':'old'};exec('def f(): return ROOT',old);new=dict(old);new['ROOT']='new';new['f']=types.FunctionType(old['f'].__code__,new)
 checks['function_globals_explicitly_rebound']=old['f']()=='old' and new['f']()=='new'
 runtime=(ROOT/'runtime.py').read_text();orchestrator=(ROOT/'orchestrate.py').read_text();prep=(ROOT/'prepare.py').read_text()
 checks['strict_new_loader_before_physical_pin']='io.load_checked(g[\'SENSOR\'],g[\'SENSOR_SHA\'],require_final=True)' in runtime and "if 'physics' in sys.modules or 'mujoco' in sys.modules" in runtime
 checks['final500_all12_independent_gate_required']="r['all_targets_pass'] is True" in prep and "v['integrity_all_pass'] is True" in prep and "checkpoint_000500.pt" in prep and 'len(r[\'rows\'])==12' in prep
 checks['B_only_new_noodor_no_A_execution']="B=('B_noodor'" in orchestrator and 'for trial in B:' in orchestrator and 'for name in A:' not in orchestrator
 checks['unknown_SHA_no_admission_written']=not (ROOT/'INPUT_BINDINGS.json').exists() and not (ROOT/'FROZEN.json').exists() and not (ROOT/'trials').exists()
 heavy=('torch','numpy','mujoco','physics','common','runtime')
 checks['no_heavy_imports']=all(x not in sys.modules for x in heavy)
 result=dict(status='CODE_ONLY_WRAPPERS_READY_OFFLINE_ADMISSION_PENDING',all_passed=all(checks.values()),checks=checks,source_hashes={str(ROOT/name):sha(ROOT/name) for name in FILES},model_loads=0,forward_calls=0,array_loads=0,Core_constructions=0,physics_steps=0,admission_generated=False)
 with (ROOT/'CODEONLY_CHECK.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result,indent=2));assert all(checks.values())
if __name__=='__main__':main()
