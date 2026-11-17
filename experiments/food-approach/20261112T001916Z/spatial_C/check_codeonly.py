"""Stdlib-only Cproposal/sourceadapter checks; no data/model/physical execution."""
from pathlib import Path
import ast,json,sys,hashlib,importlib.util
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 names=('contracts.py','spatial.py','runtime.py','runner.py','audit.py','prepare.py','preflight.py','summary.py','orchestrate.py','check_codeonly.py');checks={}
 for name in names:ast.parse((ROOT/name).read_text());compile((ROOT/name).read_text(),str(ROOT/name),'exec')
 checks['all10syntaxcompile']=True
 sys.path.insert(0,str(ROOT));spec=importlib.util.spec_from_file_location('_spatial_C_codeonly_runtime',ROOT/'runtime.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 for script in ('runner.py','audit.py'):compile(m.source(script),'C::'+script,'exec')
 checks['all_exactsourceadaptationscompile']=True
 runner=m.source('runner.py');audit=m.source('audit.py');tree=ast.parse(runner)
 checks['Cstage_not_A_or_B_gate']="stage='C'" in runner
 checks['actual_source_anchor_once_before_handover']='sourcecenter=anchor_from_head' in runner and 'geometry[-1].update(environment_record' in runner
 checks['causal_filter_inherited']='sensory=filter.advance(concentration_at(' in runner and 'g[\'sensor_xyz\'][999+5*k]' in audit
 writes=[n for n in ast.walk(tree) if isinstance(n,(ast.Assign,ast.AugAssign,ast.AnnAssign)) and any(t in ast.unparse(n).split('=')[0] for t in ('.qpos','.qvel'))]
 checks['no_actual_posewrites']=not writes
 checks['arrival_does_not_control_actions']='arrived_endpoint_proxy' not in runner and 'arrival_metrics' in (ROOT/'runtime.py').read_text()
 pr=json.loads((ROOT/'PREREG_PROPOSAL.json').read_text());checks['fixed6proposal_nonewreuse']=len(pr['layouts'])==6 and 'runnew6th' in pr['source']['noodor']
 checks['conservative201pointspancontract']='>=201' in (ROOT/'spatial.py').read_text() and 'span>=.2-1e-8' in (ROOT/'spatial.py').read_text()
 checks['B_allpass_prepare_admission']="s['all_functional_pass'] is True" in (ROOT/'contracts.py').read_text() and 'hashes=admission()' in (ROOT/'prepare.py').read_text()
 checks['no_heavy_imports']=not any(n in sys.modules for n in ('numpy','torch','mujoco','physics','scipy'))
 checks['not_executionfrozen_or_run']=not any((ROOT/n).exists() for n in ('FROZEN.json','INPUT_BINDINGS_C.json','PREFLIGHT.json','trials','JOBS_C.json'))
 assert all(checks.values()),checks
 with (ROOT/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(status='C6_CODEONLY_READY_B_ADMISSION_AND_PARENT_RELEASE_PENDING',all_passed=True,checks=checks,source_hashes={str(ROOT/n):sha(ROOT/n) for n in names+('PREREG_PROPOSAL.json',)},Core_constructions=0,model_loads=0,forward_calls=0,arrays_loaded=0,physics_steps=0,training_updates=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=False,execution_frozen=False,physics_steps=0)))
if __name__=='__main__':main()
