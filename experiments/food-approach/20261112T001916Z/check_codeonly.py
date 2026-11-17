"""Stdlib-only exact source adapters andfixedpairedsampler contracts; no model execution."""
from pathlib import Path
import ast,json,hashlib,importlib.util,sys
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 names=('common.py','data.py','engine.py','generate_data.py','preflight.py','pair_checks.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py','adapt_source.py');trees={n:ast.parse((HERE/n).read_text()) for n in names};checks={}
 for name in names:compile((HERE/name).read_text(),str(HERE/name),'exec')
 checks['all12_AST_compile']=True
 spec=importlib.util.spec_from_file_location('_paired_codeonly_adapters',HERE/'adapt_source.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);adapted={name:m.build(name) for name in ('preflight.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py')}
 for name,s in adapted.items():compile(s,str(HERE/name)+'::adapted','exec')
 checks['all6_exactadaptations_compile']=True
 tr=ast.parse(adapted['train.py']);calls=[n for n in ast.walk(tr) if isinstance(n,ast.Call)];choice=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='choice']
 checks['one_uniform_choice12pairs2']=len(choice)==1 and [ast.literal_eval(n) for n in choice[0].args]==[12,2] and any(k.arg=='replace' and ast.literal_eval(k.value) is False for k in choice[0].keywords)
 checks['no_extra_shuffle']=not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('shuffle','permutation') for n in calls)
 checks['fixed500range']=any(isinstance(n.func,ast.Name) and n.func.id=='range' and len(n.args)==2 and all(isinstance(x,ast.Constant) for x in n.args) and [x.value for x in n.args]==[1,501] for n in calls)
 checks['savedpairandexpandedhistories']="pair_schedule=pair_schedule" in adapted['train.py'] and 'actual_pair_schedule_exact' in adapted['verify_saved.py'] and 'actual_expanded_batch_schedule_exact' in adapted['verify_saved.py']
 checks['validation_only500']='checkpoint_000500.pt' in adapted['evaluate.py'] and 'VERIFY_000500.json' in adapted['evaluate.py']
 checks['runtimearchitecture_not_redefined']=not any(isinstance(n,ast.ClassDef) for n in trees['common.py'].body)
 checks['descriptivestats_not_removeRMS']='offset_signed_mean' in adapted['evaluate.py'] and 'flags[\'offset_RMS\']=rms<=.05' in adapted['evaluate.py']
 checks['no_heavy_imports_duringcheck']=not any(n in sys.modules for n in ('numpy','torch','mujoco','physics','scipy'))
 checks['no_actualoutputs_created']=not any((HERE/n).exists() for n in ('TRAIN','VALIDATION','DATASET.json','FROZEN.json','PREFLIGHT.json','train.npz','validation.npz','JOBS.json'))
 pr=json.loads((HERE/'PREREG.json').read_text());checks['fixedfc3a500budget']=pr['training']['updates']==500 and pr['parent_sha256']=='fc3a0c20f854afe162afdf3c87cbddea2b8879a7e29d3cde97f487640e101641'
 assert all(checks.values()),checks
 with (HERE/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(status='CODEONLY_READY_PARENT_RUNS',all_passed=True,checks=checks,source_hashes={str(HERE/n):sha(HERE/n) for n in names+('PREREG.json','check_codeonly.py')},heavy_imports=0,model_loads=0,Core_constructions=0,forward_calls=0,optimizer_updates=0,physics_steps=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=False,forward_calls=0,optimizer_updates=0,physics_steps=0)))
if __name__=='__main__':main()
