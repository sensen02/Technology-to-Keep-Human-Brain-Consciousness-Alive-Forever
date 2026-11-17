"""Stdlib-only source and exact-adapter validation, no numerical/model imports."""
from pathlib import Path
import ast,json,hashlib,importlib.util,sys
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 names=('common.py','data.py','engine.py','generate_data.py','preflight.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py','adapt_source.py');trees={n:ast.parse((HERE/n).read_text()) for n in names}
 pr=json.loads((HERE/'PREREG.json').read_text());checks={}
 for name in names:compile((HERE/name).read_text(),str(HERE/name),'exec')
 checks['all11_AST_compile']=True
 spec=importlib.util.spec_from_file_location('_codeonly_segment_adapters',HERE/'adapt_source.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
 adapted={name:a.build(name) for name in ('preflight.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py')}
 for name,source in adapted.items():compile(source,str(HERE/name)+'::exactinherited','exec')
 checks['all6_minimal_adaptations_compile']=True
 tr=ast.parse(adapted['train.py']);loop=next(n for n in ast.walk(tr) if isinstance(n,ast.For) and isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='range' and any(isinstance(arg,ast.Constant) and arg.value==501 for arg in n.iter.args));checks['exact_updates1_to500']=ast.unparse(loop.iter)=='range(1, 501)'
 checks['save_steps250_500']='if step in (250,500)' in adapted['train.py'] or 'if step in (250, 500)' in adapted['train.py']
 checks['finalcheckpoint500_only']='checkpoint_000500.pt' in adapted['evaluate.py'] and 'VERIFY_000500.json' in adapted['evaluate.py'] and 'checkpoint_001000.pt' not in adapted['evaluate.py']
 checks['saveaudit_0_250_500']='choices=(0,250,500)' in adapted['verify_saved.py'] or 'choices=(0, 250, 500)' in adapted['verify_saved.py']
 checks['pipeline250_500_only']="['--step','250']" in adapted['run_candidate.py'] and "['--step','1000']" not in adapted['run_candidate.py']
 checks['parent4be0_fixed']=pr['parent_sha256']=='4be0cd6a9d1f77e3be17323713f3fc8783adedd4b9e8a010df69922c7d1a09e7'
 checks['same500budget_andfourgroups']=pr['training']['updates']==500 and [pr['pulse_segments'][k]['mass'] for k in ('ON','TRANSITION','RECOVERY','BASELINE')]==[.3,.2,.4,.1]
 checks['two_newvalidationconfigs12']=len(pr['data']['motifs'])*len(pr['data']['validation_variants'])==12
 checks['reuse_frozen_originalengine']=not any(isinstance(n,ast.ClassDef) for n in trees['engine.py'].body) and 'objective=legacy.objective' in (HERE/'engine.py').read_text()
 checks['no_behavior_architecture_redefinition']=not any(isinstance(n,ast.ClassDef) for n in trees['common.py'].body)
 checks['no_heavy_imported_duringcheck']=not any(n in sys.modules for n in ('numpy','torch','mujoco','physics','scipy'))
 checks['no_actual_artifacts_generated']=not any((HERE/n).exists() for n in ('TRAIN','VALIDATION','FROZEN.json','PREFLIGHT.json','DATASET.json','train.npz','validation.npz','JOBS.json'))
 assert all(checks.values()),checks
 with (HERE/'CODEONLY_CHECK.json').open('x',encoding='utf8') as f:json.dump(dict(status='CODEONLY_READY_PARENT_EXECUTES_LATER',all_passed=True,checks=checks,source_hashes={str(HERE/n):sha(HERE/n) for n in names+('PREREG.json','check_codeonly.py')},numerical_imports=0,arrays_generated=0,model_loads=0,forward_calls=0,optimizer_updates=0,Core_constructions=0,physics_steps=0),f,indent=2,ensure_ascii=False);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=False,model_loads=0,optimizer_updates=0,physics_steps=0)))
if __name__=='__main__':main()
