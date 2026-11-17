"""Stdlib-onlyD27 syntax/exactsource/intervention contract checks, no numericalimports."""
from pathlib import Path
import ast,json,sys,hashlib,importlib.util
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 names=('contracts.py','runtime.py','runner.py','audit.py','prepare.py','preflight.py','summary.py','orchestrate.py','check_codeonly.py');checks={}
 for n in names:ast.parse((ROOT/n).read_text());compile((ROOT/n).read_text(),str(ROOT/n),'exec')
 checks['all9syntaxcompile']=True;sys.path.insert(0,str(ROOT));spec=importlib.util.spec_from_file_location('_Dcodeonlyruntime',ROOT/'runtime.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 runner=m.source('runner.py');audit=m.source('audit.py');compile(runner,'D::runner','exec');compile(audit,'D::audit','exec');checks['all_exactsourceadaptationscompile']=True
 spec=importlib.util.spec_from_file_location('_Dcodeonlycontracts',ROOT/'contracts.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);cfg=c.trials()
 checks['fixed27rows']=len(cfg)==27 and all(sum(r['control']==control for r in cfg)==n for control,n in [('normal',8),('cut',8),('swap',8),('noodor',1),('replay',2)])
 checks['no_actualposewrites']=not any(isinstance(n,(ast.Assign,ast.AugAssign,ast.AnnAssign)) and any(x in ast.unparse(n).split('=')[0] for x in ('.qpos','.qvel')) for n in ast.walk(ast.parse(runner)))
 checks['physicalenvironment_receptors_preserved']="actor_input(args.trial,elapsed,sensory['receptors'])" in runner and 'raw_command=raw[0].numpy().copy()' in runner
 checks['appliedcommand_override_inbothactualandnewprocessaudit']='command=applied_command(args.trial,k,raw,project)' in runner and 'u=applied_command(trial,k,raw,project)' in audit
 checks['cuttimefixed3_noreset']='elapsed>=3.-1e-10' in (ROOT/'runtime.py').read_text() and "result['cut_precondition']=cut_precondition(rows,args.trial)" in runner and 'effective_response_before_cut' in (ROOT/'runtime.py').read_text()
 checks['sameDgateunchanged6of8anddifferences4']="counts['normal']>=6" in (ROOT/'summary.py').read_text() and all("counts['normal']-counts['"+control+"']>=4" in (ROOT/'summary.py').read_text() for control in ('cut','swap','noodor'))
 checks['C6allstable_admission_notrelaxed']="s['all_integrity_pass'] and s['all_stable_body']" in (ROOT/'contracts.py').read_text()
 checks['noheavyimports']=not any(n in sys.modules for n in ('numpy','torch','mujoco','physics','scipy'))
 checks['no_executionfreezeoractual']=not any((ROOT/n).exists() for n in ('FROZEN.json','INPUT_BINDINGS_D.json','PREFLIGHT.json','trials','JOBS_D.json'))
 assert all(checks.values()),checks
 with (ROOT/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(status='D27_CODEONLY_READY_CADMISSION_PARENTLAYOUTSCREEN_AND_RELEASE_PENDING',all_passed=True,checks=checks,source_hashes={str(ROOT/n):sha(ROOT/n) for n in names+('PREREG_PROPOSAL.json',)},model_loads=0,arrays_loaded=0,Core_constructions=0,forward_calls=0,physics_steps=0,training_updates=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=False,execution_frozen=False)))
if __name__=='__main__':main()
