"""Stdlib syntax and exact-adapter checks only; no numerical imports or output reads."""
from pathlib import Path
import ast,json,sys,hashlib,importlib.util
ROOT=Path(__file__).resolve().parent

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 checks={};paths=[ROOT/'PREREG.json']+list((ROOT/'library').glob('*.py'))+list((ROOT/'policy').glob('*.py'))
 for p in paths:
  if p.suffix=='.py':ast.parse(p.read_text());checks[p.relative_to(ROOT).as_posix()+'_syntax']=True
 spec=importlib.util.spec_from_file_location('_far_adapter_codeonly',ROOT/'policy/adapt_source.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 for name in ('preflight.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py'):
  src=m.build(name);ast.parse(src);compile(src,name+'::codeonly','exec');checks[name+'_exactadaptation_compiles']=True
 pr=json.loads((ROOT/'PREREG.json').read_text());pp=json.loads((ROOT/'policy/PREREG.json').read_text())
 checks['sixconfigs_sevenscenes_42pairs']=len(pr['library']['configs'])*len(pr['library']['stimuli'])==42
 checks['newvalidationposes']=[c['saved_pose_index'] for c in pr['library']['configs'][-2:]]==[7299,14799]
 checks['fixed500_not_extension']=pp['training']['updates']==500 and pp['training']['save_steps']==[0,250,500] and pp['training']['validation_only_step']==500 and pp['training']['no_extension']
 src=m.build('train.py');checks['paired18sampler']=src.count('gen.choice(18,2,replace=False)')==1 and 'range(1,501)' in src
 src=m.build('preflight.py');checks['allproof_before_hash']=src.index('checks.update(extra_checks(m,parent))')<src.index('hashes={str(p):sha(p)')
 checks['physical_not_launched_by_pipeline']='orchestrate' not in m.build('run_candidate.py')
 heavy={k:k in sys.modules for k in ('torch','numpy','mujoco','physics','flygym')};checks['heavy_imports_absent']=not any(heavy.values())
 if not all(checks.values()):raise ValueError(checks)
 report=dict(all_passed=True,checks=checks,source_hashes={str(p):sha(p) for p in paths},heavy_imports=heavy,arrays_read=0,Core_constructions=0,render_calls=0,model_forward_calls=0,training_updates=0,physical_steps=0)
 target=ROOT/'CODEONLY_CHECK.json'
 with target.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=heavy)))
if __name__=='__main__':main()
