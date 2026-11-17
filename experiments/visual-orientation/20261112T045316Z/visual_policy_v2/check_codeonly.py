"""Stdlibonly v2source syntax/exactinheritance/budget review, no heavymoduleimports."""
from pathlib import Path
import ast,sys,json,hashlib
from adapt_source import build
HERE=Path(__file__).resolve().parent
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 files=[HERE/n for n in ('PREREG.json','common.py','model.py','data.py','generate_data.py','extra_checks.py','preflight.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py','adapt_source.py')];checks={}
 for f in files:
  if f.suffix=='.py':ast.parse(f.read_text());checks[f.name+'_syntax']=True
 for name in ('preflight.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py'):
  compile(build(name),str(HERE/name)+'::actualinheritedsource','exec');checks[name+'_actualadaptationcompile']=True
 t=build('train.py');checks['fixed500freshAdam']=('range(1,501)' in t and 'lr=1e-4' in t and 'gen.choice(18,2,replace=False)' in t and 'shuffle' not in t)
 checks['strictf1ainitialization']='m,parent=initialize();opt=' in t and 'parent_sha256=PARENT_SHA' in t
 p=build('preflight.py');checks['newdomainchecks_beforehashfreeze']=p.index('checks.update(extra_checks(m,parent))')<p.index('hashes={str(p):sha(p)')
 checks['unchangedthresholds']=build('evaluate.py')==(HERE.parent/'visual_policy_v1/evaluate.py').read_text()
 checks['newkind_library126']='VISION_STATIC_HEAD_LIBRARY_V2' in (HERE/'common.py').read_text() and 'rgb.shape!=(126,2,32,32,3)' in (HERE/'common.py').read_text()
 table=[[8+i,12+i] for i in range(4)]+[[16+i,20+i] for i in range(4)]+[[i,4+i] for i in range(4)]+[[24+i,28+i] for i in range(4)]+[[32,33],[34,35]];checks['18paircover36']=len(table)==18 and sorted(x for p in table for x in p)==list(range(36))
 checks['noheavyimports']=not any(n in sys.modules for n in ('numpy','torch','model','common','mujoco','physics'))
 if not all(checks.values()):raise ValueError(checks)
 with (HERE/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes={str(p):sha(p) for p in files},heavy_imports=False,array_loads=0,model_loads=0,forward_calls=0,training_updates=0,Core_constructions=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=False,forward_calls=0)))
if __name__=='__main__':main()
