"""Stdlib exactadapter/scope/scaledobjective checks,no numericalimports."""
from pathlib import Path
import ast,json,sys,hashlib,importlib.util
ROOT=Path(__file__).resolve().parent

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 paths=[ROOT/'PREREG.json']+list((ROOT/'library').glob('*.py'))+list((ROOT/'policy').glob('*.py'));checks={}
 for p in paths:
  if p.suffix=='.py':ast.parse(p.read_text());checks[p.relative_to(ROOT).as_posix()+'_syntax']=True
 s=importlib.util.spec_from_file_location('_neutral_adapter_stdlib',ROOT/'policy/adapt_source.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 sources={}
 for name in ('preflight.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py'):
  sources[name]=m.build(name);compile(sources[name],name+'::codeonly','exec');checks[name+'_exactadaptercompile']=True
 pp=json.loads((ROOT/'policy/PREREG.json').read_text());pr=json.loads((ROOT/'PREREG.json').read_text());checks['fixed1000only']=pp['training']['updates']==1000 and pp['training']['save_steps']==[0,500,1000] and pp['training']['validation_only_step']==1000 and pp['training']['no_extension'];checks['newactual56old616']=len(pr['library']['new_geometries'])*len(pr['library']['new_pose_indices'])*7==56 and pr['library']['raw_pair_count']==672
 for name,src in sources.items():
  for node in ast.walk(ast.parse(src)):
   if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='batch':
    checks[name+'_batch_unpack4_line'+str(node.lineno)]=isinstance(node.targets[0],ast.Tuple) and len(node.targets[0].elts)==4
 checks['mixedscalepaircheck']='expand([0,160])' in sources['preflight.py'] and 'expand([0,160])' in sources['train.py'];checks['manual_scaled_objective']='manual_scaled_objective' in sources['preflight.py'] and 'w[:,:2],s[:2]' in sources['preflight.py'];checks['no_scaledweight_normalization']='episode_scale*per_episode' in (ROOT/'policy/model.py').read_text() and 'episode_scale.sum' not in (ROOT/'policy/model.py').read_text()
 checks['allproofbeforefreeze']=sources['preflight.py'].index('checks.update(extra_checks(m,parent))')<sources['preflight.py'].index('hashes={str(p):sha(p)');checks['1000range_only']='range(1,1001)' in sources['train.py'] and 'range(1,2001)' not in sources['train.py'];checks['uniform360unchanged']=sources['train.py'].count('gen.choice(360,2,replace=False)')==1;checks['finalpath1000']='checkpoint_001000.pt' in sources['evaluate.py'] and 'VERIFY_001000.json' in sources['evaluate.py']
 heavy={k:k in sys.modules for k in ('numpy','torch','mujoco','physics','flygym')};checks['noheavyimports']=not any(heavy.values())
 if not all(checks.values()):raise ValueError(checks)
 with (ROOT/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes={str(p):sha(p) for p in paths},heavy_imports=heavy,arrays_read=0,model_loads=0,forward_calls=0,training_updates=0,Core_constructions=0,physical_steps=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=heavy)))
if __name__=='__main__':main()
