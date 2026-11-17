"""Stdlib-onlynewencoder/staticadapter/single1000budgetchecks;nomodelimports."""
from pathlib import Path
import ast,json,sys,hashlib,importlib.util
ROOT=Path(__file__).resolve().parent

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 paths=[ROOT/'PREREG.json']+list((ROOT/'library').glob('*.py'))+list((ROOT/'policy').glob('*.py'));checks={}
 for p in paths:
  if p.suffix=='.py':ast.parse(p.read_text());checks[p.relative_to(ROOT).as_posix()+'_syntax']=True
 spec=importlib.util.spec_from_file_location('_opponent_adapter_stdlib',ROOT/'policy/adapt_source.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);sources={}
 for name in ('preflight.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py'):sources[name]=mod.build(name);compile(sources[name],name+'::codeonly','exec');checks[name+'_exactadaptercompile']=True
 model=(ROOT/'policy/model.py').read_text();common=(ROOT/'policy/common.py').read_text();checks['fixedneuralbuffer_notparameter']="register_buffer('opponent_weight'" in model and "register_buffer('opponent_bias'" in model and 'nn.Parameter' not in model;checks['conv1singlechannel']='nn.Conv2d(1,8,3' in model;checks['newproduction_andscan_pixelpath']=model.count('torch.nn.functional.conv2d(x,self.opponent_weight,self.opponent_bias)')==2 and 'f*b*2,3,32,32' in model;checks['newidentity']='VISION_FIXED_NEURAL_OPPONENT_ARCH_V7' in model;checks['parentclassalias_restoredbefore_newconstruction']=common.index("sys.modules['model']=original")<common.index("architecture=module('_opponent_new_architecture'") and 'finally:' in common
 checks['mixedscaleobjective']='expand([0,160])' in sources['preflight.py'] and 'manual_scaled_objective' in sources['preflight.py'];checks['fixed1000range']='range(1,1001)' in sources['train.py'] and 'range(1,2001)' not in sources['train.py'];checks['sampler360unchanged']=sources['train.py'].count('gen.choice(360,2,replace=False)')==1;checks['savedfinal1000']='checkpoint_001000.pt' in sources['evaluate.py'] and 'VERIFY_001000.json' in sources['evaluate.py'];checks['allproofbeforefreeze']=sources['preflight.py'].index('checks.update(extra_checks(m,parent))')<sources['preflight.py'].index('hashes={str(p):sha(p)')
 pr=json.loads((ROOT/'PREREG.json').read_text());pp=json.loads((ROOT/'policy/PREREG.json').read_text());checks['actualnew56old672']=pr['library']['raw_pair_count']==728 and len(pr['library']['new_geometries'])*len(pr['library']['new_pose_indices'])*7==56;checks['onlyfinal1000budget']=pp['training']['updates']==1000 and pp['training']['save_steps']==[0,500,1000] and pp['training']['validation_only_step']==1000 and pp['training']['no_extension']
 heavy={k:k in sys.modules for k in ('numpy','torch','mujoco','physics','flygym')};checks['noheavy']=not any(heavy.values())
 if not all(checks.values()):raise ValueError(checks)
 with (ROOT/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes={str(p):sha(p) for p in paths},heavy_imports=heavy,arrays_read=0,model_loads=0,forward_calls=0,Core_constructions=0,training_updates=0,physical_steps=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=heavy)))
if __name__=='__main__':main()
