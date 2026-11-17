"""Stdlib-only exactadapter/syntax/budget checks; never imports numericalpipeline."""
from pathlib import Path
import ast,json,hashlib,importlib.util,sys
ROOT=Path(__file__).resolve().parent

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 paths=[ROOT/'PREREG.json']+list((ROOT/'library').glob('*.py'))+list((ROOT/'policy').glob('*.py'));checks={}
 for p in paths:
  if p.suffix=='.py':ast.parse(p.read_text());checks[p.relative_to(ROOT).as_posix()+'_syntax']=True
 spec=importlib.util.spec_from_file_location('_mirror_adapter_stdlib',ROOT/'policy/adapt_source.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 for name in ('preflight.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py'):
  src=mod.build(name);compile(src,name+'::codeonly','exec');checks[name+'_exactadapter_compiles']=True
 pr=json.loads((ROOT/'PREREG.json').read_text());pp=json.loads((ROOT/'policy/PREREG.json').read_text());checks['56actualraw_and14new']=pr['library']['raw_pair_count']==56 and len(pr['library']['new_configs'])*len(pr['library']['stimuli'])==14
 checks['newactualposes']=[c['saved_pose_index'] for c in pr['library']['new_configs']]==[8599,11999];checks['500onlyfinal']=pp['training']['updates']==500 and pp['training']['save_steps']==[0,250,500] and pp['training']['validation_only_step']==500 and pp['training']['no_extension']
 src=mod.build('train.py');checks['36pairsampler_once_noextension']=src.count('gen.choice(36,2,replace=False)')==1 and 'range(1,501)' in src
 src=mod.build('preflight.py');checks['proof_before_sourcefreeze']=src.index('checks.update(extra_checks(m,parent))')<src.index('hashes={str(p):sha(p)')
 src=(ROOT/'policy/data.py').read_text();checks['correct_eye_width_axes']='torch.flip(rgb,dims=(2,4))' in src;checks['augmentedpixels_not_saved']='np.savez' not in src;checks['targets_negated_only_split']=src.count("-value if key=='target' else value")==1
 heavy={k:k in sys.modules for k in ('torch','numpy','mujoco','physics','flygym')};checks['no_heavy_import']=not any(heavy.values())
 if not all(checks.values()):raise ValueError(checks)
 report=dict(all_passed=True,checks=checks,source_hashes={str(p):sha(p) for p in paths},heavy_imports=heavy,arrays_read=0,model_loads=0,forward_calls=0,Core_constructions=0,render_calls=0,training_updates=0,physical_steps=0)
 with (ROOT/'CODEONLY_CHECK.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=heavy)))
if __name__=='__main__':main()
