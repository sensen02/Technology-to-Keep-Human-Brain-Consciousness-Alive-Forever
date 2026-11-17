"""Stdlibcodeonly syntax/616shape/720pair/fixed1000sourcechecks, no numericalimports."""
from pathlib import Path
import ast,json,sys,hashlib,importlib.util
ROOT=Path(__file__).resolve().parent

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 checks={};paths=[ROOT/'PREREG.json']+list((ROOT/'library').glob('*.py'))+list((ROOT/'policy').glob('*.py'))
 for p in paths:
  if p.suffix=='.py':ast.parse(p.read_text());checks[p.relative_to(ROOT).as_posix()+'_syntax']=True
 s=importlib.util.spec_from_file_location('_coverage_adapter_stdlib',ROOT/'policy/adapt_source.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 for name in ('preflight.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py'):compile(m.build(name),name+'::codeonly','exec');checks[name+'_exactadaptercompile']=True
 p=json.loads((ROOT/'PREREG.json').read_text());q=json.loads((ROOT/'policy/PREREG.json').read_text());checks['88config616frames']=(len(p['library']['train_geometries'])*len(p['library']['train_pose_indices'])+len(p['library']['validation_geometries'])*len(p['library']['validation_pose_indices']))*7==616
 checks['posesdisjoint']=not set(p['library']['train_pose_indices'])&set(p['library']['validation_pose_indices']);checks['fixed1000']=q['training']['updates']==1000 and q['training']['save_steps']==[0,500,1000] and q['training']['validation_only_step']==1000 and q['training']['no_extension']
 src=m.build('train.py');checks['trainrange1000only']='range(1,1001)' in src and 'range(1,2001)' not in src;checks['save500and1000']='if step in (500,1000):checkpoint(step)' in src;checks['uniform360paironce']=src.count('gen.choice(360,2,replace=False)')==1
 checks['directional_save_replay_pair']=src.count('expand([0,80])')==1 and m.build('preflight.py').count('expand([0,80])')==1;checks['actualcoveragecounts_saved']='pair_coverage=pair_coverage(pairs)' in src and 'actual_pair_coverage_counts_exact' in m.build('verify_saved.py')
 checks['correctfinal1000path']='checkpoint_001000.pt' in m.build('evaluate.py') and 'VERIFY_001000.json' in m.build('evaluate.py');checks['pipelineonlysaved0_500_1000']="['--step','1000']" in m.build('run_candidate.py') and "['--step','250']" not in m.build('run_candidate.py')
 src=m.build('preflight.py');checks['allproofbeforefreeze']=src.index('checks.update(extra_checks(m,parent))')<src.index('hashes={str(p):sha(p)')
 heavy={k:k in sys.modules for k in ('numpy','torch','mujoco','physics','flygym')};checks['heavyabsent']=not any(heavy.values())
 if not all(checks.values()):raise ValueError(checks)
 with (ROOT/'CODEONLY_CHECK.json').open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes={str(p):sha(p) for p in paths},heavy_imports=heavy,arrays_read=0,Core_constructions=0,forward_calls=0,render_calls=0,training_updates=0,physical_steps=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=heavy)))
if __name__=='__main__':main()
