"""Pre-Core path/recovery guard plus inherited full37-check evidence."""
from pathlib import Path
import json,sys,ast
import evaluate_motor as ev
HERE=Path(__file__).resolve().parent

def main():
 old=ev.original;prior=json.loads((ev.ORIGINAL_EVAL/'READINESS.json').read_text())
 checks=dict(inherited37=prior['all_passed'] and len(prior['checks'])==37)
 for p,h in prior['source_hashes'].items():
  if ev.sha256(p)!=h:raise ValueError('original preparation drift '+p)
 paths=ev.source_paths()
 checks['every_actual_source_path_exists_before_Core']=all(p.is_file() for p in paths)
 checks['recovery_output_scope']=ev.evaluate.__globals__['HERE']==HERE
 checks['same_code_object_physics_score']=ev.evaluate.__code__ is old.evaluate.__code__
 checks['same_original_checkpoint']=ev.CHECKPOINT==HERE.parent/'train/TRAIN/checkpoint_010000.pt'
 checks['same_frozen_right_parent']=ev.PARENT_SHA=='c6b395f29676ef933e2f55ca5a92b7c5b55183a2d432226487b39e916b51acdb'
 checks['fixed_three_commands']=list(ev.COMMANDS)==['neutral','left','right']
 checks['root_prereg_exists']=(HERE.parent/'PREREG.json').is_file()
 checks['no_physics_backend_import']=all(n not in sys.modules for n in ('mujoco','physics','student_wave'))
 for p in HERE.glob('*.py'):ast.parse(p.read_text())
 checks['all_new_syntax']=True
 sourcepaths=list(dict.fromkeys(paths+[HERE/'check_prepared.py']))
 report=dict(all_passed=all(checks.values()),checks=checks,source_hashes={str(p):ev.sha256(p) for p in sourcepaths if p.name!='READINESS.json'},inherited_readiness_sha256=ev.sha256(ev.ORIGINAL_EVAL/'READINESS.json'),physical_steps=0,Core_constructions=0,optimizer_updates=0,source_paths_exist_checked=len(paths))
 (HERE/'READINESS.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(dict(all_passed=report['all_passed'],checks=checks,source_paths_exist_checked=len(paths))))
 if not report['all_passed']:raise ValueError('recovery preflight failed')

if __name__=='__main__':main()
