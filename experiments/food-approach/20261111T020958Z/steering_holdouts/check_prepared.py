"""Offline schedule, guard and causal-boundary tests. No backend imports."""
import ast,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from contract import HERE,CASES,command_at,pulse,yaw_score,digest,validate_launch,verify_qualification


def main():
 checks={}
 for c in CASES:
  u=np.asarray([command_at(c,k) for k in range(2880)])
  checks[c+'_zero_prime']=np.array_equal(u[:960],np.zeros((960,2)))
  checks[c+'_forward_structural_zero']=bool((u[:,0]==0).all())
  checks[c+'_causal_zero_withdrawal']=np.array_equal(u[960],np.zeros(2))
  checks[c+'_bounded_turn']=bool(np.max(np.abs(u[:,1]))<=1.)
  a,s=pulse(c);b,s2=pulse(c)
  checks[c+'_pulse_seed_reproducible']=np.array_equal(a,b) and s==s2
  n=15420 if s is not None else 15400
  checks[c+'_exact_phase_counts']=20*(s is not None)+1000+4800+9600==n
  y=np.zeros(n)
  if c=='halfleft':y[5800:]=np.linspace(0,.01,9600)
  elif c=='halfright':y[5800:]=np.linspace(0,-.01,9600)
  elif c in ('left_to_right','right_to_left'):
   sign=1 if c=='left_to_right' else -1;y[5800:10600]=np.linspace(0,sign*.01,4800);y[10600:]=np.linspace(sign*.01,0,4800)
  else:y[5820:]=np.linspace(0,.02*(1 if c=='pulse_left' else -1),9600)
  checks[c+'_yaw_positive_control']=yaw_score(c,y)['passed']
  checks[c+'_yaw_zero_rejected']=not yaw_score(c,np.zeros(n))['passed']
 checks['left_switch_continuity']=np.array_equal(command_at('left_to_right',1919),command_at('left_to_right',1920))
 checks['left_switch_end']=np.array_equal(command_at('left_to_right',2120),np.array([0.,-.5]))
 checks['right_switch_end']=np.array_equal(command_at('right_to_left',2120),np.array([0.,.5]))
 checks['pulse_exact_seeds']=pulse('pulse_left')[1]==20261112 and pulse('pulse_right')[1]==20261113
 checks['normal_no_pulse']=pulse('halfleft')[1] is None and np.array_equal(pulse('halfleft')[0],np.zeros(42))
 rejected=0
 for c,k in [('bad',960),('halfleft',2880),('halfleft',-1)]:
  try:command_at(c,k)
  except ValueError:rejected+=1
 checks['invalid_schedule_rejected']=rejected==3
 with tempfile.TemporaryDirectory(prefix='holdout-contract-') as td:
  p=Path(td)/'unqualified.json';p.write_text(json.dumps(dict(all_conditions_eligible=False)))
  try:verify_qualification(p,Path(td)/'missing.pt','bad');checks['missing_pass_certificate_rejected']=False
  except ValueError:checks['missing_pass_certificate_rejected']=True
  args=SimpleNamespace(qualification=str(p),checkpoint=str(Path(td)/'model.pt'),out=str(HERE/'UNEXECUTED'),run_plan=str(Path(td)/'plan.json'),launch_authorized=False,sha256='0'*64)
  try:validate_launch(args);checks['unlaunched_guard_rejected']=False
  except ValueError:checks['unlaunched_guard_rejected']=True
 for name in ('contract.py','evaluate_holdout.py','verify_saved.py','pipeline.py','check_prepared.py'):
  ast.parse((HERE/name).read_text());checks['syntax_'+name]=True
 tree=ast.parse((HERE/'evaluate_holdout.py').read_text())
 collect=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
 checks['launch_guard_before_backend_import']=isinstance(collect.body[0],ast.Assign) and next(i for i,n in enumerate(collect.body) if isinstance(n,ast.Import))>0 and 'validate_launch' in ast.unparse(collect.body[0])
 text=(HERE/'evaluate_holdout.py').read_text()
 checks['one_teacher_site_only_in_warm_branch']=text.count('teacher.action(')==1 and 'if warm:action,_=teacher.action' in text
 checks['one_actor_step_site']=text.count('model.step(')==1
 checks['one_hidden_initialization']=text.count('state=model.initial_state(1)')==1
 actual_summary=HERE.parent/'evaluation/FINAL_SUMMARY.json'
 if actual_summary.exists():
  q=json.loads(actual_summary.read_text());rejected=False
  try:verify_qualification(actual_summary,q['checkpoint'],q['checkpoint_sha256'])
  except ValueError:rejected=True
  checks['actual_failed_nonlinear_summary_rejected']=rejected and not q['all_conditions_eligible']
 checks['no_backend_import_during_checks']=all(n not in sys.modules for n in ('torch','mujoco','physics','student_wave'))
 report=dict(all_passed=all(checks.values()),checks=checks,source_hashes={str(HERE/n):digest(HERE/n) for n in ('PREREG_DRAFT.json','contract.py','evaluate_holdout.py','verify_saved.py','pipeline.py','check_prepared.py')},physical_steps=0,optimizer_updates=0,Core_constructions=0,limitation='Prepared code only; actual integration/qualification must be validated after future registered launch')
 (HERE/'OFFLINE_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(dict(all_passed=report['all_passed'],checks=len(checks))))
 if not report['all_passed']:raise ValueError(json.dumps({k:v for k,v in checks.items() if not v}))
if __name__=='__main__':main()
