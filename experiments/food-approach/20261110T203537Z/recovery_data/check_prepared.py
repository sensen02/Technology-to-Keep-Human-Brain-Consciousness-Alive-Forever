"""Offline schedule/contract/mock continuity checks; no physical/neural imports."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from contract import source_at,metadata,admit
from collect import validate_plan,jsonable

HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

checks={}
seq=[source_at(k) for k in range(1920)]
checks['prime960_teacher']=seq[:960]==[2]*960
checks['student120_decisions']=seq[960:].count(3)==120
checks['recovery840_decisions']=seq[960:].count(5)==840
checks['exact24_bursts']=all(seq[960+i*40:960+(i+1)*40]==[3]*5+[5]*35 for i in range(24))
checks['actual10600_endpoints']=1000+5*len(seq)==10600
checks['source_endpoint_counts']={s:seq.count(s)*5 for s in (2,3,5)}=={2:4800,3:600,5:4200}
checks['phase_excluded']=all(metadata(k)['phase_is_policy_input'] is False for k in range(1920))
checks['48_exact_boundaries']=sum(metadata(k)['burst_boundary'] for k in range(1920))==48
checks['time_clock']=metadata(0)['teacher_elapsed_s']==0 and metadata(960)['teacher_elapsed_s']==4.8 and metadata(1919)['teacher_elapsed_s']==9.595
bad=0
for value in (-1,1920,1.5,True):
    try:source_at(value)
    except ValueError:bad+=1
checks['schedule_rejects_invalid']=bad==4
safe=dict(nonfoot_force_max_native=0.,foot_support_mean=1.,com_height_min=1.25,slip_ratio_max=.49)
checks['safe_full_admission']=admit(safe,True,None,None)
checks['partial_never_admitted']=not admit(safe,False,None,None)
checks['stop_never_admitted']=not admit(safe,True,dict(reason='stop'),None)
checks['exception_never_admitted']=not admit(safe,True,None,'error')
checks['slip_failure_rejected']=not admit(dict(safe,slip_ratio_max=.501),True,None,None)
checks['nonfoot_failure_rejected']=not admit(dict(safe,nonfoot_force_max_native=.001),True,None,None)
# A fake recurrent state increments at EVERY decision regardless of applied source.
state=0;observed=[];queries=0
for k in range(1920):
    before=state;state+=1;queries+=1
    observed.append((before,state,source_at(k)))
checks['mock_state_continues_all_sources']=all(observed[k][0]==observed[k-1][1] for k in range(1,1920)) and state==1920
checks['mock_priming_state_preserved']=observed[960][0]==960
checks['teacher_query_every_decision']=queries==1920
# Actual code structure, in addition to pure schedule mock.
tree=ast.parse((HERE/'collect.py').read_text())
loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='range' and len(n.iter.args)==1 and isinstance(n.iter.args[0],ast.Constant) and n.iter.args[0].value==1920]
checks['actual1920_decision_loop_present']=len(loops)==1
calls=[n for n in ast.walk(loops[0]) if isinstance(n,ast.Call)]
checks['actual_teacher_action_one_query_site']=sum(isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='teacher' and n.func.attr=='action' for n in calls)==1
checks['actual_neural_step_one_site']=sum(isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='model' and n.func.attr=='step' for n in calls)==1
checks['no_reset_in_decision_loop']=not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('reset','reset_from','initial_state') for n in calls)
checks['no_physics_imports_during_checks']=not any(n in sys.modules for n in ('mujoco','torch','physics','student_wave'))
args=SimpleNamespace(run_diagnostic=False,seed=1,checkpoint='/x',out=str(HERE/'NO_RUN'),run_plan='/absent',sha256='0'*64,condition='left')
try:validate_plan(args);guard=False
except ValueError:guard=True
checks['execution_guard_before_backend']=guard
with tempfile.TemporaryDirectory(prefix='recovery-guard-') as td:
    plan=Path(td)/'plan.json'
    plan.write_text(json.dumps(dict(status='REGISTERED_FOR_ACTUAL_DIAGNOSTIC',draft_sha256=sha(HERE/'PREREG_DRAFT.json'),runs=[dict(checkpoint='/x',checkpoint_sha256='0'*64,condition='left',seed=1)])))
    args.run_diagnostic=True;args.run_plan=str(plan)
    checks['future_exact_plan_binding']=validate_plan(args)['runs'][0]['condition']=='left'
    args.seed=2
    try:validate_plan(args);mismatch=False
    except ValueError:mismatch=True
    checks['unregistered_seed_rejected']=mismatch
checks['numpy_safe_json_scalar_contract']=json.loads(json.dumps(jsonable(dict(a=[1,2],b=True))))==dict(a=[1,2],b=True)
report=dict(all_passed=all(checks.values()),checks=checks,physical_steps=0,optimizer_updates=0,backend_loaded=False,
            source_hashes={str(HERE/n):sha(HERE/n) for n in ('collect.py','contract.py','PREREG_DRAFT.json','check_prepared.py')},
            limitation='Preparation only. Mock continuity is not an actual physical/controller integration verification. No dataset has been collected or admitted.')
(HERE/'OFFLINE_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if not report['all_passed']:raise RuntimeError('offline preparation failed')
