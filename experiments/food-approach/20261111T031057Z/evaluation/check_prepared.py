"""Structural checks then optional final8000 production replay; no Core/integration."""
import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import evaluate_motor as ev
HERE=Path(__file__).resolve().parent
FILES=[HERE/n for n in ('evaluate_motor.py','verify_actual.py','pipeline.py','EVAL_PREREG.json','check_prepared.py')]


def functions(text):
 return {n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}


def run(digest=None):
 checks={}
 for p in FILES:
  ast.parse(p.read_text()) if p.suffix=='.py' else json.loads(p.read_text())
  checks['syntax:'+p.name]=True
 old=functions((ev.PRIMARY/'evaluate_motor.py').read_text());new=functions(ev.derived_source())
 for name in ('jsonable','command_at','score','pin','flatten','parser'):
  checks['immutable_function:'+name]=ast.dump(old[name],include_attributes=False)==ast.dump(new[name],include_attributes=False)
 old_loops=[ast.dump(n,include_attributes=False) for n in ast.walk(old['evaluate']) if isinstance(n,(ast.For,ast.While))]
 new_loops=[ast.dump(n,include_attributes=False) for n in ast.walk(new['evaluate']) if isinstance(n,(ast.For,ast.While))]
 checks['actual_physics_and_inference_loops_AST_identical']=old_loops==new_loops
 checks['strict_final8000_path']=ev.CHECKPOINT==HERE.parent/'train/TRAIN/checkpoint_008000.pt'
 checks['root_prereg_exists']=(HERE.parent/'PREREG.json').is_file()
 checks['head_audit_exists']=ev.HEAD_AUDIT.is_file()
 prereg=json.loads((HERE/'EVAL_PREREG.json').read_text())
 checks['registered_step_conditions_window']=prereg['checkpoint_step']==8000 and prereg['conditions']==ev.COMMANDS and prereg['timing']['autonomous_ms']==9600
 for condition,target in ev.COMMANDS.items():
  checks['first_autonomous_zero:'+condition]=np.array_equal(ev.command_at(condition,960),np.zeros(2))
  checks['command1s_target:'+condition]=np.array_equal(ev.command_at(condition,1160),np.array(target,dtype=float))
 teacher=HERE.parent.parent/'20261110T191449Z/teacher'
 for condition in ('left','right'):
  with np.load(teacher/(condition+'_full')/'SAMPLES_5MS.npz',allow_pickle=False) as z:u=z['command2'].copy()
  actual=np.asarray([ev.command_at(condition,k) for k in range(2880)])
  checks['all2880_teacher_command:'+condition]=bool(np.max(np.abs(actual-u))<=1e-14)
 good=dict(all_gates=True,nonfoot_force_max_native=0.)
 checks['score_pass']=ev.score('left',good,dict(all_legs_completed=True),.01)['eligible']
 checks['score_bad_direction_rejected']=not ev.score('right',good,dict(all_legs_completed=True),.01)['eligible']
 checks['score_bad_cycles_rejected']=not ev.score('left',good,dict(all_legs_completed=False),.02)['eligible']
 checks['score_nonfoot_rejected']=not ev.score('left',dict(all_gates=True,nonfoot_force_max_native=.001),dict(all_legs_completed=True),.02)['eligible']
 checks['numpy_scalar_json']=json.loads(json.dumps(ev.jsonable(dict(flag=np.bool_(True)))))==dict(flag=True)
 spec=importlib.util.spec_from_file_location('_weighted_prepare_model_io',ev.TRAIN/'model_io.py')
 io=importlib.util.module_from_spec(spec);sys.modules[spec.name]=io;spec.loader.exec_module(io)
 class Forbidden:
  def load_checked(self,*args,**kwargs):raise AssertionError('wrong path reached neural loader')
 rejected=False
 try:ev.checked_load(Forbidden(),str(ev.CHECKPOINT.with_name('checkpoint_005000.pt')),'0'*64)
 except ValueError:rejected=True
 checks['wrong5000_path_rejected_before_loader']=rejected
 comparison=None
 if digest:
  import torch
  torch.set_num_threads(1)
  model,payload=ev.checked_load(io,str(ev.CHECKPOINT),digest)
  checks['actual_final8000_model']=payload['curve'][-1]['step']==8000 and payload['config']['physical_eval_only_step']==8000
  checks['same_legacy_readout_architecture']=sum(p.numel() for p in model.parent.parameters())==194878 and sum(p.numel() for p in model.adapter_parameters())==2730
  checks['frozen_parent']=all(not p.requires_grad for p in model.parent.parameters())
  qualified=ev.PARENT_RIGHT.parent
  loaded={}
  for condition in ('neutral','right'):
   d=ev.arrays(qualified/('EVAL_'+condition)/'neural_decisions.npz');loaded[condition]=d
   state=model.initial_state(1);means=[];states=[];actions=[]
   with torch.no_grad():
    for x,u in zip(d['X238'],d['command']):
     mean,state=model.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(u[None],dtype=torch.float64),state)
     means.append(mean[0].numpy());states.append(torch.cat(state,dim=-1)[0].numpy());actions.append(mean[0].tanh().numpy())
   checks['actual2880_'+condition+'_means_parent_exact']=np.array_equal(means,d['mean'])
   checks['actual2880_'+condition+'_actions_parent_exact']=np.array_equal(actions,d['policy_action'])
   checks['actual2880_'+condition+'_192states_parent_exact']=np.array_equal(states,d['hidden_after'])
  sys.path.insert(0,str(ev.BASE));import identity
  d=loaded['right'];t=ev.arrays(ev.PARENT_RIGHT/'trace_1ms.npz');g=ev.arrays(ev.PARENT_RIGHT/'geometry_1ms.npz')
  ident=json.loads((ev.PARENT_RIGHT/'IDENTITY.json').read_text())
  comparison=ev.verify_parent_right(t,d,g,ident,identity)
  checks['full_parent_right_array_comparator']=comparison['all_exact'] and all(comparison['checks'].values())
  bad=dict(d);bad['hidden_after']=d['hidden_after'].copy();bad['hidden_after'][1000,191]+=.01
  rejected=False
  try:ev.verify_parent_right(t,bad,g,ident,identity)
  except ValueError:rejected=True
  checks['comparator_rejects_residual_state_change']=rejected
 # Readiness itself is a source artifact, so create a false placeholder only
 # before the first source-path existence check, then replace with the report.
 if not (HERE/'READINESS.json').exists():
  (HERE/'READINESS.json').write_text(json.dumps(dict(all_passed=False,reason='pending final model checks'))+'\n')
 paths=ev.source_paths()
 checks['every_actual_source_path_exists_before_Core']=all(p.is_file() for p in paths)
 import verify_actual as audit
 auditfunctions=functions(audit.derived_source());old_audit=functions((ev.PRIMARY/'verify_actual.py').read_text())
 for name in ('arrays','ROOT_HEAD_AUDIT','main'):
  checks['immutable_audit_function:'+name]=ast.dump(auditfunctions[name],include_attributes=False)==ast.dump(old_audit[name],include_attributes=False)
 checks['no_Core_or_physics_backend_import']=all(name not in sys.modules for name in ('mujoco','physics','student_wave'))
 all_checks=all(checks.values())
 report=dict(all_passed=bool(digest is not None and all_checks),code_checks_passed=all_checks,checks=checks,
             source_hashes={str(p):ev.sha256(p) for p in FILES},actual_model=None if digest is None else str(ev.CHECKPOINT),actual_sha256=digest,
             source_paths_checked=[str(p) for p in paths],parent_right_comparison=comparison,
             physical_steps=0,optimizer_updates=0,Core_constructions=0,
             limitation='No actual body run. Final READINESS is false without actual final8000 checkpoint even if code checks pass.')
 target='READINESS.json' if digest else 'CODE_CHECKS.json'
 (HERE/target).write_text(json.dumps(ev.jsonable(report),indent=2)+'\n')
 print(json.dumps(dict(code_checks_passed=all_checks,all_passed=report['all_passed'],checks=len(checks),physical_steps=0)))
 if not all_checks:raise ValueError('preparation check failed '+json.dumps(checks))
 return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sha256');a=p.parse_args();run(a.sha256)
