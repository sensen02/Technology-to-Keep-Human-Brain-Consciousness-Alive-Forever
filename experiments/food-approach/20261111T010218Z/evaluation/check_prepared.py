"""Offline preparation with actual final-model inference; never constructs Core."""
import ast
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch
import evaluate_motor as ev

HERE=Path(__file__).resolve().parent
SHA='eeced49c5bdd50a7e5bce2cfd63d56d9929b3bf2d2cb00c1985019d91321e6df'


def functions(source):
 return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}


def main():
 torch.set_num_threads(1)
 checks={}
 files=[HERE/n for n in ('evaluate_motor.py','verify_actual.py','pipeline.py','EVAL_PREREG.json','check_prepared.py')]
 for p in files:
  if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
  else:json.loads(p.read_text())
  checks['syntax:'+p.name]=True
 original=functions((ev.PRIMARY/'evaluate_motor.py').read_text());derived=functions(ev.derived_source())
 for name in ('jsonable','command_at','score','pin','flatten','parser'):
  checks['original_function_unchanged:'+name]=original[name]==derived[name]
 checks['checked_load_final2000']="payload['curve'][-1]['step']!=2000" in ev.derived_source() and "physical_eval_only_step']!=2000" in ev.derived_source()
 audit_source=(HERE/'verify_actual.py').read_text()
 checks['auditor_uses_new_checked_loader']='from evaluate_motor import' in (ev.PRIMARY/'verify_actual.py').read_text() and 'current.PRIMARY' in audit_source
 prereg=json.loads((HERE/'EVAL_PREREG.json').read_text())
 checks['fixed2000']=prereg['checkpoint_step']==2000
 checks['three_conditions']=list(prereg['conditions'])==['neutral','left','right']
 checks['full_window']=prereg['timing']['autonomous_ms']==9600
 checks['native_dt']=prereg['timing']['native_dt_s']==.0000125
 checks['first_autonomous_command_neutral']=all(np.array_equal(ev.command_at(c,960),np.zeros(2)) for c in ev.COMMANDS)
 checks['full_command_after1second']=all(np.array_equal(ev.command_at(c,1160),np.asarray(v)) for c,v in ev.COMMANDS.items())
 good=dict(all_gates=True,nonfoot_force_max_native=0.)
 checks['score_good_left']=ev.score('left',good,dict(all_legs_completed=True),.01)['eligible']
 checks['score_bad_direction_rejected']=not ev.score('right',good,dict(all_legs_completed=True),.01)['eligible']
 checks['score_bad_cycles_rejected']=not ev.score('left',good,dict(all_legs_completed=False),.02)['eligible']
 checks['score_nonfoot_rejected']=not ev.score('left',dict(all_gates=True,nonfoot_force_max_native=.001),dict(all_legs_completed=True),.02)['eligible']
 checks['strict_numpy_json']=json.loads(json.dumps(ev.jsonable(dict(flag=np.bool_(True),n=np.int64(2)))))==dict(flag=True,n=2)
 spec=importlib.util.spec_from_file_location('_offline_positive_readout_io',ev.TRAIN/'model_io.py')
 io=importlib.util.module_from_spec(spec);sys.modules[spec.name]=io;spec.loader.exec_module(io)
 model,payload=ev.checked_load(io,str(ev.CHECKPOINT),SHA)
 checks['actual_new2000_model_identity']=payload['curve'][-1]['step']==2000 and payload['identity']['parent_sha256']==ev.PARENT_SHA
 checks['frozen_parent194878']=sum(p.numel() for p in model.parent.parameters())==194878 and all(not p.requires_grad for p in model.parent.parameters())
 checks['readout2730']=sum(p.numel() for p in model.readout.parameters())==2730
 rejected=False
 try:ev.checked_load(io,str(ev.CHECKPOINT.with_name('checkpoint_001000.pt')),SHA)
 except ValueError:rejected=True
 checks['wrong_checkpoint_path_rejected_before_load']=rejected
 d=ev.arrays(ev.PARENT_RIGHT/'neural_decisions.npz');t=ev.arrays(ev.PARENT_RIGHT/'trace_1ms.npz');g=ev.arrays(ev.PARENT_RIGHT/'geometry_1ms.npz')
 state=model.initial_state(1);means=[];states=[];actions=[]
 with torch.no_grad():
  for x,u in zip(d['X238'],d['command']):
   mean,state=model.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(u[None],dtype=torch.float64),state)
   means.append(mean[0].numpy());states.append(torch.cat(state,dim=-1)[0].numpy());actions.append(mean[0].tanh().numpy())
 checks['all2880_new_right_means_parent_bitexact']=np.array_equal(np.asarray(means),d['mean'])
 checks['all2880_new_right_actions_parent_bitexact']=np.array_equal(np.asarray(actions),d['policy_action'])
 checks['all2880_new_right_both_hidden_parent_bitexact']=np.array_equal(np.asarray(states),d['hidden_after'])
 sys.path.insert(0,str(ev.BASE))
 import identity
 ident=json.loads((ev.PARENT_RIGHT/'IDENTITY.json').read_text())
 comparison=ev.verify_parent_right(t,d,g,ident,identity)
 checks['actual_right_reference_full_array_comparator']=comparison['all_exact'] and all(comparison['checks'].values())
 bad=dict(d);bad['hidden_after']=d['hidden_after'].copy();bad['hidden_after'][1000,191]+=.01
 rejected=False
 try:ev.verify_parent_right(t,bad,g,ident,identity)
 except ValueError:rejected=True
 checks['comparator_rejects_residual_memory_mismatch']=rejected
 import verify_actual as audit
 generated=functions(audit.derived_source());original_audit=functions((ev.PRIMARY/'verify_actual.py').read_text())
 for name in ('arrays','ROOT_HEAD_AUDIT','main'):checks['auditor_original_function_unchanged:'+name]=generated[name]==original_audit[name]
 report=dict(all_passed=all(checks.values()),checks=checks,source_hashes={str(p):ev.sha256(p) for p in files},
             actual_new_checkpoint=str(ev.CHECKPOINT),actual_new_checkpoint_sha256=SHA,
             parent_right_audit_sha256=ev.sha256(ev.PARENT_AUDIT),parent_right_reference_comparison=comparison,
             physical_steps=0,optimizer_updates=0,Core_constructions=0,
             limitation='Saved parent-right observations and arrays are replayed; this is not a new physical trajectory or final qualification.')
 (HERE/'READINESS.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(dict(all_passed=report['all_passed'],checks=len(checks),physics_steps=0,optimizer_updates=0)))
 if not report['all_passed']:raise ValueError('preparation checks failed '+json.dumps(checks))

if __name__=='__main__':main()
