"""Actual final5000 production-size offline checks, never constructs Core."""
import argparse,ast,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
import evaluate_motor as ev
HERE=Path(__file__).resolve().parent

def funcs(source):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}

def main(args):
 torch.set_num_threads(1);checks={}
 files=[HERE/n for n in ('evaluate_motor.py','verify_actual.py','pipeline.py','EVAL_PREREG.json','check_prepared.py')]
 for p in files:
  ast.parse(p.read_text()) if p.suffix=='.py' else json.loads(p.read_text())
  checks['syntax:'+p.name]=True
 original=funcs((ev.PRIMARY/'evaluate_motor.py').read_text());derived=funcs(ev.derived_source())
 for n in ('jsonable','command_at','score','pin','flatten','parser'):checks['unchanged_physical_function:'+n]=original[n]==derived[n]
 checks['strict_final5000']=ev.CHECKPOINT==HERE.parent/'train/TRAIN/checkpoint_005000.pt' and "physical_eval_only_step']!=5000" in ev.derived_source()
 checks['root_and_head_paths']=ev.ROOT==HERE.parent and (ev.ROOT/'PREREG.json').is_file() and ev.HEAD_AUDIT.is_file()
 prereg=json.loads((HERE/'EVAL_PREREG.json').read_text())
 checks['fixed_conditions_and_window']=prereg['checkpoint_step']==5000 and list(prereg['conditions'])==['neutral','left','right'] and prereg['timing']['autonomous_ms']==9600 and prereg['timing']['native_dt_s']==.0000125
 for c in ev.COMMANDS:
  checks['neutral_prime_and_initial:'+c]=np.array_equal(ev.command_at(c,959),np.zeros(2)) and np.array_equal(ev.command_at(c,960),np.zeros(2))
  checks['full_command_1s:'+c]=np.array_equal(ev.command_at(c,1160),np.asarray(ev.COMMANDS[c]))
 for c in ('left','right'):
  teacher=HERE.parent.parent/'20261110T191449Z/teacher'/f'{c}_full'/'SAMPLES_5MS.npz'
  with np.load(teacher,allow_pickle=False) as z:u=z['command2'].copy()
  computed=np.asarray([ev.command_at(c,i) for i in range(2880)])
  checks['full2880_teacher_command:'+c]=np.max(np.abs(computed-u))<=1e-14
 good=dict(all_gates=True,nonfoot_force_max_native=0.)
 checks['score_pass']=ev.score('left',good,dict(all_legs_completed=True),.01)['eligible']
 checks['score_direction_reject']=not ev.score('right',good,dict(all_legs_completed=True),.01)['eligible']
 checks['score_cycles_reject']=not ev.score('left',good,dict(all_legs_completed=False),.02)['eligible']
 checks['score_nonfoot_reject']=not ev.score('left',dict(all_gates=True,nonfoot_force_max_native=.001),dict(all_legs_completed=True),.02)['eligible']
 checks['numpy_json']=json.loads(json.dumps(ev.jsonable(dict(a=np.bool_(True),b=np.int64(2)))))==dict(a=True,b=2)
 spec=importlib.util.spec_from_file_location('_nonlinear_eval_offline_io',ev.TRAIN/'model_io.py');io=importlib.util.module_from_spec(spec);sys.modules[spec.name]=io;spec.loader.exec_module(io)
 model,p=ev.checked_load(io,str(ev.CHECKPOINT),args.sha256)
 checks['actual_final5000']=p['curve'][-1]['step']==5000 and p['config']['physical_eval_only_step']==5000
 checks['actual_parent_frozen']=hasattr(model,'parent') and sum(v.numel() for v in model.parent.parameters())==197608 and all(not v.requires_grad for v in model.parent.parameters())
 checks['explicit_state128_64']=[list(s.shape) for s in model.initial_state(1)]==[[1,128],[1,64]]
 reject=False
 try:ev.checked_load(io,str(ev.CHECKPOINT.with_name('checkpoint_002500.pt')),args.sha256)
 except ValueError:reject=True
 checks['wrong_path_rejected']=reject
 with np.load(ev.PARENT_RIGHT/'neural_decisions.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 t=ev.arrays(ev.PARENT_RIGHT/'trace_1ms.npz');g=ev.arrays(ev.PARENT_RIGHT/'geometry_1ms.npz')
 state=model.initial_state(1);means=[];states=[];actions=[]
 with torch.no_grad():
  for x,u in zip(d['X238'],d['command']):
   mean,state=model.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(u[None],dtype=torch.float64),state)
   means.append(mean[0].numpy());states.append(torch.cat(state,dim=-1)[0].numpy());actions.append(mean[0].tanh().numpy())
 checks['all2880_right_means_exact']=np.array_equal(np.asarray(means),d['mean'])
 checks['all2880_right_actions_exact']=np.array_equal(np.asarray(actions),d['policy_action'])
 checks['all2880_right_bothstates_exact']=np.array_equal(np.asarray(states),d['hidden_after'])
 neutral_run=HERE.parent.parent/'20261110T235830Z/two_only_evaluation/EVAL_neutral'
 nd=ev.arrays(neutral_run/'neural_decisions.npz');state=model.initial_state(1);nm=[];nh=[];na=[]
 with torch.no_grad():
  for x,u in zip(nd['X238'],nd['command']):
   mean,state=model.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(u[None],dtype=torch.float64),state)
   nm.append(mean[0].numpy());nh.append(torch.cat(state,dim=-1)[0].numpy());na.append(mean[0].tanh().numpy())
 checks['all2880_neutral_means_exact']=np.array_equal(np.asarray(nm),nd['mean'])
 checks['all2880_neutral_actions_exact']=np.array_equal(np.asarray(na),nd['policy_action'])
 checks['all2880_neutral_bothstates_exact']=np.array_equal(np.asarray(nh),nd['hidden_after'])
 checks['new_mlp_parameter_count22378']=sum(v.numel() for v in model.adapter_parameters())==22378
 sys.path.insert(0,str(ev.BASE));import identity
 ident=json.loads((ev.PARENT_RIGHT/'IDENTITY.json').read_text());comparison=ev.verify_parent_right(t,d,g,ident,identity)
 checks['full_parent_right_comparator']=comparison['all_exact']
 bad=dict(d);bad['hidden_after']=d['hidden_after'].copy();bad['hidden_after'][1000,191]+=.01
 reject=False
 try:ev.verify_parent_right(t,bad,g,ident,identity)
 except ValueError:reject=True
 checks['right_memory_corruption_rejected']=reject
 import verify_actual as audit
 # audit wrapper inherits unchanged whole saved-run audit; the only inserted clause
 # checks actual full parent-right preservation and records its separately bound identity.
 source=(ev.PRIMARY/'verify_actual.py').read_text();checks['audit_no_core_construction']='Core(' not in source and 'mj_step(' not in source
 checks['preCore_all_source_paths_exist']=all(p.is_file() for p in ev.source_paths())
 checks['no_backend_import']=all(n not in sys.modules for n in ('mujoco','physics','student_wave'))
 report=dict(all_passed=all(checks.values()),checks=checks,source_hashes={str(p):ev.sha256(p) for p in files},actual_final_model=str(ev.CHECKPOINT),actual_final_sha256=args.sha256,parent_right_comparison=comparison,actual_source_paths=[str(p) for p in ev.source_paths()],physical_steps=0,optimizer_updates=0,Core_constructions=0,scope='Actual final5000 inference on saved right observations, no new physical result')
 (HERE/'READINESS.json').write_text(json.dumps(ev.jsonable(report),indent=2)+'\n')
 print(json.dumps(dict(all_passed=report['all_passed'],checks=len(checks),physical_steps=0,optimizer_updates=0)))
 if not report['all_passed']:raise ValueError('preparation failed '+json.dumps(checks))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sha256',required=True);main(p.parse_args())
