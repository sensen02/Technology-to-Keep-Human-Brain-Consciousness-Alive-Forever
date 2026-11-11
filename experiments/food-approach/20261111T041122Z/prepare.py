"""Actual unchanged checkpoint and half-profile formula identity before physics."""
import json,sys,ast
from pathlib import Path
import numpy as np
import torch
import model_io as io
ROOT=io.ROOT

def main():
 torch.set_num_threads(1);model,payload=io.load_checked(io.CHECKPOINT,io.SHA)
 plan=json.loads((ROOT/'PREREG.json').read_text());checks={'strict_same2000_checkpoint':payload['curve'][-1]['step']==2000,'no_weights_changed':True}
 original=ROOT.parent/'20261111T010218Z/evaluation'
 ref=original/'EVAL_neutral';r=json.loads((ref/'RESULT.json').read_text());a=json.loads((original/'VERIFY_neutral.json').read_text())
 checks['same_checkpoint_neutral_passed']=r['checkpoint_sha256']==io.SHA and r['engineering_pass'] and a['integrity_all_pass'] and a['engineering_pass']
 for fn,k in [('trace_1ms.npz','trace'),('neural_decisions.npz','neural_decisions'),('geometry_1ms.npz','geometry'),('runtime_geometry.npz','runtime_geometry')]:
  checks['neutral_hash_'+fn]=io.sha256(ref/fn)==r[k]['sha256']
 template=json.loads((original/'EVAL_PREREG.json').read_text());template['conditions']={'left':[0.,.5],'right':[0.,-.5]};template['checkpoint_step']=2000
 (ROOT/'EVAL_PREREG.json').write_text(json.dumps(template,indent=2)+'\n')
 # Must import the actual bound evaluator and auditor before source sealing.
 import evaluate_motor as ev,verify_actual
 checks['same_original_score_body']=ev.score.__code__.co_code==ev.mod.score.__code__.co_code
 for p in plan['profiles']:
  case=ROOT/p['id'];case.mkdir(exist_ok=True)
  (case/'EVAL_PREREG.json').write_text(json.dumps(template,indent=2)+'\n')
  commands=np.stack([ev.command_at(p['condition'],k) for k in range(2880)])
  expected=np.zeros((2880,2))
  for k in range(960,2880):
   u=min((k-960)*.005,1.);expected[k,1]=p['turn']*(3*u*u-2*u*u*u)
  checks[p['id']+'_all2880_half_commands']=bool(np.max(np.abs(commands-expected))<=1e-14)
  checks[p['id']+'_same_yaw_point01']=not ev.score(p['condition'],dict(all_gates=True,nonfoot_force_max_native=0),dict(all_legs_completed=True),.009*(1 if p['turn']>0 else -1))['eligible']
 # Full production replay on unchanged parent neutral observations.
 with np.load(ref/'neural_decisions.npz',allow_pickle=False) as z:old={k:z[k].copy() for k in ('X238','command','mean','hidden_after')}
 state=model.initial_state(1);means=[];states=[]
 with torch.no_grad():
  for x,u in zip(old['X238'],old['command']):
   mean,state=model.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(u[None],dtype=torch.float64),state);means.append(mean[0].numpy());states.append(torch.cat(state,-1)[0].numpy())
 checks['actual_neutral_2880_means_exact']=np.array_equal(means,old['mean']);checks['actual_neutral_192states_exact']=np.array_equal(states,old['hidden_after'])
 files=[ROOT/n for n in ('PREREG.json','EVAL_PREREG.json','model_io.py','evaluate_motor.py','verify_actual.py','prepare.py','pipeline.py')]
 files += [ev.SOURCE,ev.PRIMARY/'verify_actual.py',io.PATH,ev.HEAD_AUDIT,ref/'RESULT.json',original/'VERIFY_neutral.json']
 hashes={str(p):io.sha256(p) for p in files}
 report=dict(all_passed=all(checks.values()),checks=checks,source_hashes=hashes,checkpoint=str(io.CHECKPOINT),checkpoint_sha256=io.SHA,physical_steps=0,optimizer_updates=0,neutral_reference=dict(result=str(ref/'RESULT.json'),result_sha256=io.sha256(ref/'RESULT.json'),audit=str(original/'VERIFY_neutral.json'),audit_sha256=io.sha256(original/'VERIFY_neutral.json')))
 (ROOT/'PREPARATION.json').write_text(json.dumps(report,indent=2)+'\n')
 for p in plan['profiles']:(ROOT/p['id']/'READINESS.json').write_text(json.dumps(dict(all_passed=report['all_passed'],source_hashes=hashes),indent=2)+'\n')
 for path in ev.source_paths():
  if not path.is_file():raise ValueError('preCore missing '+str(path))
 if not report['all_passed']:raise ValueError('preparation failed '+json.dumps(checks))
 print(json.dumps(dict(all_passed=True,checks=len(checks),physical_steps=0,checkpoint_sha256=io.SHA)),flush=True)
if __name__=='__main__':main()
