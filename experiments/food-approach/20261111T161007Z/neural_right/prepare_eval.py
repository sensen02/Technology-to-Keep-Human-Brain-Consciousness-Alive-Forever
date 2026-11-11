"""Offline preCore paths/strictload/adaptation preflight, after independent saved audits."""
import json,ast,inspect
from pathlib import Path
import torch
import model as io
import evaluate_motor as ev
HERE=io.HERE
def main():
 torch.set_num_threads(1)
 checks={}
 for step in (1000,2000):
  p=HERE/'TRAIN'/f'VERIFY_{step:06d}.json';r=json.loads(p.read_text());checks['saved_audit_'+str(step)]=r['all_passed']
  checks['saved_digest_'+str(step)]=r['sha256']==io.sha256(HERE/'TRAIN'/f'checkpoint_{step:06d}.pt')
 # Construct evaluator function only: no Core creation, no integration.
 fn=ev.build();checks['adaptations_exact']=callable(fn)
 source=ev.PRIMARY.read_text();checks['no_parent_right_preservation']='verify_parent_right' not in source
 checks['RIGHT_ONLY_commands']=ev.COMMANDS=={'neutral':[0.,0.],'right':[0.,-1.]}
 checks['boundary_ramp']=all((ev.command_at('right',k)==[0.,0.]).all() for k in range(961)) and (ev.command_at('right',1160)==[0.,-1.]).all()
 # Model validation and dataset parsing are offline, never construct a plant here.
 m,p=ev.checked_load(io,ev.CHECKPOINT,io.sha256(ev.CHECKPOINT))
 checks['only_final2000']=p['curve'][-1]['step']==2000
 checks['neutral_gate']=m.identity()['gate']=='relu(-command[:,1:2])'
 # All paths except this not-yet-written readiness are required before any Core construction.
 paths=[HERE/n for n in ('model.py','train.py','verify_saved.py','evaluate_motor.py','verify_actual.py','prepare_eval.py','PREREG.json','EVAL_PREREG.json')]
 paths += [ev.PRIMARY,ev.PEAK_SOURCE,ev.PEAK_SOURCE.parent/'turning_teacher.py',ev.HEAD_AUDIT,HERE.parent.parent/'20261110T235830Z/evaluation/verify_actual.py']
 paths += list(io.BASE.glob('*.py'))
 checks['all_preCore_paths_exist']=all(x.is_file() and x.is_absolute() for x in paths)
 result=dict(all_passed=all(checks.values()),checks=checks,source_hashes={str(x):io.sha256(x) for x in paths},physics_steps=0,Core_constructions=0,checkpoint_sha256=io.sha256(ev.CHECKPOINT))
 (HERE/'READINESS.json').write_text(json.dumps(ev.jsonable(result),indent=2)+'\n');print(json.dumps(ev.jsonable(dict(all_passed=result['all_passed'],checks=checks))),flush=True)
 if not result['all_passed']:raise ValueError('evaluation preparation failed')
if __name__=='__main__':main()
