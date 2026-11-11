"""Postsave zero-Core actual evaluator/loader preflight for dense corrective head."""
import argparse,json
from pathlib import Path
import torch
import model_io as io
import evaluate_motor as ev
import verify_actual
HERE=io.HERE
def main():
 a=argparse.ArgumentParser();a.add_argument('--parent-resource-released',action='store_true');args=a.parse_args()
 if not args.parent_resource_released:raise ValueError('parent sharedresource release required')
 torch.set_num_threads(1);checks={}
 for step in (2000,4000):
  r=json.loads((HERE/'TRAIN'/f'VERIFY_{step:06d}.json').read_text());checks['saved_audit_'+str(step)]=r['all_passed'];checks['saved_sha_'+str(step)]=r['sha256']==io.sha256(HERE/'TRAIN'/f'checkpoint_{step:06d}.pt')
 checks['oracle_audit']=json.loads((HERE/'ORACLE/VERIFY.json').read_text())['all_passed'];checks['evaluator_adaptation']=callable(ev.build());checks['actual_audit_adaptation']=callable(verify_actual.build());m,p=ev.checked_load(io,ev.CHECKPOINT,io.sha256(ev.CHECKPOINT));checks['final4000']=p['curve'][-1]['step']==4000;checks['densehead2730']=sum(q.numel() for q in m.parameters() if q.requires_grad)==2730;checks['equal_episodes']=p['config']['episode_weights']==[.5,.5]
 paths=[HERE/n for n in ('model_io.py','check_train.py','train.py','verify_saved.py','evaluate_motor.py','verify_actual.py','prepare_eval.py','PREREG.json','EVAL_PREREG.json','oracle.py','verify_oracle.py')]+[io.PARENT_DIR/'model_io.py',HERE.parent/'model.py',HERE.parent/'evaluate_motor.py',HERE.parent/'verify_actual.py',ev.PRIMARY,verify_actual.PRIMARY];paths+=list(io.BASE.glob('*.py'));checks['preCore_paths']=all(p.is_file() and p.is_absolute() for p in paths)
 r=dict(all_passed=all(checks.values()),checks=checks,source_hashes={str(p):io.sha256(p) for p in paths},Core_constructions=0,physics_steps=0,checkpoint_sha256=io.sha256(ev.CHECKPOINT));(HERE/'READINESS.json').write_text(json.dumps(r,indent=2)+'\n')
 if not r['all_passed']:raise ValueError('densecorrective actualpreparation failed')
 ev.source_paths();print(json.dumps(dict(all_passed=True,checks=checks)),flush=True)
if __name__=='__main__':main()
