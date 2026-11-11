"""Offline post-training exact adapter/strictloader/source preCore preparation."""
import argparse,json
from pathlib import Path
import torch
import model_io as io
import evaluate_motor as ev
import verify_actual
HERE=io.HERE
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');args=p.parse_args()
 if not args.parent_resource_released:raise ValueError('parent shared resource release required')
 torch.set_num_threads(1);checks={}
 for step in (2000,4000):
  v=json.loads((HERE/'TRAIN'/f'VERIFY_{step:06d}.json').read_text());checks['saved_audit_'+str(step)]=v['all_passed'];checks['saved_sha_'+str(step)]=v['sha256']==io.sha256(HERE/'TRAIN'/f'checkpoint_{step:06d}.pt')
 checks['evaluation_adaptation_builds']=callable(ev.build());checks['independent_audit_adaptation_builds']=callable(verify_actual.build())
 m,payload=ev.checked_load(io,ev.CHECKPOINT,io.sha256(ev.CHECKPOINT));checks['final4000']=payload['curve'][-1]['step']==4000;checks['head2730']=sum(p.numel() for p in m.parameters() if p.requires_grad)==2730
 paths=[HERE/n for n in ('model_io.py','train.py','verify_saved.py','evaluate_motor.py','verify_actual.py','prepare_eval.py','PREREG.json','EVAL_PREREG.json')]+[io.PARENT_DIR/'model.py',io.PARENT_DIR/'evaluate_motor.py',io.PARENT_DIR/'verify_actual.py',ev.PRIMARY,verify_actual.PRIMARY]
 paths+=list(io.BASE.glob('*.py'));checks['preCore_paths_exist']=all(p.is_file() and p.is_absolute() for p in paths)
 r=dict(all_passed=all(checks.values()),checks=checks,source_hashes={str(p):io.sha256(p) for p in paths},Core_constructions=0,physics_steps=0,checkpoint_sha256=io.sha256(ev.CHECKPOINT));(HERE/'READINESS.json').write_text(json.dumps(r,indent=2)+'\n')
 if not r['all_passed']:raise ValueError('head eval preparation failed')
 ev.source_paths();print(json.dumps(r),flush=True)
if __name__=='__main__':main()
