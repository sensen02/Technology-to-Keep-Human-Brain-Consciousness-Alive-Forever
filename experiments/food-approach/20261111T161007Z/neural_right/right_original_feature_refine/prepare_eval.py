"""Future posttraining saved model/adaptation/sourcepaths preCore validation."""
import argparse,json
import torch
import model_io as io
import evaluate_motor as ev
import verify_actual
HERE=io.HERE
def main():
 a=argparse.ArgumentParser();a.add_argument('--parent-resource-released',action='store_true');args=a.parse_args()
 if not args.parent_resource_released:raise ValueError('parent sharedresource release required')
 torch.set_num_threads(1);checks={}
 for step in (1000,2000):
  r=json.loads((HERE/'TRAIN'/f'VERIFY_{step:06d}.json').read_text());checks['saved_audit_'+str(step)]=r['all_passed'];checks['saved_sha_'+str(step)]=r['sha256']==io.sha256(HERE/'TRAIN'/f'checkpoint_{step:06d}.pt')
 checks['evaluator_build']=callable(ev.build());checks['actualaudit_build']=callable(verify_actual.build());m,p=ev.checked_load(io,ev.CHECKPOINT,io.sha256(ev.CHECKPOINT));checks['fixedfinal2000']=p['curve'][-1]['step']==2000;checks['ALL8_43242']=len(m.adapter_parameters())==8 and sum(q.numel() for q in m.adapter_parameters())==43242;checks['qualified_only_loss']=p['config']['training_episode_count']==1 and p['config']['training_dataset']==str(io.DATA) and p['config']['actual_stream_not_in_loss']
 paths=[HERE/n for n in ('model_io.py','train.py','verify_saved.py','evaluate_motor.py','verify_actual.py','prepare_eval.py','PREREG.json','EVAL_PREREG.json')]+[io.PARENT_DIR/'model_io.py',HERE.parent/'model.py',HERE.parent/'evaluate_motor.py',HERE.parent/'verify_actual.py',ev.PRIMARY,verify_actual.PRIMARY];paths+=list(io.BASE.glob('*.py'));checks['all_preCore_paths']=all(p.is_absolute() and p.is_file() for p in paths)
 r=dict(all_passed=all(checks.values()),checks=checks,source_hashes={str(p):io.sha256(p) for p in paths},Core_constructions=0,physics_steps=0,checkpoint_sha256=io.sha256(ev.CHECKPOINT));(HERE/'READINESS.json').write_text(json.dumps(r,indent=2)+'\n')
 if not r['all_passed']:raise ValueError('RIGHT originalfeature actual preparation failed')
 ev.source_paths();print(json.dumps(dict(all_passed=True,checks=checks)),flush=True)
if __name__=='__main__':main()
