"""Purestdlib beforeTRAIN recovery; reuse completed data, new journal/log names."""
from pathlib import Path
import subprocess,sys,json,os,time
HERE=Path(__file__).resolve().parent
def atomic(path,value):
 q=Path(str(path)+'.tmp');q.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');q.replace(path)
def main():
 if (HERE/'JOBS_V2.json').exists():raise FileExistsError('inspect existingV2journal before restart')
 if (HERE/'FROZEN.json').exists() or (HERE/'TRAIN').exists():raise FileExistsError('recovery only beforefreezeandTRAIN; no repeated training')
 if not all((HERE/n).is_file() for n in ('DATASET.json','train.npz','validation.npz')):raise FileNotFoundError('completed prior dataset must exist; do not regenerate')
 stages=[('preflight_v2.py',[],'PREFLIGHT_V2'),('train.py',[],'TRAIN_V2'),('verify_saved.py',['--step','0'],'VERIFY_INITIAL_V2'),('verify_saved.py',['--step','250'],'VERIFY_250_V2'),('verify_saved.py',['--step','500'],'VERIFY_500_V2'),('evaluate.py',[],'VALIDATION_V2'),('audit_validation.py',[],'AUDIT_VALIDATION_V2')];jobs=[]
 for script,args,label in stages:
  cmd=[sys.executable,'-B',str(HERE/script),*args];log=HERE/(label+'_EXECUTION.log');atomic(HERE/'CURRENT_STATE.json',dict(status='RECOVERY_RUNNING',stage=label,parent_pid=os.getpid(),previous_failure='preflight source-hash order; no prior training/physics',prior_journal_preserved=str(HERE/'JOBS.json')))
  with log.open('x') as f:
   proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);row=dict(stage=label,command=cmd,pid=proc.pid,log=str(log),status='RUNNING',started_epoch_s=time.time());jobs.append(row);atomic(HERE/'JOBS_V2.json',jobs);code=proc.wait()
  row.update(exit_code=code,status='COMPLETED' if code==0 else 'FAILED',ended_epoch_s=time.time());atomic(HERE/'JOBS_V2.json',jobs)
  if code:
   atomic(HERE/'CURRENT_STATE.json',dict(status='RECOVERY_EXECUTION_FAILED',stage=label,exit_code=code,log=str(log),no_automatic_retry=True));raise RuntimeError(label+' failed; inspect '+str(log))
 r=json.loads((HERE/'VALIDATION/RESULT.json').read_text());a=json.loads((HERE/'VALIDATION/VERIFY.json').read_text());assert a['integrity_all_pass']
 atomic(HERE/'CURRENT_STATE.json',dict(status='OFFLINE_PAIRED_TARGETS_PASS' if r['all_targets_pass'] else 'OFFLINE_PAIRED_TARGETS_FAILED',validation_result=str(HERE/'VALIDATION/RESULT.json'),validation_audit=str(HERE/'VALIDATION/VERIFY.json'),all_targets_pass=r['all_targets_pass'],integrity_all_pass=True,checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],physics_status='NOT_RUN_PARENT_COORDINATES_IF_ALLPASS',jobs_v2_all_collected=True,no_training_extension=True))
 print(json.dumps(dict(all_targets_pass=r['all_targets_pass'],integrity_all_pass=True,checkpoint_sha256=r['checkpoint_sha256'],physics_steps=0)),flush=True)
if __name__=='__main__':main()
