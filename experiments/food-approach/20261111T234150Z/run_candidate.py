"""Parent-managed purestdlib serial candidate pipeline, never launches physical trials."""
from pathlib import Path
import subprocess,sys,json,os,time
HERE=Path(__file__).resolve().parent
def atomic(path,value):
 q=Path(str(path)+'.tmp');q.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');q.replace(path)
def main():
 if (HERE/'JOBS.json').exists():raise FileExistsError('inspect existingjobs/artifacts before any continuation, no blindpipeline rerun')
 jobs=[];stages=[('preflight.py',[],'PREFLIGHT'),('train.py',[],'TRAIN'),('verify_saved.py',['--step','0'],'VERIFY_INITIAL'),('verify_saved.py',['--step','500'],'VERIFY_500'),('verify_saved.py',['--step','1000'],'VERIFY_1000'),('evaluate.py',[],'VALIDATION'),('audit_validation.py',[],'AUDIT_VALIDATION')]
 for script,args,label in stages:
  log=HERE/(label+'_EXECUTION.log');cmd=[sys.executable,'-B',str(HERE/script),*args]
  atomic(HERE/'CURRENT_STATE.json',dict(status='RUNNING',stage=label,parent_pid=os.getpid(),physics_steps=0,motor_model_loads=0))
  with log.open('x') as f:
   p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);row=dict(stage=label,command=cmd,pid=p.pid,log=str(log),started_epoch_s=time.time(),status='RUNNING');jobs.append(row);atomic(HERE/'JOBS.json',jobs);code=p.wait()
  row.update(ended_epoch_s=time.time(),exit_code=code,status='COMPLETED' if code==0 else 'FAILED');atomic(HERE/'JOBS.json',jobs)
  if code:
   atomic(HERE/'CURRENT_STATE.json',dict(status='EXECUTION_FAILED',stage=label,exit_code=code,log=str(log),physics_steps=0,no_automatic_retry=True));raise RuntimeError(label+' failed; inspect '+str(log))
 r=json.loads((HERE/'VALIDATION/RESULT.json').read_text());a=json.loads((HERE/'VALIDATION/VERIFY.json').read_text());assert a['integrity_all_pass']
 atomic(HERE/'CURRENT_STATE.json',dict(status='OFFLINE_RECOVERY_TARGETS_PASS' if r['all_targets_pass'] else 'OFFLINE_RECOVERY_TARGETS_FAILED',validation_result=str(HERE/'VALIDATION/RESULT.json'),validation_audit=str(HERE/'VALIDATION/VERIFY.json'),all_targets_pass=r['all_targets_pass'],integrity_all_pass=True,checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],physics_status='NOT_RUN_PARENT_MUST_SCHEDULE_SEPARATEIFALLPASS',jobs_all_collected=True,no_automatic_training_extension=True))
 print(json.dumps(dict(all_targets_pass=r['all_targets_pass'],integrity_all_pass=True,checkpoint_sha256=r['checkpoint_sha256'],physics_steps=0)),flush=True)
if __name__=='__main__':main()
