"""Parent-managed serial offline visualcandidate only; no automatic physicalruns."""
from pathlib import Path
import subprocess,sys,json,time,os
HERE=Path(__file__).resolve().parent
def atomic(path,value):
 tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def main():
 if (HERE/'JOBS.json').exists():raise FileExistsError('inspect previous run, no blindpipeline rerun')
 stages=[('preflight.py',[],'PREFLIGHT'),('train.py',[],'TRAIN'),('verify_saved.py',['--step','0'],'VERIFY_0'),('verify_saved.py',['--step','250'],'VERIFY_250'),('verify_saved.py',['--step','500'],'VERIFY_500'),('evaluate.py',[],'VALIDATION'),('audit_validation.py',[],'AUDIT_VALIDATION')];jobs=[]
 for script,args,label in stages:
  command=[sys.executable,'-B',str(HERE/script),*args];log=HERE/(label+'_EXECUTION.log');atomic(HERE/'CURRENT_STATE.json',dict(status='RUNNING',stage=label,parent_pid=os.getpid(),Core_constructions=0,physics_steps=0))
  with log.open('x') as f:
   p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT);row=dict(stage=label,command=command,pid=p.pid,log=str(log),status='RUNNING',started_epoch_s=time.time());jobs.append(row);atomic(HERE/'JOBS.json',jobs);code=p.wait()
  row.update(exit_code=code,ended_epoch_s=time.time(),status='COMPLETED' if code==0 else 'FAILED');atomic(HERE/'JOBS.json',jobs)
  if code:
   atomic(HERE/'CURRENT_STATE.json',dict(status='EXECUTION_FAILED',stage=label,exit_code=code,log=str(log),no_automatic_retry=True));raise RuntimeError(label+' failed; inspect saved evidence before recovery')
 r=json.loads((HERE/'VALIDATION/RESULT.json').read_text());a=json.loads((HERE/'VALIDATION/VERIFY.json').read_text())
 if not a['integrity_all_pass']:raise ValueError('independent final visual audit required')
 atomic(HERE/'CURRENT_STATE.json',dict(status='OFFLINE_VISUAL_TARGETS_PASS' if r['all_targets_pass'] else 'OFFLINE_VISUAL_TARGETS_FAILED',all_targets_pass=r['all_targets_pass'],integrity_all_pass=True,checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],validation_result=str(HERE/'VALIDATION/RESULT.json'),validation_audit=str(HERE/'VALIDATION/VERIFY.json'),all_jobs_collected=True,physical_status='NOT_RUN',no_automatic_budget_extension=True))
 print(json.dumps(dict(all_targets_pass=r['all_targets_pass'],integrity_all_pass=True,checkpoint_sha256=r['checkpoint_sha256'],physics_steps=0)))
if __name__=='__main__':main()
