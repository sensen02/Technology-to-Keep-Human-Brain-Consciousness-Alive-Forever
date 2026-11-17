"""Exclusive parent serial stages; no numerical import without explicitflag."""
from pathlib import Path
import argparse,subprocess,sys,json,time,os
HERE=Path(__file__).resolve().parent

def atomic(p,v):
 temp=Path(str(p)+'.tmp');temp.write_text(json.dumps(v,indent=2)+'\n');temp.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-parent-aggregation',action='store_true');p.add_argument('--stage',choices=('preflight','train','verify','evaluate','audit'));p.add_argument('--step',type=int);a=p.parse_args()
 if not a.execute_parent_aggregation:print('CODE_ONLY no arrays/models/training');return
 if a.stage:
  if a.stage=='preflight':from preflight import main as work;work()
  elif a.stage=='train':from train import main as work;work()
  elif a.stage=='verify':
   if a.step not in (0,200,400):raise ValueError('Declared savedsteps only')
   from verify_saved import main as work;work(a.step)
  elif a.stage=='evaluate':from evaluate import main as work;work()
  else:from evaluate import audit;audit()
  return
 if (HERE/'JOBS.json').exists():raise FileExistsError('Inspect existing workflow evidence, no blindrerun')
 jobs=[];stages=[('preflight',None),('train',None),('verify',0),('verify',200),('verify',400),('evaluate',None),('audit',None)]
 for stage,step in stages:
  label=stage.upper()+('' if step is None else '_'+str(step));cmd=[sys.executable,'-B',str(HERE/'run.py'),'--execute-parent-aggregation','--stage',stage]+([] if step is None else ['--step',str(step)]);log=HERE/(label+'_EXECUTION.log');atomic(HERE/'CURRENT_STATE.json',dict(status='RUNNING',stage=label,full_neck_training_allowed=False,new_physics_steps=0))
  with log.open('x') as f:
   child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);row=dict(stage=label,command=cmd,pid=child.pid,log=str(log),status='RUNNING',start_epoch_s=time.time());jobs.append(row);atomic(HERE/'JOBS.json',jobs);code=child.wait()
  row.update(status='COMPLETED' if code==0 else 'FAILED',exit_code=code,end_epoch_s=time.time());atomic(HERE/'JOBS.json',jobs)
  if code:atomic(HERE/'CURRENT_STATE.json',dict(status='EXECUTION_FAILED',stage=label,exit_code=code,no_automatic_retry=True,no_automatic_batch_reduction=True));raise RuntimeError('Inspect stage evidence before recovery '+label)
 from common import read,sha
 result=read(HERE/'EVALUATION/RESULT.json');audit=read(HERE/'EVALUATION/VERIFY.json')
 if not audit['integrity_all_pass'] or audit['result_sha256']!=sha(HERE/'EVALUATION/RESULT.json'):raise ValueError('Independent finalmetrics audit required')
 atomic(HERE/'CURRENT_STATE.json',dict(status='FIXED400_AGGREGATION_ENGINEERING_METRICS_COMPLETE_NOT_PHYSICAL',checkpoint=result['checkpoint'],checkpoint_sha256=result['checkpoint_sha256'],integrity_all_pass=True,training_updates=400,all_jobs_collected=True,new_physics_steps=0,full_neck_training_allowed=False,new_walking_verified=False));print('FIXED400_OFFLINE_AGGREGATION_COMPLETE_NO_NEW_BODY_CLAIM')
if __name__=='__main__':main()
