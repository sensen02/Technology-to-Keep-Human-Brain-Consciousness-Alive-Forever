"""Parent-managed exclusive serial workflow. Imports only stdlib until explicitflag."""
from pathlib import Path
import argparse,subprocess,sys,json,time,os
HERE=Path(__file__).resolve().parent

def atomic(p,value):
 temp=Path(str(p)+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute-parent-calibration',action='store_true');ap.add_argument('--stage',choices=('preflight','train','verify','evaluate','audit'));ap.add_argument('--step',type=int);a=ap.parse_args()
 if not a.execute_parent_calibration:print('CODE_ONLY no arrays/models/training executed');return
 if a.stage:
  if a.stage=='preflight':from preflight import main as stage;stage()
  elif a.stage=='train':from train import main as stage;stage()
  elif a.stage=='verify':
   if a.step not in (0,100,200):raise ValueError('Declared saved steps only')
   from verify_saved import main as stage;stage(a.step)
  elif a.stage=='evaluate':from evaluate import main as stage;stage()
  else:from evaluate import audit; audit()
  return
 if (HERE/'JOBS.json').exists():raise FileExistsError('Inspect saved stage evidence, never blindrerun')
 stages=[('preflight',None),('train',None),('verify',0),('verify',100),('verify',200),('evaluate',None),('audit',None)];jobs=[]
 for stage,step in stages:
  label=stage.upper()+('' if step is None else '_'+str(step));cmd=[sys.executable,'-B',str(HERE/'run.py'),'--execute-parent-calibration','--stage',stage]+([] if step is None else ['--step',str(step)]);log=HERE/(label+'_EXECUTION.log');atomic(HERE/'CURRENT_STATE.json',dict(status='RUNNING',stage=label,full_neck_training_allowed=False,physical_steps=0))
  with log.open('x') as f:
   p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);row=dict(stage=label,command=cmd,pid=p.pid,log=str(log),status='RUNNING',start_epoch_s=time.time());jobs.append(row);atomic(HERE/'JOBS.json',jobs);exitcode=p.wait()
  row.update(status='COMPLETED' if exitcode==0 else 'FAILED',exit_code=exitcode,end_epoch_s=time.time());atomic(HERE/'JOBS.json',jobs)
  if exitcode:atomic(HERE/'CURRENT_STATE.json',dict(status='EXECUTION_FAILED',stage=label,exit_code=exitcode,no_automatic_retry=True,full_neck_training_allowed=False));raise RuntimeError('Inspect failed stage before any recovery: '+label)
 from common import read,sha
 r=read(HERE/'EVALUATION/RESULT.json');v=read(HERE/'EVALUATION/VERIFY.json')
 if not v['integrity_all_pass'] or v['result_sha256']!=sha(HERE/'EVALUATION/RESULT.json'):raise ValueError('Independent final metric audit required')
 atomic(HERE/'CURRENT_STATE.json',dict(status='SCOPED_ENGINEERING_CALIBRATION_COMPLETE_METRICS_REPORTED',checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],integrity_all_pass=True,training_updates=200,all_jobs_collected=True,full_neck_training_allowed=False,new_body_physics_steps=0,native_DN_function_established=False,native_AN_tuning_established=False,new_walking_verified=False));print('SCOPED_OFFLINE200_COMPLETE_NO_NEW_BODY_OR_BIOLOGICAL_CLAIM')
if __name__=='__main__':main()
