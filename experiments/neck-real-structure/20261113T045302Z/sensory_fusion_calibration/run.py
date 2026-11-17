"""Parent-managed serial pipeline. Stdlib until explicit parent execution flag."""
from pathlib import Path
import argparse,subprocess,sys,json,time
HERE=Path(__file__).resolve().parent
def atomic(p,v):
 q=Path(str(p)+'.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-parent-fusion-calibration',action='store_true');p.add_argument('--stage',choices=('preflight','train','verify','evaluate','audit'));p.add_argument('--step',type=int);a=p.parse_args()
 if not a.execute_parent_fusion_calibration:print('CODE_ONLY no arrays/models/training');return
 if a.stage:
  if a.stage=='preflight':from preflight import main as work;work()
  elif a.stage=='train':from train import main as work;work()
  elif a.stage=='verify':
   if a.step not in (0,200,400):raise ValueError('Declared savedstepsonly')
   from verify_saved import main as work;work(a.step)
  elif a.stage=='evaluate':from evaluate import main as work;work()
  else:from evaluate import audit;audit()
  return
 if (HERE/'JOBS.json').exists():raise FileExistsError('Inspect partialexecution neverblindrepeat')
 from common import read,verify,sha
 check=read(HERE/'CODEONLY_CHECK.json')
 if not check['all_passed'] or not all(check['checks'].values()):raise ValueError('Actualcodeonlycontractrequired')
 verify(check['source_hashes']);jobs=[]
 for stage,step in [('preflight',None),('train',None),('verify',0),('verify',200),('verify',400),('evaluate',None),('audit',None)]:
  label=stage.upper()+('' if step is None else '_'+str(step));cmd=[sys.executable,'-B',str(HERE/'run.py'),'--execute-parent-fusion-calibration','--stage',stage]+([] if step is None else ['--step',str(step)]);log=HERE/(label+'_EXECUTION.log');atomic(HERE/'CURRENT_STATE.json',dict(status='RUNNING',stage=label,full_neck_training_allowed=False,new_physics_steps=0))
  with log.open('x') as f:
   proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);row=dict(stage=label,command=cmd,pid=proc.pid,log=str(log),status='RUNNING',start_epoch_s=time.time());jobs.append(row);atomic(HERE/'JOBS.json',jobs);code=proc.wait()
  row.update(status='COMPLETED' if code==0 else 'FAILED',exit_code=code,end_epoch_s=time.time());atomic(HERE/'JOBS.json',jobs)
  if code:atomic(HERE/'CURRENT_STATE.json',dict(status='EXECUTION_FAILED',stage=label,exit_code=code,no_automatic_retry=True,no_effective_batch_reduction=True));raise RuntimeError('Inspect actualfailure '+label)
 result=read(HERE/'EVALUATION/RESULT.json');audit=read(HERE/'EVALUATION/VERIFY.json')
 if not audit['integrity_all_pass'] or audit['result_sha256']!=sha(HERE/'EVALUATION/RESULT.json'):raise ValueError('Actualfinalauditrequired')
 atomic(HERE/'CURRENT_STATE.json',dict(status='FINAL400_ENGINEERING_SENSORY_FUSION_CALIBRATION_COMPLETE_NOT_BODY',checkpoint=result['checkpoint'],checkpoint_sha256=result['checkpoint_sha256'],integrity_all_pass=True,training_updates=400,trainable_parameter_tensors=10,effective_batch=27,microbatch=9,visual_sha256=result['visual_sha256'],odor_sha256=result['odor_sha256'],full_neck_training_allowed=False,new_body_physics_steps=0,all_jobs_collected=True));print('FIXED400_SOURCE_NECK_SENSORY_FUSION_OFFLINE_COMPLETE_NO_BODYCLAIM')
if __name__=='__main__':main()
