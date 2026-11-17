"""Parent-managed serial fixed250 pipeline; default invocation is stdlib CODE_ONLY."""
from pathlib import Path
import argparse,subprocess,sys,json,time
HERE=Path(__file__).resolve().parent
def atomic(p,v):
 q=Path(str(p)+'.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-parent-odor-memory-repair',action='store_true');p.add_argument('--stage',choices=('preflight','train','verify','evaluate','audit'));p.add_argument('--step',type=int);a=p.parse_args()
 if not a.execute_parent_odor_memory_repair:print('CODE_ONLY no arrays/models/training');return
 if a.stage:
  if a.stage=='preflight':from preflight import main as work;work()
  elif a.stage=='train':from train import main as work;work()
  elif a.stage=='verify':
   if a.step not in (0,125,250):raise ValueError('Onlypreregisteredsavedsteps')
   from verify_saved import main as work;work(a.step)
  elif a.stage=='evaluate':from evaluate import main as work;work()
  else:from evaluate import audit;audit()
  return
 if (HERE/'JOBS.json').exists():raise FileExistsError('Inspectpriorrun neverblindrepeat')
 from common import read,verify,sha
 checked=read(HERE/'CODEONLY_CHECK.json')
 if not checked['all_passed'] or not all(checked['checks'].values()):raise ValueError('ActualCODEONLYchecks required')
 verify(checked['source_hashes']);jobs=[]
 for stage,step in [('preflight',None),('train',None),('verify',0),('verify',125),('verify',250),('evaluate',None),('audit',None)]:
  label=stage.upper()+('' if step is None else '_'+str(step));cmd=[sys.executable,'-B',str(HERE/'run.py'),'--execute-parent-odor-memory-repair','--stage',stage]+([] if step is None else ['--step',str(step)]);log=HERE/(label+'_EXECUTION.log');atomic(HERE/'CURRENT_STATE.json',dict(status='RUNNING',stage=label,new_physics_steps=0))
  with log.open('x') as f:
   proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);row=dict(stage=label,command=cmd,pid=proc.pid,log=str(log),status='RUNNING',start_epoch_s=time.time());jobs.append(row);atomic(HERE/'JOBS.json',jobs);code=proc.wait()
  row.update(status='COMPLETED' if code==0 else 'FAILED',exit_code=code,end_epoch_s=time.time());atomic(HERE/'JOBS.json',jobs)
  if code:atomic(HERE/'CURRENT_STATE.json',dict(status='EXECUTION_FAILED',stage=label,exit_code=code,no_automatic_retry=True,no_batch_reduction=True));raise RuntimeError('Inspectactualfailure '+label)
 r=read(HERE/'EVALUATION/RESULT.json');v=read(HERE/'EVALUATION/VERIFY.json')
 if not v['integrity_all_pass'] or v['result_sha256']!=sha(HERE/'EVALUATION/RESULT.json'):raise ValueError('Independentfinalaudit required')
 atomic(HERE/'CURRENT_STATE.json',dict(status='FINAL250_ODOR_NEURAL_MEMORY_REPAIR_COMPLETE_NOT_BODY',checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],integrity_all_pass=True,training_updates=250,trainable_parameter_tensors=8,effective_batch=16,microbatch=8,all_jobs_collected=True,new_body_physics_steps=0,source_ALPHA_unchanged=True));print('FIXED250_ODOR_MEMORY_REPAIR_OFFLINE_COMPLETE_NO_PHYSICALCLAIM')
if __name__=='__main__':main()
