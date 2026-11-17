"""Parent explicitflag serial bounded sevenstage execution, no numericimports bydefault."""
from pathlib import Path
import argparse,subprocess,sys,json,time
HERE=Path(__file__).resolve().parent

def atomic(p,v):
 q=Path(str(p)+'.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-parent-fusion-repair',action='store_true');p.add_argument('--stage',choices=('preflight','train','verify','evaluate','audit'));p.add_argument('--step',type=int);a=p.parse_args()
 if not a.execute_parent_fusion_repair:print('CODE_ONLY no numericalexecution');return
 if a.stage:
  if a.stage=='preflight':from preflight import main as work;work()
  elif a.stage=='train':from train import main as work;work()
  elif a.stage=='verify':
   if a.step not in (0,200,400):raise ValueError('Onlydeclaredsavedsteps')
   from verify_saved import main as work;work(a.step)
  elif a.stage=='evaluate':from evaluate import main as work;work()
  else:from evaluate import audit;audit()
  return
 if (HERE/'JOBS.json').exists():raise FileExistsError('Inspectpriorstageevidencebeforerecovery')
 from common import read,sha,verify,odor_binding
 c=read(HERE/'CODEONLY_CHECK.json')
 if not c['all_passed'] or not all(c['checks'].values()):raise ValueError('Codeonlysourcechecksrequired')
 verify(c['source_hashes']);odor_binding();jobs=[]
 for stage,step in [('preflight',None),('train',None),('verify',0),('verify',200),('verify',400),('evaluate',None),('audit',None)]:
  label=stage.upper()+('' if step is None else '_'+str(step));cmd=[sys.executable,'-B',str(HERE/'run.py'),'--execute-parent-fusion-repair','--stage',stage]+([] if step is None else ['--step',str(step)]);log=HERE/(label+'_EXECUTION.log');atomic(HERE/'CURRENT_STATE.json',dict(status='RUNNING',stage=label,full_neck_training_allowed=False,new_physics_steps=0))
  with log.open('x') as stream:
   process=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT);row=dict(stage=label,command=cmd,pid=process.pid,log=str(log),status='RUNNING',start_epoch_s=time.time());jobs.append(row);atomic(HERE/'JOBS.json',jobs);code=process.wait()
  row.update(status='COMPLETED' if code==0 else 'FAILED',exit_code=code,end_epoch_s=time.time());atomic(HERE/'JOBS.json',jobs)
  if code:atomic(HERE/'CURRENT_STATE.json',dict(status='EXECUTION_FAILED',stage=label,exit_code=code,no_automatic_retry=True,no_batch_reduction=True));raise RuntimeError('Inspectsavedstagefailure '+label)
 r=read(HERE/'EVALUATION/RESULT.json');v=read(HERE/'EVALUATION/VERIFY.json')
 if not v['integrity_all_pass'] or v['result_sha256']!=sha(HERE/'EVALUATION/RESULT.json'):raise ValueError('Finalmetricsindependentauditrequired')
 atomic(HERE/'CURRENT_STATE.json',dict(status='FUSION_AGGREGATION_FIXED400_COMPLETE_NOT_NEW_BODY',checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],odor_sha256=r['odor_sha256'],integrity_all_pass=True,all_jobs_collected=True,training_updates=400,full_neck_training_allowed=False,new_physics_steps=0));print('FIXED400_FUSION_AGGREGATION_COMPLETE_NOT_PHYSICAL_VALIDATION')
if __name__=='__main__':main()
