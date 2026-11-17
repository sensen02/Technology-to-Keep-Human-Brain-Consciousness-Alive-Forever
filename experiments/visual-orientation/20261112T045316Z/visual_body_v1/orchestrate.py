"""Parent-owned stdlib-only sequential6 visualbody experiments and independent audits."""
from runtime import *
import argparse,subprocess,time

def atomic(p,value):
 tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');a=p.parse_args()
 if not a.parent_resource_released:print('CODE_ONLY no body execution');return
 verify()
 if read(ROOT/'PREFLIGHT.json')['all_passed'] is not True:raise ValueError('actual preflight required')
 if (ROOT/'JOBS.json').exists():raise FileExistsError('inspect prior jobs/artifacts before any continuation')
 jobs=[]
 def run(script,args,label):
  cmd=[sys.executable,'-B',str(ROOT/script),*args,'--parent-resource-released'];log=ROOT/(label+'.log');atomic(ROOT/'CURRENT_STATE.json',dict(status='RUNNING',stage=label,physical_trial_budget=6,no_parallel_heavy=True))
  with log.open('x') as stream:
   child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT);row=dict(label=label,command=cmd,pid=child.pid,log=str(log),status='RUNNING',start_epoch_s=time.time());jobs.append(row);atomic(ROOT/'JOBS.json',jobs);code=child.wait()
  row.update(exit_code=code,end_epoch_s=time.time(),status='COMPLETED' if code==0 else 'FAILED');atomic(ROOT/'JOBS.json',jobs)
  if code:raise RuntimeError(label+' failed; preserve saved trial/no automatic rerun')
 for condition in IDS:
  run('runner.py',['--condition',condition],condition+'_PHYSICS');run('audit.py',['--condition',condition],condition+'_AUDIT');r=read(ROOT/'trials'/condition/'RESULT.json');print(json.dumps(dict(condition=condition,stable_body=r['functional_stable_body'],slip=r['autonomous']['slip_ratio_max'],yaw_rad=r['yaw_change_rad'],captures=r['capture_count'])),flush=True)
 # Summary has no resource flag because it is only reached here after all child audits exit.
 label='SUMMARY';cmd=[sys.executable,'-B',str(ROOT/'summary.py')];log=ROOT/'SUMMARY.log'
 with log.open('x') as stream:
  child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT);row=dict(label=label,command=cmd,pid=child.pid,log=str(log),status='RUNNING',start_epoch_s=time.time());jobs.append(row);atomic(ROOT/'JOBS.json',jobs);code=child.wait()
 row.update(exit_code=code,end_epoch_s=time.time(),status='COMPLETED' if code==0 else 'FAILED');atomic(ROOT/'JOBS.json',jobs)
 if code:raise RuntimeError('SUMMARY failed; recover existing arrays, no physics rerun')
 r=read(ROOT/'VISUAL_BODY_SUMMARY.json');atomic(ROOT/'CURRENT_STATE.json',dict(status='ALL6_PHYSICAL_AUDITS_COMPLETE',all_functional_pass=r['all_functional_pass'],summary=str(ROOT/'VISUAL_BODY_SUMMARY.json'),all_jobs_collected=True,no_training=True))
if __name__=='__main__':main()
