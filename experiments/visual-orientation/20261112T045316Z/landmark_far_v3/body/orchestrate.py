"""Parent-owned eight-trial far bounded validation, exclusive logs and exact final fields."""
from runtime import *
import argparse,subprocess,time,os

def atomic(path,value):
 temporary=Path(str(path)+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');args=p.parse_args()
 if not args.parent_resource_released:print('CODE_ONLY no far physical execution');return
 verify()
 if read(ROOT/'PREFLIGHT.json')['all_passed'] is not True:raise ValueError('actual far zero-Core preflight required')
 if (ROOT/'JOBS.json').exists():raise FileExistsError('inspect saved stages before continuation, no blind rerun')
 jobs=[]
 def run(script,argv,label):
  command=[sys.executable,'-B',str(ROOT/script),*argv];log=ROOT/(label+'.log')
  atomic(ROOT/'CURRENT_STATE.json',dict(status='RUNNING',stage=label,physical_trial_budget=8,scope='FAR_BOUNDED_ONLY',no_parallel_heavy=True))
  with log.open('x') as stream:
   child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT);row=dict(label=label,command=command,pid=child.pid,log=str(log),status='RUNNING',start_epoch_s=time.time());jobs.append(row);atomic(ROOT/'JOBS.json',jobs);code=child.wait()
  row.update(exit_code=code,end_epoch_s=time.time(),status='COMPLETED' if code==0 else 'FAILED');atomic(ROOT/'JOBS.json',jobs)
  if code:
   atomic(ROOT/'CURRENT_STATE.json',dict(status='EXECUTION_FAILED',stage=label,exit_code=code,log=str(log),no_automatic_rerun=True))
   raise RuntimeError(label+' failed; preserve saved arrays and audit, never blindly repeat completed physics')
 for condition in IDS:
  run('runner.py',['--condition',condition,'--parent-resource-released'],condition+'_PHYSICS')
  run('audit.py',['--condition',condition,'--parent-resource-released'],condition+'_AUDIT')
  r=read(ROOT/'trials'/condition/'RESULT.json');a=read(ROOT/'trials'/condition/'VERIFY.json')
  print(json.dumps(dict(condition=condition,stable_body=r['functional_stable_body'],slip=r['autonomous']['slip_ratio_max'],yaw_rad=r['yaw_change_rad'],captures=r['capture_count'],integrity_all_pass=a['integrity_all_pass'])),flush=True)
 run('summary.py',[],'SUMMARY')
 result=read(ROOT/'VISUAL_BODY_SUMMARY.json')
 atomic(ROOT/'CURRENT_STATE.json',dict(status='FAR_ALL8_PHYSICAL_AND_AUDITS_COMPLETE',all_declared_targets_pass=result['all_declared_targets_pass'],normal6_all_functional_pass=result['normal6_all_functional_pass'],control2_all_pass=result['control2_all_pass'],summary=str(ROOT/'VISUAL_BODY_SUMMARY.json'),all_jobs_collected=True,no_training=True,near_domain_repaired=False))
 print(json.dumps(dict(all_declared_targets_pass=result['all_declared_targets_pass'],normal6_all_functional_pass=result['normal6_all_functional_pass'],control2_all_pass=result['control2_all_pass'],all_jobs_collected=True)),flush=True)
if __name__=='__main__':main()
