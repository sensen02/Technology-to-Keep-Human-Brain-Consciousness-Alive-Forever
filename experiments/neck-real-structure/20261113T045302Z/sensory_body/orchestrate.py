"""Parent serial14physical+14independentaudit+summary+review, stops on failure; no training."""
from pathlib import Path
import argparse,json,sys,subprocess,time
ROOT=Path(__file__).resolve().parent

def atomic(p,v):
 q=Path(str(p)+'.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');a=p.parse_args()
 if not a.parent_resource_released:print('CODE_ONLY no bodyexecution');return
 from runtime import IDS,read,sha,SUMMARY_KIND
 if (ROOT/'JOBS.json').exists():raise FileExistsError('Inspect priorstagesbeforeanyrecovery')
 if not read(ROOT/'PREFLIGHT.json')['all_passed']:raise ValueError('ActualzeroCorepreflightneeded')
 jobs=[];stages=[]
 for name in IDS:stages.extend([('PHYSICS',name,'runner.py'),('AUDIT',name,'audit.py')])
 stages.extend([('SUMMARY',None,'summary.py'),('REVIEW',None,'review.py')])
 for stage,condition,script in stages:
  label=stage if condition is None else condition+'_'+stage;cmd=[sys.executable,'-B',str(ROOT/script)]+([] if condition is None else ['--condition',condition,'--parent-resource-released']);log=ROOT/(label+'.log');atomic(ROOT/'CURRENT_STATE.json',dict(status='RUNNING',stage=label,state_total=336,whole_neck_verified=False))
  with log.open('x') as stream:
   p=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT);row=dict(stage=label,command=cmd,pid=p.pid,log=str(log),status='RUNNING',start_epoch_s=time.time());jobs.append(row);atomic(ROOT/'JOBS.json',jobs);exitcode=p.wait()
  row.update(status='COMPLETED' if exitcode==0 else 'FAILED',exit_code=exitcode,end_epoch_s=time.time());atomic(ROOT/'JOBS.json',jobs)
  if exitcode:atomic(ROOT/'CURRENT_STATE.json',dict(status='EXECUTION_FAILED',stage=label,exit_code=exitcode,no_automatic_retry=True));raise RuntimeError('Inspect savedpartialsourcebefore recovery '+label)
 s=read(ROOT/'SENSORY_BODY_SUMMARY.json');r=read(ROOT/'PARENT_READY_REVIEW.json')
 if not r['integrity_all_pass'] or r['summary_sha256']!=sha(ROOT/'SENSORY_BODY_SUMMARY.json') or s['kind']!=SUMMARY_KIND:raise ValueError('Actual14pairedreview needed')
 atomic(ROOT/'CURRENT_STATE.json',dict(status='ACTUAL14_SENSORY_BODY_INTEGRITY_COMPLETE_FUNCTIONAL_RESULTS_SEPARATE',unique_physical_trials=14,all30_stages_exit0=True,state_total=336,whole_neck_verified=False,foodarrival_or_parking=False));print('ACTUAL14_CASES_AND_PAIRED_REVIEW_COMPLETE')
if __name__=='__main__':main()
