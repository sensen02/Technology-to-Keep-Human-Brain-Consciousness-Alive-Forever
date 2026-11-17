"""Parent managed serial10physics/10fresh-replay/summary/review; never rerun finishedtrials."""
from pathlib import Path
import argparse,subprocess,sys,json,time
from runtime import ROOT,IDS,read,sha

def atomic(p,v):
 tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');a=p.parse_args()
 if not a.parent_resource_released:print('CODE_ONLY no resource launch');return
 if (ROOT/'JOBS.json').exists():raise FileExistsError('Inspect partial/finishedjobs, no automatic physical retry')
 stages=[]
 for name in IDS:stages.extend([('PHYSICS',name,'runner.py'),('AUDIT',name,'audit.py')])
 stages.extend([('SUMMARY',None,'summary.py'),('REVIEW',None,'review.py')]);jobs=[]
 for kind,name,entry in stages:
  label=kind if name is None else name+'_'+kind;cmd=[sys.executable,'-B',str(ROOT/entry)]+([] if name is None else ['--condition',name,'--parent-resource-released']);log=ROOT/(label+'.log');atomic(ROOT/'CURRENT_STATE.json',dict(status='RUNNING',stage=label,full_neck_qualified=False))
  with log.open('x') as stream:
   child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT);row=dict(stage=label,command=cmd,pid=child.pid,log=str(log),status='RUNNING',start_epoch_s=time.time());jobs.append(row);atomic(ROOT/'JOBS.json',jobs);exitcode=child.wait()
  row.update(status='COMPLETED' if exitcode==0 else 'FAILED',exit_code=exitcode,end_epoch_s=time.time());atomic(ROOT/'JOBS.json',jobs)
  if exitcode:atomic(ROOT/'CURRENT_STATE.json',dict(status='EXECUTION_FAILED',stage=label,exit_code=exitcode,physical_retries_authorized=False,no_automatic_retry=True));raise RuntimeError('Inspect failed stage; never blindlyrepeat physical '+label)
  if kind=='AUDIT':
   v=read(ROOT/'trials'/name/'VERIFY.json');r=read(ROOT/'trials'/name/'RESULT.json');print(json.dumps(dict(condition=name,integrity=v['integrity_all_pass'],stable=r['functional_stable_body'],slip=r['autonomous']['slip_ratio_max'],yaw_rad=r['yaw_change_rad'])),flush=True)
 final=read(ROOT/'PARENT_READY_REVIEW.json');summary=read(ROOT/'NEURAL_VISUAL_BODY_SUMMARY.json')
 if not final['integrity_all_pass'] or final['summary_sha256']!=sha(ROOT/'NEURAL_VISUAL_BODY_SUMMARY.json'):raise ValueError('Independent actualsummaryreview required')
 atomic(ROOT/'CURRENT_STATE.json',dict(status='ACTUAL_VISUAL_SOURCE_NECK_TENCASE_COMPLETE_METRICS_REPORTED',all_jobs_collected=True,physical_trials=10,normal6_functional_pass=summary['normal6_all_functional_pass'],control2_pass=summary['control2_all_pass'],full_neck_qualified=False,olfaction_connected=False,new_visual_goal_acceptance_requires_parent_review=True));print('TEN_ACTUAL_RGB_NECK_BODY_TRIALS_AND_INDEPENDENT_REPLAYS_COMPLETED_NO_FULLBIOLOGYCLAIM')
if __name__=='__main__':main()
