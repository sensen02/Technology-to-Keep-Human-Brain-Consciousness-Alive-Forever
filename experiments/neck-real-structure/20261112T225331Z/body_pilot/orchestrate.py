"""Parent-managed serial six-case physics/audits; function failures remain data, no blind retry."""
from pathlib import Path
import argparse,sys,subprocess,json,time,os
ROOT=Path(__file__).resolve().parent
IDS=('P_empty','P_left_pulse','P_right_pulse','P_left_DN_cut','P_left_AN_cut','P_left_cuezero')
def atomic(p,v):
 tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');a=p.parse_args()
 if not a.parent_resource_released:print('CODE_ONLY no physical tasks');return
 if (ROOT/'JOBS.json').exists():raise FileExistsError('Inspect existing child stages; do not rerun completed physics')
 stages=[(name,['--condition',condition,'--parent-resource-released'],condition+'_'+label) for condition in IDS for name,label in [('runner.py','PHYSICS'),('audit.py','AUDIT')]]+[('summary.py',[],'SUMMARY')];jobs=[]
 for script,args,label in stages:
  cmd=[sys.executable,'-B',str(ROOT/script),*args];log=ROOT/(label+'.log');atomic(ROOT/'CURRENT_STATE.json',dict(status='RUNNING',stage=label,parent_pid=os.getpid()))
  with log.open('x') as f:
   process=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);row=dict(label=label,command=cmd,pid=process.pid,log=str(log),status='RUNNING',start_epoch_s=time.time());jobs.append(row);atomic(ROOT/'JOBS.json',jobs);code=process.wait()
  row.update(exit_code=code,status='COMPLETED' if code==0 else 'FAILED',end_epoch_s=time.time());atomic(ROOT/'JOBS.json',jobs)
  if code:
   atomic(ROOT/'CURRENT_STATE.json',dict(status='EXECUTION_FAILED',stage=label,exit_code=code,log=str(log),completed_physics_not_to_be_repeated=True));raise RuntimeError(label+' failed; inspect saved evidence before limited recovery')
  if label.endswith('_AUDIT'):
   condition=label[:-6];r=json.loads((ROOT/'trials'/condition/'RESULT.json').read_text());v=json.loads((ROOT/'trials'/condition/'VERIFY.json').read_text());print(json.dumps(dict(condition=condition,integrity_all_pass=v['integrity_all_pass'],stable_body=r['functional_stable_body'],slip=r['autonomous']['slip_ratio_max'],yaw_rad=r['yaw_change_rad'])),flush=True)
 s=json.loads((ROOT/'BODY_SUMMARY.json').read_text());atomic(ROOT/'CURRENT_STATE.json',dict(status='SIX_ACTUAL_BODY_CASES_AND_AUDITS_COMPLETE',integrity_all6=s['all6_integrity_verified'],stable_body_count=s['stable_body_count'],original_engineering_pass_count=s['original_engineering_pass_count'],all6functionalPASS_claim=False,summary=str(ROOT/'BODY_SUMMARY.json'),all_jobs_collected=True,new_training_updates=0,whole_neck_qualified=False));print(json.dumps(dict(status='COMPLETE_SCOPED_BODY_PILOT',integrity_all6=True,stable_body_count=s['stable_body_count'],original_engineering_pass_count=s['original_engineering_pass_count'])),flush=True)
if __name__=='__main__':main()
