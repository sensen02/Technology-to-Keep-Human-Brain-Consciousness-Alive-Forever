"""Three-stage resume300to600 only; no originalscope retry or total1200updates."""
import argparse,datetime,json,os,subprocess,sys
from sources import ROOT,read,save,sha,verify
PYTHON='/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python'
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-parent-order-resume',action='store_true');a=p.parse_args()
 if not a.execute_parent_order_resume:p.error('Explicitorderresume required')
 if sys.executable!=PYTHON:raise ValueError('Usequalified125712Torch Python')
 b=read(ROOT/'INPUT_BINDINGS.json');pre=read(ROOT/'PREFLIGHT.json');verify(b['source_hashes']);verify(b['data_hashes'])
 if not pre['all_passed'] or pre['timing_status']!='READY_RESUME300_TO600' or pre['source_hashes']!=b['source_hashes'] or pre['data_hashes']!=b['data_hashes']:raise ValueError('Actual300fullorderpreflightrequired')
 path=ROOT/'JOBS.json'
 if path.exists():raise FileExistsError('No overwrite/retry')
 jobs=[]
 def journal():
  temp=path.with_suffix('.tmp')
  with temp.open('x') as f:json.dump(jobs,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(temp,path)
 journal()
 for stage,script,flag in (('RESUME301_TO600_AND_FRESH300_450_600','train.py','--execute-parent-resume300-to600'),('FINAL600_EVALUATION','evaluate.py','--execute-parent-evaluation'),('INDEPENDENT_SAVED_METRICS','audit_eval.py','--execute-parent-saved-audit')):
  verify(b['source_hashes']);verify(b['data_hashes']);command=[sys.executable,'-B',str(ROOT/script),flag];row=dict(stage=stage,status='RUNNING',command=command,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log=str(ROOT/(stage+'.log')));jobs.append(row);journal()
  with open(row['log'],'x') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT),check=False)
  row.update(status='COMPLETED' if r.returncode==0 else 'FAILED',exit_code=r.returncode,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());journal()
  if r.returncode!=0:raise RuntimeError('Orderrepair stagefailed, noautomaticextension')
 tr=read(ROOT/'TRAIN/RESULT.json');v=read(ROOT/'EVALUATION_VERIFY.json')
 if tr['new_updates']!=300 or tr['total_updates']!=600 or not v['integrity_all_pass']:raise ValueError('Actualtotal600resumequalificationrequired')
 save(ROOT/'STAGE_VERIFICATION.json',dict(kind='RESUME300_ORDER_REPAIR_TO600',all_stages_exit0=True,train_updates=600,new_updates=300,checkpoint=tr['checkpoint'],checkpoint_sha256=tr['sha256'],evaluation_sha256=sha(ROOT/'EVALUATION.json'),physical_steps=0,body_behavior_pass=None,new_neck_deployed=False))
if __name__=='__main__':main()
