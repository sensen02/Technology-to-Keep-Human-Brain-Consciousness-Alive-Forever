"""Onefinite300 statealignedaction experiment, restored600→900, nooriginalretry."""
import argparse,datetime,json,os,subprocess,sys,time,signal
from sources import ROOT,read,save,sha,verify
PYTHON='/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python'
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-parent-state-resume',action='store_true');a=p.parse_args()
 if not a.execute_parent_state_resume:p.error('Explicitstatealignedresume required')
 if sys.executable!=PYTHON:raise ValueError('Usequalified125712TorchPython')
 b=read(ROOT/'INPUT_BINDINGS.json');pre=read(ROOT/'PREFLIGHT.json');verify(b['source_hashes']);verify(b['data_hashes'])
 if not pre['all_passed'] or pre['timing_status']!='READY_STATE_ALIGNED600_TO900' or pre['source_hashes']!=b['source_hashes'] or pre['data_hashes']!=b['data_hashes']:raise ValueError('Actualstatealignedpreflightrequired')
 path=ROOT/'JOBS.json'
 if path.exists():raise FileExistsError('Nooverwrite/retry')
 jobs=[];pipeline_start=time.monotonic()
 def journal():
  temp=path.with_suffix('.tmp')
  with temp.open('x') as f:json.dump(jobs,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(temp,path)
 journal()
 for stage,script,flag in (('RESUME601_TO900_AND_FRESH600_750_900','train.py','--execute-parent-resume600-to900'),('FINAL900_STATE_ALIGNED_EVALUATION','evaluate.py','--execute-parent-evaluation'),('INDEPENDENT_SAVED_METRICS','audit_eval.py','--execute-parent-saved-audit')):
  verify(b['source_hashes']);verify(b['data_hashes']);command=[sys.executable,'-B',str(ROOT/script),flag];row=dict(stage=stage,status='RUNNING',command=command,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log=str(ROOT/(stage+'.log')));jobs.append(row);journal()
  remaining=7200-(time.monotonic()-pipeline_start)
  if remaining<=0:row.update(status='FAILED',exit_code=124,reason='pipelinewall7200expired');journal();raise RuntimeError('Pipelinebudgetexpired,noextension')
  with open(row['log'],'x') as log:
   process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT),start_new_session=True)
   try:code=process.wait(timeout=remaining)
   except subprocess.TimeoutExpired:
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    process.wait();code=124
  row.update(status='COMPLETED' if code==0 else 'FAILED',exit_code=code,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pipeline_elapsed_s=time.monotonic()-pipeline_start);journal()
  if code!=0:raise RuntimeError('Statealignedstagefailed,noextension')
 tr=read(ROOT/'TRAIN/RESULT.json');v=read(ROOT/'EVALUATION_VERIFY.json')
 if tr['new_updates']!=300 or tr['total_updates']!=900 or not v['integrity_all_pass']:raise ValueError('Actualtotal900qualificationrequired')
 save(ROOT/'STAGE_VERIFICATION.json',dict(kind='STATE_ALIGNED_ACTION_RESUME600_TO900',all_stages_exit0=True,train_updates=900,new_updates=300,checkpoint=tr['checkpoint'],checkpoint_sha256=tr['sha256'],evaluation_sha256=sha(ROOT/'EVALUATION.json'),physical_steps=0,body_behavior_pass=None,new_neck_deployed=False))
if __name__=='__main__':main()
