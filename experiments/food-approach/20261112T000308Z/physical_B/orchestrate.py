"""Parent-owned serial six NEW B episodes only. No resident numerical/model modules."""
from pathlib import Path
import argparse,subprocess,sys,json,hashlib,time
ROOT=Path(__file__).resolve().parent
B=('B_noodor','B_equal','B_left_pulse','B_right_pulse','B_left_to_right','B_right_to_left')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic(p,v):
 q=Path(str(p)+'.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');a=p.parse_args()
 if not a.parent_resource_released:print('CODE_ONLY: B6 not launched');return
 binding=read(ROOT/'INPUT_BINDINGS.json');frozen=read(ROOT/'FROZEN.json')
 for path,h in frozen['source_hashes'].items():
  if sha(path)!=h:raise ValueError('prelaunch frozen input changed '+path)
 if not read(ROOT/'PREFLIGHT.json')['all_passed']:raise ValueError('new strict loader preflight required')
 jobs=[]
 def run(script,args,label):
  log=ROOT/(label+'.log');cmd=[sys.executable,'-B',str(ROOT/script),*args]
  with log.open('x') as out:
   proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT);row=dict(command=cmd,pid=proc.pid,label=label,log=str(log),start=time.time(),status='RUNNING');jobs.append(row);atomic(ROOT/'JOBS.json',jobs);code=proc.wait()
  row.update(status='COMPLETED' if code==0 else 'FAILED',exit_code=code,end=time.time());atomic(ROOT/'JOBS.json',jobs)
  if code:raise RuntimeError(label+' failed exit'+str(code)+'; preserve partial output')
 for trial in B:
  folder=ROOT/'trials'/trial
  if folder.exists():
   if not (folder/'RESULT.json').is_file() or not (folder/'VERIFY.json').is_file():raise ValueError('partial trial exists; inspect/recover, no blind rerun '+trial)
  else:run('runner.py',['--trial',trial],trial+'_PHYSICS');run('audit.py',['--trial',trial],trial+'_AUDIT')
  r=read(folder/'RESULT.json');v=read(folder/'VERIFY.json')
  if not v['integrity_all_pass'] or not all(v['checks'].values()) or v['result_sha256']!=sha(folder/'RESULT.json') or r['behavior_checkpoint_sha256']!=binding['behavior_sha256'] or v['behavior_checkpoint_sha256']!=binding['behavior_sha256']:raise ValueError('saved newbehavior trial binding mismatch')
  print(json.dumps(dict(condition=trial,slip=r['autonomous']['slip_ratio_max'],yaw_rad=r['yaw_change_rad'],checks=len(v['checks']))),flush=True)
 if not (ROOT/'B_FUNCTIONAL_SUMMARY.json').exists():run('summarize_B.py',[],'B_FUNCTIONAL_SUMMARY')
 atomic(ROOT/'CURRENT_STATE.json',dict(status='NEW_BEHAVIOR_B6_COMPLETE_AUDITED',A_reused=3,new_B_trials=6,all_jobs_collected=True,behavior_sha256=binding['behavior_sha256'],summary=str(ROOT/'B_FUNCTIONAL_SUMMARY.json'),video_status='PENDING_PARENT',food_success=False))
if __name__=='__main__':main()
