"""Pure-stdlib sequential A gate then B. No resident Torch or physics model."""
from pathlib import Path
import subprocess,sys,os,json,hashlib,time
ROOT=Path(__file__).resolve().parent
A=('A_left_half','A_right_half','A_switch');B=('B_noodor','B_equal','B_left_pulse','B_right_pulse','B_left_to_right','B_right_to_left')
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic(p,obj):
 q=Path(str(p)+'.tmp');q.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');q.replace(p)
def fresh(p,obj):
 with p.open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n')
def main():
 jobs=[]
 def run(script,args,label):
  log=ROOT/(label+'.log');cmd=[sys.executable,'-B',str(ROOT/script),*args];started=time.time()
  with log.open('x') as f:
   proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);row=dict(label=label,command=cmd,pid=proc.pid,log=str(log),started_epoch_s=started,status='RUNNING');jobs.append(row);atomic(ROOT/'JOBS.json',jobs);code=proc.wait()
  row.update(exit_code=code,ended_epoch_s=time.time(),status='COMPLETED' if code==0 else 'FAILED');atomic(ROOT/'JOBS.json',jobs)
  if code:raise RuntimeError(label+' exited '+str(code)+'; see '+str(log))
 def trial(name):
  folder=ROOT/'trials'/name
  if (folder/'RESULT.json').exists() and (folder/'VERIFY.json').exists():
   r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json')
   if not a['integrity_all_pass'] or not all(a['checks'].values()) or sha(folder/'RESULT.json')!=a['result_sha256']:raise ValueError('saved completed trial identity mismatch '+name)
   return r,a
  if folder.exists():raise ValueError('partial trial exists: inspect and recover, do NOT blind rerun '+name)
  atomic(ROOT/'CURRENT_STATE.json',dict(stage=name,status='PHYSICS_RUNNING',owner_pid=os.getpid(),no_parallel_heavy=True))
  run('runner.py',['--trial',name],name+'_PHYSICS')
  run('audit.py',['--trial',name],name+'_AUDIT')
  r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json')
  print(json.dumps(dict(trial=name,phase='COMPLETE_AND_AUDITED',functional=r['functional'],slip=r['autonomous']['slip_ratio_max'],yaw_deg=r['yaw_change_rad']*180/3.141592653589793,integrity=a['integrity_all_pass'],checks=len(a['checks'])),ensure_ascii=False),flush=True)
  return r,a
 if not (ROOT/'FROZEN.json').exists():run('preflight.py',[],'PREFLIGHT')
 rows=[]
 for name in A:
  r,a=trial(name);rows.append(dict(condition=name,functional_pass=r['functional_pass'],integrity_all_pass=a['integrity_all_pass'],result_sha256=sha(ROOT/'trials'/name/'RESULT.json'),audit_sha256=sha(ROOT/'trials'/name/'VERIFY.json')))
 gate=dict(all_A_integrity_and_functional_pass=all(r['functional_pass'] and r['integrity_all_pass'] for r in rows),rows=rows,no_training=True,physics_trials=3,baseline_reused=True,B_executed=False)
 if not (ROOT/'A_GATE.json').exists():fresh(ROOT/'A_GATE.json',gate)
 elif read(ROOT/'A_GATE.json')!=gate:raise ValueError('A gate changed')
 if not gate['all_A_integrity_and_functional_pass']:
  atomic(ROOT/'CURRENT_STATE.json',dict(status='A_FUNCTIONAL_GATE_FAILED_B_NOT_RUN',A_gate=str(ROOT/'A_GATE.json'),B_status='NOT_RUN',next='Diagnose saved A functional failure; no automatic training or hidden gate relaxation.',all_jobs_collected=True))
  print('A_FUNCTIONAL_GATE_FAILED_B_NOT_RUN',flush=True);return
 for name in B:trial(name)
 run('summarize_B.py',[],'B_FUNCTIONAL_SUMMARY')
 atomic(ROOT/'CURRENT_STATE.json',dict(status='A_AND_B_COMPLETE_AUDITED',A_gate=str(ROOT/'A_GATE.json'),B_summary=str(ROOT/'B_FUNCTIONAL_SUMMARY.json'),all_jobs_collected=True,video_status='PENDING_PARENT_RENDER',no_training=True))
if __name__=='__main__':main()
