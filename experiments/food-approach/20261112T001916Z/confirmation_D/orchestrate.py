"""Purestdlib sequential27Dtrials/audits; fixedreplaydonor preconditions, no training."""
from contracts import *
import argparse,subprocess,time,os

def atomic(p,v):
 q=Path(str(p)+'.tmp');q.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');q.replace(p)
def fresh(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
def bind_replay(cfg):
 folder=ROOT/'trials'/cfg['donor'];r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json');require(a['integrity_all_pass'] and all(a['checks'].values()) and a['result_sha256']==sha(folder/'RESULT.json'),'fixedreplaydonorintegrity')
 if not r['functional']['stable_body']:
  fresh(ROOT/('REPLAY_NOT_RUN_'+cfg['id']+'.json'),dict(id=cfg['id'],status='NOT_RUN_DONOR_UNSTABLE',donor=cfg['donor'],counts_as_new_closed_loop_success=False));return False
 hashes={str(folder/n):sha(folder/n) for n in ('RESULT.json','VERIFY.json','trace_1ms.npz','neural_decisions.npz','geometry_1ms.npz')}
 fresh(ROOT/('REPLAY_BINDING_'+cfg['id']+'.json'),dict(recipient=cfg['id'],donor=cfg['donor'],donor_result=str(folder/'RESULT.json'),neural_decisions=str(folder/'neural_decisions.npz'),source_hashes=hashes,not_newclosedloop=True));return True

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');a=p.parse_args();require(a.parent_resource_released,'parentrelease');verify();require(read(ROOT/'PREFLIGHT.json')['all_passed'],'actualzeroCorepreflight needed');require(not (ROOT/'JOBS_D.json').exists(),'inspectexistingDjobs; no blindrerun');jobs=[]
 def run(script,args,label):
  cmd=[sys.executable,'-B',str(ROOT/script),*args];log=ROOT/(label+'.log');atomic(ROOT/'CURRENT_STATE.json',dict(status='RUNNING',stage=label,physicsbudget=27))
  with log.open('x') as f:
   proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);row=dict(stage=label,command=cmd,pid=proc.pid,log=str(log),status='RUNNING',start_epoch_s=time.time());jobs.append(row);atomic(ROOT/'JOBS_D.json',jobs);code=proc.wait()
  row.update(status='COMPLETED' if code==0 else 'FAILED',exit_code=code,end_epoch_s=time.time());atomic(ROOT/'JOBS_D.json',jobs)
  if code:raise RuntimeError(label+' exited'+str(code)+'; recover saved evidence, no automaticrerun')
 for cfg in trials():
  if cfg['control']=='replay' and not bind_replay(cfg):continue
  name=cfg['id'];run('runner.py',['--trial',name],name+'_PHYSICS');run('audit.py',['--trial',name],name+'_AUDIT')
  r=read(ROOT/'trials'/name/'RESULT.json');print(json.dumps(dict(condition=name,stable=r['functional']['stable_body'],arrived=r['spatial_metrics']['arrived_endpoint_proxy'],cut_precondition=r['cut_precondition'],slip=r['autonomous']['slip_ratio_max'])),flush=True)
 run('summary.py',[],'D_SUMMARY');atomic(ROOT/'CURRENT_STATE.json',dict(status='D_CONFIRMATION_COMPLETE',summary=str(ROOT/'D_CONFIRMATION_SUMMARY.json'),all_jobs_collected=True,no_training=True))
if __name__=='__main__':main()
