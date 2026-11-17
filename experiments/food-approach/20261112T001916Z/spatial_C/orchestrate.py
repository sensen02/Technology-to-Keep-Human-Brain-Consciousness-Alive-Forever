"""Purestdlib parent-owned C6sequentialphysical+independentaudit; no training."""
from contracts import *
import argparse,subprocess,time,os

def atomic(p,v):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');a=p.parse_args();require(a.parent_resource_released,'parentCscopeandresource release needed');verify();require(read(ROOT/'PREFLIGHT.json')['all_passed'],'realCpreflight needed');require(not (ROOT/'JOBS_C.json').exists(),'inspectexistingjobsbeforeblindrerun');jobs=[]
 def run(script,args,label):
  cmd=[sys.executable,'-B',str(ROOT/script),*args];log=ROOT/(label+'.log');atomic(ROOT/'CURRENT_STATE.json',dict(stage=label,status='RUNNING',physicsbudget=6))
  with log.open('x') as f:
   proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);row=dict(stage=label,command=cmd,pid=proc.pid,log=str(log),status='RUNNING',start_epoch_s=time.time());jobs.append(row);atomic(ROOT/'JOBS_C.json',jobs);code=proc.wait()
  row.update(status='COMPLETED' if code==0 else 'FAILED',exit_code=code,end_epoch_s=time.time());atomic(ROOT/'JOBS_C.json',jobs)
  if code:raise RuntimeError(label+' exited'+str(code)+'; no automaticrerun')
 for cfg in read(ROOT/'PREREG_PROPOSAL.json')['layouts']:
  name=cfg['id'];run('runner.py',['--trial',name],name+'_PHYSICS');run('audit.py',['--trial',name],name+'_AUDIT')
  r=read(ROOT/'trials'/name/'RESULT.json');print(json.dumps(dict(condition=name,stable_body=r['functional']['stable_body'],metrics=r['spatial_metrics'],slip=r['autonomous']['slip_ratio_max'])),flush=True)
 run('summary.py',[],'C_SUMMARY');atomic(ROOT/'CURRENT_STATE.json',dict(status='C6_DEVELOPMENT_COMPLETE',summary=str(ROOT/'C_DEVELOPMENT_SUMMARY.json'),all_jobs_collected=True,no_training=True,not_D_confirmation=True))
if __name__=='__main__':main()
