"""Exact three missing physical conditions in new recovery scope."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,subprocess,sys,os,json
import evaluate_motor as ev
HERE=Path(__file__).resolve().parent
SHA='3bac5af7aa5058a617383eb9deef4a848778d04ae47f66600ddd53788703344e'

def main():
 p=argparse.ArgumentParser();p.add_argument('--training-complete',action='store_true');p.add_argument('--sha256',required=True);a=p.parse_args()
 if not a.training_complete or a.sha256!=SHA:raise ValueError('explicit fixed10000 authorization required')
 ready=json.loads((HERE/'READINESS.json').read_text())
 if not ready['all_passed']:raise ValueError('recovery preflight required')
 for f,h in ready['source_hashes'].items():
  if ev.sha256(f)!=h:raise ValueError('source drift '+f)
 for f in ev.source_paths():
  if not f.is_file():raise FileNotFoundError(f)
 if (HERE/'FINAL_SUMMARY.json').exists():raise FileExistsError('fresh scope required')
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 results={}
 def one(c):
  out=HERE/('EVAL_'+c);audit=HERE/('VERIFY_'+c+'.json')
  args=[HERE/'evaluate_motor.py','--checkpoint',ev.CHECKPOINT,'--sha256',SHA,'--condition',c,'--out',out,'--training-complete']
  with (HERE/('EVAL_'+c+'.log')).open('x') as log:rc=subprocess.run([sys.executable,'-B',*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
  if rc:return dict(engineering_pass=False,integrity_all_pass=False,execution_returncode=rc)
  with (HERE/('VERIFY_'+c+'.log')).open('x') as log:arc=subprocess.run([sys.executable,'-B',str(HERE/'verify_actual.py'),'--run',str(out),'--out',str(audit)],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
  r=json.loads((out/'RESULT.json').read_text());v={} if arc else json.loads(audit.read_text())
  return dict(engineering_pass=bool(r['engineering_pass'] and arc==0 and v.get('integrity_all_pass')),integrity_all_pass=bool(v.get('integrity_all_pass')),execution_returncode=rc,audit_returncode=arc,result=str(out/'RESULT.json'),autonomous=r['autonomous'],yaw_change_rad=r['yaw_change_rad'],criteria=r['criteria'])
 def persist():
  (HERE/'FINAL_SUMMARY.json').write_text(json.dumps(dict(status='COMPLETE' if len(results)==3 else 'RUNNING',checkpoint=str(ev.CHECKPOINT),checkpoint_sha256=SHA,conditions=results,all_conditions_eligible=len(results)==3 and all(r['engineering_pass'] for r in results.values()),source_freeze=ready),indent=2)+'\n')
 persist();results['neutral']=one('neutral');persist();print(json.dumps(dict(condition='neutral',**results['neutral'])),flush=True)
 if results['neutral']['engineering_pass']:
  with ThreadPoolExecutor(max_workers=2) as pool:
   futures={c:pool.submit(one,c) for c in ('left','right')}
   for c,f in futures.items():results[c]=f.result();persist();print(json.dumps(dict(condition=c,**results[c])),flush=True)
 else:
  for c in ('left','right'):results[c]=dict(engineering_pass=False,integrity_all_pass=False,status='NOT_RUN_NEUTRAL_FAILED')
  persist()
 for f,h in ready['source_hashes'].items():
  if ev.sha256(f)!=h:raise ValueError('source drift '+f)
 if any(r.get('execution_returncode',0)!=0 or r.get('audit_returncode',0)!=0 for r in results.values()):raise SystemExit(1)

if __name__=='__main__':main()
