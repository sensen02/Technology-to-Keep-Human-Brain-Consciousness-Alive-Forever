"""Future exactly-sixcase runner. No execution without explicit frozen plan/qualification."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse,json,os,subprocess,sys
from contract import HERE,CASES,validate_launch,digest
from evaluate_holdout import parser

def main():
 p=argparse.ArgumentParser();p.add_argument('--launch-authorized',action='store_true');p.add_argument('--qualification',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True);p.add_argument('--run-plan',required=True);a=p.parse_args()
 from types import SimpleNamespace
 for case in CASES:validate_launch(SimpleNamespace(**vars(a),profile=case,out=str(HERE/('EVAL_'+case))))
 summary=HERE/'QUALIFICATION_SUMMARY.json'
 if summary.exists():raise FileExistsError(summary)
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');results={}
 def persist():
  value=dict(status='COMPLETE' if len(results)==6 else 'RUNNING',checkpoint=a.checkpoint,checkpoint_sha256=a.sha256,conditions=results,all_six_qualified=len(results)==6 and all(v['engineering_pass'] and v['integrity_all_pass'] for v in results.values()),food_qualification=False,run_plan_sha256=digest(a.run_plan),qualification_sha256=digest(a.qualification))
  tmp=summary.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(summary)
 def command(label,argv):
  with (HERE/(label+'.log')).open('x') as f:return subprocess.run([sys.executable,'-B',*map(str,argv)],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 def one(case):
  out=HERE/('EVAL_'+case);audit=HERE/('VERIFY_'+case+'.json')
  argv=[HERE/'evaluate_holdout.py','--launch-authorized','--qualification',a.qualification,'--checkpoint',a.checkpoint,'--sha256',a.sha256,'--run-plan',a.run_plan,'--profile',case,'--out',out]
  rc=command('EVAL_'+case,argv);value=dict(execution_exit_code=rc,engineering_pass=False,integrity_all_pass=False)
  if rc==0:
   ar=command('VERIFY_'+case,[HERE/'verify_saved.py','--run',out,'--out',audit]);r=json.loads((out/'RESULT.json').read_text());value.update(audit_exit_code=ar,engineering_pass=r['engineering_pass'],result=str(out/'RESULT.json'),result_sha256=digest(out/'RESULT.json'))
   if ar==0:value.update(integrity_all_pass=json.loads(audit.read_text())['integrity_all_pass'],audit=str(audit),audit_sha256=digest(audit))
  return value
 persist()
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures={pool.submit(one,c):c for c in CASES}
  for future in as_completed(futures):
   c=futures[future]
   try:results[c]=future.result()
   except BaseException as exc:results[c]=dict(error=repr(exc),engineering_pass=False,integrity_all_pass=False)
   persist();print(json.dumps(dict(profile=c,**results[c])),flush=True)
 if any(v.get('execution_exit_code',1)!=0 or v.get('audit_exit_code',1)!=0 for v in results.values()):raise SystemExit(1)
if __name__=='__main__':main()
