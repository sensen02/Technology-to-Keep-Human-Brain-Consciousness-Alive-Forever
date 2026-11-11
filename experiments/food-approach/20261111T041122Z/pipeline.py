"""Exactly two fixed half inputs; no further amplitudes or parameter changes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import os,sys,json,subprocess
import model_io as io
ROOT=io.ROOT

def main():
 prep=json.loads((ROOT/'PREPARATION.json').read_text())
 if not prep['all_passed']:raise ValueError('preparation failed')
 for path,h in prep['source_hashes'].items():
  if io.sha256(path)!=h:raise ValueError('source drift '+path)
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');results={}
 def cmd(label,args):
  with (ROOT/(label+'.log')).open('x') as f:rc=subprocess.run([sys.executable,'-B',*map(str,args)],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
  if rc:raise RuntimeError(label+' exit '+str(rc))
 def one(p):
  case=ROOT/p['id'];out=case/('EVAL_'+p['condition']);audit=case/'VERIFY.json'
  cmd(p['id'],[ROOT/'evaluate_motor.py','--checkpoint',io.CHECKPOINT,'--sha256',io.SHA,'--condition',p['condition'],'--out',out,'--training-complete'])
  cmd(p['id']+'_audit',[ROOT/'verify_actual.py','--run',out,'--out',audit])
  r=json.loads((out/'RESULT.json').read_text());a=json.loads(audit.read_text())
  v=dict(profile=p['id'],engineering_pass=r['engineering_pass'],integrity_all_pass=a['integrity_all_pass'],gates=r['autonomous']['gates'],criteria=r['criteria'],slip=r['autonomous']['slip_ratio_max'],forward=r['autonomous']['forward_signed_native_per_s'],yaw=r['yaw_change_rad'],cycles=r['autonomous_cycles']['complete_per_leg'],audit_checks=len(a['checks']),result=str(out/'RESULT.json'),audit=str(audit))
  print(json.dumps(v),flush=True);return v
 with ThreadPoolExecutor(max_workers=2) as pool:
  fs={pool.submit(one,p):p['id'] for p in json.loads((ROOT/'PREREG.json').read_text())['profiles']}
  for f in as_completed(fs):
   results[fs[f]]=f.result();(ROOT/'PROGRESS.json').write_text(json.dumps(results,indent=2)+'\n')
 final=dict(status='COMPLETE',checkpoint=str(io.CHECKPOINT),checkpoint_sha256=io.SHA,profiles=results,both_fixed_profiles_pass=all(r['engineering_pass'] and r['integrity_all_pass'] for r in results.values()),neutral_reference=prep['neutral_reference'],scope='ONLY fixed +/-0.5 and neutral, no continuous range/switch/holdout/native/food qualification')
 (ROOT/'FINAL_SUMMARY.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final),flush=True)
if __name__=='__main__':main()
