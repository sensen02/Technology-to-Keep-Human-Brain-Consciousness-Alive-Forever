"""Exactly four registered ANN calibrations; optional selected N/R confirmation."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import subprocess,sys,os,json
from model_io import ROOT,sha256

def main():
 plan=json.loads((ROOT/'PREREG.json').read_text());prep=json.loads((ROOT/'PREPARATION.json').read_text())
 if not prep['all_passed']:raise ValueError('preparation failed')
 for p,h in prep['source_hashes'].items():
  if sha256(p)!=h:raise ValueError('source changed '+p)
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');results={}
 def command(label,argv):
  with (ROOT/(label+'.log')).open('x') as f:rc=subprocess.run([sys.executable,'-B',*map(str,argv)],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
  if rc:raise RuntimeError(label+' exit '+str(rc))
 def one(cid,condition):
  folder=ROOT/cid;ck=folder/'candidate.pt';out=folder/('EVAL_'+condition);audit=folder/('VERIFY_'+condition+'.json');digest=prep['candidates'][cid]['sha256']
  command(cid+'_'+condition,[ROOT/'evaluate_motor.py','--checkpoint',ck,'--sha256',digest,'--condition',condition,'--out',out,'--training-complete'])
  command(cid+'_audit_'+condition,[ROOT/'verify_actual.py','--run',out,'--out',audit])
  r=json.loads((out/'RESULT.json').read_text());a=json.loads(audit.read_text())
  value=dict(candidate=cid,condition=condition,checkpoint=str(ck),checkpoint_sha256=digest,engineering_pass=bool(r['engineering_pass'] and a['integrity_all_pass']),integrity_all_pass=a['integrity_all_pass'],slip=r['autonomous']['slip_ratio_max'],forward=r['autonomous']['forward_signed_native_per_s'],yaw=r['yaw_change_rad'],gates=r['autonomous']['gates'],cycles=r['autonomous_cycles']['complete_per_leg'],result=str(out/'RESULT.json'),audit=str(audit),audit_checks=len(a['checks']))
  print(json.dumps(value),flush=True);return value
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures={pool.submit(one,c['id'],'left'):c['id'] for c in plan['candidates']}
  for f in as_completed(futures):
   cid=futures[f];results[cid]=f.result();(ROOT/'LEFT_PROGRESS.json').write_text(json.dumps(results,indent=2)+'\n')
 selected=next((c['id'] for c in plan['candidates'] if results[c['id']]['engineering_pass']),None)
 confirmations=[]
 if selected:
  with ThreadPoolExecutor(max_workers=2) as pool:confirmations=list(pool.map(lambda c:one(selected,c),['neutral','right']))
 final=dict(status='COMPLETE',left_results=results,selected=selected,confirmations=confirmations,selected_all_three_pass=bool(selected and all(v['engineering_pass'] for v in confirmations)),scope='finite ANN weight calibration fixed-condition engineering only, no food/native qualification')
 (ROOT/'FINAL_SUMMARY.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final),flush=True)
if __name__=='__main__':main()
