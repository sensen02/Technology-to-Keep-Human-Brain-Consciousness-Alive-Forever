"""One fixed generation/four sequential actualRIGHT candidates, audit before selection.
Pure Python orchestrator: no torch/model imports, no parallel physics.
"""
from pathlib import Path
import argparse,json,hashlib,subprocess,sys,os,time,math
ROOT=Path(__file__).resolve().parent
def sha256(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic(path,value):
 tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def choose(results):
 if len(results)!=4 or sorted(r['index'] for r in results)!=list(range(4)) or not all(r['integrity_all_pass'] for r in results):raise ValueError('all4 audited actual candidates required before any selection')
 passing=[r for r in results if r['engineering_pass']];return None if not passing else min(passing,key=lambda r:(r['slip'],r['index']))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-physics',action='store_true');p.add_argument('--parent-resource-released',action='store_true');args=p.parse_args()
 if not args.run_physics or not args.parent_resource_released:raise ValueError('explicit actual physics+parent resource release required')
 prep=json.loads((ROOT/'PREPARATION.json').read_text());frozen=json.loads((ROOT/'FROZEN.json').read_text())
 if not prep['all_passed'] or len(frozen['candidates'])!=4 or prep['frozen_sha256']!=sha256(ROOT/'FROZEN.json'):raise ValueError('complete all4 pre-actual freeze/preflight required')
 for path,digest in {**prep['source_hashes'],**prep['input_stream_hashes']}.items():
  if sha256(path)!=digest:raise ValueError('preflight source/input drift '+path)
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');started=time.monotonic()
 atomic(ROOT/'RUNTIME.json',dict(pid=os.getpid(),python=sys.executable,start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),mode='sequential4RIGHT_onlyselectedneutral',argv=sys.argv,frozen_sha256=sha256(ROOT/'FROZEN.json')))
 def command(label,argv):
  log=ROOT/(label+'.log');attempt=0
  while log.exists():attempt+=1;log=ROOT/(label+f'.retry{attempt}.log')
  with log.open('x') as stream:rc=subprocess.run([sys.executable,'-B',*map(str,argv)],env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
  with (ROOT/'COMMANDS.jsonl').open('a') as stream:stream.write(json.dumps(dict(argv=[sys.executable,'-B',*map(str,argv)],log=str(log),exit_code=rc))+'\n')
  if rc:raise RuntimeError(label+' exit '+str(rc)+' see '+str(log))
 def one(c,condition):
  for item in frozen['candidates']:
   if sha256(item['checkpoint'])!=item['sha256']:raise ValueError('a frozen candidate changed')
  case=ROOT/c['id'];out=case/('EVAL_'+condition);audit=case/('VERIFY_'+condition+'.json')
  if not (out/'RESULT.json').exists():
   if out.exists():raise RuntimeError('existing incomplete/analysis-failed physical arrays retained; recover without rerunning '+str(out))
   command(c['id']+'_'+condition,[ROOT/'evaluate_motor.py','--candidate',c['id'],'--condition',condition,'--parent-resource-released'])
  if not audit.exists():command(c['id']+'_audit_'+condition,[ROOT/'verify_actual.py','--candidate',c['id'],'--condition',condition])
  r=json.loads((out/'RESULT.json').read_text());a=json.loads(audit.read_text())
  if not a['integrity_all_pass'] or not all(a['checks'].values()) or r['checkpoint_sha256']!=c['sha256'] or a['checkpoint_sha256']!=c['sha256']:raise ValueError('actual/audit/checkpoint integrity mismatch')
  for key,filename in [('trace','trace_1ms.npz'),('neural_decisions','neural_decisions.npz'),('geometry','geometry_1ms.npz'),('runtime_geometry','runtime_geometry.npz')]:
   if sha256(out/filename)!=r[key]['sha256']:raise ValueError('completed artifact drift '+filename)
  flags=r['criteria'];expected=all(v for k,v in flags.items() if k!='eligible')
  if r['engineering_pass']!=expected or flags['eligible']!=expected:raise ValueError('all gates selection mismatch')
  row=dict(id=c['id'],index=c['index'],delta=c['delta'],checkpoint=c['checkpoint'],sha256=c['sha256'],condition=condition,integrity_all_pass=True,engineering_pass=r['engineering_pass'],slip=r['autonomous']['slip_ratio_max'],yaw_rad=r['yaw_change_rad'],yaw_deg=math.degrees(r['yaw_change_rad']),forward=r['autonomous']['forward_signed_native_per_s'],criteria=r['criteria'],cycles=r['autonomous_cycles']['complete_per_leg'],native_nonfoot_peak=r['native']['post_nonfoot_peak_native'],audit_checks=len(a['checks']),result=str(out/'RESULT.json'),result_sha256=sha256(out/'RESULT.json'),audit=str(audit),audit_sha256=sha256(audit))
  print(json.dumps(row),flush=True);return row
 results=[]
 for c in frozen['candidates']:
  results.append(one(c,'right'));atomic(ROOT/'LIVE_PROGRESS.json',dict(completed_RIGHT=len(results),results=results,selected=None,selection_pending_all4=True))
 selected=choose(results);proof=dict(rule='all4 independently audited, minimum(slip,index) among ALL originalgates+all6+strictnativeNF0+signedRIGHTyaw>=30',all4_results=results,selected=selected,frozen_sha256=sha256(ROOT/'FROZEN.json'),proposal_sha256=sha256(ROOT/'PROPOSAL.json'))
 selected_path=ROOT/'SELECTED.json'
 if selected_path.exists():
  if json.loads(selected_path.read_text())!=proof:raise ValueError('immutable existing selection mismatch')
 else:
  with selected_path.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
 neutral=None
 if selected is not None:neutral=one(next(c for c in frozen['candidates'] if c['id']==selected['id']),'neutral')
 summary=dict(status='COMPLETE',actual_RIGHT_rounds=4,results=results,selected=None if selected is None else selected['id'],selected_checkpoint=None if selected is None else selected['checkpoint'],selected_sha256=None if selected is None else selected['sha256'],neutral_confirmation=neutral,selected_all_required_pass=bool(selected is not None and neutral is not None and neutral['engineering_pass']),wall_s=time.monotonic()-started,extra_generations=0,optimizer_training_updates=0)
 atomic(ROOT/'FINAL_SUMMARY.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
