"""Archive hash-bound admission results without neural/body execution."""
from pathlib import Path
import json,hashlib,subprocess,sys,copy
from engine.embodied.runtime_pool_guard import load_runtime_pool,RuntimePoolRejected,HISTORICAL_POOL,HISTORICAL_SHA256
from test_runtime_pool_guard import CANDIDATE
O=Path('/home/sensen/Desktop/engine修复验收/s3-runtime-pool-guard/20261027T203000Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False);P=Path(__file__).parent;runner=P/'run_neural_muscle_closure.py';old=runner.read_text().replace('    from engine.embodied.runtime_pool_guard import load_runtime_pool\n    manifest = build_motor_manifest(load_runtime_pool(POOLS),','    manifest = build_motor_manifest(json.loads(POOLS.read_text()),');oldhash=hashlib.sha256(old.encode()).hexdigest();previous=Path('/home/sensen/Desktop/engine修复验收/s2-solver-step-observation/20261027T202000Z/PREREGISTRATION.json');assert oldhash==json.loads(previous.read_text())['source_hashes'][str(runner)]
 inputs={str(p):sha(p) for p in [HISTORICAL_POOL,CANDIDATE,P/'check_pool_version_contract.py',P/'generate_explicit_identity_pools.py']};assert inputs[str(HISTORICAL_POOL)]==HISTORICAL_SHA256
 dump(O/'PREREGISTRATION.json',{'scope':'S3/P3 runtime admission hard rejection only; no neural/body','trust_anchor':HISTORICAL_SHA256,'inputs_sha256':inputs,'no_self_attested_candidate_approval':True})
 rows=[]
 def record(label,fn):
  try:fn();rows.append({'case':label,'status':'HISTORICAL_COMPATIBILITY_ALLOWED','runtime_allowed':True})
  except RuntimePoolRejected as e:rows.append({'case':label,'status':e.status,'runtime_allowed':False})
 record('historical',lambda:load_runtime_pool());record('actual_candidate',lambda:load_runtime_pool(CANDIDATE));contract={'input_sha256':{str(HISTORICAL_POOL):HISTORICAL_SHA256},'authoritative_namespaces':{},'hash_bound_namespace_evidence':[]}
 record('missing_version_evidence',lambda:load_runtime_pool(version_contract=contract));conflict=copy.deepcopy(contract);conflict['authoritative_namespaces']={'annotation':'A','connection':'B'};record('namespace_conflict',lambda:load_runtime_pool(version_contract=conflict));changed=copy.deepcopy(contract);changed['input_sha256'][str(HISTORICAL_POOL)]='0'*64;record('changed_input_hash',lambda:load_runtime_pool(version_contract=changed))
 for kind in ['forged_candidate','explicit_runtime_false','missing_version_status']:
  pool=json.loads((CANDIDATE if kind=='forged_candidate' else HISTORICAL_POOL).read_text())
  if kind=='forged_candidate':pool['status']='APPROVED';pool['runtime_allowed']=True;pool['contract']['runtime_allowed']=True;pool['contract']['published_release_confirmed']=True
  elif kind=='explicit_runtime_false':pool['runtime_allowed']=False
  else:pool['status']='VERSION_EVIDENCE_MISSING'
  path=O/(kind+'.json');dump(path,pool);record(kind,lambda path=path:load_runtime_pool(path))
 dump(O/'BLOCK_RESULTS.json',{'bound_input_sha256':inputs,'cases':rows});assert rows[0]['runtime_allowed'] and all(not r['runtime_allowed'] for r in rows[1:])
 cmd=[sys.executable,'-m','unittest','test_runtime_pool_guard','-v'];r=subprocess.run(cmd,cwd=P,capture_output=True,text=True);(O/'TEST.log').write_text(r.stdout+r.stderr);assert r.returncode==0
 files=[runner,P/'engine/embodied/runtime_pool_guard.py',P/'test_runtime_pool_guard.py',Path(__file__)];ledger=[]
 for f in files:
  ledger.append({'path':str(f),'old_sha256':oldhash if f==runner else None,'new_sha256':sha(f),'reason':'guard default pool loading' if f==runner else 'new scoped guard/regression/evidence file'});(O/f.name).write_bytes(f.read_bytes())
 dump(O/'CHANGE_LEDGER.json',ledger);assert all(sha(Path(p))==v for p,v in inputs.items());dump(O/'EXECUTION.json',{'command':' '.join(cmd),'exit_code':r.returncode,'earlier_failure':{'command':'python -m unittest test_runtime_pool_guard -v','exit_code':127,'reason':'python alias not installed'},'models':0,'forward':0,'body_step':0,'neural_step':0,'failures':['shell python alias exit127; project interpreter tests pass']});print('runtime guard evidence verified',len(rows),'cases')
if __name__=='__main__':main()
