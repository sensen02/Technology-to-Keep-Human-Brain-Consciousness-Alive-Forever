"""No neural/body imports or execution; admission tests against actual archived pool."""
from pathlib import Path
import copy,json,tempfile,unittest,hashlib
from engine.embodied.runtime_pool_guard import load_runtime_pool,RuntimePoolRejected,HISTORICAL_POOL,HISTORICAL_SHA256
CANDIDATE=Path('/home/sensen/Desktop/engine修复验收/s2-explicit-pool-identity/20261027T172000Z/CANDIDATE_POOLS.json')
class Tests(unittest.TestCase):
 def reject(self,status,fn):
  with self.assertRaises(RuntimePoolRejected) as cm:fn()
  self.assertEqual(cm.exception.status,status)
 def test_history(self):self.assertEqual(load_runtime_pool(),json.loads(HISTORICAL_POOL.read_text()))
 def test_candidate(self):self.reject('CANDIDATE_NOT_RUNTIME',lambda:load_runtime_pool(CANDIDATE))
 def test_flags_and_forgery(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'pool.json';pool=json.loads(CANDIDATE.read_text());pool.pop('status',None);pool['runtime_allowed']=True;pool['contract']['runtime_allowed']=True;pool['contract']['published_release_confirmed']=True;p.write_text(json.dumps(pool));self.reject('UNTRUSTED_POOL_CONTENT',lambda:load_runtime_pool(p))
   historical=json.loads(HISTORICAL_POOL.read_text());historical['runtime_allowed']=False;p.write_text(json.dumps(historical));self.reject('RUNTIME_EXPLICITLY_DISABLED',lambda:load_runtime_pool(p))
   historical['runtime_allowed']=True;historical['status']='VERSION_EVIDENCE_MISSING';p.write_text(json.dumps(historical));self.reject('VERSION_EVIDENCE_MISSING',lambda:load_runtime_pool(p))
 def test_contract_cases(self):
  h=hashlib.sha256(HISTORICAL_POOL.read_bytes()).hexdigest();contract={'input_sha256':{str(HISTORICAL_POOL):h},'authoritative_namespaces':{},'hash_bound_namespace_evidence':[]}
  self.reject('VERSION_EVIDENCE_MISSING',lambda:load_runtime_pool(version_contract=contract))
  conflict=copy.deepcopy(contract);conflict['authoritative_namespaces']={'annotation':'A','connection':'B'};self.reject('IDENTITY_NAMESPACE_CONFLICT',lambda:load_runtime_pool(version_contract=conflict))
  changed=copy.deepcopy(contract);changed['input_sha256'][str(HISTORICAL_POOL)]='0'*64;self.reject('INPUT_HASH_CHANGED',lambda:load_runtime_pool(version_contract=changed))
 def test_default_entry_uses_guard(self):
  import ast
  path=Path(__file__).parent/'run_neural_muscle_closure.py';tree=ast.parse(path.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');calls=[n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='build_motor_manifest'];self.assertEqual(len(calls),1);self.assertIsInstance(calls[0].args[0],ast.Call);self.assertEqual(calls[0].args[0].func.id,'load_runtime_pool')
if __name__=='__main__':unittest.main()
