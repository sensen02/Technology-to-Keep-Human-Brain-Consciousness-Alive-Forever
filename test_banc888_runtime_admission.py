"""Actual snapshot888 admission contracts; no neural/physical evaluation."""
from pathlib import Path
import copy,json,tempfile,unittest
from engine.embodied.runtime_pool_guard import load_runtime_pool,RuntimePoolRejected,HISTORICAL_POOL
from check_pool_version_contract import check
BASE=Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/20261101T220018Z')
POOL=BASE/'MN_POOLS_SNAPSHOT888.json'
CANDIDATE=Path('/home/sensen/Desktop/engine修复验收/s4-banc888-interface-candidates/20261101T212622Z/mn_pools_root888.candidate.json')
class Tests(unittest.TestCase):
 def setUp(self):self.contract=json.loads((BASE/'VERSION_CONTRACT.json').read_text())
 def reject(self,status,fn):
  with self.assertRaises(RuntimePoolRejected) as cm:fn()
  self.assertEqual(cm.exception.status,status)
 def test_explicit_verified_contract(self):
  self.assertTrue(check(self.contract)['runtime_allowed'])
  pool=load_runtime_pool(POOL,version_contract=self.contract)
  self.assertEqual(pool['measured']['source']['root_id_column'],'root_888')
 def test_no_default_switch(self):
  self.assertEqual(load_runtime_pool(),json.loads(HISTORICAL_POOL.read_text()))
  self.reject('VERSION_CONTRACT_REQUIRED',lambda:load_runtime_pool(POOL))
 def test_old_candidate_still_rejected(self):self.reject('CANDIDATE_NOT_RUNTIME',lambda:load_runtime_pool(CANDIDATE))
 def test_flags_cannot_grant_admission(self):
  with tempfile.TemporaryDirectory(prefix='banc888_admission_') as temporary:
   pool=json.loads(POOL.read_text());pool['forged_extra_field']=True;p=Path(temporary)/'pool.json';p.write_text(json.dumps(pool))
   self.reject('UNTRUSTED_POOL_CONTENT',lambda:load_runtime_pool(p))
   self.reject('CONTRACT_POOL_HASH_MISMATCH',lambda:load_runtime_pool(p,version_contract=self.contract))
 def test_contract_scope_forgery(self):
  forged=copy.deepcopy(self.contract);forged['scope']['physical_behavior']='PASS'
  self.assertEqual(check(forged)['status'],'EVIDENCE_REQUIRES_REVIEW')
 def test_input_mismatch_and_namespace(self):
  bad=copy.deepcopy(self.contract);key=next(iter(bad['input_sha256']));bad['input_sha256'][key]='0'*64
  self.assertEqual(check(bad)['status'],'INPUT_HASH_CHANGED')
  bad=copy.deepcopy(self.contract);bad['authoritative_namespaces']['annotation']='another_snapshot'
  self.assertEqual(check(bad)['status'],'IDENTITY_NAMESPACE_CONFLICT')
 def test_wrong_pool_contract_pair(self):self.reject('CONTRACT_POOL_HASH_MISMATCH',lambda:load_runtime_pool(HISTORICAL_POOL,version_contract=self.contract))
if __name__=='__main__':unittest.main()
