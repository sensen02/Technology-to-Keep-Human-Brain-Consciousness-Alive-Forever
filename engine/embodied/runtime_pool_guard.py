"""Fail-closed runtime admission; historical compatibility is NOT version certification."""
from pathlib import Path
import hashlib,json
from check_pool_version_contract import check
HISTORICAL_POOL=Path(__file__).resolve().parents[2]/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json'
HISTORICAL_SHA256='3244c8c1470ad3009dd3e12fc64b11f97d1672b8bda811f9ceacf0221e1dd94c'
class RuntimePoolRejected(ValueError):
 def __init__(self,status):self.status=status;super().__init__(status)
def digest(data):return hashlib.sha256(data).hexdigest()
def load_runtime_pool(path=HISTORICAL_POOL,*,version_contract=None):
 raw=Path(path).read_bytes();pool=json.loads(raw)
 if version_contract is not None:
  result=check(version_contract)
  if not result['runtime_allowed']:raise RuntimePoolRejected(result['status'])
 status=pool.get('status');contract=pool.get('contract',{})
 if status=='CANDIDATE_NOT_RUNTIME':raise RuntimePoolRejected('CANDIDATE_NOT_RUNTIME')
 if pool.get('runtime_allowed') is False or contract.get('runtime_allowed') is False:raise RuntimePoolRejected('RUNTIME_EXPLICITLY_DISABLED')
 if status=='VERSION_EVIDENCE_MISSING' or contract.get('status')=='VERSION_EVIDENCE_MISSING':raise RuntimePoolRejected('VERSION_EVIDENCE_MISSING')
 # Trust anchor is outside pool JSON; editing its flags or self-declared hashes grants nothing.
 if digest(raw)!=HISTORICAL_SHA256:raise RuntimePoolRejected('UNTRUSTED_POOL_CONTENT')
 return pool
