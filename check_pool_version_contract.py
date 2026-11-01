"""Hash-bound interface version decision; self-declared candidate flags grant nothing.
The reviewed fingerprint below binds snapshot888 identity and full neural replay,
not physiological calibration, physical standing or autonomous walking.
"""
from pathlib import Path
import json,hashlib,argparse

REVIEWED_CONTRACT_FINGERPRINTS=frozenset({
 '24aeb1d777a43fd77838f95d7dfc793e8fd4b311cf19abedfa09ae8528a5fd2d',
})

def contract_fingerprint(contract):
 return hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()

def file_digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()

def check(contract):
 try:
  inputs=contract['input_sha256']
  changed=[p for p,h in inputs.items() if not Path(p).is_file() or file_digest(p)!=h]
  if changed:return dict(status='INPUT_HASH_CHANGED',changed_inputs=changed,runtime_allowed=False)
  namespaces=contract.get('authoritative_namespaces',{})
  if namespaces.get('annotation') and namespaces.get('connection') and namespaces['annotation']!=namespaces['connection']:return dict(status='IDENTITY_NAMESPACE_CONFLICT',runtime_allowed=False)
  proof=contract.get('hash_bound_namespace_evidence',[])
  if not proof or not namespaces.get('annotation') or not namespaces.get('connection'):return dict(status='VERSION_EVIDENCE_MISSING',runtime_allowed=False,identity_namespace_consistent='NOT_ESTABLISHED',exact_snapshot_consistent='NOT_ESTABLISHED',missing=contract.get('missing_proof',[]))
  fingerprint=contract_fingerprint(contract)
  if fingerprint in REVIEWED_CONTRACT_FINGERPRINTS:
   return dict(status='VERIFIED_SNAPSHOT888_INTERFACE',runtime_allowed=True,identity_namespace_consistent=True,exact_snapshot_consistent=True,reviewed_contract_fingerprint=fingerprint,pool_sha256=contract['pool_sha256'],sensory_lookup_sha256=contract['sensory_lookup_sha256'],input_hashes_verified=len(inputs),validation_scope='Snapshot identity and numerical neural continuity; physical and biological outcomes remain unestablished')
  return dict(status='EVIDENCE_REQUIRES_REVIEW',runtime_allowed=False)
 except (KeyError,TypeError,ValueError,AttributeError,OSError):
  return dict(status='MALFORMED_VERSION_CONTRACT',runtime_allowed=False)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('contract',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=check(json.loads(a.contract.read_text()));a.output.write_text(json.dumps(r,indent=2));print(json.dumps(r));raise SystemExit(0 if r['runtime_allowed'] or r['status']=='VERSION_EVIDENCE_MISSING' else 2)
