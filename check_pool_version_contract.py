"""Hash-bound candidate-version decision; missing evidence fails closed, not whole project."""
from pathlib import Path
import json,hashlib,argparse

def check(contract):
 changed=[p for p,h in contract['input_sha256'].items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
 if changed:return dict(status='INPUT_HASH_CHANGED',changed_inputs=changed,runtime_allowed=False)
 namespaces=contract.get('authoritative_namespaces',{})
 if namespaces.get('annotation') and namespaces.get('connection') and namespaces['annotation']!=namespaces['connection']:return dict(status='IDENTITY_NAMESPACE_CONFLICT',runtime_allowed=False)
 proof=contract.get('hash_bound_namespace_evidence',[])
 if not proof or not namespaces.get('annotation') or not namespaces.get('connection'):return dict(status='VERSION_EVIDENCE_MISSING',runtime_allowed=False,identity_namespace_consistent='NOT_ESTABLISHED',exact_snapshot_consistent='NOT_ESTABLISHED',missing=contract.get('missing_proof',[]))
 # No permissive fallback: this bounded checker does not implement an unreviewed approval branch.
 return dict(status='EVIDENCE_REQUIRES_REVIEW',runtime_allowed=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('contract',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=check(json.loads(a.contract.read_text()));a.output.write_text(json.dumps(r,indent=2));print(json.dumps(r));raise SystemExit(0 if r['status']=='VERSION_EVIDENCE_MISSING' else 2)
