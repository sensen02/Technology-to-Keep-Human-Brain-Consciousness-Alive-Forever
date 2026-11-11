"""Final source/artifact SHA256 manifest; no physics or training."""
from pathlib import Path
import json
import hashlib
HERE=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    frozen=json.loads((HERE/'INPUT_IDENTITY.json').read_text())
    drift=[p for p,h in frozen['source_hashes'].items() if sha(Path(p))!=h]
    verification=json.loads((HERE/'VERIFICATION.json').read_text())
    artifact_drift=[p for p,h in verification['artifact_sha256'].items() if sha(Path(p))!=h]
    if drift or artifact_drift:raise RuntimeError(dict(input_drift=drift,artifact_drift=artifact_drift))
    alignment=json.loads((HERE/'ALIGNMENT_VERIFICATION.json').read_text())
    if not verification['all_integrity_checks_passed'] or not alignment['all_passed']:raise RuntimeError('verification failed')
    files={str(p):dict(sha256=sha(p),size=p.stat().st_size) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='ARTIFACT_SHA256.json'}
    result=dict(format_version=1,all_frozen_inputs_and_previous_evidence_unchanged=True,
        all_independent_artifacts_unchanged=True,completed_fixed_conditions=2,physical_job='bash-188 exit0 collected',
        independent_replay_job='bash-192 exit0 collected',independent_checks=46,alignment_checks=12,
        scope='Exactly two algorithm teacher cases; BOTH all6cycle FAIL and original forward FAIL retained; no training or food behavior',files=files)
    (HERE/'ARTIFACT_SHA256.json').write_text(json.dumps(result,indent=2)+'\n')
    check=json.loads((HERE/'ARTIFACT_SHA256.json').read_text())
    assert all(sha(Path(p))==old['sha256'] for p,old in check['files'].items())
    print(json.dumps(dict(files_hashed=len(files),manifest_sha256=sha(HERE/'ARTIFACT_SHA256.json'),all_hashes_verified=True)))
if __name__=='__main__':main()
