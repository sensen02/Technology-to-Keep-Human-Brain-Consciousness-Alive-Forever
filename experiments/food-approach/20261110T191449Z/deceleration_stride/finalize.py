"""Final read-only input recheck and new-directory artifact hashing."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    frozen=json.loads((HERE/'INPUT_IDENTITY.json').read_text())
    verification=json.loads((HERE/'VERIFICATION.json').read_text())
    for p,h in frozen['source_hashes'].items():assert sha(p)==h,p
    for p,h in verification['artifact_sha256'].items():assert sha(p)==h,p
    assert verification['all_integrity_checks_passed']
    for file in ('REFERENCE_CHECKS.json','REFERENCE_REPLAY.json'):assert json.loads((HERE/file).read_text())['all_passed']
    files={str(p):dict(sha256=sha(p),size=p.stat().st_size) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='ARTIFACT_SHA256.json'}
    result=dict(format_version=1,all_frozen_sources_previous_teacher_and_completed_deceleration_unchanged=True,
        completed_fixed_conditions=2,physical_job='bash-193 exit0 collected',saved_replay_job='bash-197 exit0 collected',
        forward_only_contact_diagnosis_job='bash-198 exit0 collected',independent_checks=49,preflight_checks=12,offline_reference_checks=10,
        scope='Two fixedcadence stride cases bothall6PASS/bothslipFAIL; no training/food/neuralautonomy; no thirdconfiguration',files=files)
    p=HERE/'ARTIFACT_SHA256.json';p.write_text(json.dumps(result,indent=2)+'\n')
    assert all(sha(p)==v['sha256'] for p,v in result['files'].items())
    print(json.dumps(dict(files_hashed=len(files),all_hashes_verified=True,manifest_sha256=sha(p))))
if __name__=='__main__':main()
