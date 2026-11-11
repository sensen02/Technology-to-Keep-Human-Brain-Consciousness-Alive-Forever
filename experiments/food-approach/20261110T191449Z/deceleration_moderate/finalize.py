"""Final integrity check/hash manifest, never runs physics or training."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    frozen=json.loads((HERE/'INPUT_IDENTITY.json').read_text());v=json.loads((HERE/'VERIFICATION.json').read_text())
    assert v['all_integrity_checks_passed']
    for p,h in frozen['source_hashes'].items():assert sha(p)==h,p
    for p,h in v['artifact_sha256'].items():assert sha(p)==h,p
    reference=json.loads((HERE/'REFERENCE_CHECKS.json').read_text());assert reference['all_passed'] and all(reference['checks'].values())
    for p,h in reference['source_hashes'].items():assert sha(p)==h,p
    files={str(p):dict(sha256=sha(p),size=p.stat().st_size) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='ARTIFACT_SHA256.json'}
    value=dict(format_version=1,previous_teacher_deceleration_stride_and_frozeninputs_unchanged=True,
        preregistration='AMENDMENT_BEFORE_EXECUTION.json supersedes initialhistoricalreference beforeanyphysics',
        completed_physical_trajectories=3,matched_controls=1,fixed_candidates=2,
        independent_checks=len(v['checks']),preflight_checks=len(reference['checks']),
        control_forward=v['matched_control_forward'],deceleration_maximum_forward=v['deceleration_maximum_forward'],
        functional_passes={n:r['all_functional_gates'] for n,r in v['recomputed'].items()},files=files)
    p=HERE/'ARTIFACT_SHA256.json';p.write_text(json.dumps(value,indent=2)+'\n')
    assert all(sha(p)==r['sha256'] for p,r in value['files'].items())
    print(json.dumps(dict(files_hashed=len(files),all_hashes_verified=True,manifest_sha256=sha(p))))
if __name__=='__main__':main()
