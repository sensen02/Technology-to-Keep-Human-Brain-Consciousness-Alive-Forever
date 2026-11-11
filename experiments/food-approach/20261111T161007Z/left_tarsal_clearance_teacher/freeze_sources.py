"""Strict pre-run source freeze; stdlib only, no overwrite/reanchoring."""
import json
import time
import os
from pathlib import Path
from candidate_identity import HERE,EXPECTED_ROOT,FREEZE,sha256


def main():
    if HERE!=EXPECTED_ROOT:raise RuntimeError('ROOT mismatch')
    if FREEZE.exists() or (HERE/'left_yaw_p060').exists():raise RuntimeError('Freeze/output exists; never overwrite')
    allowed=os.sched_getaffinity(0);os.sched_setaffinity(0,{1 if 1 in allowed else min(allowed)})
    prereg=json.loads((HERE/'PREREG.json').read_text());old=Path(prereg['parent_run']);ident=old/'IDENTITY.json'
    m=json.loads(ident.read_text())
    bad=[p for p,info in m['files'].items() if not Path(p).is_file() or sha256(p)!=info['sha256']]
    if bad:raise RuntimeError(f'Inherited executed identity drift: {bad[:8]}')
    paths={Path(p) for p in m['files']};paths.update(HERE.glob('*.py'));paths.add(HERE/'PREREG.json');paths.add(ident)
    prior=HERE.parent/'left_tarsal_clearance_probe'/'SOURCEHASH_STATIC_PROBE.json';paths.add(prior)
    pm=json.loads(prior.read_text())
    bad=[p for p,h in pm['files'].items() if not Path(p).is_file() or sha256(p)!=h]
    if bad:raise RuntimeError(f'Static probe frozen-input drift: {bad[:8]}')
    paths.update(Path(p) for p in pm['files']);paths.add(Path(prereg['static_evidence']))
    paths.update((HERE.parent/'left_tarsal_clearance_probe').glob('AUDIT_STATIC_PROBE_*.json'))
    for name in ('TRACE_1MS.npz','RESULT.json','REFERENCE_WIDER_RADIUS.json','NATIVE_MANIFEST.json'):
        paths.add(old/name)
    value=dict(status='STRICT_FROZEN_BEFORE_NEW_TEACHER_PHYSICS',root=str(HERE),created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        inherited_actual_identity=str(ident),static_evidence=prereg['static_evidence'],files={str(p.resolve()):sha256(p) for p in sorted(paths)},
        excludes='Generated new reports/logs/diagnostic outputs/audits and mutable project tracker/bias-candidate files; old outputs remain read-only')
    with FREEZE.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=value['status'],files=len(value['files']),sha256=sha256(FREEZE),path=str(FREEZE))))


if __name__=='__main__':main()
