"""One-time strict source/input freeze for actual guard; stdlib only."""
import json
import time
import os
from pathlib import Path
from guard_identity import HERE,EXPECTED_ROOT,FREEZE,sha256


def main():
    if HERE!=EXPECTED_ROOT:raise RuntimeError('ROOT mismatch')
    if FREEZE.exists() or (HERE/'left_yaw_p060').exists():raise RuntimeError('Existing freeze/output, no reanchoring')
    allowed=os.sched_getaffinity(0);os.sched_setaffinity(0,{1 if 1 in allowed else min(allowed)})
    p=json.loads((HERE/'PREREG.json').read_text());old=Path(p['parent_run']);ident=old/'IDENTITY.json';m=json.loads(ident.read_text())
    bad=[name for name,info in m['files'].items() if not Path(name).is_file() or sha256(name)!=info['sha256']]
    if bad:raise RuntimeError(f'Inherited executed source drift: {bad[:8]}')
    paths={Path(name) for name in m['files']};paths.update(HERE.glob('*.py'));paths.add(HERE/'PREREG.json');paths.add(ident)
    prior=HERE.parent/'left_tarsal_clearance_teacher'/'SOURCEHASH_CLEARANCE_TEACHER.json';paths.add(prior);previous=json.loads(prior.read_text())
    bad=[name for name,h in previous['files'].items() if not Path(name).is_file() or sha256(name)!=h]
    if bad:raise RuntimeError(f'Prior frozen input drift: {bad[:8]}')
    paths.update(Path(name) for name in previous['files'])
    for name in ('TRACE_1MS.npz','RESULT.json','REFERENCE_CLEARANCE_TEACHER.json','CLEARANCE_DIAGNOSTICS.jsonl','NATIVE_MANIFEST.json'):
        paths.add(old/name)
    paths.add(HERE.parent/'left_tarsal_clearance_teacher'/'CANDIDATE_CLEARANCE_TEACHER.json')
    paths.update((HERE.parent/'left_tarsal_clearance_teacher').glob('AUDIT_CLEARANCE_TEACHER_*.json'))
    value=dict(status='STRICT_FROZEN_BEFORE_ACTUAL_GUARD_PHYSICS',root=str(HERE),created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        inherited_actual_identity=str(ident),files={str(name.resolve()):sha256(name) for name in sorted(paths)},
        excludes='Generated new reports/logs/audits/diagnostic outputs and mutable tracker/right-Dagger files; old86 evidence and all old sources immutable')
    with FREEZE.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=value['status'],files=len(value['files']),path=str(FREEZE),sha256=sha256(FREEZE))))


if __name__=='__main__':main()
