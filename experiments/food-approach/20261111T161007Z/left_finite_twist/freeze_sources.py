"""One-time pre-run source/input freeze; no physics imports.

--replace-codeonly only permits revisions while no physical candidate exists;
previous manifest is retained as source-freeze revision evidence.
"""
import argparse
import json
import os
import time
from pathlib import Path
from source_identity import HERE, EXPECTED_ROOT, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replace-codeonly',action='store_true')
    args=p.parse_args()
    if HERE!=EXPECTED_ROOT:raise RuntimeError('ROOT mismatch')
    if (HERE/'left_yaw_p060').exists():raise RuntimeError('Physical directory exists; source freeze is immutable')
    out=HERE/'SOURCEHASH.json'
    if out.exists() and not args.replace_codeonly:raise RuntimeError('Freeze already exists')
    allowed=os.sched_getaffinity(0);os.sched_setaffinity(0,{1 if 1 in allowed else min(allowed)})
    old_identity=HERE.parent/'attempt02'/'left_yaw_p060'/'IDENTITY.json'
    old=json.loads(old_identity.read_text())
    paths={Path(k) for k in old['files']}
    baseline=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
    paths.update(baseline.glob('*.py'))
    paths.update(HERE.glob('*.py'))
    paths.add(HERE/'PREREG.json')
    paths.update(HERE.parent/'attempt02'/name for name in ('diagnose.py','turning_teacher.py','verify_saved.py','PREREG.json'))
    paths.add(old_identity)
    paths.add(HERE.parent/'attempt02'/'left_yaw_p060'/'TRACE_1MS.npz')
    paths.add(Path('/run/media/sensen/Data2/cell_wound_prototype/references/flybody-measured-2025/selected_reference.npz'))
    files={str(path.resolve()):sha256(path) for path in sorted(paths)}
    if out.exists():
        previous=json.loads(out.read_text())
        archive=HERE/('SOURCEHASH_CODEONLY_SUPERSEDED_'+sha256(out)[:16]+'.json')
        if not archive.exists():archive.write_text(json.dumps(previous,indent=2)+'\n')
    value=dict(status='FROZEN_BEFORE_PHYSICS',root=str(HERE),created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
               inherited_dependency_manifest=str(old_identity),files=files,
               note='All legacy captured source/assets/environment inputs plus baseline modules/new executable sources/prereg/measured reference/actualrest trace; current hashes fixed before new Core. Reports and mutable project tracker excluded.')
    out.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status=value['status'],files=len(files),manifest=str(out),sha256=sha256(out))))


if __name__=='__main__':main()
