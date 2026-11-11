"""One-time wider-radius source/input freeze. No overwrite/reanchor option."""
import json
import os
import time
from pathlib import Path
from source_identity_wider import HERE,EXPECTED_ROOT,FREEZE,sha256


def main():
    if HERE!=EXPECTED_ROOT:raise RuntimeError('ROOT mismatch')
    if FREEZE.exists():raise RuntimeError('Strict freeze already exists; never overwrite')
    if (HERE/'left_yaw_p060').exists():raise RuntimeError('Physical output exists; cannot freeze retroactively')
    allowed=os.sched_getaffinity(0);os.sched_setaffinity(0,{1 if 1 in allowed else min(allowed)})
    old=HERE.parent/'left_finite_twist'/'left_yaw_p060'
    identity_path=old/'IDENTITY.json';identity=json.loads(identity_path.read_text())
    # Refuse drift in the inherited executed source/input identity as well.
    inherited_bad=[p for p,info in identity['files'].items() if not Path(p).is_file() or sha256(p)!=info['sha256']]
    if inherited_bad:raise RuntimeError(f'Inherited physical identity drift: {inherited_bad[:8]}')
    paths={Path(p) for p in identity['files']}
    paths.update(HERE.glob('*.py'));paths.add(HERE/'PREREG.json');paths.add(identity_path)
    baseline=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
    paths.update(baseline.glob('*.py'))
    paths.add(HERE.parent/'left_finite_twist'/'turning_teacher.py')
    for name in ('RESULT.json','REFERENCE.json','TRACE_1MS.npz','SAMPLES_5MS.npz','YAW_1MS.npz','NATIVE_MANIFEST.json','VERIFY.json'):
        paths.add(old/name)
    for name in ('diagnose.py','turning_teacher.py','PREREG.json'):
        paths.add(HERE.parent/'attempt02'/name)
    paths.add(Path('/run/media/sensen/Data2/cell_wound_prototype/references/flybody-measured-2025/selected_reference.npz'))
    files={str(p.resolve()):sha256(p) for p in sorted(paths)}
    value=dict(status='STRICT_FROZEN_BEFORE_NEW_PHYSICS',root=str(HERE),created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        inherited_actual_identity=str(identity_path),files=files,
        excludes='new generated reports/audits/logs and all mutable project tracker/head_refit files; old captured VERIFY is read-only input, never a writer destination')
    with FREEZE.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=value['status'],files=len(files),path=str(FREEZE),sha256=sha256(FREEZE))))


if __name__=='__main__':main()
