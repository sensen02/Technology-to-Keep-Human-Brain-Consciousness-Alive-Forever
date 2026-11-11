"""Single LEFT physical adapter. CODEONLY by default; never launches on import.

Reuse the frozen parent's one-condition implementation and lightweight native
PeakRecorder without copying it. Its output root and factory are explicitly
rebound and checked; its sweep/main entry is never called.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import sys
import time
from source_identity import HERE, verify_sources, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-physics', action='store_true', help='Parent-only later sequential physical execution')
    args = parser.parse_args()
    frozen, prereg = verify_sources()
    if not args.run_physics:
        print(json.dumps(dict(status='CODEONLY_READY_NO_PHYSICS', root=str(HERE), conditions=prereg['conditions'], frozen_files=len(frozen['files']))))
        return
    for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(var) != '1':
            raise RuntimeError(f'{var}=1 required before numerical imports')
    allowed = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {1 if 1 in allowed else min(allowed)})
    if (HERE/'left_yaw_p060').exists():
        raise RuntimeError('Candidate output already exists; never overwrite or silently rerun')
    import turning_teacher as finite
    if Path(finite.__file__).resolve() != HERE/'turning_teacher.py':
        raise RuntimeError('Wrong horizontal teacher import')
    finite.pin_baseline()
    legacy_path = HERE.parent/'attempt02'/'diagnose.py'
    spec = importlib.util.spec_from_file_location('_left_finite_parent_diagnostic', legacy_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if mod.make_teacher is not finite.make_teacher:
        raise RuntimeError('Legacy adapter did not import the new finite factory')
    mod.HERE = HERE
    if mod.HERE.resolve() != Path(prereg['root']).resolve():
        raise RuntimeError('Output ROOT mismatch')
    def factory(core, **kwargs):
        teacher = finite.make_teacher(core, **kwargs)
        mod.dump(HERE/'left_yaw_p060'/'REFERENCE.json', dict(rest_body_feet=teacher.reference.base.rest_body_feet.tolist(), reference=teacher.reference.base.metadata, finite_twist=teacher.reference.metadata))
        return teacher
    mod.make_teacher = factory
    mod.sw.seed_all(prereg['seed'])
    mod.dump(HERE/'RUNTIME.json', dict(pid=os.getpid(), cpu_affinity=list(os.sched_getaffinity(0)), python=sys.executable,
        sourcehash_sha256=sha256(HERE/'SOURCEHASH.json'), legacy_runner=str(legacy_path), root=str(HERE),
        environment={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    # Exactly one fixed LEFT. Never invoke mod.main() or its parent sweep.
    mod.one(dict(prereg['conditions'][0]))
    verify_sources()
    result_path = HERE/'left_yaw_p060'/'RESULT.json'
    result = json.loads(result_path.read_text())
    eligible = bool(result['all_gates'] and result['cycles']['all_legs_completed']
                    and result['native']['post_nonfoot_peak_native'] == 0.
                    and result['actual_signed_yaw_rad'] >= prereg['acceptance']['left_actual_yaw_rad_min']
                    and result['early_stop'] is None)
    result.update(teacher_eligible=eligible, sourcehash_sha256=sha256(HERE/'SOURCEHASH.json'),
                  artifact_hashes={name:sha256(HERE/'left_yaw_p060'/name) for name in ('YAW_1MS.npz','REFERENCE.json','NATIVE_MANIFEST.json','INITIAL_RUNTIME.npz','IDENTITY.json')},
                  mapping='finite SE2 horizontal-only', independent_saved_audit='PENDING')
    mod.dump(result_path, result)
    mod.dump(HERE/'CANDIDATE_RESULT.json', dict(condition=prereg['conditions'][0], teacher_eligible=eligible,
        result=str(result_path), result_sha256=sha256(result_path), actual_signed_yaw_deg=result['actual_signed_yaw_deg'],
        slip=result['slip_ratio_max'], native_peak=result['native']['post_nonfoot_peak_native'],
        cycles=result['cycles']['complete_per_leg'], independent_saved_audit='PENDING'))
    print(json.dumps(dict(condition=prereg['conditions'][0], teacher_eligible=eligible, independent_saved_audit='PENDING')))


if __name__ == '__main__':
    main()
